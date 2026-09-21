from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable

from infra.silence_window import SilenceWindow
from service.cadence import CadenceCalculator, parse_iso8601, utc_now
from service.message_builder import FollowupMessageBuilder
from service.telegram_gateway import GatewayError

logger = logging.getLogger(__name__)


def log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str))


class FollowupService:
    """Orquestrador do follow-up automático (U6, FR8).

    Pipeline por lead (conversa mais recente do Contrato 5): passo vencido da
    cadência (FR8.1) → janela de silêncio (FR8.3) → guarda anti-spam (passo já
    enviado / lead respondeu) → contexto resumido (FR8.2, PII-safe) → mensagem
    PT-BR → envio via Telegram → persistir próximo passo. Lead inválido, sem
    contexto, sem canal ou com cadência expirada sem conclusão recebe drop
    explícito com log. Falha de envio/estado não grava o passo (retry no
    próximo tick).
    """

    def __init__(
        self,
        conversations: Any,
        state_store: Any,
        guard: Any,
        cadence: CadenceCalculator,
        silence_window: SilenceWindow,
        builder: FollowupMessageBuilder | None = None,
        telegram: Any | None = None,
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        self.conversations = conversations
        self.state_store = state_store
        self.guard = guard
        self.cadence = cadence
        self.silence_window = silence_window
        self.builder = builder or FollowupMessageBuilder()
        self.telegram = telegram
        self.now_fn = now_fn

    def run(self) -> dict[str, Any]:
        log_event("followup_run_started")
        try:
            conversations = self.conversations.list_conversations()
            profiles = self.conversations.get_lead_profiles()
        except Exception:
            log_event("conversation_read_failed")
            raise
        latest = self._latest_by_lead(conversations)
        window_open = self.silence_window.is_open()
        if not window_open:
            log_event("silence_window_closed", leads=len(latest))
        summary = {
            "leads": len(latest),
            "due": 0,
            "sent": 0,
            "deferred": 0,
            "skipped_already_followed": 0,
            "skipped_replied": 0,
            "dropped": 0,
            "errors": 0,
            "not_due": 0,
        }
        for lead_id, conversation in sorted(latest.items()):
            self._process_lead(
                str(lead_id), conversation, profiles.get(str(lead_id)) or {}, window_open, summary
            )
        log_event("followup_run_completed", **summary)
        return summary

    def _process_lead(
        self,
        lead_id: str,
        conversation: dict[str, Any],
        profile: dict[str, Any],
        window_open: bool,
        summary: dict[str, Any],
    ) -> None:
        created_at = conversation.get("created_at")
        state = self._safe_state(lead_id, summary)
        last_step = int(state.get("last_step", 0)) if state else 0
        if not parse_iso8601(created_at):
            summary["dropped"] += 1
            log_event("lead_dropped", lead_id=lead_id, reason="invalid_created_at")
            return
        if self.cadence.is_expired(created_at):
            if last_step >= self.cadence.last_day:
                summary["not_due"] += 1
            else:
                summary["dropped"] += 1
                log_event("lead_dropped", lead_id=lead_id, reason="cadence_exhausted")
            return
        step = self.cadence.due_step(created_at)
        if step is None:
            summary["not_due"] += 1
            return
        summary["due"] += 1
        if not window_open:
            summary["deferred"] += 1
            log_event("followup_deferred", lead_id=lead_id, step=step, reason="silence_window")
            return
        reason = self.guard.evaluate(lead_id, step, self._last_lead_message_at(conversation))
        if reason:
            key = "skipped_replied" if reason == "lead_replied" else "skipped_already_followed"
            summary[key] += 1
            log_event("followup_suppressed", lead_id=lead_id, step=step, reason=reason)
            return
        context_summary = self._context_summary(conversation, profile, step)
        if context_summary is None:
            summary["dropped"] += 1
            log_event("lead_dropped", lead_id=lead_id, reason="no_context")
            return
        chat_id = profile.get("telegram_user_id")
        if not chat_id:
            summary["dropped"] += 1
            log_event("lead_dropped", lead_id=lead_id, reason="no_channel")
            return
        message = self.builder.build(context_summary)
        try:
            self.telegram.send_message(chat_id, message)
        except GatewayError as exc:
            summary["errors"] += 1
            log_event("followup_send_failed", lead_id=lead_id, step=step, error=str(exc))
            return
        next_step = self.cadence.next_step(step)
        try:
            self.state_store.record_followup(lead_id, step, self.now_fn().isoformat(), next_step)
        except Exception:
            summary["errors"] += 1
            log_event("followup_state_write_failed", lead_id=lead_id, step=step)
            return
        summary["sent"] += 1
        log_event("followup_sent", lead_id=lead_id, step=step, next_step=next_step)

    def _latest_by_lead(self, conversations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for conversation in conversations:
            lead_id = conversation.get("lead_id")
            if not lead_id:
                logger.warning("conversation without lead_id; skipped")
                continue
            current = latest.get(str(lead_id))
            if current is None or self._created_key(conversation) > self._created_key(current):
                latest[str(lead_id)] = conversation
        return latest

    @staticmethod
    def _created_key(conversation: dict[str, Any]) -> tuple[int, str]:
        created = parse_iso8601(conversation.get("created_at"))
        if created is None:
            return (0, str(conversation.get("created_at") or ""))
        return (1, created.isoformat())

    @staticmethod
    def _last_lead_message_at(conversation: dict[str, Any]) -> str | None:
        timestamps = [
            str(message.get("ts"))
            for message in conversation.get("messages") or []
            if message.get("role") == "lead" and message.get("ts")
        ]
        return max(timestamps) if timestamps else None

    @staticmethod
    def _context_summary(
        conversation: dict[str, Any], profile: dict[str, Any], step: int
    ) -> dict[str, Any] | None:
        """Resumo PII-safe: apenas passo, intenção, estado e volume de mensagens."""
        messages = conversation.get("messages") or []
        context = conversation.get("context") or {}
        intent = profile.get("intent") or context.get("intent")
        if not messages and not intent:
            return None
        return {
            "step": step,
            "intent": intent,
            "state": conversation.get("current_state"),
            "message_count": len(messages),
        }

    def _safe_state(self, lead_id: str, summary: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return self.state_store.get_state(lead_id)
        except Exception as exc:
            summary["errors"] += 1
            log_event("followup_state_read_failed", lead_id=lead_id, error=str(exc))
            return None
