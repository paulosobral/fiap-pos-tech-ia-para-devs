from __future__ import annotations

import json
import logging
from typing import Any

from infra.dedupe_store import DedupeError
from infra.session_store import SessionError
from service.email_parser import EmailParser
from service.router_gateway import RouterError, RouterGateway

logger = logging.getLogger(__name__)

OUTCOME_OK = "ok"
OUTCOME_RETRY = "retry"
OUTCOME_DROP = "drop"

OUTCOMES = (OUTCOME_OK, OUTCOME_RETRY, OUTCOME_DROP)


class PendingRetryError(Exception):
    """Sinaliza falha transitória ao Lambda (async retry) — nunca para e-mail inválido."""


def log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str))


class ContactIngest:
    """Ingestão de contato assíncrona (U4): SES → sessão → ConversationRouter.

    Fluxo por e-mail (FR12): parse heurístico → dedupe (put condicional,
    anti spam/duplicata) → abrir sessão (Contrato 5) → re-injetar a 1ª
    mensagem no router como se o lead a tivesse digitado. Outcomes:
    `ok`, `retry` (falha transitória; rollback da marca de dedupe permite o
    reprocesso idempotente) e `drop` explícito (e-mail não parseável /
    duplicata) — e-mail inválido nunca derruba o evento (NFR4.1).
    """

    def __init__(
        self,
        parser: EmailParser,
        dedupe: Any,
        sessions: Any,
        router: RouterGateway,
    ) -> None:
        self.parser = parser
        self.dedupe = dedupe
        self.sessions = sessions
        self.router = router

    def handle_event(self, event: dict[str, Any]) -> dict[str, Any]:
        results: list[dict[str, str]] = []
        counts = {outcome: 0 for outcome in OUTCOMES}
        for record in event.get("Records", []):
            mail = (record.get("ses") or {}).get("mail") or {}
            identifier = str(mail.get("messageId") or "")
            outcome = self.process_mail(mail)
            counts[outcome] += 1
            results.append({"message_id": identifier, "outcome": outcome})
            log_event("record_processed", message_id=identifier, outcome=outcome)
        log_event("event_processed", **counts)
        return {"results": results, **counts}

    def process_mail(self, mail: Any) -> str:
        if not isinstance(mail, dict) or not mail:
            log_event("missing_ses_mail")
            return OUTCOME_DROP
        try:
            return self._ingest(mail)
        except Exception as exc:
            logger.error("unexpected failure: %s", exc)
            return OUTCOME_RETRY

    def _ingest(self, mail: dict[str, Any]) -> str:
        message_id = str(mail.get("messageId") or "")
        parsed = self.parser.parse(mail)
        if parsed is None:
            log_event("unparseable_email", message_id=message_id)
            return OUTCOME_DROP
        try:
            first = self.dedupe.put_first(message_id, self._source_domain(mail))
        except DedupeError as exc:
            log_event("dedupe_unavailable", message_id=message_id, error=str(exc))
            return OUTCOME_RETRY
        if not first:
            log_event("duplicate_suppressed", message_id=message_id)
            return OUTCOME_DROP
        try:
            session = self.sessions.open_session(
                {"name": parsed.name, "email": parsed.email, "phone": parsed.phone}
            )
        except SessionError as exc:
            self._rollback(message_id)
            log_event("session_open_failed", message_id=message_id, error=str(exc))
            return OUTCOME_RETRY
        try:
            self.router.reinject(
                telegram_user_id=session["telegram_user_id"],
                session_id=session["session_id"],
                text=parsed.message_text,
            )
        except RouterError as exc:
            self._rollback(message_id)
            log_event(
                "router_reinject_failed",
                message_id=message_id,
                session_id=session["session_id"],
                error=str(exc),
            )
            return OUTCOME_RETRY
        log_event(
            "contact_ingested",
            message_id=message_id,
            session_id=session["session_id"],
            lead_id=session["lead_id"],
        )
        return OUTCOME_OK

    def _rollback(self, message_id: str) -> None:
        """Remove a marca de dedupe (best-effort) para permitir o reprocesso."""
        try:
            self.dedupe.delete(message_id)
        except Exception as exc:
            logger.error("dedupe rollback failed: %s", exc)

    @staticmethod
    def _source_domain(mail: dict[str, Any]) -> str | None:
        """Domínio do remetente (PII-safe): o endereço completo nunca é gravado."""
        source = str(mail.get("source") or "")
        if "@" not in source:
            return None
        return source.rsplit("@", 1)[-1].strip().lower() or None
