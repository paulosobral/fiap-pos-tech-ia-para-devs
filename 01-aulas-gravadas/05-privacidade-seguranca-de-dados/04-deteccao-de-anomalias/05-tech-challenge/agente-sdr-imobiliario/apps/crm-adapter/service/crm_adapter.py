from __future__ import annotations

import json
import logging
from typing import Any

from service.crm_gateway import CrmError, CrmGateway
from service.flow_gateway import FlowError, FlowGateway
from service.pii_mask import mask_pii
from service.status_sync import StatusSync

logger = logging.getLogger(__name__)

OUTCOME_OK = "ok"
OUTCOME_RETRY = "retry"
OUTCOME_DROP = "drop"

REQUIRED_FIELDS = ("message_id", "lead_id", "lead_data", "session_id", "timestamp")
# Forma real do produtor (Contract 4 atualizado pela u1): `name` (de contact["NOME"],
# com fallback seguro) e `urgency` (low/medium/high) vêm sempre; `email`/`phone`
# podem vir ausentes/null (registro de PII sem contato) e `score` pode ser null.
REQUIRED_LEAD_FIELDS = ("name", "urgency")
OPTIONAL_LEAD_FIELDS = ("email", "phone", "intent", "budget", "deadline", "area")

DEFAULT_MAX_RECEIVES = 3


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, default=str))


class CrmAdapter:
    """Consumidor SQS do Contract 4: lead → CRM (via gateway) → esteira Kanban → fluxo.

    DLQ-friendly: `retry` → `batchItemFailures` (redrive SQS → DLQ, Contract 4);
    `drop` explícito para schema inválido / lead desconhecido / política de
    tentativas esgotada; o lote nunca quebra (NFR4.1). Msg sem email/phone é
    registrada com campos vazios e flag `contact_info_partial` no log — nunca descartada.
    """

    def __init__(
        self,
        crm: CrmGateway,
        status: StatusSync,
        flow: FlowGateway,
        sessions: Any | None = None,
        max_receives: int = DEFAULT_MAX_RECEIVES,
    ) -> None:
        self.crm = crm
        self.status = status
        self.flow = flow
        self.sessions = sessions
        self.max_receives = max_receives

    def handle_event(self, event: dict[str, Any]) -> dict[str, Any]:
        failures: list[dict[str, str]] = []
        for record in event.get("Records", []):
            identifier = record.get("messageId", "")
            outcome = self._process_record(record)
            if outcome == OUTCOME_RETRY and self._receive_count(record) >= self.max_receives:
                log_event(
                    "retry_policy_exhausted",
                    message_id=identifier,
                    receives=self._receive_count(record),
                )
                outcome = OUTCOME_DROP
            log_event("record_processed", message_id=identifier, outcome=outcome)
            if outcome == OUTCOME_RETRY:
                failures.append({"itemIdentifier": identifier})
        return {"batchItemFailures": failures}

    def _process_record(self, record: dict[str, Any]) -> str:
        try:
            message = json.loads(record.get("body") or "{}")
        except json.JSONDecodeError:
            body = str(record.get("body") or "")
            # Preview do body cru pode conter PII do Contract 4 — mascarado com o
            # padrão oficial da u1 (NFR2.1) e truncado antes de ir ao CloudWatch.
            log_event("invalid_body", body_length=len(body), body_preview=mask_pii(body)[:120])
            return OUTCOME_DROP
        try:
            return self.process_message(message)
        except Exception as exc:
            log_event("unexpected_failure", level=logging.ERROR, error=str(exc))
            return OUTCOME_RETRY

    def process_message(self, message: Any) -> str:
        error = self._validate(message)
        if error:
            log_event("message_rejected", reason=error)
            return OUTCOME_DROP
        lead_id = message["lead_id"]
        session_id = message["session_id"]
        if self.sessions is not None and self.sessions.get_session(lead_id, session_id) is None:
            log_event("lead_unknown", session_id=session_id, lead_id=lead_id)
            return OUTCOME_DROP
        lead_data = message["lead_data"]
        contact_partial = not (lead_data.get("email") and lead_data.get("phone"))
        try:
            record = self.crm.upsert_lead(
                {
                    "lead_id": lead_id,
                    "session_id": session_id,
                    "name": lead_data.get("name"),
                    "email": lead_data.get("email"),
                    "phone": lead_data.get("phone"),
                    "score": lead_data.get("score"),
                    "urgency": lead_data.get("urgency"),
                    "intent": lead_data.get("intent"),
                    "budget": lead_data.get("budget"),
                    "deadline": lead_data.get("deadline"),
                    "area": lead_data.get("area"),
                }
            )
        except CrmError as exc:
            log_event(
                "crm_unreachable",
                level=logging.ERROR,
                session_id=session_id,
                lead_id=lead_id,
                error=str(exc),
            )
            return OUTCOME_RETRY
        try:
            result = self.status.sync(lead_id, session_id, lead_data)
        except CrmError as exc:
            log_event(
                "stage_sync_failed",
                level=logging.ERROR,
                session_id=session_id,
                lead_id=lead_id,
                error=str(exc),
            )
            return OUTCOME_RETRY
        except FlowError as exc:
            log_event(
                "flow_notify_failed",
                level=logging.ERROR,
                session_id=session_id,
                lead_id=lead_id,
                error=str(exc),
            )
            return OUTCOME_RETRY
        log_event(
            "lead_synced",
            session_id=session_id,
            lead_id=lead_id,
            message_id=message["message_id"],
            crm_id=record.get("crm_id"),
            stage=result["stage"],
            contact_info_partial=contact_partial,
        )
        return OUTCOME_OK

    @staticmethod
    def _validate(message: Any) -> str | None:
        if not isinstance(message, dict):
            return "payload is not an object"
        for field in REQUIRED_FIELDS:
            if not message.get(field):
                return f"missing field {field}"
        lead_data = message.get("lead_data")
        if not isinstance(lead_data, dict):
            return "lead_data must be an object"
        for field in REQUIRED_LEAD_FIELDS:
            value = lead_data.get(field)
            if not isinstance(value, str) or not value.strip():
                return f"lead_data.{field} must be a non-empty string"
        for field in OPTIONAL_LEAD_FIELDS:
            value = lead_data.get(field)
            if value is not None and not isinstance(value, str):
                return f"lead_data.{field} must be a string or null"
        score = lead_data.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
            return "lead_data.score must be a number or null"
        return None

    @staticmethod
    def _receive_count(record: dict[str, Any]) -> int:
        raw = (record.get("attributes") or {}).get("ApproximateReceiveCount")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0
