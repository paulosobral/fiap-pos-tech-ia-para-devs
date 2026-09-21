from __future__ import annotations

import json
import logging
from typing import Any

from service.crm_gateway import CrmError, CrmGateway
from service.flow_gateway import FlowError, FlowGateway
from service.status_sync import StatusSync

logger = logging.getLogger(__name__)

OUTCOME_OK = "ok"
OUTCOME_RETRY = "retry"
OUTCOME_DROP = "drop"

REQUIRED_FIELDS = ("message_id", "lead_id", "lead_data", "session_id", "timestamp")
REQUIRED_LEAD_FIELDS = ("name", "email", "phone", "urgency", "intent")

DEFAULT_MAX_RECEIVES = 3


def log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str))


class CrmAdapter:
    """Consumidor SQS do Contract 4: lead → CRM (via gateway) → esteira Kanban → fluxo.

    DLQ-friendly: `retry` → `batchItemFailures` (redrive SQS → DLQ, Contract 4);
    `drop` explícito para schema inválido / lead desconhecido / política de
    tentativas esgotada; o lote nunca quebra (NFR4.1).
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
            log_event("invalid_body", body_preview=str(record.get("body"))[:120])
            return OUTCOME_DROP
        try:
            return self.process_message(message)
        except Exception as exc:
            logger.error("unexpected failure: %s", exc)
            return OUTCOME_RETRY

    def process_message(self, message: Any) -> str:
        error = self._validate(message)
        if error:
            log_event("message_rejected", reason=error)
            return OUTCOME_DROP
        lead_id = message["lead_id"]
        session_id = message["session_id"]
        if self.sessions is not None and self.sessions.get_session(session_id, lead_id) is None:
            log_event("lead_unknown", session_id=session_id, lead_id=lead_id)
            return OUTCOME_DROP
        lead_data = message["lead_data"]
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
                }
            )
        except CrmError as exc:
            log_event("crm_unreachable", session_id=session_id, lead_id=lead_id, error=str(exc))
            return OUTCOME_RETRY
        try:
            result = self.status.sync(lead_id, session_id, lead_data)
        except CrmError as exc:
            log_event("stage_sync_failed", session_id=session_id, lead_id=lead_id, error=str(exc))
            return OUTCOME_RETRY
        except FlowError as exc:
            log_event("flow_notify_failed", session_id=session_id, lead_id=lead_id, error=str(exc))
            return OUTCOME_RETRY
        log_event(
            "lead_synced",
            session_id=session_id,
            lead_id=lead_id,
            message_id=message["message_id"],
            crm_id=record.get("crm_id"),
            stage=result["stage"],
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
        score = lead_data.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return "lead_data.score must be a number"
        return None

    @staticmethod
    def _receive_count(record: dict[str, Any]) -> int:
        raw = (record.get("attributes") or {}).get("ApproximateReceiveCount")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0