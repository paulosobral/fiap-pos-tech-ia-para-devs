import json
from datetime import datetime, timezone
from typing import Any, Callable

from logs import log_event
from service.kpi_calc import latest_conversation_by_lead, parse_iso8601
from service.kpis import utc_now

LEADS_LIMIT = 200
_UNNAMED = "Lead (nome não informado)"


class LeadNotFoundError(Exception):
    pass


def _first(contact: dict[str, list[str]], label: str) -> str | None:
    values = contact.get(label) or []
    return values[0] if values else None


def _text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


class LeadService:
    """Lista de leads do backoffice (nome + contato + qualificação) e reenvio ao CRM.

    O contato vem do registro de PII decifrado (KMS) — exposto só a usuário autenticado no
    Cognito. O reenvio publica na MESMA fila do handoff (Contrato 4), então o crm-adapter
    trata o botão do dashboard e a transição automática do mesmo jeito.
    """

    def __init__(
        self,
        conversations: Any,
        pii: Any,
        sqs_client: Any,
        crm_queue_url: str,
        now_fn: Callable[[], Any] = utc_now,
    ) -> None:
        self._conversations = conversations
        self._pii = pii
        self._sqs = sqs_client
        self._queue = crm_queue_url
        self._now = now_fn

    def list_leads(self) -> list[dict[str, Any]]:
        profiles = self._conversations.get_lead_profiles()
        latest = latest_conversation_by_lead(self._conversations.list_conversations())
        rows = [self._row(lead_id, profile, latest.get(lead_id)) for lead_id, profile in profiles.items()]
        rows.sort(key=lambda r: parse_iso8601(r["updated_at"]) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        log_event("leads_listed", count=len(rows))
        return rows[:LEADS_LIMIT]

    def send_to_crm(self, lead_id: str) -> dict[str, Any]:
        profiles = self._conversations.get_lead_profiles()
        profile = profiles.get(lead_id)
        if profile is None:
            raise LeadNotFoundError(lead_id)
        conversation = latest_conversation_by_lead(self._conversations.list_conversations()).get(lead_id)
        if conversation is None or not conversation.get("session_id"):
            raise LeadNotFoundError(lead_id)
        row = self._row(lead_id, profile, conversation)
        message = {
            "message_id": f"{conversation['session_id']}-dashboard-{int(self._now().timestamp())}",
            "lead_id": lead_id,
            "session_id": conversation["session_id"],
            "timestamp": self._now().isoformat(),
            "lead_data": {
                "name": row["name"],
                "email": row["email"],
                "phone": row["phone"],
                "score": row["score"],
                "urgency": row["urgency"],
                "intent": row["intent"],
                "budget": row["budget"],
                "deadline": row["deadline"],
                "area": row["area"],
            },
        }
        self._sqs.send_message(QueueUrl=self._queue, MessageBody=json.dumps(message))
        log_event("lead_sent_to_crm", lead_id=lead_id)
        return {"lead_id": lead_id, "queued": True}

    def _row(self, lead_id: str, profile: dict[str, Any], conversation: dict[str, Any] | None) -> dict[str, Any]:
        session_id = (conversation or {}).get("session_id")
        contact = self._pii.load(str(session_id)) if session_id else {}
        return {
            "lead_id": lead_id,
            "name": _first(contact, "NOME") or _UNNAMED,
            "email": _first(contact, "EMAIL"),
            "phone": _first(contact, "TELEFONE"),
            "score": profile.get("score"),
            "urgency": _text(profile.get("urgency")),
            "intent": _text(profile.get("intent")),
            "budget": _text(profile.get("budget")),
            "area": _text(profile.get("area")),
            "region": _text(profile.get("region")),
            "deadline": _text(profile.get("deadline")),
            "state": _text((conversation or {}).get("current_state")),
            "created_at": _text(profile.get("created_at")),
            "updated_at": _text(profile.get("updated_at")) or _text(profile.get("created_at")),
        }
