import json
from datetime import datetime, timezone
from typing import Any, Callable

from logs import log_event
from service.kpi_calc import latest_conversation_by_lead, parse_iso8601
from service.kpis import utc_now

LEADS_LIMIT = 200
_UNNAMED = "Lead (nome não informado)"


def format_price(price: Any, mode: Any = None, fallback: Any = None) -> str | None:
    """Preço exato em reais, no formato brasileiro: R$ 850.000 ou R$ 4.500/mês. O `price_text` do catálogo
    é arredondado ("R$ 0.8 milhão"), por isso só serve de plano B quando não há preço numérico."""
    try:
        value = float(price)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0:
        return str(fallback) if fallback not in (None, "") else None
    text = f"R$ {value:,.0f}".replace(",", ".")
    return f"{text}/mês" if str(mode or "").lower() == "rent" else text


def chosen_property_price(context: dict[str, Any]) -> str | None:
    """Valor do imóvel que o lead escolheu: procura o favorito entre os imóveis já exibidos na conversa."""
    title = context.get("favorite_property")
    if not title:
        return None
    for prop in context.get("properties") or []:
        if isinstance(prop, dict) and prop.get("title") == title:
            return format_price(prop.get("price"), prop.get("mode"), prop.get("price_text"))
    return None


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
                "region": row["region"],
                "property": row["property"],
                "property_price": row["property_price"],
            },
        }
        self._sqs.send_message(QueueUrl=self._queue, MessageBody=json.dumps(message))
        log_event("lead_sent_to_crm", lead_id=lead_id)
        return {"lead_id": lead_id, "queued": True}

    def _row(self, lead_id: str, profile: dict[str, Any], conversation: dict[str, Any] | None) -> dict[str, Any]:
        session_id = (conversation or {}).get("session_id")
        contact = self._pii.load(str(session_id)) if session_id else {}
        context = (conversation or {}).get("context")
        context = context if isinstance(context, dict) else {}
        info = context.get("lead_info") if isinstance(context.get("lead_info"), dict) else {}

        def field(key: str) -> str | None:
            # O perfil só é completado no envio ao CRM; antes disso o dado vive na conversa.
            return _text(profile.get(key)) or _text(info.get(key))

        budget = field("budget") or (f"a partir de {_text(info.get('budget_min'))}" if info.get("budget_min") else None)
        return {
            "lead_id": lead_id,
            "name": _first(contact, "NOME") or _UNNAMED,
            "email": _first(contact, "EMAIL"),
            "phone": _first(contact, "TELEFONE"),
            "score": profile.get("score"),
            "urgency": _text(profile.get("urgency")),
            "intent": field("intent"),
            "budget": budget,
            "area": field("area"),
            "region": field("region"),
            "deadline": field("deadline"),
            "property": _text(context.get("favorite_property")),
            "property_price": chosen_property_price(context),
            "state": _text((conversation or {}).get("current_state")),
            "created_at": _text(profile.get("created_at")),
            "updated_at": _text(profile.get("updated_at")) or _text(profile.get("created_at")),
        }
