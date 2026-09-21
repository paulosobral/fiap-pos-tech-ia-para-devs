from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def conversation_ttl_seconds() -> int:
    return 90 * 24 * 60 * 60


class Lead:
    def __init__(
        self,
        lead_id: str,
        telegram_user_id: int,
        status: str = "new",
        score: float | None = None,
        urgency: str | None = None,
        intent: str | None = None,
        budget: str | None = None,
        area: str | None = None,
        region: str | None = None,
        deadline: str | None = None,
        people_count: int | None = None,
        decision_maker: str | None = None,
        route: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        self.lead_id = lead_id
        self.telegram_user_id = telegram_user_id
        self.status = status
        self.score = score
        self.urgency = urgency
        self.intent = intent
        self.budget = budget
        self.area = area
        self.region = region
        self.deadline = deadline
        self.people_count = people_count
        self.decision_maker = decision_maker
        self.route = route
        self.created_at = created_at or utc_now_iso()
        self.updated_at = updated_at or utc_now_iso()

    def to_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "lead_id": self.lead_id,
            "telegram_user_id": self.telegram_user_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": utc_now_iso(),
        }
        for attr in (
            "score",
            "urgency",
            "intent",
            "budget",
            "area",
            "region",
            "deadline",
            "people_count",
            "decision_maker",
            "route",
        ):
            value = getattr(self, attr)
            if value is not None:
                item[attr] = value
        return item

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> Lead:
        return cls(
            lead_id=item["lead_id"],
            telegram_user_id=item["telegram_user_id"],
            status=item.get("status", "new"),
            score=item.get("score"),
            urgency=item.get("urgency"),
            intent=item.get("intent"),
            budget=item.get("budget"),
            area=item.get("area"),
            region=item.get("region"),
            deadline=item.get("deadline"),
            people_count=item.get("people_count"),
            decision_maker=item.get("decision_maker"),
            route=item.get("route"),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
        )


class Conversation:
    def __init__(
        self,
        session_id: str,
        lead_id: str,
        messages: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        current_state: str = "greeting",
        pii_masked: bool = False,
        consent_recorded: bool = False,
        created_at: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.lead_id = lead_id
        self.messages = messages if messages is not None else []
        self.context = context if context is not None else {}
        self.current_state = current_state
        self.pii_masked = pii_masked
        self.consent_recorded = consent_recorded
        self.created_at = created_at or utc_now_iso()

    @property
    def ttl(self) -> int:
        return conversation_ttl_seconds()

    def to_item(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "lead_id": self.lead_id,
            "messages": self.messages,
            "context": self.context,
            "current_state": self.current_state,
            "pii_masked": self.pii_masked,
            "consent_recorded": self.consent_recorded,
            "created_at": self.created_at,
            "ttl": self.ttl,
        }

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> Conversation:
        return cls(
            session_id=item["session_id"],
            lead_id=item["lead_id"],
            messages=item.get("messages", []),
            context=item.get("context", {}),
            current_state=item.get("current_state", "greeting"),
            pii_masked=item.get("pii_masked", False),
            consent_recorded=item.get("consent_recorded", False),
            created_at=item.get("created_at"),
        )


VALID_STATES = (
    "greeting",
    "elicitation",
    "intent",
    "qualification",
    "recommendation",
    "scheduling",
    "handoff",
    "followup",
)

# Esteira Kanban do CRM (FR7.3/FR11.3) — espelha KANBAN_STAGES do u3; usada para
# tornar a atualização de estágio via /internal/crm-status monotônica (sem regressão).
KANBAN_STAGE_ORDER = (
    "novo",
    "qualificado",
    "contato-feito",
    "visita-agendada",
    "handoff",
    "ganho",
    "perdido",
)

# Status internos do Lead ("new"/"qualified") projetados na esteira Kanban.
KANBAN_STATUS_MAP = {
    "new": "novo",
    "qualified": "qualificado",
}

_PHONE_RE = re.compile(r"\+55\s?\d{2}\s?\d{4,5}-?\d{4}")


def looks_like_phone(text: str) -> bool:
    return bool(_PHONE_RE.search(text))
