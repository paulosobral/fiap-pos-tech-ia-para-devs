from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from service.crm_gateway import CrmGateway
from service.flow_gateway import FlowGateway

logger = logging.getLogger(__name__)

# Esteira Kanban da POC (FR7.3/FR11.3) — estágios ordenados do pipeline.
KANBAN_STAGES = (
    "novo",
    "qualificado",
    "contato-feito",
    "visita-agendada",
    "handoff",
    "ganho",
    "perdido",
)

QUALIFY_SCORE = 70
URGENT_LEVEL = "alta"


def stage_for_lead(lead_data: dict[str, Any]) -> str:
    """Regra determinística da POC: lead qualificado avança para `qualificado`."""
    try:
        score = float(lead_data.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    urgency = str(lead_data.get("urgency") or "").strip().lower()
    if score >= QUALIFY_SCORE or urgency == URGENT_LEVEL:
        return "qualificado"
    return "novo"


class StatusSync:
    """Sincroniza o status da esteira Kanban no CRM e devolve ao fluxo."""

    def __init__(
        self,
        crm: CrmGateway,
        flow: FlowGateway | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.crm = crm
        self.flow = flow
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def sync(self, lead_id: str, session_id: str, lead_data: dict[str, Any]) -> dict[str, Any]:
        stage = stage_for_lead(lead_data)
        self.crm.update_stage(lead_id, stage)
        status = {
            "lead_id": lead_id,
            "session_id": session_id,
            "stage": stage,
            "synced_at": self._clock().isoformat(),
        }
        if self.flow is not None:
            self.flow.notify_status(lead_id, session_id, stage)
        return status