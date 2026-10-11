from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from service.crm_gateway import CrmGateway
from service.flow_gateway import FlowGateway

logger = logging.getLogger(__name__)

# Esteira Kanban da POC (FR7.3/FR11.3) — MESMA ordem monotônica da u1
# (KANBAN_STAGE_ORDER em apps/conversation-router/service/entities.py, espelhada
# aqui porque a fronteira Lambda proíbe import cross-app).
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
# Rótulos OFICIAIS de urgência do produtor (u1, LeadQualifier.urgency): low/medium/high.
URGENT_LEVEL = "high"


def _stage_index(stage: Any) -> int:
    return KANBAN_STAGES.index(stage) if stage in KANBAN_STAGES else -1


def stage_for_lead(lead_data: dict[str, Any]) -> str:
    """Regra determinística da POC: score >= 70 ou urgência 'high' → `qualificado`.

    O estágio produzido é sempre membro de KANBAN_STAGES (ordem oficial da u1).
    """
    try:
        score = float(lead_data.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    urgency = str(lead_data.get("urgency") or "").strip().lower()
    if score >= QUALIFY_SCORE or urgency == URGENT_LEVEL:
        return "qualificado"
    return "novo"


class StatusSync:
    """Sincroniza o status da esteira Kanban no CRM e devolve ao fluxo.

    Monotônico (FR7.3/FR11.3, espelho do receptor /internal/crm-status da u1):
    o estágio já avançado no CRM — inclusive movido pelo corretor — nunca regride
    na ordem KANBAN_STAGES; redelivery at-least-once ou re-enfileiramento da u1
    não derruba a esteira.
    """

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
        target = stage_for_lead(lead_data)
        current = self._current_stage(lead_id)
        stage = target
        if _stage_index(current) > _stage_index(target):
            # Já avançado (ex.: movido pelo corretor no CRM): mantém, sem regressão.
            stage = current
        else:
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

    def _current_stage(self, lead_id: str) -> Any:
        existing = self.crm.get_lead(lead_id)
        if not existing:
            return None
        return existing.get("stage")
