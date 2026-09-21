from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SchedulingGate:
    """Restrição de agendamento para leads suspeitos (FR9.4).

    U5 é owner apenas da tabela de alertas (Contrato 7), então a restrição é
    materializada no item da anomalia e exposta como check consultável — o
    Scheduler (u1) e o Build-and-Test consultam `is_restricted(lead_id)`
    antes de confirmar visitas de leads sinalizados.
    """

    def __init__(self, alerts: Any) -> None:
        self.alerts = alerts

    def restrict(self, lead_id: str, anomaly_id: str) -> None:
        if not lead_id or not anomaly_id:
            raise ValueError("lead_id and anomaly_id are required to restrict scheduling")
        self.alerts.restrict_scheduling(anomaly_id)
        logger.info(
            "scheduling restricted: lead_id=%s anomaly_id=%s", lead_id, anomaly_id
        )

    def is_restricted(self, lead_id: str) -> bool:
        if not lead_id:
            return False
        return self.alerts.is_scheduling_restricted(lead_id)

    def restriction_reason(self, lead_id: str) -> str | None:
        if not lead_id:
            return None
        item = self.alerts.find_open_restriction(lead_id)
        if item is None:
            return None
        return str(item.get("restriction_reason") or item.get("action_taken") or "")
