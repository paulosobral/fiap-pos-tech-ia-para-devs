from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from infra.alert_store import project_alert
from service import kpi_calc
from service.kpi_calc import (
    alerts_last_24h,
    count_new_leads,
    days_ago,
    intent_volume,
    latest_conversation_by_lead,
    qualification_rate,
    route_distribution,
    scheduled_visits_count,
    start_of_today,
    state_funnel,
)

logger = logging.getLogger(__name__)

ALERTS_PAYLOAD_LIMIT = 100


def log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class KpiService:
    """Orquestrador da agregação de KPIs (U7 DashAPI, Contrato 2, FR7).

    Lê sessões do Contrato 5 (conversas + perfis), alertas do Contrato 7 e
    métricas operacionais via `meter` injetado (CloudWatch). Tabelas vazias ou
    itens inválidos viram zeros/estado vazio (nunca quebram o snapshot);
    falha de leitura de store propaga (handler → 500, Contrato 2). Resposta
    agregada apenas — nunca texto bruto de mensagem (NFR2.1/NFR3).
    """

    def __init__(
        self,
        conversations: Any,
        alerts: Any,
        meter: Any,
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        self.conversations = conversations
        self.alerts = alerts
        self.meter = meter
        self.now_fn = now_fn

    def snapshot(self) -> dict[str, Any]:
        log_event("kpi_aggregation_started")
        conversations = self._read(self.conversations.list_conversations, "conversation_read_failed")
        profiles = self._read(self.conversations.get_lead_profiles, "profiles_read_failed")
        alerts = self._read(self.alerts.list_alerts, "alerts_read_failed")
        metrics = self._read_metrics()

        now = self.now_fn()
        latest_by_lead = latest_conversation_by_lead(conversations)
        recent_alerts = alerts_last_24h(alerts, now)

        snapshot: dict[str, Any] = {
            "leads_today": count_new_leads(profiles, start_of_today(now), now),
            "leads_week": count_new_leads(profiles, days_ago(now, 7), now),
            "response_time_p90": metrics.get("response_time_p90") or 0.0,
            "qualification_rate": qualification_rate(profiles, latest_by_lead),
            "scheduled_visits": scheduled_visits_count(latest_by_lead),
            "anomalies_count": len(recent_alerts),
            "cost_monthly": metrics.get("cost_monthly") or 0.0,
            "generated_at": now.isoformat(),
            "intents": intent_volume(profiles),
            "route_distribution": route_distribution(profiles),
            "funnel": state_funnel(latest_by_lead),
            "alerts": [
                project_alert(alert) for alert in recent_alerts[:ALERTS_PAYLOAD_LIMIT]
            ],
        }
        log_event(
            "kpi_aggregation_completed",
            leads=len(profiles),
            conversations=len(conversations),
            anomalies=snapshot["anomalies_count"],
        )
        return snapshot

    def _read(self, reader: Callable[[], Any], failure_event: str) -> Any:
        try:
            return reader()
        except Exception:
            log_event(failure_event)
            raise

    def _read_metrics(self) -> dict[str, Any]:
        try:
            metrics = self.meter.read()
        except Exception as exc:
            log_event("metrics_read_failed", error=str(exc))
            return {}
        if not isinstance(metrics, dict):
            log_event("metrics_read_invalid")
            return {}
        return metrics
