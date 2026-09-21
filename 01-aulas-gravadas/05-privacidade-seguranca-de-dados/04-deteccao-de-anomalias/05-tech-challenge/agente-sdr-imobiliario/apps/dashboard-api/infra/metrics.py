from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


class CloudWatchMeter:
    """Leitor de métricas operacionais (Contrato 2: CloudWatch) — resposta p90
    e custo mensal. Cliente injetado (handler cria boto3 com preguiça); falta
    de datapoint ou erro de leitura devolve `None` (zeros no snapshot) —
    degradação graciosa em POC, nunca derruba a agregação."""

    def __init__(
        self,
        client: Any,
        namespace: str = "SdrApp",
        response_metric: str = "ResponseTimeP90",
        cost_metric: str = "CostMonthly",
        now_fn: Any = None,
        response_lookback_seconds: int = 2 * 24 * 60 * 60,
        cost_lookback_seconds: int = 31 * 24 * 60 * 60,
    ) -> None:
        self._client = client
        self._namespace = namespace
        self._response_metric = response_metric
        self._cost_metric = cost_metric
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._response_lookback = response_lookback_seconds
        self._cost_lookback = cost_lookback_seconds

    def read(self) -> dict[str, float | None]:
        end = self._now_fn()
        try:
            response_value = self._latest_datapoint(
                "response_p90",
                self._response_metric,
                "p90",
                self._response_lookback,
                end,
            )
            cost_value = self._latest_datapoint(
                "cost_monthly", self._cost_metric, "Maximum", self._cost_lookback, end
            )
        except Exception as exc:
            logger.error("metrics read failed: %s", exc)
            return {"response_time_p90": None, "cost_monthly": None}
        return {
            "response_time_p90": _to_float(response_value),
            "cost_monthly": _to_float(cost_value),
        }

    def _latest_datapoint(
        self, query_id: str, metric_name: str, statistic: str, lookback: int, end: datetime
    ) -> float | None:
        start = end - timedelta(seconds=lookback)
        period = max(lookback // 2, 300)
        response = self._client.get_metric_data(
            MetricDataQueries=[
                {
                    "Id": query_id,
                    "MetricStat": {
                        "Metric": {"Namespace": self._namespace, "MetricName": metric_name},
                        "Period": period,
                        "Stat": statistic,
                    },
                    "ReturnData": True,
                }
            ],
            StartTime=start,
            EndTime=end,
        )
        results = response.get("MetricDataResults") or [{}]
        raw_values = results[0].get("Values", []) if isinstance(results[0], dict) else []
        values = [value for value in (_to_float(item) for item in raw_values) if value is not None]
        if not values:
            return None
        return max(values) if statistic == "Maximum" else values[-1]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
