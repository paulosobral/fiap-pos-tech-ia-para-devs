from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ALERT_STATUS_OPEN = "open"


def unmarshal_alert_item(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw in item.items():
        if isinstance(raw, dict) and len(raw) == 1:
            kind, value = next(iter(raw.items()))
            if kind == "BOOL":
                out[key] = value
            elif kind == "N":
                number = float(value)
                out[key] = int(number) if number.is_integer() else number
            else:
                out[key] = value
        else:
            out[key] = raw
    return out


class DynamoRestrictionCheck:
    """Checker de restrição de agendamento (FR9.4) consumido pelo fluxo da U1.

    Lê a MESMA tabela de alertas da U5 (Contrato 7, owner U5): itens com
    `scheduling_restricted=True` e `status="open"` para o lead, via GSI
    `lead-index`. Env `ALERTS_TABLE` (mesma variável da U5; default `sdr-alerts`).
    Fail-open: erro de consulta = NÃO restrito (indisponibilidade não bloqueia o lead).
    """

    def __init__(self, dynamodb_client: Any, table_name: str | None = None) -> None:
        self._client = dynamodb_client
        self._table = table_name or os.environ.get("ALERTS_TABLE", "sdr-alerts")

    def __call__(self, lead_id: str) -> bool:
        if not lead_id:
            return False
        try:
            response = self._client.query(
                TableName=self._table,
                IndexName="lead-index",
                KeyConditionExpression="lead_id = :lid",
                ExpressionAttributeValues={":lid": {"S": lead_id}},
            )
        except Exception as exc:
            logger.error("restriction check unavailable (fail-open): %s", exc)
            return False
        for raw in response.get("Items", []):
            item = unmarshal_alert_item(raw)
            if (
                item.get("lead_id") == lead_id
                and item.get("scheduling_restricted") is True
                and item.get("status") == ALERT_STATUS_OPEN
            ):
                return True
        return False
