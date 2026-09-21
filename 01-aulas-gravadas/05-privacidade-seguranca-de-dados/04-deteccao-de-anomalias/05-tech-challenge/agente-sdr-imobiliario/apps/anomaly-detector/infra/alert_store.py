from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

ALERT_STATUS_OPEN = "open"
ACTION_ALERT_ISSUED = "alert_issued"
ACTION_SCHEDULE_RESTRICTED = "schedule_restricted"


class AlertStoreError(Exception):
    pass


def marshal_item(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, bool):
            out[key] = {"BOOL": value}
        elif isinstance(value, (int, float)):
            out[key] = {"N": str(value)}
        elif isinstance(value, (list, dict)):
            out[key] = {"S": json.dumps(value, default=str)}
        else:
            out[key] = {"S": str(value)}
    return out


def unmarshal_item(item: dict[str, Any]) -> dict[str, Any]:
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
                out[key] = _decode_json_field(key, str(value))
        else:
            out[key] = raw
    return out


def _decode_json_field(key: str, value: str) -> Any:
    if key in ("features",) and value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class AlertStore:
    """Owner do Contrato 7 (alertas) — U5 escreve, U7 lê (dashboard).

    PK do item é o próprio `anomaly_id` (schema do contrato); a restrição de
    agendamento (FR9.4) é materializada no item da anomalia
    (`scheduling_restricted=True` + `action_taken="schedule_restricted"`) e
    consultável por lead via GSI `lead-index` — U5 não escreve na tabela de
    sessões (owner U1).
    """

    def __init__(self, dynamodb_client: Any, table_name: str = "sdr-alerts") -> None:
        self._client = dynamodb_client
        self._table = table_name

    def save_anomaly(self, anomaly: dict[str, Any]) -> dict[str, Any]:
        if not anomaly.get("anomaly_id"):
            raise AlertStoreError("anomaly_id required")
        if not anomaly.get("lead_id"):
            raise AlertStoreError("lead_id required")
        item = marshal_item({"PK": anomaly["anomaly_id"], **anomaly})
        try:
            self._client.put_item(TableName=self._table, Item=item)
        except Exception as exc:
            logger.error("alert store unavailable: %s", exc)
            raise AlertStoreError("alert store unavailable") from exc
        return anomaly

    def restrict_scheduling(self, anomaly_id: str) -> None:
        try:
            self._client.update_item(
                TableName=self._table,
                Key={"PK": {"S": anomaly_id}},
                UpdateExpression=(
                    "SET scheduling_restricted = :t, "
                    "action_taken = :a, "
                    "restriction_reason = :r"
                ),
                ExpressionAttributeValues={
                    ":t": {"BOOL": True},
                    ":a": {"S": ACTION_SCHEDULE_RESTRICTED},
                    ":r": {"S": ACTION_SCHEDULE_RESTRICTED},
                },
            )
        except Exception as exc:
            logger.error("alert store unavailable: %s", exc)
            raise AlertStoreError("alert store unavailable") from exc

    def find_open_restriction(self, lead_id: str) -> dict[str, Any] | None:
        try:
            response = self._client.query(
                TableName=self._table,
                IndexName="lead-index",
                KeyConditionExpression="lead_id = :lid",
                ExpressionAttributeValues={":lid": {"S": lead_id}},
            )
        except Exception as exc:
            logger.error("alert store unavailable: %s", exc)
            raise AlertStoreError("alert store unavailable") from exc
        items = [
            unmarshal_item(raw)
            for raw in response.get("Items", [])
            if unmarshal_item(raw).get("lead_id") == lead_id
        ]
        items.sort(key=lambda item: str(item.get("detected_at")), reverse=True)
        for item in items:
            if item.get("scheduling_restricted") and item.get("status") == ALERT_STATUS_OPEN:
                return item
        return None

    def is_scheduling_restricted(self, lead_id: str) -> bool:
        return self.find_open_restriction(lead_id) is not None
