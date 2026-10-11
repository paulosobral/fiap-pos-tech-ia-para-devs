from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from infra.logging_utils import log_event

ALERT_STATUS_OPEN = "open"
ALERT_STATUS_RESOLVED = "resolved"
RESTRICTION_AUTO_RESOLVE_REASON = "not_anomalous_in_run"
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
    """Owner do Contrato 7 (alertas) — U5 escreve, U1 e U7 leem.

    PK do item é o próprio `anomaly_id` (schema do contrato); a restrição de
    agendamento (FR9.4) é materializada no item da anomalia
    (`scheduling_restricted=True` + `action_taken="schedule_restricted"` +
    `status="open"`) e consultável por lead via GSI `lead-index` — shape
    consumido pelo checker real da U1 (`DynamoRestrictionCheck` em
    apps/conversation-router/service/restriction.py, que exige
    `scheduling_restricted is True` e `status == "open"`).

    Ciclo de vida da restrição (sem bloqueio eterno): `open` → `resolved` via
    `resolve_restriction` (motivo + timestamp) ou `resolve_open_restriction`
    (auto-resolução quando o job diário re-corre e o lead NÃO é pontuado como
    anômalo). U5 não escreve na tabela de sessões (owner U1).
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
            log_event("alert_store_unavailable", operation="save_anomaly", error=str(exc))
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
            log_event("alert_store_unavailable", operation="restrict_scheduling", error=str(exc))
            raise AlertStoreError("alert store unavailable") from exc

    def resolve_restriction(self, anomaly_id: str, reason: str, resolved_at: str) -> None:
        """Fecha a restrição: `status` → resolved com motivo e timestamp."""
        try:
            self._client.update_item(
                TableName=self._table,
                Key={"PK": {"S": anomaly_id}},
                UpdateExpression="SET #st = :s, resolved_at = :t, resolved_reason = :r",
                ExpressionAttributeNames={"#st": "status"},
                ExpressionAttributeValues={
                    ":s": {"S": ALERT_STATUS_RESOLVED},
                    ":t": {"S": resolved_at},
                    ":r": {"S": reason},
                },
            )
        except Exception as exc:
            log_event("alert_store_unavailable", operation="resolve_restriction", error=str(exc))
            raise AlertStoreError("alert store unavailable") from exc

    def resolve_open_restriction(
        self,
        lead_id: str,
        reason: str = RESTRICTION_AUTO_RESOLVE_REASON,
        resolved_at: str | None = None,
    ) -> list[str]:
        """Resolve todas as restrições abertas do lead; retorna os anomaly_id resolvidos."""
        if resolved_at is None:
            resolved_at = datetime.now(timezone.utc).isoformat()
        resolved: list[str] = []
        for raw in self._query_lead_items(lead_id):
            item = unmarshal_item(raw)
            if (
                item.get("lead_id") == lead_id
                and item.get("scheduling_restricted")
                and item.get("status") == ALERT_STATUS_OPEN
                and item.get("anomaly_id")
            ):
                anomaly_id = str(item["anomaly_id"])
                self.resolve_restriction(anomaly_id, reason, resolved_at)
                resolved.append(anomaly_id)
        return resolved

    def find_open_restriction(self, lead_id: str) -> dict[str, Any] | None:
        items = []
        for raw in self._query_lead_items(lead_id):
            item = unmarshal_item(raw)
            if item.get("lead_id") == lead_id:
                items.append(item)
        items.sort(key=lambda item: str(item.get("detected_at")), reverse=True)
        for item in items:
            if item.get("scheduling_restricted") and item.get("status") == ALERT_STATUS_OPEN:
                return item
        return None

    def _query_lead_items(self, lead_id: str) -> list[dict[str, Any]]:
        """Query paginada no GSI `lead-index` (ExclusiveStartKey até esgotar)."""
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "TableName": self._table,
            "IndexName": "lead-index",
            "KeyConditionExpression": "lead_id = :lid",
            "ExpressionAttributeValues": {":lid": {"S": lead_id}},
        }
        try:
            while True:
                response = self._client.query(**kwargs)
                items.extend(response.get("Items", []))
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                kwargs["ExclusiveStartKey"] = last_key
        except Exception as exc:
            log_event("alert_store_unavailable", operation="query", error=str(exc))
            raise AlertStoreError("alert store unavailable") from exc
        return items

    def is_scheduling_restricted(self, lead_id: str) -> bool:
        return self.find_open_restriction(lead_id) is not None
