from __future__ import annotations

import json
from typing import Any

from infra.structured_log import log_event

FOLLOWUP_SK = "FOLLOWUP"


class FollowupStateError(Exception):
    pass


def _marshal(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in item.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = {"BOOL": value}
        elif isinstance(value, (int, float)):
            out[key] = {"N": str(value)}
        elif isinstance(value, (list, dict)):
            out[key] = {"S": json.dumps(value, default=str)}
        else:
            out[key] = {"S": str(value)}
    return out


def _unmarshal(item: dict[str, Any]) -> dict[str, Any]:
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


class FollowupStateStore:
    """Estado próprio de cadência do follow-up (U6, FR8.1/FR8.3).

    Item `PK=LEAD#<lead_id>, SK=FOLLOWUP` numa tabela dedicada (`sdr-followup-state`):
    U6 nunca escreve na tabela de sessões (owner U1). Guarda o último passo da
    cadência enviado, o instante do último follow-up e o próximo passo — base
    da idempotência do tick e da guarda anti-spam (lead respondeu → para).
    """

    def __init__(self, dynamodb_client: Any, table_name: str = "sdr-followup-state") -> None:
        self._client = dynamodb_client
        self._table = table_name

    def get_state(self, lead_id: str) -> dict[str, Any] | None:
        try:
            response = self._client.query(
                TableName=self._table,
                KeyConditionExpression="PK = :pk AND SK = :sk",
                ExpressionAttributeValues={":pk": {"S": f"LEAD#{lead_id}"}, ":sk": {"S": FOLLOWUP_SK}},
            )
        except FollowupStateError:
            raise
        except Exception as exc:
            log_event("followup_state_unavailable", error=str(exc))
            raise FollowupStateError("followup state store unavailable") from exc
        items = response.get("Items", [])
        return _unmarshal(items[0]) if items else None

    def record_followup(self, lead_id: str, step: int, sent_at: str, next_step: int | None) -> None:
        item = _marshal(
            {
                "PK": f"LEAD#{lead_id}",
                "SK": FOLLOWUP_SK,
                "lead_id": lead_id,
                "cadence_step": step,
                "last_step": step,
                "last_followup_at": sent_at,
                "next_step": next_step,
                "done": next_step is None,
            }
        )
        try:
            self._client.put_item(TableName=self._table, Item=item)
        except Exception as exc:
            log_event("followup_state_write_failed", error=str(exc))
            raise FollowupStateError("followup state write failed") from exc
