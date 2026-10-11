from __future__ import annotations

import json
from typing import Any

from infra.logging_utils import log_event

CONVERSATION_SK_PREFIX = "CONV#"
_JSON_FIELDS = ("messages", "context")


class ConversationStoreError(Exception):
    pass


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
    if key in _JSON_FIELDS and value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class ConversationStore:
    """Leitor do Contrato 6 (conversas) — U5 é reader, U1 é owner.

    Varredura paginada com filtro `begins_with(SK, CONV#)` sobre a tabela de
    sessões compartilhada; itens sem `session_id`/`lead_id` são ignorados com
    log (nunca quebram o job). DynamoDB throttling/erros viram
    `ConversationStoreError` (retry automático da Lambda, NFR4).
    """

    def __init__(self, dynamodb_client: Any, table_name: str = "sdr-sessions") -> None:
        self._client = dynamodb_client
        self._table = table_name

    def list_conversations(self) -> list[dict[str, Any]]:
        conversations: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "TableName": self._table,
            "FilterExpression": "begins_with(SK, :prefix)",
            "ExpressionAttributeValues": {":prefix": {"S": CONVERSATION_SK_PREFIX}},
        }
        try:
            while True:
                response = self._client.scan(**kwargs)
                for raw in response.get("Items", []):
                    item = unmarshal_item(raw)
                    if not item.get("session_id") or not item.get("lead_id"):
                        log_event("conversation_item_skipped", reason="missing_identifiers")
                        continue
                    conversations.append(item)
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                kwargs["ExclusiveStartKey"] = last_key
        except ConversationStoreError:
            raise
        except Exception as exc:
            log_event("conversation_store_unavailable", error=str(exc))
            raise ConversationStoreError("conversation store unavailable") from exc
        return conversations
