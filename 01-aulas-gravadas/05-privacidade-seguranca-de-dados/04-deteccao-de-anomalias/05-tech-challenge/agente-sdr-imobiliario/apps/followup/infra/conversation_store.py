from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

CONVERSATION_SK_PREFIX = "CONV#"
PROFILE_SK = "PROFILE"
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
    """Leitor do Contrato 5 (sessões) — U6 é reader, U1 é owner.

    Duas varreduras paginadas sobre a tabela compartilhada: `CONV#` devolve as
    conversas e `PROFILE` o mapa `lead_id → perfil` (telegram_user_id, intent).
    Itens sem identificadores são ignorados com log (nunca quebram o tick).
    DynamoDB throttling/erros viram `ConversationStoreError` (retry da Lambda).
    """

    def __init__(self, dynamodb_client: Any, table_name: str = "sdr-sessions") -> None:
        self._client = dynamodb_client
        self._table = table_name

    def list_conversations(self) -> list[dict[str, Any]]:
        return self._scan_by_prefix(CONVERSATION_SK_PREFIX)

    def get_lead_profiles(self) -> dict[str, dict[str, Any]]:
        profiles: dict[str, dict[str, Any]] = {}
        for item in self._scan_by_prefix(PROFILE_SK):
            lead_id = item.get("lead_id")
            if not lead_id:
                logger.warning("lead profile item missing lead_id; skipped")
                continue
            profiles[str(lead_id)] = item
        return profiles

    def _scan_by_prefix(self, prefix: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "TableName": self._table,
            "FilterExpression": "begins_with(SK, :prefix)",
            "ExpressionAttributeValues": {":prefix": {"S": prefix}},
        }
        try:
            while True:
                response = self._client.scan(**kwargs)
                for raw in response.get("Items", []):
                    item = unmarshal_item(raw)
                    if prefix == PROFILE_SK:
                        items.append(item)
                        continue
                    if not item.get("session_id") or not item.get("lead_id"):
                        logger.warning("conversation item missing identifiers; skipped")
                        continue
                    items.append(item)
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                kwargs["ExclusiveStartKey"] = last_key
        except ConversationStoreError:
            raise
        except Exception as exc:
            logger.error("conversation store unavailable: %s", exc)
            raise ConversationStoreError("conversation store unavailable") from exc
        return items
