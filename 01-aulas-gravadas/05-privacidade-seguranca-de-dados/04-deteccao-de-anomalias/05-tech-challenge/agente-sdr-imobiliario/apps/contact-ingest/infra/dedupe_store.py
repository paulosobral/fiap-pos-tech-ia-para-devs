from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class DedupeError(Exception):
    pass


class DedupeStore:
    """Marca anti spam/duplicata por mensagem SES (put condicional).

    `put_first` grava `message_id` com `attribute_not_exists` — `True` na
    primeira entrega, `False` em duplicata. `delete` é o rollback best-effort
    usado pelo orquestrador quando uma falha transitória exige reprocesso.
    """

    def __init__(self, dynamodb_client: Any, table_name: str) -> None:
        self._client = dynamodb_client
        self._table = table_name

    def put_first(self, message_id: str, source: str | None = None) -> bool:
        item: dict[str, Any] = {
            "message_id": {"S": message_id},
            "received_at": {"S": datetime.now(timezone.utc).isoformat()},
        }
        if source:
            item["source"] = {"S": source}
        try:
            self._client.put_item(
                TableName=self._table,
                Item=item,
                ConditionExpression="attribute_not_exists(message_id)",
            )
            return True
        except self._client.exceptions.ConditionalCheckFailedException:
            return False
        except Exception as exc:
            logger.error("dedupe store unavailable: %s", exc)
            raise DedupeError("dedupe store unavailable") from exc

    def delete(self, message_id: str) -> bool:
        try:
            self._client.delete_item(TableName=self._table, Key={"message_id": {"S": message_id}})
            return True
        except Exception as exc:
            logger.error("dedupe delete failed: %s", exc)
            return False
