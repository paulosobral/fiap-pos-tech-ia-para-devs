from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class DedupeError(Exception):
    pass


STATUS_INGESTED = "INGESTED"
STATUS_QUARANTINE = "QUARANTINE"


class DedupeStore:
    """Marca anti spam/duplicata por mensagem SES (put condicional).

    `put_first` grava `message_id` com `attribute_not_exists` — `True` na
    primeira entrega, `False` em duplicata. `delete` é o rollback best-effort
    usado pelo orquestrador quando uma falha transitória exige reprocesso.
    Quando nem o rollback é possível, `mark_quarantine` move a marca para o
    estado explícito `QUARANTINE` (poison de tentativa falha, com motivo
    PII-safe) — a marca não vira drop silencioso no reprocesso: `take_quarantined`
    distingue "duplicata de ingest concluída" de "marca obsoleta de tentativa
    falha" e libera esta última (delete + `True`) para o reprocesso.
    """

    def __init__(self, dynamodb_client: Any, table_name: str) -> None:
        self._client = dynamodb_client
        self._table = table_name

    def put_first(self, message_id: str, source: str | None = None) -> bool:
        item: dict[str, Any] = {
            "message_id": {"S": message_id},
            "status": {"S": STATUS_INGESTED},
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
            logger.error("dedupe store unavailable: %s", type(exc).__name__)
            raise DedupeError("dedupe store unavailable") from exc

    def delete(self, message_id: str) -> bool:
        try:
            self._client.delete_item(TableName=self._table, Key={"message_id": {"S": message_id}})
            return True
        except Exception as exc:
            logger.error("dedupe delete failed: %s", type(exc).__name__)
            return False

    def mark_quarantine(self, message_id: str, reason: str) -> bool:
        """Move a marca para `QUARANTINE` (poison explícito de tentativa falha).

        Sobrescreve o item existente com o estado de quarentena; motivo deve
        ser PII-safe (tipo da exceção, nunca `str(exc)` verbatim). Best-effort:
        erro só é registrado e `False` devolvido.
        """
        try:
            self._client.put_item(
                TableName=self._table,
                Item={
                    "message_id": {"S": message_id},
                    "status": {"S": STATUS_QUARANTINE},
                    "quarantine_reason": {"S": reason},
                    "quarantined_at": {"S": datetime.now(timezone.utc).isoformat()},
                },
            )
            return True
        except Exception as exc:
            logger.error("dedupe quarantine failed: %s", type(exc).__name__)
            return False

    def take_quarantined(self, message_id: str) -> bool:
        """Libera (delete) uma marca `QUARANTINE` e reporta `True`.

        `False` para item ausente ou marca comum de ingest concluída — só a
        marca obsoleta de tentativa falha é liberada para reprocesso.
        """
        try:
            result = self._client.get_item(
                TableName=self._table, Key={"message_id": {"S": message_id}}
            )
            item = result.get("Item") or {}
            if item.get("status", {}).get("S") != STATUS_QUARANTINE:
                return False
            self._client.delete_item(TableName=self._table, Key={"message_id": {"S": message_id}})
            return True
        except Exception as exc:
            logger.error("dedupe quarantine release failed: %s", type(exc).__name__)
            return False
