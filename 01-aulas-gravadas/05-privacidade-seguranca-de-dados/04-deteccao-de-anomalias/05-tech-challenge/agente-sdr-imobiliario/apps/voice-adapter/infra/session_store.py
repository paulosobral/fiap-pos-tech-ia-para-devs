from __future__ import annotations

from typing import Any


class SessionLookup:
    """Leitor somente-leitura da tabela de sessões da U1 (Contract 5).

    Espelha o acesso real da U1 (apps/conversation-router/infra/session_store.py):
    chave composta PK/SK (`LEAD#<lead_id>/PROFILE`, `LEAD#<lead_id>/CONV#<session_id>`)
    com GSIs `telegram-user-index` e `lead-index`. A GSI `session-index` NÃO existe
    — o lead é resolvido pelo `telegram_user_id` no GSI e a conversa é lida pela
    chave composta, como em `get_by_telegram_user`/`get_conversation` da U1.
    """

    def __init__(self, dynamodb_client: Any, table_name: str) -> None:
        self._client = dynamodb_client
        self._table = table_name

    def get_session(self, session_id: str, telegram_user_id: int | None = None) -> dict[str, Any] | None:
        lead_id = self._lead_id_by_telegram_user(telegram_user_id)
        if not lead_id:
            return None
        item = self._client.get_item(
            TableName=self._table,
            Key={"PK": {"S": f"LEAD#{lead_id}"}, "SK": {"S": f"CONV#{session_id}"}},
        ).get("Item")
        if not item:
            return None
        return self._unwrap(item)

    def _lead_id_by_telegram_user(self, telegram_user_id: int | None) -> str | None:
        if telegram_user_id is None:
            return None
        response = self._client.query(
            TableName=self._table,
            IndexName="telegram-user-index",
            KeyConditionExpression="telegram_user_id = :uid",
            ExpressionAttributeValues={":uid": {"N": str(telegram_user_id)}},
        )
        items = response.get("Items", [])
        if not items:
            return None
        return self._scalar(items[0].get("lead_id"))

    @staticmethod
    def _scalar(raw: Any) -> str | None:
        if isinstance(raw, dict):
            return next(iter(raw.values()), None)
        return raw

    @staticmethod
    def _unwrap(item: dict[str, Any]) -> dict[str, Any]:
        import json

        out: dict[str, Any] = {}
        for key, raw in item.items():
            for type_key, value in raw.items():
                if type_key == "N":
                    out[key] = float(value) if "." in str(value) else int(value)
                elif type_key == "BOOL":
                    out[key] = value
                else:
                    out[key] = (
                        json.loads(value)
                        if isinstance(value, str) and value[:1] in ("[", "{")
                        else value
                    )
        return out
