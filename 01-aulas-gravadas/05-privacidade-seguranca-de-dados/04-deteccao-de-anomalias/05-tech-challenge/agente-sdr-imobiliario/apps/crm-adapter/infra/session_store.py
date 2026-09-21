from __future__ import annotations

from typing import Any


class SessionLookup:
    """Leitor do Contract 5 espelhando o acesso REAL do dono da tabela (u1).

    `sdr-sessions` é tabela de chave composta (PK/SK): itens `LEAD#<lead_id>/PROFILE`
    (lead) e `LEAD#<lead_id>/CONV#<session_id>` (conversa), como em
    apps/conversation-router/infra/session_store.py. Os GSIs existentes são
    `telegram-user-index` e `lead-index` (caminhos da própria u1) — NÃO existe
    GSI por session_id; a validação lead+sessão usa dois get_item com a chave
    composta (uma leitura do CONV# já confere lead e sessão ao mesmo tempo).
    """

    def __init__(self, dynamodb_client: Any, table_name: str) -> None:
        self._client = dynamodb_client
        self._table = table_name

    def get_session(self, lead_id: str, session_id: str) -> dict[str, Any] | None:
        lead_item = self._client.get_item(
            TableName=self._table,
            Key={"PK": {"S": f"LEAD#{lead_id}"}, "SK": {"S": "PROFILE"}},
        ).get("Item")
        if not lead_item:
            return None
        item = self._client.get_item(
            TableName=self._table,
            Key={"PK": {"S": f"LEAD#{lead_id}"}, "SK": {"S": f"CONV#{session_id}"}},
        ).get("Item")
        if not item:
            return None
        return self._unwrap(item)

    @staticmethod
    def _unwrap(item: dict[str, Any]) -> dict[str, Any]:
        return {key: next(iter(raw.values())) for key, raw in item.items()}
