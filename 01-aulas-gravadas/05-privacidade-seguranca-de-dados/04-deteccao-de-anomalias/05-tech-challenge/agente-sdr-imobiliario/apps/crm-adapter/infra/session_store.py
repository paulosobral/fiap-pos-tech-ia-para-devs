from __future__ import annotations

from typing import Any


class SessionLookup:
    """Leitor do Contract 5 (sessões): valida que o lead/sessão existe."""

    def __init__(self, dynamodb_client: Any, table_name: str) -> None:
        self._client = dynamodb_client
        self._table = table_name

    def get_session(self, session_id: str, lead_id: str | None = None) -> dict[str, Any] | None:
        response = self._client.query(
            TableName=self._table,
            IndexName="session-index",
            KeyConditionExpression="session_id = :sid",
            ExpressionAttributeValues={":sid": {"S": session_id}},
        )
        items = response.get("Items", [])
        if not items:
            return None
        session = self._unwrap(items[0])
        if lead_id is not None:
            stored = session.get("lead_id")
            if stored is not None and str(stored) != str(lead_id):
                return None
        return session

    @staticmethod
    def _unwrap(item: dict[str, Any]) -> dict[str, Any]:
        return {key: next(iter(raw.values())) for key, raw in item.items()}