from __future__ import annotations

from decimal import Decimal
from typing import Any

from service.entities import Conversation, Lead


class SessionStore:
    def __init__(self, dynamodb_client: Any, table_name: str) -> None:
        self._client = dynamodb_client
        self._table = table_name

    def get_by_telegram_user(self, telegram_user_id: int) -> tuple[Lead | None, Conversation | None]:
        response = self._client.query(
            TableName=self._table,
            IndexName="telegram-user-index",
            KeyConditionExpression="telegram_user_id = :uid",
            ExpressionAttributeValues={":uid": {"N": str(telegram_user_id)}},
        )
        items = response.get("Items", [])
        if not items:
            return None, None
        return self._load(self._scalar(items[0].get("lead_id")))

    @staticmethod
    def _scalar(raw: Any) -> str | None:
        if isinstance(raw, dict):
            return next(iter(raw.values()), None)
        return raw

    def _load(self, lead_id: str | None) -> tuple[Lead | None, Conversation | None]:
        if not lead_id:
            return None, None
        lead_item = self._client.get_item(
            TableName=self._table, Key={"PK": {"S": f"LEAD#{lead_id}"}}
        ).get("Item")
        if not lead_item:
            return None, None
        response = self._client.query(
            TableName=self._table,
            IndexName="lead-index",
            KeyConditionExpression="lead_id = :lid",
            ExpressionAttributeValues={":lid": {"S": lead_id}},
        )
        conv_items = response.get("Items", [])
        return Lead.from_item(self._unmarshal(lead_item)), (
            Conversation.from_item(self._unmarshal(conv_items[0])) if conv_items else None
        )

    def create(self, telegram_user_id: int) -> tuple[Lead, Conversation]:
        from service.entities import new_id

        lead = Lead(lead_id=new_id(), telegram_user_id=telegram_user_id)
        conversation = Conversation(session_id=new_id(), lead_id=lead.lead_id)
        self.save(lead, conversation)
        return lead, conversation

    def get_or_create(self, telegram_user_id: int) -> tuple[Lead, Conversation, bool]:
        lead, conversation = self.get_by_telegram_user(telegram_user_id)
        if lead is not None and conversation is not None:
            return lead, conversation, False
        lead, conversation = self.create(telegram_user_id)
        return lead, conversation, True

    def save(self, lead: Lead, conversation: Conversation) -> None:
        self._client.put_item(
            TableName=self._table, Item=self._marshal({"PK": f"LEAD#{lead.lead_id}", "SK": "PROFILE", **lead.to_item()})
        )
        self._client.put_item(
            TableName=self._table,
            Item=self._marshal(
                {
                    "PK": f"LEAD#{lead.lead_id}",
                    "SK": f"CONV#{conversation.session_id}",
                    **conversation.to_item(),
                }
            ),
        )

    def _unmarshal(self, item: dict[str, Any]) -> dict[str, Any]:
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

    def _marshal(self, item: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(value, bool):
                out[key] = {"BOOL": value}
            elif isinstance(value, (int, float)):
                out[key] = {"N": str(value)}
            elif isinstance(value, (list, dict)):
                import json

                out[key] = {"S": json.dumps(value, default=str)}
            else:
                out[key] = {"S": str(value)}
        return out
