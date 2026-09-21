import json

import pytest

from infra.conversation_store import ConversationStore, ConversationStoreError


def s(value):
    return {"S": str(value)}


def n(value):
    return {"N": str(value)}


def bool_(value):
    return {"BOOL": value}


def conv_item(session_id="s1", lead_id="l1", state="greeting", created_at="2026-09-20T10:00:00Z"):
    return {
        "PK": s(f"LEAD#{lead_id}"),
        "SK": s(f"CONV#{session_id}"),
        "session_id": s(session_id),
        "lead_id": s(lead_id),
        "current_state": s(state),
        "created_at": s(created_at),
        "messages": {"S": json.dumps([{"role": "lead", "text": "oi"}])},
        "context": {"S": json.dumps({"lead_qualified": True})},
        "pii_masked": bool_(False),
        "consent_recorded": bool_(True),
        "ttl": n(1789000000),
    }


def profile_item(lead_id="l1", status="new", created_at="2026-09-20T09:00:00Z"):
    return {
        "PK": s(f"LEAD#{lead_id}"),
        "SK": s("PROFILE"),
        "lead_id": s(lead_id),
        "telegram_user_id": n(42),
        "status": s(status),
        "created_at": s(created_at),
        "updated_at": s(created_at),
    }


class FakeScanClient:
    def __init__(self, pages_by_table):
        self._pages = pages_by_table
        self._cursor: dict[str, int] = {}
        self.calls: list[dict] = []

    def scan(self, **kwargs):
        self.calls.append(kwargs)
        table = kwargs["TableName"]
        pages = self._pages.get(table, [])
        index = self._cursor.get(table, 0)
        if index >= len(pages):
            return {"Items": []}
        self._cursor[table] = index + 1
        payload = pages[index]
        response = {"Items": payload.get("Items", [])}
        if payload.get("LastEvaluatedKey"):
            response["LastEvaluatedKey"] = payload["LastEvaluatedKey"]
        return response


class FailingClient:
    def scan(self, **kwargs):
        raise RuntimeError("dynamodb down")


def make_store(pages, table="sdr-sessions"):
    return ConversationStore(FakeScanClient(pages), table)


class TestConversations:
    def test_lists_and_unmarshals_conversations(self):
        store = make_store({"sdr-sessions": [{"Items": [conv_item()]}]})
        conversations = store.list_conversations()
        assert len(conversations) == 1
        conversation = conversations[0]
        assert conversation["session_id"] == "s1"
        assert conversation["ttl"] == 1789000000
        assert conversation["pii_masked"] is False
        assert conversation["messages"] == [{"role": "lead", "text": "oi"}]
        assert conversation["context"] == {"lead_qualified": True}

    def test_pagination_follows_last_evaluated_key(self):
        pages = {
            "sdr-sessions": [
                {"Items": [conv_item(session_id="s1")], "LastEvaluatedKey": {"SK": {"S": "CONV#s1"}}},
                {"Items": [conv_item(session_id="s2")]},
            ]
        }
        conversations = make_store(pages).list_conversations()
        assert [c["session_id"] for c in conversations] == ["s1", "s2"]

    def test_skips_conversation_without_identifiers(self, caplog):
        broken = conv_item(session_id="s9")
        broken.pop("lead_id")
        store = make_store({"sdr-sessions": [{"Items": [broken, conv_item()]}]})
        conversations = store.list_conversations()
        assert [c["session_id"] for c in conversations] == ["s1"]

    def test_client_error_raises_conversation_store_error(self):
        with pytest.raises(ConversationStoreError):
            ConversationStore(FailingClient()).list_conversations()


class TestProfiles:
    def test_profile_map_and_missing_lead_id_skipped(self):
        broken = profile_item(lead_id="l9")
        broken.pop("lead_id")
        store = make_store(
            {"sdr-sessions": [{"Items": [profile_item(lead_id="l1"), broken]}]}
        )
        profiles = store.get_lead_profiles()
        assert set(profiles) == {"l1"}
        assert profiles["l1"]["telegram_user_id"] == 42

    def test_profile_scan_filters_by_prefix(self):
        store = make_store({"sdr-sessions": [{"Items": []}]})
        store.get_lead_profiles()
        call = store._client.calls[0]
        assert call["FilterExpression"] == "begins_with(SK, :prefix)"
        assert call["ExpressionAttributeValues"] == {":prefix": {"S": "PROFILE"}}
