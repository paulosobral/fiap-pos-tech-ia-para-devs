import json
from unittest.mock import MagicMock

import pytest

from infra.conversation_store import (
    CONVERSATION_SK_PREFIX,
    ConversationStore,
    ConversationStoreError,
    unmarshal_item,
)


def raw_conversation(session_id, lead_id, messages=None):
    return {
        "PK": {"S": f"LEAD#{lead_id}"},
        "SK": {"S": f"CONV#{session_id}"},
        "session_id": {"S": session_id},
        "lead_id": {"S": lead_id},
        "current_state": {"S": "greeting"},
        "messages": {"S": json.dumps(messages if messages is not None else [])},
        "context": {"S": json.dumps({"channel": "telegram"})},
        "pii_masked": {"BOOL": False},
        "ttl": {"N": "7776000"},
    }


def make_store(pages):
    client = MagicMock()
    responses = list(pages)
    client.scan.side_effect = responses
    return ConversationStore(client, "sdr-sessions"), client


class TestConversationStore:
    def test_list_conversations_scans_with_conv_prefix_filter(self):
        store, client = make_store([{"Items": [raw_conversation("s1", "l1")]}])
        conversations = store.list_conversations()
        assert len(conversations) == 1
        kwargs = client.scan.call_args.kwargs
        assert kwargs["FilterExpression"] == "begins_with(SK, :prefix)"
        assert kwargs["ExpressionAttributeValues"][":prefix"] == {"S": CONVERSATION_SK_PREFIX}

    def test_scan_pagination_collects_all_pages(self):
        page_one = {"Items": [raw_conversation("s1", "l1")], "LastEvaluatedKey": {"PK": {"S": "LEAD#l1"}}}
        page_two = {"Items": [raw_conversation("s2", "l2")]}
        store, client = make_store([page_one, page_two])
        conversations = store.list_conversations()
        assert [c["session_id"] for c in conversations] == ["s1", "s2"]
        assert client.scan.call_count == 2
        assert client.scan.call_args_list[1].kwargs["ExclusiveStartKey"] == {"PK": {"S": "LEAD#l1"}}

    def test_items_without_identifiers_are_skipped(self):
        page = {"Items": [raw_conversation("s1", "l1"), {"SK": {"S": "CONV#broken"}}]}
        store, _ = make_store([page])
        conversations = store.list_conversations()
        assert [c["session_id"] for c in conversations] == ["s1"]

    def test_profile_items_are_not_returned(self):
        page = {
            "Items": [
                {"PK": {"S": "LEAD#l1"}, "SK": {"S": "PROFILE"}, "lead_id": {"S": "l1"}},
                raw_conversation("s1", "l1"),
            ]
        }
        store, _ = make_store([page])
        assert [c["session_id"] for c in store.list_conversations()] == ["s1"]

    def test_client_failure_raises_conversation_store_error(self):
        client = MagicMock()
        client.scan.side_effect = RuntimeError("dynamo down")
        store = ConversationStore(client, "sdr-sessions")
        with pytest.raises(ConversationStoreError, match="conversation store unavailable"):
            store.list_conversations()

    def test_unmarshal_converts_types_and_decodes_json_fields(self):
        item = unmarshal_item(raw_conversation("s1", "l1", messages=[{"text": "oi"}]))
        assert item["messages"] == [{"text": "oi"}]
        assert item["context"] == {"channel": "telegram"}
        assert item["pii_masked"] is False
        assert item["ttl"] == 7776000
