import json

import pytest

from infra.conversation_store import ConversationStore, ConversationStoreError, unmarshal_item


class FakeScanClient:
    def __init__(self, items=(), error=None, page_size=None):
        self._items = list(items)
        self._error = error
        self._page_size = page_size
        self.scans = 0

    def scan(self, **kwargs):
        self.scans += 1
        if self._error:
            raise self._error
        prefix = kwargs["ExpressionAttributeValues"][":prefix"]["S"]
        start = kwargs.get("ExclusiveStartKey")
        keyed = [(i["PK"]["S"] + "|" + i["SK"]["S"], i) for i in self._items]
        keyed.sort()
        items = [item for key, item in keyed if item.get("SK", {}).get("S", "").startswith(prefix)]
        if start is not None:
            start_key = start["PK"]["S"] + "|" + start["SK"]["S"]
            items = [item for key, item in keyed if item.get("SK", {}).get("S", "").startswith(prefix) and key > start_key]
        if self._page_size and len(items) > self._page_size:
            page, rest = items[: self._page_size], items[self._page_size :]
            last = page[-1]
            return {
                "Items": page,
                "LastEvaluatedKey": {"PK": last["PK"], "SK": last["SK"]},
            }
        return {"Items": items}


def conversation_raw(lead_id="lead-1", session_id="sess-1", created_at="2026-09-18T10:00:00+00:00"):
    return {
        "PK": {"S": f"LEAD#{lead_id}"},
        "SK": {"S": f"CONV#{session_id}"},
        "session_id": {"S": session_id},
        "lead_id": {"S": lead_id},
        "messages": {"S": json.dumps([{"role": "lead", "text": "quero locar", "at": created_at}])},
        "context": {"S": json.dumps({"channel": "telegram"})},
        "current_state": {"S": "qualification"},
        "pii_masked": {"BOOL": False},
        "created_at": {"S": created_at},
        "ttl": {"N": "7776000"},
    }


def profile_raw(lead_id="lead-1", telegram_user_id=42, intent="locação"):
    return {
        "PK": {"S": f"LEAD#{lead_id}"},
        "SK": {"S": "PROFILE"},
        "lead_id": {"S": lead_id},
        "telegram_user_id": {"N": str(telegram_user_id)},
        "intent": {"S": intent},
        "status": {"S": "new"},
    }


class TestConversationStore:
    def test_list_conversations_unmarshals_shared_schema(self):
        store = ConversationStore(FakeScanClient([conversation_raw()]))
        conversations = store.list_conversations()
        assert len(conversations) == 1
        item = conversations[0]
        assert item["lead_id"] == "lead-1"
        assert item["messages"][0]["text"] == "quero locar"
        assert item["context"] == {"channel": "telegram"}
        assert item["pii_masked"] is False
        assert item["ttl"] == 7776000

    def test_conversation_without_identifiers_is_skipped_with_log(self):
        broken = {"PK": {"S": "LEAD#x"}, "SK": {"S": "CONV#s1"}, "messages": {"S": "[]"}}
        store = ConversationStore(FakeScanClient([broken, conversation_raw()]))
        assert len(store.list_conversations()) == 1

    def test_skipped_item_logs_are_structured_json(self, caplog):
        import logging

        caplog.set_level(logging.INFO)
        broken = {"PK": {"S": "LEAD#x"}, "SK": {"S": "CONV#s1"}, "messages": {"S": "[]"}}
        ConversationStore(FakeScanClient([broken, conversation_raw()])).list_conversations()
        events = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
        assert any(e.get("event") == "conversation_item_skipped" for e in events)

    def test_scan_pagination_follows_last_evaluated_key(self):
        items = [conversation_raw(f"lead-{i}", f"sess-{i}") for i in range(3)]
        store = ConversationStore(FakeScanClient(items, page_size=2))
        assert len(store.list_conversations()) == 3

    def test_scan_failure_becomes_conversation_store_error(self):
        with pytest.raises(ConversationStoreError, match="unavailable"):
            ConversationStore(FakeScanClient(error=RuntimeError("throttled"))).list_conversations()

    def test_get_lead_profiles_builds_map_by_lead_id(self):
        store = ConversationStore(FakeScanClient([profile_raw("lead-1", 42), profile_raw("lead-2", 7)]))
        profiles = store.get_lead_profiles()
        assert profiles["lead-1"]["telegram_user_id"] == 42
        assert profiles["lead-2"]["intent"] == "locação"

    def test_profile_without_lead_id_is_skipped(self):
        broken = {"PK": {"S": "LEAD#x"}, "SK": {"S": "PROFILE"}, "telegram_user_id": {"N": "1"}}
        store = ConversationStore(FakeScanClient([broken, profile_raw()]))
        assert set(store.get_lead_profiles()) == {"lead-1"}

    def test_profile_scan_failure_becomes_conversation_store_error(self):
        with pytest.raises(ConversationStoreError, match="unavailable"):
            ConversationStore(FakeScanClient(error=RuntimeError("throttled"))).get_lead_profiles()

    def test_unmarshal_item_passthrough_for_plain_values(self):
        assert unmarshal_item({"PK": {"S": "LEAD#1"}}) == {"PK": "LEAD#1"}
        assert unmarshal_item({"note": "texto simples"}) == {"note": "texto simples"}
