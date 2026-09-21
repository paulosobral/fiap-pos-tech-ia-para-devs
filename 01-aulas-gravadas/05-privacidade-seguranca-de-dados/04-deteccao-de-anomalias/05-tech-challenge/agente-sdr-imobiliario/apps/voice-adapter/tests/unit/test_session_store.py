from unittest.mock import MagicMock

from infra.session_store import SessionLookup


PROFILE_ITEM = {
    "PK": {"S": "LEAD#l1"},
    "SK": {"S": "PROFILE"},
    "lead_id": {"S": "l1"},
    "telegram_user_id": {"N": "42"},
}

CONV_ITEM = {
    "PK": {"S": "LEAD#l1"},
    "SK": {"S": "CONV#s1"},
    "session_id": {"S": "s1"},
    "lead_id": {"S": "l1"},
    "current_state": {"S": "greeting"},
    "pii_masked": {"BOOL": True},
    "ttl": {"N": "7776000"},
}


def make_client(items=(), conv_item=None):
    client = MagicMock()
    client.query.return_value = {"Items": list(items)}
    client.get_item.return_value = {"Item": conv_item} if conv_item is not None else {}
    return client


class TestSessionLookup:
    def test_resolves_lead_by_telegram_index_then_reads_composite_conversation_row(self):
        client = make_client(items=[PROFILE_ITEM], conv_item=CONV_ITEM)
        store = SessionLookup(client, "sdr-sessions")
        session = store.get_session("s1", telegram_user_id=42)
        query = client.query.call_args.kwargs
        assert query["IndexName"] == "telegram-user-index"
        assert query["KeyConditionExpression"] == "telegram_user_id = :uid"
        assert query["ExpressionAttributeValues"] == {":uid": {"N": "42"}}
        assert client.get_item.call_args.kwargs["Key"] == {
            "PK": {"S": "LEAD#l1"},
            "SK": {"S": "CONV#s1"},
        }
        assert session["session_id"] == "s1"
        assert session["lead_id"] == "l1"
        assert session["current_state"] == "greeting"

    def test_lead_not_found_returns_none(self):
        client = make_client(items=[])
        store = SessionLookup(client, "sdr-sessions")
        assert store.get_session("s1", telegram_user_id=42) is None
        client.get_item.assert_not_called()

    def test_conversation_row_not_found_returns_none(self):
        client = make_client(items=[PROFILE_ITEM])
        store = SessionLookup(client, "sdr-sessions")
        assert store.get_session("s1", telegram_user_id=42) is None

    def test_unknown_session_id_never_matches_composite_key(self):
        client = make_client(items=[PROFILE_ITEM])
        store = SessionLookup(client, "sdr-sessions")
        assert store.get_session("other-session", telegram_user_id=42) is None

    def test_missing_telegram_user_id_returns_none_without_query(self):
        client = make_client(items=[PROFILE_ITEM], conv_item=CONV_ITEM)
        store = SessionLookup(client, "sdr-sessions")
        assert store.get_session("s1") is None
        client.query.assert_not_called()

    def test_scalar_lead_id_in_gsi_item_is_supported(self):
        client = make_client(items=[{"lead_id": "l1"}], conv_item=CONV_ITEM)
        store = SessionLookup(client, "sdr-sessions")
        assert store.get_session("s1", telegram_user_id=42) is not None

    def test_never_uses_nonexistent_session_index_gsi(self):
        client = make_client(items=[PROFILE_ITEM], conv_item=CONV_ITEM)
        store = SessionLookup(client, "sdr-sessions")
        store.get_session("s1", telegram_user_id=42)
        assert client.query.call_args.kwargs["IndexName"] == "telegram-user-index"
