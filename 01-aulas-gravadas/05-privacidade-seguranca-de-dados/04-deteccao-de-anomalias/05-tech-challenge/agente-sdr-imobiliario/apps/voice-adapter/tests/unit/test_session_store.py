from unittest.mock import MagicMock

from infra.session_store import SessionLookup


ITEM = {
    "PK": {"S": "LEAD#l1"},
    "SK": {"S": "CONV#s1"},
    "session_id": {"S": "s1"},
    "telegram_user_id": {"N": "42"},
}


def make_client(items):
    client = MagicMock()
    client.query.return_value = {"Items": items}
    return client


class TestSessionLookup:
    def test_get_session_found(self):
        store = SessionLookup(make_client([ITEM]), "sdr-sessions")
        session = store.get_session("s1")
        assert session["session_id"] == "s1"
        assert session["telegram_user_id"] == "42"

    def test_get_session_not_found(self):
        store = SessionLookup(make_client([]), "sdr-sessions")
        assert store.get_session("nope") is None

    def test_user_id_mismatch_returns_none(self):
        store = SessionLookup(make_client([ITEM]), "sdr-sessions")
        assert store.get_session("s1", telegram_user_id=99) is None

    def test_user_id_match_returns_session(self):
        store = SessionLookup(make_client([ITEM]), "sdr-sessions")
        assert store.get_session("s1", telegram_user_id=42) is not None

    def test_query_uses_session_index(self):
        client = make_client([ITEM])
        store = SessionLookup(client, "sdr-sessions")
        store.get_session("s1")
        assert client.query.call_args.kwargs["IndexName"] == "session-index"
        assert client.query.call_args.kwargs["ExpressionAttributeValues"] == {":sid": {"S": "s1"}}
