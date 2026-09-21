from unittest.mock import MagicMock

from infra.session_store import SessionLookup


def dynamo_item(**pairs):
    return {
        key: {"S": value} if isinstance(value, str) else {"N": str(value)}
        for key, value in pairs.items()
    }


def make_store(items, lead_id=None):
    client = MagicMock()
    client.query.return_value = {"Items": items}
    return SessionLookup(client, "sdr-sessions"), client


class TestSessionLookup:
    def test_found_session_returns_unwrapped(self):
        store, client = make_store([dynamo_item(session_id="s1", lead_id="lead-1")])
        session = store.get_session("s1")
        assert session == {"session_id": "s1", "lead_id": "lead-1"}

    def test_missing_session_returns_none(self):
        store, _ = make_store([])
        assert store.get_session("ghost") is None

    def test_lead_id_mismatch_returns_none(self):
        store, _ = make_store([dynamo_item(session_id="s1", lead_id="lead-9")])
        assert store.get_session("s1", lead_id="lead-1") is None

    def test_lead_id_match_returns_session(self):
        store, _ = make_store([dynamo_item(session_id="s1", lead_id="lead-1")])
        assert store.get_session("s1", lead_id="lead-1") is not None

    def test_query_uses_gsi_and_table(self):
        store, client = make_store([])
        store.get_session("s1")
        client.query.assert_called_once()
        kwargs = client.query.call_args[1]
        assert kwargs["TableName"] == "sdr-sessions"
        assert kwargs["IndexName"] == "session-index"