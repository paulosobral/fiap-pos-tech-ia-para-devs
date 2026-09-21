from unittest.mock import MagicMock

from infra.session_store import SessionLookup


def item(pk, sk, **attrs):
    raw = {"PK": {"S": pk}, "SK": {"S": sk}}
    for key, value in attrs.items():
        raw[key] = {"S": value} if isinstance(value, str) else {"N": str(value)}
    return raw


def make_store(lead_item=None, conv_item=None):
    client = MagicMock()

    def get_item(TableName=None, Key=None):
        sk = Key["SK"]["S"]
        if sk == "PROFILE" and lead_item:
            return {"Item": lead_item}
        if sk.startswith("CONV#") and conv_item and sk == conv_item["SK"]["S"]:
            return {"Item": conv_item}
        return {}

    client.get_item.side_effect = get_item
    return SessionLookup(client, "sdr-sessions"), client


class TestSessionLookup:
    def test_found_lead_and_session_returns_unwrapped(self):
        lead = item("LEAD#lead-1", "PROFILE", lead_id="lead-1", status="qualified")
        conv = item("LEAD#lead-1", "CONV#s1", session_id="s1", lead_id="lead-1")
        store, _ = make_store(lead_item=lead, conv_item=conv)
        session = store.get_session("lead-1", "s1")
        assert session["session_id"] == "s1"
        assert session["lead_id"] == "lead-1"

    def test_missing_lead_returns_none(self):
        conv = item("LEAD#lead-1", "CONV#s1", session_id="s1")
        store, _ = make_store(lead_item=None, conv_item=conv)
        assert store.get_session("lead-1", "s1") is None

    def test_missing_conversation_returns_none(self):
        lead = item("LEAD#lead-1", "PROFILE", lead_id="lead-1")
        store, _ = make_store(lead_item=lead, conv_item=None)
        assert store.get_session("lead-1", "s1") is None

    def test_wrong_session_for_lead_returns_none(self):
        """Sessão só vale para o próprio lead (chave composta CONV# no lead)."""
        lead = item("LEAD#lead-1", "PROFILE", lead_id="lead-1")
        conv = item("LEAD#lead-1", "CONV#s1", session_id="s1")
        store, _ = make_store(lead_item=lead, conv_item=conv)
        assert store.get_session("lead-1", "outra-sessao") is None

    def test_uses_composite_keys_without_gsi(self):
        """Acesso espelha o dono da tabela (u1): get_item LEAD#/PROFILE + LEAD#/CONV#; nenhum GSI."""
        lead = item("LEAD#lead-1", "PROFILE", lead_id="lead-1")
        conv = item("LEAD#lead-1", "CONV#s1", session_id="s1")
        store, client = make_store(lead_item=lead, conv_item=conv)
        store.get_session("lead-1", "s1")
        assert client.get_item.call_count == 2
        calls = client.get_item.call_args_list
        assert all(call.kwargs["TableName"] == "sdr-sessions" for call in calls)
        assert calls[0].kwargs["Key"] == {"PK": {"S": "LEAD#lead-1"}, "SK": {"S": "PROFILE"}}
        assert calls[1].kwargs["Key"] == {"PK": {"S": "LEAD#lead-1"}, "SK": {"S": "CONV#s1"}}
        client.query.assert_not_called()
