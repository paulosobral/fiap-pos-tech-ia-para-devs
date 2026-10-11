import json
from unittest.mock import MagicMock

import pytest

from infra.session_store import (
    SessionError,
    SessionWriter,
    channel_user_id,
    conversation_ttl_seconds,
)


def make_store(fail=False):
    client = MagicMock()
    if fail:
        client.put_item.side_effect = RuntimeError("dynamo down")
    return SessionWriter(client, "sdr-sessions"), client


class TestSessionWriter:
    def test_open_session_writes_lead_and_conversation_items(self):
        store, client = make_store()
        session = store.open_session(
            {"name": "Ana Souza", "email": "ana@empresa.com", "phone": "11988887777"}
        )
        assert client.put_item.call_count == 2
        lead_item = client.put_item.call_args_list[0].kwargs["Item"]
        conv_item = client.put_item.call_args_list[1].kwargs["Item"]
        assert lead_item["PK"]["S"] == f"LEAD#{session['lead_id']}"
        assert lead_item["SK"] == {"S": "PROFILE"}
        assert lead_item["telegram_user_id"] == {"N": str(session["telegram_user_id"])}
        assert lead_item["status"] == {"S": "new"}
        assert conv_item["PK"] == lead_item["PK"]
        assert conv_item["SK"] == {"S": f"CONV#{session['session_id']}"}
        assert conv_item["current_state"] == {"S": "greeting"}
        assert conv_item["pii_masked"] == {"BOOL": False}
        assert conv_item["consent_recorded"] == {"BOOL": False}
        assert conv_item["messages"] == {"S": "[]"}
        assert conv_item["ttl"] == {"N": str(conversation_ttl_seconds())}
        context = json.loads(conv_item["context"]["S"])
        assert context["channel"] == "email"
        assert context["contact"]["name"] == "Ana Souza"
        assert context["contact"]["email"] == "ana@empresa.com"

    def test_session_payload_has_channel_user_id(self):
        store, _ = make_store()
        session = store.open_session({"email": "ana@empresa.com"})
        assert set(session) == {"lead_id", "session_id", "telegram_user_id"}
        assert 0 <= session["telegram_user_id"] < 1_000_000_000

    def test_open_session_requires_email(self):
        store, client = make_store()
        with pytest.raises(SessionError):
            store.open_session({"name": "Sem e-mail"})
        client.put_item.assert_not_called()

    def test_client_failure_raises_session_error(self):
        store, _ = make_store(fail=True)
        with pytest.raises(SessionError):
            store.open_session({"email": "ana@empresa.com"})

    def test_contact_context_omits_missing_fields(self):
        store, client = make_store()
        store.open_session({"email": "x@y.com"})
        conv_item = client.put_item.call_args_list[1].kwargs["Item"]
        context = json.loads(conv_item["context"]["S"])
        assert context["contact"] == {"email": "x@y.com"}

    def test_ttl_is_90_days(self):
        assert conversation_ttl_seconds() == 90 * 24 * 60 * 60


class TestChannelUserId:
    def test_deterministic_for_same_email_and_lead(self):
        assert channel_user_id("Ana@Empresa.com", "l1") == channel_user_id("ana@empresa.com", "l1")

    def test_varies_per_lead(self):
        assert channel_user_id("ana@empresa.com", "l1") != channel_user_id("ana@empresa.com", "l2")
