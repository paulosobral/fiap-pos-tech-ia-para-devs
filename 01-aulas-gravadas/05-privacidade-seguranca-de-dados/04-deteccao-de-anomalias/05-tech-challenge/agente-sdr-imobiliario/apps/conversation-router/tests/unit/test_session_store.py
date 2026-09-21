from unittest.mock import MagicMock

from infra.session_store import SessionStore


def make_client(lead_item=None, conv_items=None):
    client = MagicMock()
    client.query.return_value = {"Items": conv_items or []}
    client.get_item.return_value = {"Item": lead_item} if lead_item else {}
    return client


LEAD_ITEM = {
    "PK": {"S": "LEAD#l1"},
    "SK": {"S": "PROFILE"},
    "lead_id": {"S": "l1"},
    "telegram_user_id": {"N": "42"},
    "status": {"S": "new"},
    "created_at": {"S": "2026-09-20T00:00:00+00:00"},
    "updated_at": {"S": "2026-09-20T00:00:00+00:00"},
}
CONV_ITEM = {
    "PK": {"S": "LEAD#l1"},
    "SK": {"S": "CONV#s1"},
    "session_id": {"S": "s1"},
    "lead_id": {"S": "l1"},
    "current_state": {"S": "greeting"},
    "pii_masked": {"BOOL": False},
    "consent_recorded": {"BOOL": False},
    "created_at": {"S": "2026-09-20T00:00:00+00:00"},
    "ttl": {"N": "7776000"},
}


class TestSessionStore:
    def test_get_or_create_new_session(self):
        store = SessionStore(make_client(), "t")
        lead, conv, created = store.get_or_create(99)
        assert created is True
        assert lead.telegram_user_id == 99
        assert conv.lead_id == lead.lead_id

    def test_get_or_create_existing_session(self):
        client = make_client(lead_item=LEAD_ITEM, conv_items=[CONV_ITEM])
        client.query.side_effect = [
            {"Items": [{"lead_id": {"S": "l1"}}]},
            {"Items": [CONV_ITEM]},
        ]
        store = SessionStore(client, "t")
        lead, conv, created = store.get_or_create(42)
        assert created is False
        assert lead.lead_id == "l1"
        assert conv.session_id == "s1"

    def test_save_writes_two_items(self):
        client = make_client()
        store = SessionStore(client, "t")
        lead, conv, _ = store.get_or_create(5)
        client.put_item.reset_mock()
        store.save(lead, conv)
        assert client.put_item.call_count == 2

    def test_get_by_telegram_user_empty(self):
        store = SessionStore(make_client(), "t")
        lead, conv = store.get_by_telegram_user(123)
        assert lead is None
        assert conv is None
