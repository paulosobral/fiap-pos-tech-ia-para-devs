import json
from datetime import datetime, timezone

import pytest

import handler as handler_module
from infra.pii_reader import PiiReader
from service.leads import LeadNotFoundError, LeadService

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


class FakeConversations:
    def get_lead_profiles(self):
        return {
            "L1": {"lead_id": "L1", "score": 80, "urgency": "alta", "intent": "compra", "budget": "5 mil",
                   "updated_at": "2026-10-07T10:00:00+00:00"},
            "L2": {"lead_id": "L2", "score": 10, "updated_at": "2026-10-06T10:00:00+00:00"},
        }

    def list_conversations(self):
        return [
            {"lead_id": "L1", "session_id": "S1", "created_at": "2026-10-07T09:00:00+00:00", "current_state": "handoff"},
            {"lead_id": "L2", "session_id": "S2", "created_at": "2026-10-06T09:00:00+00:00", "current_state": "qualification"},
        ]


class FakePii:
    def load(self, session_id):
        if session_id == "S1":
            return {"NOME": ["Ana"], "EMAIL": ["ana@x.com"], "TELEFONE": ["11999990000"]}
        return {}


class FakeSqs:
    def __init__(self):
        self.sent = []

    def send_message(self, **kwargs):
        self.sent.append(kwargs)


def make(sqs=None):
    return LeadService(FakeConversations(), FakePii(), sqs or FakeSqs(), "http://queue", now_fn=lambda: NOW)


def test_list_leads_includes_contact_and_orders_by_recent():
    rows = make().list_leads()
    assert [r["lead_id"] for r in rows] == ["L1", "L2"]
    assert rows[0]["name"] == "Ana" and rows[0]["email"] == "ana@x.com" and rows[0]["phone"] == "11999990000"
    assert rows[0]["state"] == "handoff"
    assert rows[1]["name"] == "Lead (nome não informado)" and rows[1]["email"] is None


def test_send_to_crm_publishes_contract_4_message():
    sqs = FakeSqs()
    make(sqs).send_to_crm("L1")
    sent = sqs.sent[0]
    assert sent["QueueUrl"] == "http://queue"
    body = json.loads(sent["MessageBody"])
    assert body["lead_id"] == "L1" and body["session_id"] == "S1"
    assert body["lead_data"]["name"] == "Ana" and body["lead_data"]["email"] == "ana@x.com"
    assert body["lead_data"]["score"] == 80


def test_send_to_crm_unknown_lead():
    with pytest.raises(LeadNotFoundError):
        make().send_to_crm("nope")


def test_pii_reader_decrypts_and_degrades(monkeypatch):
    import base64

    class Db:
        def get_item(self, **kw):
            return {"Item": {"ciphertext": {"S": base64.b64encode(b"blob").decode()}}}

    class Kms:
        def decrypt(self, CiphertextBlob):
            return {"Plaintext": b'{"EMAIL": ["a@b.com"]}'}

    assert PiiReader(Db(), Kms()).load("S") == {"EMAIL": ["a@b.com"]}

    class Boom:
        def get_item(self, **kw):
            raise RuntimeError("down")

    assert PiiReader(Boom(), Kms()).load("S") == {}


class TestRoutes:
    def _event(self, method, path, token="t"):
        headers = {"authorization": f"Bearer {token}"} if token else {}
        return {"httpMethod": method, "path": path, "headers": headers}

    def test_requires_token(self):
        assert handler_module.handler(self._event("GET", "/api/leads", token=None))["statusCode"] == 401

    def test_list_and_send(self, monkeypatch):
        service = make()
        monkeypatch.setattr(handler_module, "build_lead_service", lambda *a, **k: service)
        import sys, types

        fake_boto3 = types.SimpleNamespace(client=lambda name: object())
        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
        listed = handler_module.handler(self._event("GET", "/api/leads"))
        assert listed["statusCode"] == 200 and len(json.loads(listed["body"])["leads"]) == 2
        sent = handler_module.handler(self._event("POST", "/api/leads/L1/crm"))
        assert sent["statusCode"] == 202
        missing = handler_module.handler(self._event("POST", "/api/leads/zzz/crm"))
        assert missing["statusCode"] == 404
