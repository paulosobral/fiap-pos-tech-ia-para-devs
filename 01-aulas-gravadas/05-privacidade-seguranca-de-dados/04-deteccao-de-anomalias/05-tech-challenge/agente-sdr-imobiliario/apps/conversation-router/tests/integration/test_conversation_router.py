import json
from unittest.mock import MagicMock

import pytest

from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow
from handler import ConversationRouter
from service.security_layer import SecurityLayer
from infra.session_store import SessionStore


class FakeDynamo:
    def __init__(self):
        self.items = {}

    def query(self, TableName, IndexName=None, **kw):
        if IndexName == "lead-index":
            return {"Items": [item for item in self.items.values() if item["SK"]["S"].startswith("CONV#")]}
        return {"Items": [item for item in self.items.values() if item["SK"]["S"] == "PROFILE"]}

    def get_item(self, TableName, Key):
        pk = Key["PK"]["S"]
        for item in self.items.values():
            if item["PK"]["S"] == pk and item["SK"]["S"] == "PROFILE":
                return {"Item": item}
        return {}

    def put_item(self, TableName, Item):
        key = Item["PK"]["S"] + "#" + Item["SK"]["S"]
        self.items[key] = Item


def make_router(secret="tok"):
    store = SessionStore(FakeDynamo(), "t")
    security = SecurityLayer()
    flow = SalesFlow(lead_qualifier=LeadQualifier())
    sqs = MagicMock()
    telegram = MagicMock()
    return (
        ConversationRouter(
            store=store,
            security_layer=security,
            sales_flow=flow,
            sqs_client=sqs,
            telegram_client=telegram,
            voice_queue_url="https://sqs/voice",
            crm_queue_url="https://sqs/crm",
            secret_token=secret,
        ),
        sqs,
        telegram,
    )


def update(text, user_id=42, chat_id=7, voice=None):
    message = {"message_id": 1, "from": {"id": user_id}, "chat": {"id": chat_id}, "text": text}
    if voice:
        message["voice"] = voice
    return {
        "headers": {"X-Telegram-Bot-Api-Secret-Token": "tok"},
        "body": json.dumps({"update_id": 1, "message": message}),
    }


class TestConversationRouter:
    def test_invalid_secret_returns_401(self):
        router, _, _ = make_router()
        event = update("olá")
        event["headers"] = {"X-Telegram-Bot-Api-Secret-Token": "wrong"}
        assert router.handle(event)["statusCode"] == 401

    def test_missing_secret_returns_401(self):
        router, _, _ = make_router()
        event = update("olá")
        event["headers"] = {}
        assert router.handle(event)["statusCode"] == 401

    def test_invalid_payload_returns_400(self):
        router, _, _ = make_router()
        event = {"headers": {"X-Telegram-Bot-Api-Secret-Token": "tok"}, "body": "not json"}
        assert router.handle(event)["statusCode"] == 400

    def test_missing_message_returns_400(self):
        router, _, _ = make_router()
        event = {"headers": {"X-Telegram-Bot-Api-Secret-Token": "tok"}, "body": json.dumps({"update_id": 1})}
        assert router.handle(event)["statusCode"] == 400

    def test_valid_message_returns_200(self):
        router, _, _ = make_router()
        result = router.handle(update("olá"))
        assert result["statusCode"] == 200

    def test_response_sent_to_telegram(self):
        router, _, telegram = make_router()
        router.handle(update("olá", chat_id=7))
        telegram.send_message.assert_called_once()
        assert telegram.send_message.call_args[0][0] == 7

    def test_pii_masked_before_llm_flow(self):
        router, _, telegram = make_router()
        router.handle(update("Meu e-mail é joao@empresa.com"))
        sent = telegram.send_message.call_args[0][1]
        assert "joao@empresa.com" not in sent

    def test_voice_enqueued_to_sqs(self):
        router, sqs, _ = make_router()
        event = update("olá", voice={"file_id": "f1", "duration": 5})
        router.handle(event)
        sqs.send_message.assert_called_once()
        body = json.loads(sqs.send_message.call_args.kwargs["MessageBody"])
        assert body["voice_file_id"] == "f1"
        assert body["session_id"]

    def test_qualified_lead_enqueued_to_crm(self):
        router, sqs, _ = make_router()
        router.handle(update("olá"))
        router.flow.invoke = lambda s: {**s, "lead_qualified": True, "score": 90, "current_state": "recommendation", "response": "ok"}
        router.handle(update("continuar"))
        calls = sqs.send_message.call_args_list
        assert any(json.loads(c.kwargs["MessageBody"]).get("lead_id") for c in calls)
