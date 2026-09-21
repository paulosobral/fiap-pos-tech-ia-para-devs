"""Fixtures compartilhadas dos testes de integração: payloads Telegram sintéticos e doubles de infraestrutura."""
import json
from unittest.mock import MagicMock

import base64

from handler import ConversationRouter
from infra.session_store import SessionStore
from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow
from service.security_layer import KmsPiiRegistry, SecurityLayer


class FakeDynamo:
    """Double do client DynamoDB (docformat nativo) respeitando chave composta PK+SK."""

    def __init__(self):
        self.items = {}

    def query(self, TableName, IndexName=None, **kw):
        if IndexName == "lead-index":
            return {"Items": [item for item in self.items.values() if item["SK"]["S"].startswith("CONV#")]}
        return {"Items": [item for item in self.items.values() if item["SK"]["S"] == "PROFILE"]}

    def get_item(self, TableName, Key):
        pk = Key["PK"]["S"]
        sk = Key.get("SK", {}).get("S")
        for item in self.items.values():
            if item["PK"]["S"] == pk and (sk is None or item["SK"]["S"] == sk):
                return {"Item": item}
        return {}

    def put_item(self, TableName, Item):
        key = Item["PK"]["S"] + "#" + Item["SK"]["S"]
        self.items[key] = Item


class FakeKms:
    """Double determinístico de KMS (base64) para validar o fluxo criptografado do registro de PII."""

    def encrypt(self, KeyId, Plaintext):
        return {"CiphertextBlob": base64.b64encode(Plaintext)}

    def decrypt(self, CiphertextBlob):
        return {"Plaintext": base64.b64decode(CiphertextBlob)}


def telegram_update(text, user_id=42, chat_id=7, voice=None, update_id=1, message_id=1):
    message = {"message_id": message_id, "from": {"id": user_id}, "chat": {"id": chat_id}, "text": text}
    if voice:
        message["voice"] = voice
    return {
        "headers": {"X-Telegram-Bot-Api-Secret-Token": "tok"},
        "body": json.dumps({"update_id": update_id, "message": message}),
    }


def webhook_event(update_body, secret="tok"):
    return {"headers": {"X-Telegram-Bot-Api-Secret-Token": secret}, "body": update_body}


def internal_event(path, body, secret="internal-tok", method="POST"):
    return {"path": path, "httpMethod": method, "headers": {"X-Internal-Secret": secret}, "body": json.dumps(body)}


def make_router(secret="tok", with_pii=False):
    store = SessionStore(FakeDynamo(), "t")
    pii_store = None
    if with_pii:
        pii_store = KmsPiiRegistry(FakeDynamo(), "pii-t", FakeKms(), key_id="key-1")
    security = SecurityLayer(pii_store=pii_store)
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
            pii_store=pii_store,
            internal_secret_token="internal-tok",
        ),
        sqs,
        telegram,
    )
