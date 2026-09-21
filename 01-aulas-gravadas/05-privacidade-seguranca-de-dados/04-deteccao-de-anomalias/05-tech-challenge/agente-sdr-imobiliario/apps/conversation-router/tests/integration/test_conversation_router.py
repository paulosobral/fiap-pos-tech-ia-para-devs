import json
import sys
import types
from unittest.mock import MagicMock

import handler as handler_module
from tests.integration.fixtures import FakeDynamo, make_router, telegram_update as update


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

    def test_production_handler_sends_telegram_reply(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
        monkeypatch.setenv("TELEGRAM_SECRET_TOKEN", "tok")
        monkeypatch.delenv("PII_KMS_KEY_ID", raising=False)
        fake_dynamo = FakeDynamo()
        fake_sqs = MagicMock()

        def fake_boto3_client(name, **kw):
            if name == "dynamodb":
                return fake_dynamo
            if name == "sqs":
                return fake_sqs
            raise AssertionError(name)

        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.client = fake_boto3_client
        telegram = MagicMock()
        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
        monkeypatch.setattr(handler_module, "TelegramApi", lambda token: telegram)
        result = handler_module.handler(update("olá"))
        assert result["statusCode"] == 200
        telegram.send_message.assert_called_once()

    def test_lead_info_extracted_end_to_end_without_seed(self):
        router, _, _ = make_router()
        router.handle(update("/start"))
        router.handle(update("sim"))
        router.handle(update("quero alugar uma sala"))
        router.handle(
            update(
                "sala de 100 m² na região de Pinheiros, orçamento R$ 60 mil, "
                "prazo de 2 meses, 20 pessoas, sou o decisor"
            )
        )
        _, conversation = router.store.get_by_telegram_user(42)
        lead_info = conversation.context["lead_info"]
        assert lead_info["area"] == "100 m²"
        assert lead_info["region"] == "Pinheiros"
        assert lead_info["budget"] == "R$ 60 mil"
        assert lead_info["deadline"].endswith("2 meses")
        assert lead_info["people_count"] == 20
        assert lead_info["decision_maker"] == "yes"

    def test_refusal_persists_consent_recorded_false(self):
        router, _, _ = make_router()
        router.handle(update("/start"))
        router.handle(update("não"))
        _, conversation = router.store.get_by_telegram_user(42)
        assert conversation.consent_recorded is False
        assert conversation.current_state == "followup"

    def test_crm_lead_data_populated_from_pii_registry(self):
        router, sqs, _ = make_router(with_pii=True)
        router.handle(update("Meu e-mail é joao@empresa.com, telefone +55 11 91234-5678"))
        router.flow.invoke = lambda s: {**s, "lead_qualified": True, "score": 90, "current_state": "handoff", "response": "ok"}
        router.handle(update("continuar"))
        crm_bodies = [json.loads(c.kwargs["MessageBody"]) for c in sqs.send_message.call_args_list]
        lead_data = [b["lead_data"] for b in crm_bodies if "lead_data" in b]
        assert lead_data[0]["email"] == "joao@empresa.com"
        assert lead_data[0]["phone"] == "+55 11 91234-5678"

    def test_route_persisted_on_lead(self):
        router, _, _ = make_router()
        router.handle(update("olá"))
        router.flow.invoke = lambda s: {**s, "lead_qualified": True, "score": 90, "current_state": "recommendation", "route": "diretor", "response": "ok"}
        router.handle(update("continuar"))
        lead, _ = router.store.get_by_telegram_user(42)
        assert lead.route == "diretor"
