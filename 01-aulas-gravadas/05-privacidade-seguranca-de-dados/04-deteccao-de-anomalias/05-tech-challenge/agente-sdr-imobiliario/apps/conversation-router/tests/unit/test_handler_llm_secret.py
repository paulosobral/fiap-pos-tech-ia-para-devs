import sys
import types
from unittest.mock import MagicMock

import pytest

import handler


def test_llm_key_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_API_SECRET_ID", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    assert handler._llm_api_key() == "env-key"


def test_llm_key_from_secret_manager(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_SECRET_ID", "arn:aws:secretsmanager:us-east-1::secret/sdr/llm-api-key")
    sm = types.SimpleNamespace(get_secret_value=lambda SecretId: {"SecretString": "sk-or-123"})
    fake = types.ModuleType("boto3")
    fake.client = lambda service: sm
    monkeypatch.setitem(sys.modules, "boto3", fake)
    assert handler._llm_api_key() == "sk-or-123"


def test_llm_key_absent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_SECRET_ID", raising=False)
    assert handler._llm_api_key() is None


def test_llm_key_secret_failure_returns_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_SECRET_ID", "arn:aws:secretsmanager:us-east-1::secret/sdr/llm-api-key")
    fake = types.ModuleType("boto3")

    def boom(service):
        raise RuntimeError("GetSecretValue indisponível")

    fake.client = boom
    monkeypatch.setitem(sys.modules, "boto3", fake)
    assert handler._llm_api_key() is None


def test_handler_function_wires_llm_router_when_key_present(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("INTERNAL_SECRET_TOKEN", "test-token")
    captured: dict = {}

    def fake_sales_flow(**kwargs):
        captured.update(kwargs)
        mock = MagicMock()
        mock.invoke.return_value = {"response": "olá", "current_state": "greeting"}
        return mock

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *args, **kwargs: MagicMock()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    monkeypatch.setattr(handler, "SalesFlow", fake_sales_flow)
    monkeypatch.setattr(handler, "ConversationRouter", lambda **kwargs: MagicMock(handle=lambda e: {"statusCode": 200}))
    monkeypatch.setattr(handler, "SessionStore", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(handler, "SecurityLayer", lambda **kwargs: MagicMock())

    handler.handler({"body": '{"message": "olá"}'}, None)
    assert captured.get("llm_router") is not None
    assert captured.get("reply_generator") is not None


def test_handler_function_leaves_llm_router_none_without_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_SECRET_ID", raising=False)
    monkeypatch.setenv("INTERNAL_SECRET_TOKEN", "test-token")
    captured: dict = {}

    def fake_sales_flow(**kwargs):
        captured.update(kwargs)
        mock = MagicMock()
        mock.invoke.return_value = {"response": "olá", "current_state": "greeting"}
        return mock

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *args, **kwargs: MagicMock()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    monkeypatch.setattr(handler, "SalesFlow", fake_sales_flow)
    monkeypatch.setattr(handler, "ConversationRouter", lambda **kwargs: MagicMock(handle=lambda e: {"statusCode": 200}))
    monkeypatch.setattr(handler, "SessionStore", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(handler, "SecurityLayer", lambda **kwargs: MagicMock())

    handler.handler({"body": '{"message": "olá"}'}, None)
    assert captured.get("llm_router") is None
    assert captured.get("reply_generator") is None
