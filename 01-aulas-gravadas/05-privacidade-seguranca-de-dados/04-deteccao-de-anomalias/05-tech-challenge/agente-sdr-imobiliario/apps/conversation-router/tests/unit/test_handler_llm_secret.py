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

    monkeypatch.setattr(
        handler, "_llm_generate_reply",
        lambda *a, **k: f"ok:{k.get('favorite_property')}:{k.get('conversation_stage')}:{k.get('shown_properties_count')}",
    )
    reply = captured["reply_generator"]
    out = reply(
        "a Torre Nova tem estacionamento?",
        "Sobre a Torre Nova: estacionamento com 2 vaga(s).",
        {"region": "Pinheiros"},
        [{"title": "Torre Nova"}],
        favorite_property="Torre Nova",
        conversation_stage="discovery",
        shown_properties_count=1,
    )
    assert out == "ok:Torre Nova:discovery:1"


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


def test_llm_route_returns_tool_contract(monkeypatch: pytest.MonkeyPatch):
    """Task 7: handler llm_route must return tool-agent dict (tool, not action)."""
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
    monkeypatch.setattr(
        handler,
        "_llm_extract_and_route",
        lambda message, lead_info, current_state, api_key: {
            "thought": "x",
            "tool": "request_options",
            "arguments": {"list_scope": "all"},
            "lead_info": {"region": "Pinheiros"},
            "memory_updates": {},
        },
    )

    handler.handler({"body": '{"message": "mostra tudo"}'}, None)
    assert captured.get("llm_router") is not None
    route = captured["llm_router"]
    out = route("mostra tudo", {}, "conversation")
    assert "tool" in out
    assert "action" not in out
    assert out["tool"] == "request_options"


def test_reply_generator_forwards_tool_agent_kwargs(monkeypatch: pytest.MonkeyPatch):
    """Task 6/7: reply kwargs include visit_interest, rejected, last_tool."""
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

    handler.handler({"body": '{"message": "oi"}'}, None)
    reply = captured["reply_generator"]
    seen: dict = {}

    def fake_gen(*a, **k):
        seen.update(k)
        return "ok"

    monkeypatch.setattr(handler, "_llm_generate_reply", fake_gen)
    reply(
        "msg",
        "canned",
        {},
        [],
        favorite_property="Torre Nova",
        conversation_stage="conversation",
        shown_properties_count=3,
        visit_interest=True,
        rejected_properties=["Velha"],
        last_tool="request_schedule",
    )
    assert seen.get("visit_interest") is True
    assert seen.get("rejected_properties") == ["Velha"]
    assert seen.get("last_tool") == "request_schedule"
