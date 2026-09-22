from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from service import llm as lm


def _make_completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_classify_intent_success(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _make_completion('{"intent": "rent", "confidence": 0.92}')

    monkeypatch.setattr(lm.litellm, "completion", fake_completion)
    intent, confidence = lm.classify_intent("você tem algo para alugar?", "key-123", model="m/1")
    assert intent == "rent"
    assert confidence == 0.92
    assert captured["model"] == "openrouter/m/1"
    assert captured["api_key"] == "key-123"
    assert captured["max_tokens"] == 40
    assert captured["temperature"] == 0


def test_classify_intent_env_model_and_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_MODEL", "env/model")

    def fake_completion(**kwargs):
        assert kwargs["model"] == "openrouter/env/model"
        return _make_completion('{"intent": "purchase", "confidence": 0.8}')

    monkeypatch.setattr(lm.litellm, "completion", fake_completion)
    intent, confidence = lm.classify_intent("quero comprar", "k")
    assert intent == "purchase"
    assert confidence == 0.8


def test_classify_intent_invalid_content_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(lm.litellm, "completion", lambda **kw: _make_completion("não é json"))
    with pytest.raises(ValueError):
        lm.classify_intent("oi", "k")


def test_classify_intent_unknown_intent_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(lm.litellm, "completion", lambda **kw: _make_completion('{"intent": "signup", "confidence": 0.9}'))
    with pytest.raises(ValueError):
        lm.classify_intent("cadastro", "k")


def test_classify_intent_http_error_propagates(monkeypatch: pytest.MonkeyPatch):
    def boom(**kw):
        raise lm.litellm.AuthenticationError("401", llm_provider="openrouter", model="m")
    monkeypatch.setattr(lm.litellm, "completion", boom)
    with pytest.raises(lm.litellm.AuthenticationError):
        lm.classify_intent("oi", "k")


def test_generate_reply_success(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _make_completion("Ótima escolha! Podemos agendar uma visita?")

    monkeypatch.setattr(lm.litellm, "completion", fake_completion)
    out = lm.generate_reply(
        message="quero ver opções na Faria Lima",
        canned_response="Encontramos estas opções:\n1. Torre Office One — Faria Lima, 500 m²\nGostaria de agendar?",
        lead_info={"intent": "rent", "region": "Faria Lima", "area": "100 m²"},
        properties=[{"title": "Torre Office One", "area_util": 500, "price": 10000}],
        api_key="key",
        model="m/1",
    )
    assert out == "Ótima escolha! Podemos agendar uma visita?"
    assert captured["max_tokens"] == 280
    assert captured["temperature"] == 0.6
    msgs = captured["messages"]
    assert msgs[0]["role"] == "system"
    assert "nunca invente" in msgs[0]["content"]
    assert "verificar com a equipe" in msgs[0]["content"]
    user_block = msgs[1]["content"]
    assert "RESPOSTA OFICIAL" in user_block
    assert "Torre Office One" in user_block


def test_generate_reply_empty_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(lm.litellm, "completion", lambda **kw: _make_completion("   "))
    with pytest.raises(ValueError):
        lm.generate_reply("oi", "resposta", {}, [], api_key="k")


def test_generate_reply_http_error_propagates(monkeypatch: pytest.MonkeyPatch):
    def boom(**kw):
        raise lm.litellm.AuthenticationError("500", llm_provider="openrouter", model="m")
    monkeypatch.setattr(lm.litellm, "completion", boom)
    with pytest.raises(lm.litellm.AuthenticationError):
        lm.generate_reply("oi", "resposta", {}, [], api_key="k")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"intent": "investment", "confidence": 0.5}', ("investment", 0.5)),
        ('{"intent": "unknown", "confidence": 0.2}', ("unknown", 0.2)),
        ('{"intent": "rent", "confidence": 1.5}', ("rent", 1.0)),
        ('{"intent": "rent", "confidence": -0.5}', ("rent", 0.0)),
        ('{"intent": "rent", "confidence": "x"}', (None, None)),
        ("", (None, None)),
        ("{invalid", (None, None)),
        ('{"intent": "purchase"}', ("purchase", 0.0)),
    ],
)
def test_parse(raw: str, expected: tuple[object, object]):
    assert lm._parse(raw) == expected
