from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from service import llm as lm


def _make_completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class TestResolveModel:
    def test_explicit_model_wins(self):
        assert lm.resolve_model(lm.TIER_PRIMARY, explicit_model="custom/m") == "custom/m"

    def test_env_per_tier(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_MODEL_PRIMARY", "env/main")
        monkeypatch.setenv("LLM_MODEL_FALLBACK", "env/backup")
        monkeypatch.setenv("LLM_MODEL_COMPLEX", "env/brain")
        assert lm.resolve_model(lm.TIER_PRIMARY) == "env/main"
        assert lm.resolve_model(lm.TIER_FALLBACK) == "env/backup"
        assert lm.resolve_model(lm.TIER_COMPLEX) == "env/brain"

    def test_defaults_by_tier(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_MODEL_PRIMARY", raising=False)
        monkeypatch.delenv("LLM_MODEL_FALLBACK", raising=False)
        monkeypatch.delenv("LLM_MODEL_COMPLEX", raising=False)
        monkeypatch.delenv("LLM_MODEL_PRIMARY_SSM", raising=False)
        monkeypatch.delenv("LLM_MODEL_FALLBACK_SSM", raising=False)
        monkeypatch.delenv("LLM_MODEL_COMPLEX_SSM", raising=False)
        lm._SSM_CACHE.clear()
        assert lm.resolve_model(lm.TIER_PRIMARY) == lm.DEFAULT_MODEL_PRIMARY
        assert lm.resolve_model(lm.TIER_FALLBACK) == lm.DEFAULT_MODEL_FALLBACK
        assert lm.resolve_model(lm.TIER_COMPLEX) == lm.DEFAULT_MODEL_COMPLEX

    def test_unknown_tier_falls_back_to_primary_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_MODEL_PRIMARY", raising=False)
        monkeypatch.delenv("LLM_MODEL_PRIMARY_SSM", raising=False)
        assert lm.resolve_model("bogus") == lm.DEFAULT_MODEL_PRIMARY


class TestClassifyIntent:
    def test_success(self, monkeypatch: pytest.MonkeyPatch):
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

    def test_env_model(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_MODEL_PRIMARY", "env/main")

        def fake_completion(**kwargs):
            assert kwargs["model"] == "openrouter/env/main"
            return _make_completion('{"intent": "purchase", "confidence": 0.8}')

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        intent, confidence = lm.classify_intent("quero comprar", "k")
        assert intent == "purchase"

    def test_fallback_on_primary_failure(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_MODEL_PRIMARY", "env/main")
        monkeypatch.setenv("LLM_MODEL_FALLBACK", "env/backup")
        call_count = [0]

        def fake_completion(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise lm.litellm.AuthenticationError("429", llm_provider="openrouter", model="env/main")
            return _make_completion('{"intent": "purchase", "confidence": 0.8}')

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        intent, _ = lm.classify_intent("quero comprar", "k")
        assert intent == "purchase"
        assert call_count[0] == 2

    def test_invalid_content_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lm.litellm, "completion", lambda **kw: _make_completion("não é json"))
        with pytest.raises(ValueError):
            lm.classify_intent("oi", "k")

    def test_unknown_intent_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lm.litellm, "completion", lambda **kw: _make_completion('{"intent": "signup", "confidence": 0.9}'))
        with pytest.raises(ValueError):
            lm.classify_intent("cadastro", "k")

    def test_both_primary_and_fallback_fail(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_MODEL_PRIMARY", "p")
        monkeypatch.setenv("LLM_MODEL_FALLBACK", "f")

        def boom(**kw):
            raise lm.litellm.AuthenticationError("down", llm_provider="openrouter", model="any")
        monkeypatch.setattr(lm.litellm, "completion", boom)
        with pytest.raises(lm.litellm.AuthenticationError):
            lm.classify_intent("oi", "k")


class TestGenerateReply:
    def test_success(self, monkeypatch: pytest.MonkeyPatch):
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
        msgs = captured["messages"]
        assert msgs[0]["role"] == "system"
        assert "NUNCA invente" in msgs[0]["content"] or "nunca invente" in msgs[0]["content"].lower()
        user_block = msgs[1]["content"]
        assert "RESPOSTA OFICIAL" in user_block
        assert "Torre Office One" in user_block

    def test_force_complex_uses_complex_model(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_MODEL_COMPLEX", "env/brain")
        captured: dict[str, object] = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _make_completion("Análise completa.")

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        lm.generate_reply("negociação complexa", "resposta", {}, [], api_key="k", force_complex=True)
        assert captured["model"] == "openrouter/env/brain"

    def test_primary_with_fallback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_MODEL_PRIMARY", "env/main")
        monkeypatch.setenv("LLM_MODEL_FALLBACK", "env/backup")
        call_count = [0]

        def fake_completion(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise lm.litellm.AuthenticationError("timeout", llm_provider="openrouter", model="env/main")
            return _make_completion("resposta do fallback")

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        out = lm.generate_reply("oi", "resposta", {}, [], api_key="k")
        assert out == "resposta do fallback"
        assert call_count[0] == 2

    def test_empty_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lm.litellm, "completion", lambda **kw: _make_completion("   "))
        with pytest.raises(ValueError):
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


class TestExtractAndRoute:
    """ADR-011: LLM extrai lead_info livre + classifica ação em enum fixo."""

    def test_success_extracts_fields_and_action(self, monkeypatch: pytest.MonkeyPatch):
        def fake_completion(**kwargs):
            return _make_completion(
                '{"lead_info": {"region": "Pinheiros", "decision_maker": "yes"}, '
                '"action": "provide_info"}'
            )

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        result = lm.extract_and_route(
            message="quero aluguel em Pinheiros, quem decide sou eu",
            lead_info={},
            current_state="qualification",
            api_key="k",
        )
        assert result["lead_info"] == {"region": "Pinheiros", "decision_maker": "yes"}
        assert result["action"] == "provide_info"

    def test_detects_options_request_action(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion('{"lead_info": {}, "action": "request_options"}'),
        )
        result = lm.extract_and_route("tem mais opções?", {}, "scheduling", api_key="k")
        assert result["action"] == "request_options"

    def test_invalid_action_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion('{"lead_info": {}, "action": "invent_transition"}'),
        )
        with pytest.raises(ValueError):
            lm.extract_and_route("oi", {}, "qualification", api_key="k")

    def test_malformed_json_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lm.litellm, "completion", lambda **kw: _make_completion("não é json"))
        with pytest.raises(ValueError):
            lm.extract_and_route("oi", {}, "qualification", api_key="k")

    def test_unknown_extracted_fields_are_dropped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion(
                '{"lead_info": {"region": "Pinheiros", "nome_do_lead": "Paulo"}, "action": "provide_info"}'
            ),
        )
        result = lm.extract_and_route("meu nome é Paulo, moro em Pinheiros", {}, "qualification", api_key="k")
        assert result["lead_info"] == {"region": "Pinheiros"}

    def test_fallback_tier_kicks_in_on_primary_failure(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_MODEL_PRIMARY", "p")
        monkeypatch.setenv("LLM_MODEL_FALLBACK", "f")
        call_count = [0]

        def fake_completion(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise lm.litellm.AuthenticationError("429", llm_provider="openrouter", model="p")
            return _make_completion('{"lead_info": {}, "action": "request_schedule"}')

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        result = lm.extract_and_route("quero agendar", {}, "recommendation", api_key="k")
        assert result["action"] == "request_schedule"
        assert call_count[0] == 2
