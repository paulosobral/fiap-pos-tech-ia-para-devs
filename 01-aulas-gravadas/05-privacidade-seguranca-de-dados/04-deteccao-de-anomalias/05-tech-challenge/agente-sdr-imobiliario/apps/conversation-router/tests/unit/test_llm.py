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


class TestGenerateReplyRichContext:
    def test_prompt_is_consultative_and_user_block_has_stage(self, monkeypatch: pytest.MonkeyPatch):
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _make_completion("Certo! Sobre a Torre Nova...")

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        out = lm.generate_reply(
            message="tem estacionamento?",
            canned_response="A Torre Nova tem 2 vagas.",
            lead_info={"region": "Pinheiros"},
            properties=[{"title": "Torre Nova", "area_util": 100}],
            api_key="k",
            favorite_property="Torre Nova",
            conversation_stage="discovery",
            shown_properties_count=3,
        )
        system = captured["messages"][0]["content"]
        assert "consultor" in system.lower()
        assert "1 pergunta" in system or "uma pergunta" in system.lower()
        user = captured["messages"][1]["content"]
        assert "ESTÁGIO DA CONVERSA: discovery" in user
        assert "Torre Nova" in user
        assert "IMÓVEIS JÁ EXIBIDOS: 3" in user
        assert out == "Certo! Sobre a Torre Nova..."

    def test_backward_compat_positional_call(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion("ok"),
        )
        assert lm.generate_reply("oi", "resposta", {}, [], api_key="k") == "ok"


class TestGenerateReplyToolAgent:
    """Task 6: visit_interest + rejected + last_tool kwargs + consultative prompt + long-list cap."""

    def test_generate_reply_accepts_new_kwargs(self):
        import inspect

        sig = inspect.signature(lm.generate_reply)
        assert "visit_interest" in sig.parameters
        assert "rejected_properties" in sig.parameters
        assert "last_tool" in sig.parameters

    def test_reply_prompt_mentions_last_tool_and_consultative(self):
        prompt = lm._REPLY_SYSTEM_PROMPT
        assert "last_tool" in prompt or "ÚLTIMA" in prompt or "TOOL" in prompt.upper()
        assert "1 pergunta" in prompt or "uma pergunta" in prompt.lower()

    def test_user_block_has_new_fields_and_long_list_cap(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _make_completion("ok")

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        props = [{"title": f"P{i}", "area_util": 100} for i in range(5)]
        lm.generate_reply(
            message="oi",
            canned_response="resposta",
            lead_info={},
            properties=props,
            api_key="k",
            visit_interest=True,
            rejected_properties=["Torre Velha"],
            last_tool="request_options",
        )
        assert captured["max_tokens"] == 800
        user = captured["messages"][1]["content"]
        assert "INTERESSE DE VISITA: sim" in user
        assert "REJEITADOS: Torre Velha" in user
        assert "ÚLTIMA TOOL: request_options" in user
        system = captured["messages"][0]["content"]
        assert "1 frase por imóvel" in system.lower() or "uma frase por imóvel" in system.lower()

    def test_short_list_keeps_280_tokens(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _make_completion("ok")

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        lm.generate_reply(
            message="oi",
            canned_response="r",
            lead_info={},
            properties=[{"title": "A"}],
            api_key="k",
        )
        assert captured["max_tokens"] == 280


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


class TestToolContract:
    """Single-step tool-call contract (spec 2026-09-23 §4)."""

    def test_valid_tools_enum(self):
        expected = {
            "request_options", "property_detail", "compare_properties",
            "refine_search", "express_visit_interest", "request_schedule",
            "request_human", "decline", "provide_info", "unclear",
        }
        assert set(lm.VALID_TOOLS) == expected
        assert "visit_interest" not in lm.VALID_TOOLS

    def test_parse_tool_call_full(self):
        raw = (
            '{"thought":"x","tool":"request_options","arguments":{"list_scope":"all"},'
            '"lead_info":{"region":"Pinheiros"},'
            '"memory_updates":{"favorite_property":null,"visit_interest":false}}'
        )
        out = lm._parse_extract_and_route(raw)
        assert out["tool"] == "request_options"
        assert out["arguments"]["list_scope"] == "all"
        assert out["lead_info"]["region"] == "Pinheiros"
        assert out["memory_updates"]["visit_interest"] is False

    def test_parse_invalid_tool_raises(self):
        with pytest.raises(ValueError):
            lm._parse_extract_and_route('{"tool":"invented_tool","thought":"x"}')

    def test_parse_invalid_json_raises(self):
        with pytest.raises(ValueError):
            lm._parse_extract_and_route("not json")

    def test_parse_missing_tool_raises(self):
        with pytest.raises(ValueError):
            lm._parse_extract_and_route('{"thought":"x","lead_info":{}}')

    def test_parse_defaults_for_optional_blocks(self):
        out = lm._parse_extract_and_route('{"tool":"unclear"}')
        assert out["thought"] == ""
        assert out["arguments"] == {}
        assert out["lead_info"] == {}
        assert out["memory_updates"] == {}

    def test_extract_and_route_returns_tool_contract(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion(
                '{"thought":"show options","tool":"request_options",'
                '"arguments":{"list_scope":"filtered"},"lead_info":{"region":"Pinheiros"},'
                '"memory_updates":{"favorite_property":null,"visit_interest":false}}'
            ),
        )
        result = lm.extract_and_route("mostra opções", {}, "conversation", api_key="k")
        assert result["tool"] == "request_options"
        assert "action" not in result
        assert result["lead_info"]["region"] == "Pinheiros"

    def test_extract_and_route_rejects_invalid_tool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion('{"tool":"invented","thought":"x"}'),
        )
        with pytest.raises(ValueError):
            lm.extract_and_route("oi", {}, "conversation", api_key="k")

    def test_router_prompt_documents_tools_and_memory(self):
        prompt = lm._ROUTER_SYSTEM_PROMPT
        assert "express_visit_interest" in prompt
        assert "property_detail" in prompt
        assert "memory_updates" in prompt
        assert "list_scope" in prompt
        assert "visit_interest" in prompt
        assert "favorito" in prompt.lower() or "favorite_property" in prompt


class TestExtractAndRoute:
    """Tool-agent: LLM extrai lead_info livre + escolhe tool do enum fixo."""

    def test_success_extracts_fields_and_tool(self, monkeypatch: pytest.MonkeyPatch):
        def fake_completion(**kwargs):
            return _make_completion(
                '{"thought":"info", "lead_info": {"region": "Pinheiros", "decision_maker": "yes"}, '
                '"tool": "provide_info", "arguments": {}, '
                '"memory_updates": {"favorite_property": null, "visit_interest": false}}'
            )

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        result = lm.extract_and_route(
            message="quero aluguel em Pinheiros, quem decide sou eu",
            lead_info={},
            current_state="conversation",
            api_key="k",
        )
        assert result["lead_info"] == {"region": "Pinheiros", "decision_maker": "yes"}
        assert result["tool"] == "provide_info"
        assert result["thought"] == "info"
        assert "action" not in result

    def test_detects_options_request_tool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion('{"lead_info": {}, "tool": "request_options", "arguments": {}}'),
        )
        result = lm.extract_and_route("tem mais opções?", {}, "scheduling", api_key="k")
        assert result["tool"] == "request_options"

    def test_invalid_tool_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion('{"lead_info": {}, "tool": "invent_transition"}'),
        )
        with pytest.raises(ValueError):
            lm.extract_and_route("oi", {}, "conversation", api_key="k")

    def test_malformed_json_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(lm.litellm, "completion", lambda **kw: _make_completion("não é json"))
        with pytest.raises(ValueError):
            lm.extract_and_route("oi", {}, "conversation", api_key="k")

    def test_unknown_extracted_fields_are_dropped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion(
                '{"lead_info": {"region": "Pinheiros", "nome_do_lead": "Paulo"}, '
                '"tool": "provide_info", "arguments": {}}'
            ),
        )
        result = lm.extract_and_route("meu nome é Paulo, moro em Pinheiros", {}, "conversation", api_key="k")
        assert result["lead_info"] == {"region": "Pinheiros"}

    def test_fallback_tier_kicks_in_on_primary_failure(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_MODEL_PRIMARY", "p")
        monkeypatch.setenv("LLM_MODEL_FALLBACK", "f")
        call_count = [0]

        def fake_completion(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise lm.litellm.AuthenticationError("429", llm_provider="openrouter", model="p")
            return _make_completion('{"lead_info": {}, "tool": "request_schedule", "arguments": {}}')

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        result = lm.extract_and_route("quero agendar", {}, "conversation", api_key="k")
        assert result["tool"] == "request_schedule"
        assert call_count[0] == 2

    def test_accepts_refine_search_tool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion(
                '{"lead_info": {"budget": "R$ 80 mil"}, "tool": "refine_search", "arguments": {}}'
            ),
        )
        result = lm.extract_and_route("tem algo mais barato?", {}, "conversation", api_key="k")
        assert result["tool"] == "refine_search"

    def test_accepts_compare_properties_tool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion('{"lead_info": {}, "tool": "compare_properties", "arguments": {}}'),
        )
        result = lm.extract_and_route("qual a diferença entre 1 e 2?", {}, "conversation", api_key="k")
        assert result["tool"] == "compare_properties"

    def test_accepts_express_visit_interest_tool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion(
                '{"lead_info": {}, "tool": "express_visit_interest", "arguments": {}, '
                '"memory_updates": {"visit_interest": false}}'
            ),
        )
        result = lm.extract_and_route("gostei da Torre Nova", {}, "conversation", api_key="k")
        assert result["tool"] == "express_visit_interest"
        assert "visit_interest" not in lm.VALID_TOOLS


def test_router_prompt_documents_tools():
    assert "express_visit_interest" in lm._ROUTER_SYSTEM_PROMPT
    assert "property_detail" in lm._ROUTER_SYSTEM_PROMPT
    assert "memory_updates" in lm._ROUTER_SYSTEM_PROMPT
    assert "list_scope" in lm._ROUTER_SYSTEM_PROMPT
    assert "APENAS com verbo explícito" in lm._ROUTER_SYSTEM_PROMPT
    assert '"tool"' in lm._ROUTER_SYSTEM_PROMPT


def test_router_prompt_splits_property_detail_vs_list_more():
    prompt = lm._ROUTER_SYSTEM_PROMPT
    segments = {}
    for chunk in prompt.split("   - ")[1:]:
        key = chunk.split(":", 1)[0].strip()
        segments[key] = chunk
    detail = segments.get("property_detail", "")
    options = segments.get("request_options", "")
    assert "estacionamento" in detail
    assert "vaga" in detail
    assert "andar" in detail
    assert "quanto custa" in detail
    assert "tem mais opções" in options
    assert "list_scope" in options
    assert "property_ref" in detail
