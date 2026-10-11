"""Decisão do roteador no log e dados de busca vindos de `arguments` (ADR-025)."""
import logging

from service import llm
from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow


class TestRouterDecisionLog:
    def test_router_decision_is_logged_with_tool_args_and_thought(self, caplog):
        def router(message, lead_info, current_state, **kw):
            return {"tool": "property_detail", "arguments": {"property_ref": "1"}, "lead_info": {},
                    "memory_updates": {}, "thought": "o lead se refere ao primeiro imóvel da lista"}

        shown = [{"id": "a", "title": "Casa Vila Gumercindo", "images": []}, {"id": "b", "title": "Apê Água Funda", "images": []}]
        flow = SalesFlow(lead_qualifier=LeadQualifier(), llm_router=router, bot_name_provider=lambda: "Cecília")
        with caplog.at_level(logging.INFO):
            flow.invoke({"current_state": "conversation", "message": "1", "context": {"properties": shown},
                         "properties": shown, "lead_info": {}})
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("roteador:")]
        assert lines, "a decisão do roteador não foi logada"
        line = lines[0]
        assert "tool=property_detail" in line and "property_ref" in line and "imoveis_exibidos=2" in line
        assert "o lead se refere ao primeiro imóvel" in line

    def test_log_shows_when_validation_changed_the_llm_choice(self, caplog):
        def router(message, lead_info, current_state, **kw):
            return {"tool": "ferramenta_inexistente", "arguments": {}, "lead_info": {}, "memory_updates": {}, "thought": "x"}

        flow = SalesFlow(lead_qualifier=LeadQualifier(), llm_router=router, bot_name_provider=lambda: "Cecília")
        with caplog.at_level(logging.INFO):
            flow.invoke({"current_state": "conversation", "message": "oi", "context": {}, "lead_info": {}})
        line = next((r.getMessage() for r in caplog.records if r.getMessage().startswith("roteador:")), "")
        assert "LLM pediu=ferramenta_inexistente" in line


class TestSearchDataFromToolArguments:
    """A LLM às vezes manda a região só em `arguments`; o dado não pode se perder (chat de 09/10)."""

    def _validate(self, raw, state=None):
        from service.validation import validate_router_output

        return validate_router_output(raw, state or {"properties": []}, message="quero ver as opções de são paulo")

    def test_region_given_only_in_arguments_reaches_lead_info(self):
        out = self._validate({"tool": "request_options", "arguments": {"list_scope": "filtered", "region": "São Paulo"},
                              "lead_info": {}, "memory_updates": {}, "thought": "x"})
        assert out["lead_info"].get("region") == "São Paulo"
        assert out["arguments"]["list_scope"] == "filtered"

    def test_explicit_lead_info_wins_over_arguments(self):
        out = self._validate({"tool": "refine_search", "arguments": {"region": "Santo André"},
                              "lead_info": {"region": "São Paulo"}, "memory_updates": {}, "thought": "x"})
        assert out["lead_info"]["region"] == "São Paulo"

    def test_non_search_arguments_and_empty_values_are_not_promoted(self):
        out = self._validate({"tool": "request_options", "arguments": {"list_scope": "all", "send_photos": True, "region": ""},
                              "lead_info": {}, "memory_updates": {}, "thought": "x"})
        assert out["lead_info"] == {}

    def test_invalid_intent_in_arguments_is_still_dropped(self):
        out = self._validate({"tool": "request_options", "arguments": {"intent": "talvez"}, "lead_info": {},
                              "memory_updates": {}, "thought": "x"})
        assert "intent" not in out["lead_info"]

    def test_flow_persists_the_region_and_searches_with_it(self, monkeypatch):
        seen = {}

        def router(message, lead_info, current_state, **kw):
            return {"tool": "request_options", "arguments": {"list_scope": "filtered", "region": "São Paulo"},
                    "lead_info": {}, "memory_updates": {}, "thought": "x"}

        def fake_search(info, top_k=9, **kwargs):
            seen["info"] = dict(info)
            return [{"id": "1", "title": "Casa Vila Gumercindo", "region": "Vila Gumercindo", "images": []}]

        import service.properties_catalog as catalog

        monkeypatch.setattr(catalog, "search_properties", fake_search)
        flow = SalesFlow(lead_qualifier=LeadQualifier(), llm_router=router, properties_rag=lambda info: [], bot_name_provider=lambda: "Cecília")
        out = flow.invoke({"current_state": "conversation", "message": "quero ver as opções de são paulo",
                           "context": {}, "lead_info": {"intent": "purchase"}})
        assert out["context"]["lead_info"].get("region") == "São Paulo"
        assert seen["info"].get("region") == "São Paulo"

    def test_router_prompt_tells_the_llm_to_keep_search_data_in_lead_info(self):
        assert "SEMPRE em lead_info" in llm._ROUTER_SYSTEM_PROMPT
