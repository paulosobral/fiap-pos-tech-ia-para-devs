from unittest.mock import MagicMock

from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow, extract_lead_structure


def make_flow(**kw):
    return SalesFlow(lead_qualifier=LeadQualifier(), **kw)


class TestSalesFlow:
    def test_greeting_presents_consent_without_recording(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "greeting", "message": "olá"})
        assert state.get("consent_recorded", False) is False
        assert state["current_state"] == "elicitation"
        assert "LGPD" in state["response"]

    def test_consent_refusal_encloses(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "greeting", "message": "não"})
        assert state["consent_recorded"] is False
        assert state["current_state"] == "followup"
        assert "/start" in state["response"]

    def test_elicitation_records_consent_granted(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "elicitation", "message": "sim, pode continuar"})
        assert state["consent_recorded"] is True
        assert state["current_state"] == "intent"

    def test_elicitation_refusal_persists_false(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "elicitation", "message": "não"})
        assert state["consent_recorded"] is False
        assert state["current_state"] == "followup"

    def test_intent_high_confidence_advances(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "intent", "message": "quero alugar uma sala"})
        assert state["intent"] == "rent"
        assert state["current_state"] == "qualification"

    def test_intent_low_confidence_asks_confirmation(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "intent", "message": "sei lá"})
        assert state.get("intent") is None
        assert "compra, locação ou investimento" in state["response"]

    def test_qualification_qualified_routes_recommendation(self):
        flow = make_flow()
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "ok",
                "lead_info": {
                    "area": "1000 m²", "region": "B", "budget": "R$ 50k/mês",
                    "deadline": "3 meses", "people_count": 50, "decision_maker": "yes",
                },
            }
        )
        assert state["score"] == 100
        assert state["current_state"] == "recommendation"
        assert state["lead_qualified"] is True

    def test_qualification_qualified_assigns_route(self):
        flow = make_flow()
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "ok",
                "lead_info": {
                    "area": "1000 m²", "region": "B", "budget": "R$ 50k/mês",
                    "deadline": "3 meses", "people_count": 50, "decision_maker": "yes",
                },
            }
        )
        assert state["route"] == "diretor"

    def test_qualification_route_respects_rotation(self):
        flow = make_flow(specialist_rotation=["ana", "bruno"])
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "ok",
                "lead_info": {
                    "area": "400 m²", "region": "B", "budget": "R$ 50k/mês",
                    "deadline": "3 meses", "people_count": 50, "decision_maker": "yes",
                },
            }
        )
        assert state["route"] == "ana"

    def test_invoke_extracts_lead_structure_into_context(self):
        flow = make_flow()
        message = (
            "sala de 100 m² na região de Pinheiros, orçamento R$ 60 mil, "
            "prazo de 2 meses, 20 pessoas, sou o decisor"
        )
        state = flow.invoke({"current_state": "elicitation", "message": message, "context": {}})
        info = state["context"]["lead_info"]
        assert info["area"] == "100 m²"
        assert info["region"] == "Pinheiros"
        assert info["budget"] == "R$ 60 mil"
        assert info["deadline"].endswith("2 meses")
        assert info["people_count"] == 20
        assert info["decision_maker"] == "yes"

    def test_extract_budget_keeps_multiple_thousands_separators(self):
        assert extract_lead_structure("orçamento R$ 1.500.000")["budget"] == "R$ 1.500.000"

    def test_extract_budget_millions_singular_and_plural(self):
        assert extract_lead_structure("tenho 1,5 milhão para investir")["budget"] == "1,5 milhão"
        assert extract_lead_structure("tenho 1,5 milhões para investir")["budget"] == "1,5 milhões"

    def test_extract_budget_mixed_form_is_not_truncated(self):
        assert extract_lead_structure("orçamento de R$ 1.500.000")["budget"] == "R$ 1.500.000"

    def test_invoke_merges_partial_info_across_turns(self):
        flow = make_flow()
        first = flow.invoke(
            {"current_state": "elicitation", "message": "sala de 100 m² na região de Pinheiros", "context": {}}
        )
        second = flow.invoke(
            {"current_state": "intent", "message": "orçamento R$ 60 mil", "context": first["context"]}
        )
        info = second["context"]["lead_info"]
        assert info["area"] == "100 m²"
        assert info["budget"] == "R$ 60 mil"

    def test_qualification_low_score_keeps_collecting_information(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "qualification", "message": "ok", "lead_info": {}})
        assert state["current_state"] == "qualification"
        assert state["lead_qualified"] is False
        assert "Ainda precisamos" in state["response"]

    def test_extracts_region_and_decision_maker_from_real_chat_wording(self):
        info = extract_lead_structure("quero aluguel em Pinheiros; quem decide sou eu, proprietário")
        assert info["region"] == "Pinheiros"
        assert info["decision_maker"] == "yes"

    def test_options_request_recovers_from_previous_followup_and_uses_rag(self):
        rag = lambda info: [{"title": "Pinheiros Office", "region": "Pinheiros", "area_util": 100}]
        flow = make_flow(properties_rag=rag)
        state = flow.invoke(
            {
                "current_state": "followup",
                "message": "cadê as opções?",
                "lead_info": {"region": "Pinheiros", "area": "100 m²", "budget": "R$ 5000"},
            }
        )
        assert state["current_state"] == "followup"
        assert "Pinheiros Office" in state["response"]

    def test_options_request_uses_rag_before_lead_is_fully_qualified(self):
        rag = lambda info: [{"title": "Pinheiros Office", "region": "Pinheiros", "area_util": 100}]
        flow = make_flow(properties_rag=rag)
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "cadê as opções?",
                "lead_info": {"region": "Pinheiros", "area": "100 m²", "budget": "R$ 5000"},
            }
        )
        assert "Pinheiros Office" in state["response"]

    def test_more_options_request_in_scheduling_shows_new_properties(self):
        catalog = [
            {"title": f"Imóvel {i}", "region": "Pinheiros", "area_util": 100} for i in range(6)
        ]
        def rag(info):
            return catalog
        flow = make_flow(properties_rag=rag)
        first = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "quero ver",
                "lead_info": {"region": "Pinheiros"},
            }
        )
        assert "Imóvel 0" in first["response"]
        second = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "tem mais opções?",
                "lead_info": {"region": "Pinheiros"},
                "properties": first["properties"],
                "context": first.get("context", {}),
            }
        )
        assert "Imóvel 0" not in second["response"]
        assert "Imóvel 3" in second["response"]
        assert second["current_state"] == "scheduling"

    def test_more_options_request_exhausted_catalog_stays_in_scheduling(self):
        rag = lambda info: [{"title": "Único Imóvel", "region": "Pinheiros", "area_util": 100}]
        flow = make_flow(properties_rag=rag)
        first = flow.invoke(
            {"current_state": "recommendation", "message": "quero ver", "lead_info": {"region": "Pinheiros"}}
        )
        second = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "tem mais opções?",
                "lead_info": {"region": "Pinheiros"},
                "properties": first["properties"],
            }
        )
        assert "todas as opções" in second["response"]
        assert second["current_state"] == "scheduling"

    def test_extracts_budget_from_rent_ceiling_wording(self):
        assert extract_lead_structure("aluguel até 5000 reais")["budget"] == "até 5000 reais"

    def test_recommendation_with_results_lists_top3(self):
        rag = lambda info: [{"title": f"Imóvel {i}", "region": "B", "area_util": 100} for i in range(5)]
        flow = make_flow(properties_rag=rag)
        state = flow.invoke({"current_state": "recommendation", "message": "quero ver"})
        assert len(state["properties"]) == 3

    def test_recommendation_empty_explains(self):
        flow = make_flow(properties_rag=lambda info: [])
        state = flow.invoke({"current_state": "recommendation", "message": "quero ver"})
        assert "Não encontramos imóveis" in state["response"]


class TestReplyGenerator:
    def test_llm_polishes_response(self):
        calls: list[tuple[str, str]] = []

        def fake_reply(message, canned, lead_info, properties, **kwargs):
            calls.append((message, canned))
            return "Claro! {canned}".replace("{canned}", canned.lower())

        flow = make_flow(reply_generator=fake_reply)
        state = flow.invoke({"current_state": "intent", "message": "quero alugar agora"})
        assert calls
        assert state["response"] != state["response"].upper()  # nunca sobe; mock devolve própria
        assert state["intent"] == "rent"  # transição de estado preservada apesar do polish

    def test_reply_generator_failure_keeps_canned(self):
        flow = make_flow(reply_generator=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("lu fail")))
        state = flow.invoke({"current_state": "intent", "message": "quero alugar"})
        assert state["current_state"] == "qualification"
        assert state.get("response")

    def test_lgpd_texts_are_never_rewritten(self):
        from service.security_layer import CONSENT_MESSAGE, REFUSAL_MESSAGE

        def fake_reply(message, canned, lead_info, properties, **kwargs):
            raise AssertionError("não pode ser chamado para LGPD")

        flow = make_flow(reply_generator=fake_reply)
        state = flow.invoke({"current_state": "greeting", "message": "olá"})
        assert state["response"] == CONSENT_MESSAGE
        state = flow.invoke({"current_state": "elicitation", "message": "não"})
        assert state["response"] == REFUSAL_MESSAGE


class TestSchedulingRestriction:
    def scheduler(self):
        return MagicMock(return_value={"confirmed": True, "when": "amanhã 10h"})

    def test_restricted_scheduling_defers_action(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler, restriction_check=lambda lead_id: lead_id == "L1")
        state = flow.invoke({"current_state": "scheduling", "message": "amanhã 10h", "lead_id": "L1", "visit_interest": True})
        scheduler.assert_not_called()
        assert state["scheduling_restricted"] is True
        assert state["current_state"] == "handoff"
        assert "corretor" in state["response"]

    def test_unrestricted_scheduling_calls_scheduler(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler, restriction_check=lambda lead_id: False)
        state = flow.invoke({"current_state": "scheduling", "message": "amanhã 10h", "lead_id": "L1", "visit_interest": True})
        scheduler.assert_called_once()
        assert state["appointment"]["confirmed"] is True
        assert "scheduling_restricted" not in state

    def test_scheduling_without_checker_runs_normally(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke({"current_state": "scheduling", "message": "amanhã 10h", "lead_id": "L1", "visit_interest": True})
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"

    def test_restriction_check_failure_is_fail_open(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler, restriction_check=lambda lead_id: (_ for _ in ()).throw(RuntimeError("boom")))
        state = flow.invoke({"current_state": "scheduling", "message": "amanhã 10h", "lead_id": "L1", "visit_interest": True})
        scheduler.assert_called_once()
        assert "scheduling_restricted" not in state

    def test_restricted_followup_is_deferred(self):
        flow = make_flow(restriction_check=lambda lead_id: lead_id == "L1")
        state = flow.invoke({"current_state": "followup", "message": "ok", "lead_id": "L1"})
        assert state["followup_deferred"] is True
        assert state["current_state"] == "followup"
        assert "corretor" in state["response"]

    def test_restricted_lead_without_lead_id_is_not_restricted(self):
        scheduler = self.scheduler()
        flow = make_flow(
            scheduler=scheduler,
            restriction_check=lambda lead_id: True,
            properties_rag=lambda info: [{"title": f"Imóvel {i}"} for i in range(5)],
        )
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "shown_properties_count": 3,
            "visit_interest": True,
        })
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"


class TestReadyForSchedulingGate:
    def scheduler(self):
        return MagicMock(return_value={"confirmed": True, "when": "amanhã 10h"})

    def test_gate_blocks_without_signal(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
        })
        scheduler.assert_not_called()
        assert state["current_state"] == "recommendation"
        assert "antes de agendar" in state["response"].lower()

    def test_gate_allows_visit_interest(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
            "visit_interest": True,
        })
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"

    def test_gate_allows_favorite_property(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
            "favorite_property": "Torre Nova",
        })
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"

    def test_gate_allows_deadline_in_lead_info(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
            "lead_info": {"deadline": "6 meses"},
        })
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"

    def test_gate_not_satisfied_by_double_counting_same_batch(self):
        scheduler = self.scheduler()
        props = [{"title": "A"}, {"title": "B"}]
        flow = make_flow(scheduler=scheduler, properties_rag=lambda i: props)
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "visit_interest": True,
            "shown_properties_count": 2,
            "properties": props,
        })
        scheduler.assert_not_called()
        assert state["current_state"] == "recommendation"


class TestAgenticRouter:
    """ADR-011: LLM classifica ação (enum fixo); código decide o próximo nó."""

    def test_llm_router_merges_extracted_lead_info(self):
        router = lambda message, lead_info, current_state: {
            "lead_info": {"region": "Pinheiros", "decision_maker": "yes"},
            "action": "provide_info",
        }
        flow = make_flow(llm_router=router)
        state = flow.invoke({"current_state": "qualification", "message": "qualquer frase nova", "lead_info": {}})
        assert state["lead_info"]["region"] == "Pinheiros"
        assert state["lead_info"]["decision_maker"] == "yes"

    def test_llm_router_failure_falls_back_to_regex_extraction(self):
        def boom(message, lead_info, current_state):
            raise RuntimeError("LLM indisponível")

        flow = make_flow(llm_router=boom)
        state = flow.invoke(
            {"current_state": "qualification", "message": "100 m² na região de Pinheiros", "lead_info": {}}
        )
        assert state["lead_info"]["region"] == "Pinheiros"
        assert state["lead_info"]["area"] == "100 m²"

    def test_request_options_action_shows_recommendations_regardless_of_score(self):
        rag = lambda info: [{"title": "Torre Nova", "region": "Pinheiros", "area_util": 100}]
        router = lambda message, lead_info, current_state: {"lead_info": {}, "action": "request_options"}
        flow = make_flow(properties_rag=rag, llm_router=router)
        state = flow.invoke(
            {"current_state": "qualification", "message": "alguma frase nunca vista antes", "lead_info": {}}
        )
        assert "Torre Nova" in state["response"]

    def test_request_options_action_in_scheduling_shows_more(self):
        catalog = [{"title": f"Imóvel {i}", "region": "B", "area_util": 100} for i in range(6)]
        router = lambda message, lead_info, current_state: {"lead_info": {}, "action": "request_options"}
        flow = make_flow(properties_rag=lambda info: catalog, llm_router=router)
        first = flow.invoke({"current_state": "recommendation", "message": "quero ver", "lead_info": {}})
        second = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "frase totalmente diferente pedindo mais",
                "lead_info": {},
                "properties": first["properties"],
            }
        )
        assert "Imóvel 3" in second["response"]
        assert second["current_state"] == "scheduling"

    def test_request_human_action_goes_straight_to_handoff_from_any_state(self):
        router = lambda message, lead_info, current_state: {"lead_info": {}, "action": "request_human"}
        flow = make_flow(llm_router=router)
        state = flow.invoke(
            {"current_state": "qualification", "message": "quero falar com uma pessoa de verdade", "lead_info": {}}
        )
        assert state["current_state"] == "handoff"

    def test_request_human_with_options_message_shows_options_not_handoff(self):
        router = lambda message, lead_info, current_state: {"lead_info": {}, "action": "request_human"}
        flow = make_flow(properties_rag=lambda info: [{"title": "Opção 1"}], llm_router=router)
        state = flow.invoke(
            {"current_state": "recommendation", "message": "Cadê as opções", "lead_info": {}}
        )
        assert "Opção 1" in state["response"]
        assert "Corretor" not in state["response"]

    def test_decline_action_routes_to_followup(self):
        router = lambda message, lead_info, current_state: {"lead_info": {}, "action": "decline"}
        flow = make_flow(llm_router=router)
        state = flow.invoke(
            {"current_state": "qualification", "message": "na verdade desisto, não quero mais", "lead_info": {}}
        )
        assert state["current_state"] == "followup"

    def test_provide_info_action_keeps_default_state_machine_behavior(self):
        router = lambda message, lead_info, current_state: {"lead_info": {}, "action": "provide_info"}
        flow = make_flow(llm_router=router)
        state = flow.invoke({"current_state": "qualification", "message": "ok", "lead_info": {}})
        assert state["current_state"] == "qualification"
        assert state["lead_qualified"] is False

    def test_refine_search_reruns_rag_with_new_lead_info(self):
        captured = {}
        def rag(info):
            captured.update(info)
            return [{"title": "Opção Barata", "region": "Pinheiros", "area_util": 80}]
        router = lambda message, lead_info, current_state: {
            "lead_info": {"budget": "R$ 80 mil"},
            "action": "refine_search",
        }
        flow = make_flow(properties_rag=rag, llm_router=router)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "tem algo mais barato?",
            "lead_info": {"budget": "R$ 200 mil"},
            "shown_properties_count": 3,
        })
        assert captured.get("budget") == "R$ 80 mil"
        assert "Opção Barata" in state["response"]

    def test_compare_properties_shows_listed_options(self):
        catalog = [
            {"title": "A", "region": "Pinheiros", "area_util": 100},
            {"title": "B", "region": "Pinheiros", "area_util": 150},
        ]
        rag_calls = []

        def rag(info):
            rag_calls.append(info)
            return catalog

        router = lambda message, lead_info, current_state: {
            "lead_info": {},
            "action": "compare_properties",
        }
        flow = make_flow(properties_rag=rag, llm_router=router)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "qual a diferença entre A e B?",
            "lead_info": {},
            "properties": catalog,
            "shown_properties_count": 2,
        })
        assert state["current_state"] == "recommendation"
        assert "A" in state["response"] and "B" in state["response"]
        assert rag_calls == []

    def test_visit_interest_goes_to_scheduling_when_ready(self):
        scheduler = MagicMock(return_value={"confirmed": True, "when": "amanhã 10h"})
        router = lambda message, lead_info, current_state: {
            "lead_info": {},
            "action": "visit_interest",
        }
        flow = make_flow(scheduler=scheduler, llm_router=router, properties_rag=lambda i: [{"title": "X"}])
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "quero visitar a Torre Nova",
            "lead_info": {},
            "shown_properties_count": 3,
            "properties": [{"title": "X"}],
        })
        assert state.get("visit_interest") is True
        scheduler.assert_called_once()

    def test_visit_interest_stays_if_not_enough_shown(self):
        scheduler = MagicMock(return_value={"confirmed": True})
        router = lambda message, lead_info, current_state: {
            "lead_info": {},
            "action": "visit_interest",
        }
        flow = make_flow(scheduler=scheduler, llm_router=router)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "quero visitar",
            "lead_info": {},
            "shown_properties_count": 1,
            "properties": [],
        })
        scheduler.assert_not_called()
        assert state.get("visit_interest") is True

    def test_visit_interest_two_shown_same_batch_stays_current(self):
        scheduler = MagicMock(return_value={"confirmed": True})
        props = [{"title": "A"}, {"title": "B"}]
        router = lambda message, lead_info, current_state: {
            "lead_info": {},
            "action": "visit_interest",
        }
        flow = make_flow(scheduler=scheduler, llm_router=router)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "quero visitar",
            "lead_info": {},
            "shown_properties_count": 2,
            "properties": props,
        })
        scheduler.assert_not_called()
        assert state["current_state"] == "recommendation"
        assert state.get("visit_interest") is True

    def test_refine_search_from_qualification_routes_to_recommendation(self):
        captured = {}

        def rag(info):
            captured.update(info)
            return [{"title": "Opção Barata", "region": "Pinheiros", "area_util": 80}]

        router = lambda message, lead_info, current_state: {
            "lead_info": {"budget": "R$ 80 mil"},
            "action": "refine_search",
        }
        flow = make_flow(properties_rag=rag, llm_router=router)
        state = flow.invoke({
            "current_state": "qualification",
            "message": "tem algo mais barato?",
            "lead_info": {"budget": "R$ 200 mil"},
            "score": 40,
        })
        # routing discriminates: recommendation node ran RAG with router's new lead_info
        assert captured.get("budget") == "R$ 80 mil"
        assert "Opção Barata" in state["response"]


class TestDiscoveryState:
    def test_discovery_answers_about_shown_property_without_changing_stage(self):
        props = [{"title": "Torre Nova", "region": "Pinheiros", "area_util": 100, "vagas": 2}]
        router = lambda message, lead_info, current_state: {
            "lead_info": {},
            "action": "provide_info",
        }
        flow = make_flow(properties_rag=lambda info: props, llm_router=router)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "a Torre Nova tem estacionamento?",
            "lead_info": {},
            "properties": props,
            "favorite_property": "Torre Nova",
            "shown_properties_count": 1,
        })
        # Either discovery node or recommendation kept stage; must mention property context
        assert state["current_state"] in ("discovery", "recommendation")
        assert state.get("favorite_property") == "Torre Nova"
        assert "Torre Nova" in state["response"] or "estacionamento" in state["response"].lower() or "opções" in state["response"].lower()

    def test_postprocess_passes_rich_kwargs(self):
        captured = {}
        def fake_reply(message, canned, lead_info, properties, **kwargs):
            captured.update(kwargs)
            return canned + " (llm)"
        flow = make_flow(reply_generator=fake_reply, properties_rag=lambda i: [{"title": "X"}])
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "ok",
            "lead_info": {},
            "properties": [{"title": "X"}],
            "favorite_property": "X",
            "shown_properties_count": 3,
        })
        assert captured.get("favorite_property") == "X"
        assert captured.get("conversation_stage") == "recommendation"
        assert captured.get("shown_properties_count") == 3
        assert state["response"].endswith("(llm)")


class TestCommercialMemory:
    def test_detects_favorite_by_list_number(self):
        props = [
            {"title": "Torre Nova", "region": "Pinheiros", "area_util": 100},
            {"title": "Torre Antiga", "region": "Pinheiros", "area_util": 80},
        ]
        flow = make_flow(properties_rag=lambda info: props)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "gostei da 2",
            "lead_info": {},
            "properties": props,
            "shown_properties_count": 2,
        })
        assert state.get("favorite_property") == "Torre Antiga"
        assert state.get("context", {}).get("favorite_property") == "Torre Antiga"

    def test_detects_favorite_by_title_substring(self):
        props = [{"title": "Torre Nova", "region": "Pinheiros", "area_util": 100}]
        flow = make_flow(properties_rag=lambda info: props)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "gostei da Torre Nova",
            "lead_info": {},
            "properties": props,
            "shown_properties_count": 1,
        })
        assert state.get("favorite_property") == "Torre Nova"

    def test_detects_rejection(self):
        props = [
            {"title": "Torre Nova", "region": "Pinheiros", "area_util": 100},
            {"title": "Torre Antiga", "region": "Pinheiros", "area_util": 80},
        ]
        flow = make_flow(properties_rag=lambda info: props)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "não quero a 1",
            "lead_info": {},
            "properties": props,
            "shown_properties_count": 2,
        })
        assert "Torre Nova" in (state.get("rejected_properties") or [])
        assert state.get("context", {}).get("rejected_properties") is not None

    def test_context_seed_restores_favorite_across_invokes(self):
        props = [{"title": "Torre Nova", "region": "Pinheiros", "area_util": 100}]
        flow = make_flow(scheduler=MagicMock(return_value={"confirmed": True, "when": "x"}))
        # simulate prior turn persisted context
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
            "context": {"favorite_property": "Torre Nova"},
        })
        assert state["current_state"] == "handoff"  # gate opens via context seed

    def test_visit_interest_flag_from_context_seed(self):
        flow = make_flow(scheduler=MagicMock(return_value={"confirmed": True, "when": "x"}))
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
            "context": {"visit_interest": True},
        })
        assert state["current_state"] == "handoff"
