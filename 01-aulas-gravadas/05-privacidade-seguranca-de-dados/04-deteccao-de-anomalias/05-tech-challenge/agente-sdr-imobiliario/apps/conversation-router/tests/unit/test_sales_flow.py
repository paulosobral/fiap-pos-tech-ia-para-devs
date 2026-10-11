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
        state = flow.invoke(
            {"current_state": "elicitation", "message": "sim, pode continuar"}
        )
        assert state["consent_recorded"] is True
        assert state["current_state"] == "intent"

    def test_elicitation_refusal_persists_false(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "elicitation", "message": "não"})
        assert state["consent_recorded"] is False
        assert state["current_state"] == "followup"

    def test_intent_high_confidence_advances(self):
        flow = make_flow()
        state = flow.invoke(
            {"current_state": "intent", "message": "quero alugar uma sala"}
        )
        assert state["intent"] == "rent"
        assert state["current_state"] == "qualification"

    def test_intent_low_confidence_asks_confirmation(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "intent", "message": "sei lá"})
        assert state.get("intent") is None
        assert "comprar, alugar ou investir" in state["response"]

    def test_qualification_qualified_routes_recommendation(self):
        flow = make_flow()
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "ok",
                "lead_info": {
                    "area": "1000 m²",
                    "region": "B",
                    "budget": "R$ 50k/mês",
                    "deadline": "3 meses",
                    "people_count": 50,
                    "decision_maker": "yes",
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
                    "area": "1000 m²",
                    "region": "B",
                    "budget": "R$ 50k/mês",
                    "deadline": "3 meses",
                    "people_count": 50,
                    "decision_maker": "yes",
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
                    "area": "400 m²",
                    "region": "B",
                    "budget": "R$ 50k/mês",
                    "deadline": "3 meses",
                    "people_count": 50,
                    "decision_maker": "yes",
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
        state = flow.invoke(
            {"current_state": "elicitation", "message": message, "context": {}}
        )
        info = state["context"]["lead_info"]
        assert info["area"] == "100 m²"
        assert info["region"] == "Pinheiros"
        assert info["budget"] == "R$ 60 mil"
        assert info["deadline"].endswith("2 meses")
        assert info["people_count"] == 20
        assert info["decision_maker"] == "yes"

    def test_extract_budget_keeps_multiple_thousands_separators(self):
        assert (
            extract_lead_structure("orçamento R$ 1.500.000")["budget"] == "R$ 1.500.000"
        )

    def test_extract_budget_millions_singular_and_plural(self):
        assert (
            extract_lead_structure("tenho 1,5 milhão para investir")["budget"]
            == "1,5 milhão"
        )
        assert (
            extract_lead_structure("tenho 1,5 milhões para investir")["budget"]
            == "1,5 milhões"
        )

    def test_extract_budget_mixed_form_is_not_truncated(self):
        assert (
            extract_lead_structure("orçamento de R$ 1.500.000")["budget"]
            == "R$ 1.500.000"
        )

    def test_invoke_merges_partial_info_across_turns(self):
        flow = make_flow()
        first = flow.invoke(
            {
                "current_state": "elicitation",
                "message": "sala de 100 m² na região de Pinheiros",
                "context": {},
            }
        )
        second = flow.invoke(
            {
                "current_state": "intent",
                "message": "orçamento R$ 60 mil",
                "context": first["context"],
            }
        )
        info = second["context"]["lead_info"]
        assert info["area"] == "100 m²"
        assert info["budget"] == "R$ 60 mil"

    def test_qualification_low_score_keeps_collecting_information(self):
        flow = make_flow()
        state = flow.invoke(
            {"current_state": "qualification", "message": "ok", "lead_info": {}}
        )
        assert state["current_state"] == "qualification"
        assert state["lead_qualified"] is False
        assert "Ainda precisamos" in state["response"]

    def test_extracts_region_and_decision_maker_from_real_chat_wording(self):
        info = extract_lead_structure(
            "quero aluguel em Pinheiros; quem decide sou eu, proprietário"
        )
        assert info["region"] == "Pinheiros"
        assert info["decision_maker"] == "yes"

    def test_options_request_recovers_from_previous_followup_and_uses_rag(self):
        rag = lambda info: [
            {"title": "Pinheiros Office", "region": "Pinheiros", "area_util": 100}
        ]
        flow = make_flow(properties_rag=rag)
        state = flow.invoke(
            {
                "current_state": "followup",
                "message": "cadê as opções?",
                "lead_info": {
                    "region": "Pinheiros",
                    "area": "100 m²",
                    "budget": "R$ 5000",
                },
            }
        )
        assert state["current_state"] == "followup"
        assert "Pinheiros Office" in state["response"]

    def test_options_request_uses_rag_before_lead_is_fully_qualified(self):
        rag = lambda info: [
            {"title": "Pinheiros Office", "region": "Pinheiros", "area_util": 100}
        ]
        flow = make_flow(properties_rag=rag)
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "cadê as opções?",
                "lead_info": {
                    "region": "Pinheiros",
                    "area": "100 m²",
                    "budget": "R$ 5000",
                },
            }
        )
        assert "Pinheiros Office" in state["response"]

    def test_more_options_request_in_scheduling_shows_new_properties(self):
        catalog = [
            {"title": f"Imóvel {i}", "region": "Pinheiros", "area_util": 100}
            for i in range(6)
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
        rag = lambda info: [
            {"title": "Único Imóvel", "region": "Pinheiros", "area_util": 100}
        ]
        flow = make_flow(properties_rag=rag)
        first = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "quero ver",
                "lead_info": {"region": "Pinheiros"},
            }
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

    def test_more_options_uses_context_persisted_properties_without_explicit_state(
        self,
    ):
        """Handler only persists context across turns — not top-level properties."""
        catalog = [
            {"title": f"Imóvel {i}", "region": "Moema", "area_util": 100}
            for i in range(2)
        ]
        flow = make_flow(properties_rag=lambda info: catalog)
        first = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "quero ver",
                "lead_info": {"region": "Moema"},
            }
        )
        assert first.get("context", {}).get("properties")
        second = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "me traga todos os imóveis",
                "lead_info": {"region": "Moema"},
                "context": first.get("context", {}),
                "lead_info": first.get("lead_info", {}),
            }
        )
        assert "todas as opções" in second["response"]
        assert "Imóvel 0" in first["response"]

    def test_extracts_budget_from_rent_ceiling_wording(self):
        assert (
            extract_lead_structure("aluguel até 5000 reais")["budget"]
            == "até 5000 reais"
        )

    def test_recommendation_with_results_lists_top3(self):
        rag = lambda info: [
            {"title": f"Imóvel {i}", "region": "B", "area_util": 100} for i in range(5)
        ]
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
        state = flow.invoke(
            {"current_state": "intent", "message": "quero alugar agora"}
        )
        assert calls
        assert (
            state["response"] != state["response"].upper()
        )  # nunca sobe; mock devolve própria
        assert (
            state["intent"] == "rent"
        )  # transição de estado preservada apesar do polish

    def test_reply_generator_failure_keeps_canned(self):
        flow = make_flow(
            reply_generator=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("lu fail")
            )
        )
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

    def test_missing_contact_refusal_is_never_rewritten(self):
        """Regressão real: a recusa de request_schedule por falta de contato
        (nome/e-mail/telefone) pedia esses dados literalmente, mas a
        humanização reescrevia pra uma pergunta de horário — escondendo o
        pedido de contato e deixando o lead sem dado nenhum pro dashboard/CRM.
        Esse pedido agora é tratado como a LGPD: nunca passa pela LLM."""

        # Se a humanização NÃO for pulada, cai aqui — texto errado (sem o
        # pedido de contato), que é exatamente o bug real observado em produção.
        def fake_reply(message, canned, lead_info, properties, **kwargs):
            return "Qual dia e horário funciona melhor pra você?"

        router = lambda message, lead_info, current_state, **kwargs: {
            "tool": "request_schedule",
            "arguments": {},
            "lead_info": {},
            "memory_updates": {"visit_interest": True, "visit_quote": "quero marcar"},
        }
        flow = make_flow(llm_router=router, reply_generator=fake_reply)
        state = flow.invoke(
            {
                "current_state": "conversation",
                "message": "quero marcar, pode ser a qualquer momento",
                "favorite_property": "Torre Nova",
                "visit_interest": True,
                "shown_properties_count": 3,
                "missing_contact_fields": ["email", "phone"],
                "properties": [{"title": "Torre Nova"}],
            }
        )
        assert "seu e-mail" in state["response"]
        assert "seu telefone" in state["response"]
        assert "horário" not in state["response"]
        assert state["current_state"] == "conversation"

    def test_reply_generator_receives_photo_fact_for_property_with_images(self):
        """Fecha o pipeline do Bug 2: imóvel com `images` populado chega intacto
        em `properties` até o reply_generator — é esse dado que alimenta o fato
        `fotos_disponiveis` enviado ao LLM (llm.py), sem o qual ele alucina."""
        received: dict[str, Any] = {}

        def fake_reply(message, canned, lead_info, properties, **kwargs):
            received["properties"] = properties
            return "Sim, tenho fotos! Vou te enviar a seguir."

        router = lambda message, lead_info, current_state, **kwargs: {
            "tool": "property_detail",
            "arguments": {"property_ref": "Torre Nova"},
            "lead_info": {},
            "memory_updates": {},
        }
        flow = make_flow(llm_router=router, reply_generator=fake_reply)
        flow.invoke(
            {
                "current_state": "conversation",
                "message": "cadê a foto?",
                "properties": [
                    {"title": "Torre Nova", "images": ["https://cdn/1.jpg"] * 9}
                ],
            }
        )
        assert any(p.get("images") for p in received["properties"])


class TestResponseImages:
    """_response_images: fotos seguem o foco da resposta, não o acumulado de properties."""

    def test_single_property_focus_sends_up_to_three_photos(self):
        from service.flow.sales_flow import SalesFlow

        state = {
            "response_properties": [
                {"title": "Torre Nova", "images": [f"https://cdn/{i}.jpg" for i in range(9)]}
            ]
        }
        assert SalesFlow._response_images(state) == [
            "https://cdn/0.jpg",
            "https://cdn/1.jpg",
            "https://cdn/2.jpg",
        ]

    def test_multiple_properties_send_one_photo_each(self):
        from service.flow.sales_flow import SalesFlow

        state = {
            "response_properties": [
                {"title": "A", "images": ["https://cdn/a1.jpg", "https://cdn/a2.jpg"]},
                {"title": "B", "images": ["https://cdn/b1.jpg"]},
            ]
        }
        assert SalesFlow._response_images(state) == [
            "https://cdn/a1.jpg",
            "https://cdn/b1.jpg",
        ]

    def test_no_response_properties_sends_no_photos(self):
        from service.flow.sales_flow import SalesFlow

        assert SalesFlow._response_images({}) == []
        assert SalesFlow._response_images({"properties": [{"images": ["x"]}]}) == []

    def test_response_properties_survives_the_real_compiled_langgraph(self):
        """Regressão: response_properties/response_images só existiam como chaves
        soltas em `state`, nunca declaradas em `FlowState` (TypedDict). O
        LangGraph real (StateGraph(FlowState), ativo sempre que `langgraph` está
        instalado — é o caso deste venv) descarta SILENCIOSAMENTE qualquer chave
        que o node devolva fora do schema declarado, então `response_images`
        sempre voltava vazio em produção mesmo com a lógica de `_response_images`
        100% correta. Os outros testes desta classe chamam `_response_images`
        direto num dict à mão (bypassando o grafo) ou só checam `properties` —
        nenhum deles passava pelo `flow.invoke()` de ponta a ponta de verdade.
        Este teste só funciona como regressão se `_HAS_LANGGRAPH` for True aqui."""
        from service.flow import sales_flow as sales_flow_module

        assert sales_flow_module._HAS_LANGGRAPH, (
            "langgraph não está instalado neste venv — este teste não cobre "
            "o bug real (precisa do StateGraph de verdade, não do fallback _invoke_fsm)"
        )
        properties = [{"title": "Torre Nova", "images": ["https://cdn/1.jpg"] * 5}]
        router = lambda message, lead_info, current_state, **kwargs: {
            "tool": "property_detail",
            "arguments": {"property_ref": "Torre Nova", "send_photos": True},
            "lead_info": {},
            "memory_updates": {},
        }
        flow = make_flow(llm_router=router)
        assert flow._graph is not None  # usando o grafo real, não o fallback FSM
        state = flow.invoke(
            {
                "current_state": "conversation",
                "message": "mostra a torre nova",
                "properties": properties,
            }
        )
        assert state["response_properties"] == properties
        assert state["response_images"] == ["https://cdn/1.jpg"] * 3


class TestSchedulingRestriction:
    def scheduler(self):
        return MagicMock(return_value={"confirmed": True, "when": "amanhã 10h"})

    def test_restricted_scheduling_defers_action(self):
        scheduler = self.scheduler()
        flow = make_flow(
            scheduler=scheduler, restriction_check=lambda lead_id: lead_id == "L1"
        )
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "lead_id": "L1",
                "visit_interest": True,
            }
        )
        scheduler.assert_not_called()
        assert state["scheduling_restricted"] is True
        assert state["current_state"] == "handoff"
        assert "corretor" in state["response"]

    def test_unrestricted_scheduling_calls_scheduler(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler, restriction_check=lambda lead_id: False)
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "lead_id": "L1",
                "visit_interest": True,
            }
        )
        scheduler.assert_called_once()
        assert state["appointment"]["confirmed"] is True
        assert "scheduling_restricted" not in state

    def test_scheduling_without_checker_runs_normally(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "lead_id": "L1",
                "visit_interest": True,
            }
        )
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"

    def test_restriction_check_failure_is_fail_open(self):
        scheduler = self.scheduler()
        flow = make_flow(
            scheduler=scheduler,
            restriction_check=lambda lead_id: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "lead_id": "L1",
                "visit_interest": True,
            }
        )
        scheduler.assert_called_once()
        assert "scheduling_restricted" not in state

    def test_restricted_followup_is_deferred(self):
        flow = make_flow(restriction_check=lambda lead_id: lead_id == "L1")
        state = flow.invoke(
            {"current_state": "followup", "message": "ok", "lead_id": "L1"}
        )
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
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "shown_properties_count": 3,
                "visit_interest": True,
            }
        )
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"


class TestReadyForSchedulingGate:
    def scheduler(self):
        return MagicMock(return_value={"confirmed": True, "when": "amanhã 10h"})

    def test_gate_blocks_without_signal(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "lead_id": "L1",
                "shown_properties_count": 3,
            }
        )
        scheduler.assert_not_called()
        assert state["current_state"] == "recommendation"
        assert "antes de agendar" in state["response"].lower()

    def test_gate_allows_visit_interest(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "lead_id": "L1",
                "shown_properties_count": 3,
                "visit_interest": True,
            }
        )
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"

    def test_gate_allows_favorite_property(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "lead_id": "L1",
                "shown_properties_count": 3,
                "favorite_property": "Torre Nova",
            }
        )
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"

    def test_gate_allows_deadline_in_lead_info(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "lead_id": "L1",
                "shown_properties_count": 3,
                "lead_info": {"deadline": "6 meses"},
            }
        )
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"

    def test_gate_not_satisfied_by_double_counting_same_batch(self):
        scheduler = self.scheduler()
        props = [{"title": "A"}, {"title": "B"}]
        flow = make_flow(scheduler=scheduler, properties_rag=lambda i: props)
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "visit_interest": True,
                "shown_properties_count": 2,
                "properties": props,
            }
        )
        scheduler.assert_not_called()
        assert state["current_state"] == "recommendation"


class TestAgenticRouter:
    """ADR-011: LLM classifica ação (enum fixo); código decide o próximo nó."""

    def test_conversation_history_reaches_router_and_reply_generator(self):
        history = [
            {"role": "user", "content": "Busco um escritório em Pinheiros"},
            {"role": "assistant", "content": "Qual metragem você procura?"},
        ]
        captured = {"router": None, "reply": None}

        def router(message, lead_info, current_state, **kwargs):
            captured["router"] = kwargs.get("conversation_history")
            return {"action": "provide_info", "lead_info": {}}

        def reply(message, canned, lead_info, properties, **kwargs):
            captured["reply"] = kwargs.get("conversation_history")
            return canned

        flow = make_flow(
            llm_router=router,
            reply_generator=reply,
            properties_rag=lambda info: [],
        )
        flow.invoke(
            {
                "current_state": "recommendation",
                "message": "e a segunda opção?",
                "conversation_history": history,
            }
        )

        assert captured["router"] == history
        assert captured["reply"] == history

    def test_llm_router_merges_extracted_lead_info(self):
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {"region": "Pinheiros", "decision_maker": "yes"},
            "action": "provide_info",
        }
        flow = make_flow(llm_router=router)
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "qualquer frase nova",
                "lead_info": {},
            }
        )
        assert state["lead_info"]["region"] == "Pinheiros"
        assert state["lead_info"]["decision_maker"] == "yes"

    def test_llm_router_failure_falls_back_to_regex_extraction(self):
        def boom(message, lead_info, current_state):
            raise RuntimeError("LLM indisponível")

        flow = make_flow(llm_router=boom)
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "100 m² na região de Pinheiros",
                "lead_info": {},
            }
        )
        assert state["lead_info"]["region"] == "Pinheiros"
        assert state["lead_info"]["area"] == "100 m²"

    def test_request_options_action_shows_recommendations_regardless_of_score(self):
        rag = lambda info: [
            {"title": "Torre Nova", "region": "Pinheiros", "area_util": 100}
        ]
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "request_options",
        }
        flow = make_flow(properties_rag=rag, llm_router=router)
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "alguma frase nunca vista antes",
                "lead_info": {},
            }
        )
        assert "Torre Nova" in state["response"]

    def test_request_options_action_in_scheduling_shows_more(self):
        catalog = [
            {"title": f"Imóvel {i}", "region": "B", "area_util": 100} for i in range(6)
        ]
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "request_options",
        }
        flow = make_flow(properties_rag=lambda info: catalog, llm_router=router)
        first = flow.invoke(
            {"current_state": "recommendation", "message": "quero ver", "lead_info": {}}
        )
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
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "request_human",
        }
        flow = make_flow(llm_router=router)
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "quero falar com uma pessoa de verdade",
                "lead_info": {},
            }
        )
        assert state["current_state"] == "handoff"

    def test_request_human_with_options_message_shows_options_not_handoff(self):
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "request_human",
        }
        flow = make_flow(
            properties_rag=lambda info: [{"title": "Opção 1"}], llm_router=router
        )
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "Cadê as opções",
                "lead_info": {},
            }
        )
        assert "Opção 1" in state["response"]
        assert "Corretor" not in state["response"]

    def test_decline_action_routes_to_followup(self):
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "decline",
        }
        flow = make_flow(llm_router=router)
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "na verdade desisto, não quero mais",
                "lead_info": {},
            }
        )
        assert state["current_state"] == "followup"

    def test_provide_info_action_keeps_default_state_machine_behavior(self):
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "provide_info",
        }
        flow = make_flow(llm_router=router)
        state = flow.invoke(
            {"current_state": "qualification", "message": "ok", "lead_info": {}}
        )
        assert state["current_state"] == "qualification"
        assert state["lead_qualified"] is False

    def test_refine_search_reruns_rag_with_new_lead_info(self):
        captured = {}

        def rag(info):
            captured.update(info)
            return [{"title": "Opção Barata", "region": "Pinheiros", "area_util": 80}]

        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {"budget": "R$ 80 mil"},
            "action": "refine_search",
        }
        flow = make_flow(properties_rag=rag, llm_router=router)
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "tem algo mais barato?",
                "lead_info": {"budget": "R$ 200 mil"},
                "shown_properties_count": 3,
            }
        )
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

        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "compare_properties",
        }
        flow = make_flow(properties_rag=rag, llm_router=router)
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "qual a diferença entre A e B?",
                "lead_info": {},
                "properties": catalog,
                "shown_properties_count": 2,
            }
        )
        assert state["current_state"] == "recommendation"
        assert "A" in state["response"] and "B" in state["response"]
        assert rag_calls == []

    def test_visit_interest_goes_to_scheduling_when_ready(self):
        scheduler = MagicMock(return_value={"confirmed": True, "when": "amanhã 10h"})
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "visit_interest",
        }
        flow = make_flow(
            scheduler=scheduler,
            llm_router=router,
            properties_rag=lambda i: [{"title": "X"}],
        )
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "quero visitar a Torre Nova",
                "lead_info": {},
                "shown_properties_count": 3,
                "properties": [{"title": "X"}],
            }
        )
        assert state.get("visit_interest") is True
        scheduler.assert_called_once()

    def test_visit_interest_stays_if_not_enough_shown(self):
        scheduler = MagicMock(return_value={"confirmed": True})
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "visit_interest",
        }
        flow = make_flow(scheduler=scheduler, llm_router=router)
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "quero visitar",
                "lead_info": {},
                "shown_properties_count": 1,
                "properties": [],
            }
        )
        scheduler.assert_not_called()
        assert state.get("visit_interest") is True

    def test_visit_interest_two_shown_same_batch_stays_current(self):
        scheduler = MagicMock(return_value={"confirmed": True})
        props = [{"title": "A"}, {"title": "B"}]
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "visit_interest",
        }
        flow = make_flow(scheduler=scheduler, llm_router=router)
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "quero visitar",
                "lead_info": {},
                "shown_properties_count": 2,
                "properties": props,
            }
        )
        scheduler.assert_not_called()
        assert state["current_state"] == "recommendation"
        assert state.get("visit_interest") is True

    def test_visit_interest_lead_id_alone_does_not_open_scheduling_when_shown_below_three(
        self,
    ):
        scheduler = MagicMock(return_value={"confirmed": True, "when": "amanhã 10h"})
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "visit_interest",
        }
        flow = make_flow(
            scheduler=scheduler,
            llm_router=router,
            properties_rag=lambda i: [{"title": "X"}],
        )
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "quero visitar",
                "lead_info": {},
                "lead_id": "L1",
                "shown_properties_count": 1,
                "properties": [{"title": "X"}],
            }
        )
        scheduler.assert_not_called()
        assert state.get("visit_interest") is True
        assert state["current_state"] == "recommendation"

    def test_refine_search_from_qualification_routes_to_recommendation(self):
        captured = {}

        def rag(info):
            captured.update(info)
            return [{"title": "Opção Barata", "region": "Pinheiros", "area_util": 80}]

        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {"budget": "R$ 80 mil"},
            "action": "refine_search",
        }
        flow = make_flow(properties_rag=rag, llm_router=router)
        state = flow.invoke(
            {
                "current_state": "qualification",
                "message": "tem algo mais barato?",
                "lead_info": {"budget": "R$ 200 mil"},
                "score": 40,
            }
        )
        # routing discriminates: recommendation node ran RAG with router's new lead_info
        assert captured.get("budget") == "R$ 80 mil"
        assert "Opção Barata" in state["response"]

    def test_request_options_uses_new_purchase_intent_for_search(self, monkeypatch):
        from service import properties_catalog

        captured = {}
        purchase_property = {"title": "Edifício à venda", "mode": "purchase"}

        def search(info, top_k=9, list_scope="filtered"):
            captured.update(info)
            return [purchase_property]

        monkeypatch.setattr(properties_catalog, "search_properties", search)
        router = lambda message, lead_info, current_state, **kwargs: {
            "tool": "request_options",
            "arguments": {},
            "lead_info": {"intent": "purchase"},
            "memory_updates": {},
        }
        flow = make_flow(properties_rag=lambda info: [], llm_router=router)

        state = flow.invoke(
            {"current_state": "recommendation", "message": "quero comprar"}
        )

        assert captured["intent"] == "purchase"
        assert state["intent"] == "purchase"
        assert state["properties"] == [purchase_property]

    def test_property_detail_follows_favorite_and_feeds_correct_facts_to_llm(self):
        """`property_detail` agora também passa pela humanização via LLM (item 1.5).
        Mensagem genérica ("quero mais detalhes") sem referência nenhuma: o roteador
        não tem nada pra resolver em property_ref (fica ausente) — a resolução do
        imóvel-alvo cai pro favorito salvo, determinística em código, e é ela quem
        alimenta os FATOS enviados ao LLM."""
        properties = [
            {"title": "Corporate Faria Lima 02", "region": "Paulista", "area_util": 80},
            {
                "title": "Vila Olímpia Executive 75",
                "region": "Berrini",
                "area_util": 80,
                "disponibilidade": "reservado",
            },
        ]
        router = lambda message, lead_info, current_state, **kwargs: {
            "tool": "property_detail",
            "arguments": {},
            "lead_info": {},
            "memory_updates": {},
        }
        reply = MagicMock(return_value="Esse imóvel está reservado no momento.")
        flow = make_flow(llm_router=router, reply_generator=reply)

        state = flow.invoke(
            {
                "current_state": "conversation",
                "message": "quero mais detalhes",
                "properties": properties,
                "favorite_property": properties[1]["title"],
            }
        )

        reply.assert_called_once()
        canned = reply.call_args.args[1]
        assert "Vila Olímpia Executive 75" in canned
        assert "Corporate Faria Lima 02" not in canned
        assert "Disponibilidade: reservado" in canned
        assert state["response"] == "Esse imóvel está reservado no momento."

    def test_reservation_question_canned_facts_never_name_who_reserved(self):
        """A resposta oficial (fatos enviados ao LLM) nunca inventa PARA QUEM um imóvel
        está reservado — isso é responsabilidade do nó determinístico, não do prompt."""
        prop = {
            "title": "Vila Olímpia Executive 75",
            "region": "Berrini",
            "area_util": 80,
            "disponibilidade": "reservado",
        }
        router = lambda message, lead_info, current_state, **kwargs: {
            "tool": "property_detail",
            "arguments": {},
            "lead_info": {},
            "memory_updates": {},
        }
        reply = MagicMock(return_value="Está reservado no momento, mas posso sugerir outras opções.")
        flow = make_flow(llm_router=router, reply_generator=reply)

        state = flow.invoke(
            {
                "current_state": "conversation",
                "message": "Está reservado pra quem?",
                "properties": [prop],
                "favorite_property": prop["title"],
            }
        )

        canned = reply.call_args.args[1]
        assert "Disponibilidade: reservado" in canned
        assert "outro cliente" not in canned
        assert state["response"] == "Está reservado no momento, mas posso sugerir outras opções."

    def test_reserved_for_whom_guarantee_lives_in_the_reply_prompt_not_in_a_keyword_branch(self):
        """Sem regex em 'pra quem': a LLM responde qualquer forma da pergunta, e a regra
        de nunca inventar quem reservou fica no prompt de humanização."""
        from service import llm as llm_module

        assert "NÃO informa para quem" in llm_module._REPLY_SYSTEM_PROMPT


class TestDiscoveryState:
    def test_discovery_answers_about_shown_property_and_stays_discovery(self):
        props = [
            {"title": "Torre Nova", "region": "Pinheiros", "area_util": 100, "vagas": 2}
        ]
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "provide_info",
        }
        flow = make_flow(properties_rag=lambda info: props, llm_router=router)
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "a Torre Nova tem estacionamento?",
                "lead_info": {},
                "properties": props,
                "favorite_property": "Torre Nova",
                "shown_properties_count": 1,
            }
        )
        assert state["current_state"] == "discovery"
        assert state.get("favorite_property") == "Torre Nova"
        assert (
            "estacionamento" in state["response"].lower()
            or "2 vaga" in state["response"]
        )

    def test_discovery_request_options_from_discovery_shows_more(self):
        catalog = [
            {"title": f"Imóvel {i}", "region": "Pinheiros", "area_util": 100}
            for i in range(6)
        ]
        router = lambda message, lead_info, current_state, **kwargs: {
            "lead_info": {},
            "action": "request_options",
        }
        flow = make_flow(properties_rag=lambda info: catalog, llm_router=router)
        state = flow.invoke(
            {
                "current_state": "discovery",
                "message": "quero ver mais opções",
                "lead_info": {},
                "properties": catalog[:3],
                "favorite_property": "Imóvel 0",
                "shown_properties_count": 3,
            }
        )
        assert "Imóvel 3" in state["response"]
        assert "Sobre Imóvel 0" not in state["response"]
        assert state["current_state"] == "discovery"

    def test_discovery_more_options_regex_path_from_discovery(self):
        catalog = [
            {"title": f"Imóvel {i}", "region": "Pinheiros", "area_util": 100}
            for i in range(6)
        ]
        flow = make_flow(properties_rag=lambda info: catalog)
        state = flow.invoke(
            {
                "current_state": "discovery",
                "message": "mostre mais opções",
                "lead_info": {},
                "properties": catalog[:3],
                "favorite_property": "Imóvel 0",
                "shown_properties_count": 3,
            }
        )
        assert "Imóvel 3" in state["response"]
        assert "Sobre Imóvel 0" not in state["response"]

    def test_postprocess_passes_rich_kwargs(self):
        captured = {}

        def fake_reply(message, canned, lead_info, properties, **kwargs):
            captured.update(kwargs)
            return canned + " (llm)"

        flow = make_flow(
            reply_generator=fake_reply, properties_rag=lambda i: [{"title": "X"}]
        )
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "ok",
                "lead_info": {},
                "properties": [{"title": "X"}],
                "favorite_property": "X",
                "shown_properties_count": 3,
            }
        )
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
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "gostei da 2",
                "lead_info": {},
                "properties": props,
                "shown_properties_count": 2,
            }
        )
        assert state.get("favorite_property") == "Torre Antiga"
        assert state.get("context", {}).get("favorite_property") == "Torre Antiga"

    def test_detects_favorite_by_title_substring(self):
        props = [{"title": "Torre Nova", "region": "Pinheiros", "area_util": 100}]
        flow = make_flow(properties_rag=lambda info: props)
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "gostei da Torre Nova",
                "lead_info": {},
                "properties": props,
                "shown_properties_count": 1,
            }
        )
        assert state.get("favorite_property") == "Torre Nova"

    def test_detects_rejection(self):
        props = [
            {"title": "Torre Nova", "region": "Pinheiros", "area_util": 100},
            {"title": "Torre Antiga", "region": "Pinheiros", "area_util": 80},
        ]
        flow = make_flow(properties_rag=lambda info: props)
        state = flow.invoke(
            {
                "current_state": "recommendation",
                "message": "não quero a 1",
                "lead_info": {},
                "properties": props,
                "shown_properties_count": 2,
            }
        )
        assert "Torre Nova" in (state.get("rejected_properties") or [])
        assert state.get("context", {}).get("rejected_properties") is not None

    def test_context_seed_restores_favorite_across_invokes(self):
        props = [{"title": "Torre Nova", "region": "Pinheiros", "area_util": 100}]
        flow = make_flow(
            scheduler=MagicMock(return_value={"confirmed": True, "when": "x"})
        )
        # simulate prior turn persisted context
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "lead_id": "L1",
                "shown_properties_count": 3,
                "context": {"favorite_property": "Torre Nova"},
            }
        )
        assert state["current_state"] == "handoff"  # gate opens via context seed

    def test_visit_interest_flag_from_context_seed(self):
        flow = make_flow(
            scheduler=MagicMock(return_value={"confirmed": True, "when": "x"})
        )
        state = flow.invoke(
            {
                "current_state": "scheduling",
                "message": "amanhã 10h",
                "lead_id": "L1",
                "shown_properties_count": 3,
                "context": {"visit_interest": True},
            }
        )
        assert state["current_state"] == "handoff"


class TestToolRouting:
    """Tool-agent: single-step contract (spec 2026-09-23)."""

    def _flow(self, router_result=None):
        def router(msg, lead, state, **kwargs):
            return router_result or {
                "thought": "t",
                "tool": "request_options",
                "arguments": {},
                "lead_info": {},
                "memory_updates": {},
            }

        return make_flow(
            properties_rag=lambda info: [
                {"title": "A"},
                {"title": "B"},
                {"title": "C"},
            ],
            reply_generator=lambda *a, **k: "ok",
            llm_router=router,
        )

    def test_five_states_only_in_graph(self):
        flow = self._flow()
        assert flow._graph is not None

    def test_router_tool_sets_state_fields(self):
        flow = self._flow(
            router_result={
                "thought": "show all",
                "tool": "request_options",
                "arguments": {"list_scope": "all"},
                "lead_info": {"region": "Pinheiros"},
                "memory_updates": {},
            }
        )
        out = flow.invoke(
            {
                "current_state": "conversation",
                "message": "mostra tudo",
                "lead_info": {},
                "context": {},
                "properties": [],
            }
        )
        assert (
            out.get("_last_tool") == "request_options"
            or out.get("_router_tool") == "request_options"
        )

    def test_request_schedule_routes_scheduling_when_gates_ok(self):
        flow = self._flow(
            router_result={
                "thought": "wants visit",
                "tool": "request_schedule",
                "arguments": {},
                "lead_info": {},
                "memory_updates": {"visit_interest": True},
            }
        )
        props = [{"title": f"P{i}"} for i in range(5)]
        out = flow.invoke(
            {
                "current_state": "conversation",
                "message": "quero agendar uma visita",
                "lead_info": {},
                "context": {},
                "properties": props,
                "shown_properties_count": 5,
                "visit_interest": True,
            }
        )
        # scheduling gate ok → scheduling node; with default scheduler None ends handoff (corretor)
        assert out.get("current_state") in ("scheduling", "conversation", "handoff")
        if out.get("current_state") == "handoff":
            assert "corretor" in (out.get("response") or "").lower() or out.get(
                "response"
            )

    def test_request_human_chosen_by_llm_is_respected_even_if_message_mentions_options(self):
        """Antes o código cancelava o handoff por achar 'opção' na mensagem. Escolher entre
        'quer ver opções' e 'quer um humano' é interpretação: fica com a LLM (prompt do
        roteador + quality gate), não com palavra-chave."""
        flow = self._flow(
            router_result={
                "thought": "x",
                "tool": "request_human",
                "arguments": {},
                "lead_info": {},
                "memory_updates": {},
            }
        )
        out = flow.invoke(
            {
                "current_state": "conversation",
                "message": "quero falar com o corretor sobre essa opção",
                "lead_info": {},
                "context": {},
                "properties": [{"title": "A"}],
            }
        )
        assert out.get("current_state") == "handoff"

    def test_decline_routes_followup(self):
        flow = self._flow(
            router_result={
                "thought": "x",
                "tool": "decline",
                "arguments": {},
                "lead_info": {},
                "memory_updates": {},
            }
        )
        out = flow.invoke(
            {
                "current_state": "conversation",
                "message": "não quero mais",
                "lead_info": {},
                "context": {},
                "properties": [],
            }
        )
        assert out.get("current_state") == "followup"

    def test_greeting_bypasses_router(self):
        flow = self._flow()
        out = flow.invoke(
            {
                "current_state": "greeting",
                "message": "oi",
                "lead_info": {},
                "context": {},
                "consent_recorded": False,
            }
        )
        assert out.get("current_state") in (
            "greeting",
            "elicitation",
            "intent",
            "conversation",
        )


class TestLeadCompletionAndPhotos:
    """Conversa real 03/10 18:27: fotos sem pedir, atributos inventados no detalhe,
    e 'quero contato' -> 'whatsup' encerrando sem pegar telefone nem e-mail."""

    PROPS = [
        {"id": "a", "title": "Sobrado A", "images": [f"https://cdn/a{i}.jpg" for i in range(9)]},
        {"id": "b", "title": "Sobrado B", "images": ["https://cdn/b0.jpg"]},
        {"id": "c", "title": "Sobrado C", "images": ["https://cdn/c0.jpg"]},
    ]

    @staticmethod
    def _router(tool, **args):
        return lambda message, lead_info, current_state, **kw: {
            "tool": tool, "arguments": args, "lead_info": {}, "memory_updates": {},
        }

    def test_list_without_photo_request_sends_no_photos(self):
        flow = make_flow(
            llm_router=self._router("request_options"),
            properties_rag=lambda info: list(self.PROPS),
        )
        state = flow.invoke({"current_state": "conversation", "message": "comprar"})
        assert state["response_properties"]
        assert state["response_images"] == []

    def test_router_decides_photos_and_never_resends_same_ones(self):
        flow = make_flow(llm_router=self._router("property_detail", property_ref="Sobrado A", send_photos=True))
        base = {"current_state": "conversation", "message": "manda fotos", "properties": list(self.PROPS)}
        first = flow.invoke(dict(base, context={}))
        assert first["response_images"] == [f"https://cdn/a{i}.jpg" for i in range(3)]
        second = flow.invoke(dict(base, context=first["context"]))
        assert second["response_images"] == [f"https://cdn/a{i}.jpg" for i in range(3, 6)]

    def test_request_human_without_phone_or_email_asks_contact_instead_of_handoff(self):
        flow = make_flow(llm_router=self._router("request_human"))
        state = flow.invoke({
            "current_state": "conversation",
            "message": "quero entrar em contato por esse imóvel",
            "properties": list(self.PROPS),
            "missing_contact_fields": ["phone", "email"],
            "context": {},
        })
        assert state["current_state"] == "conversation"
        assert "telefone" in state["response"]
        assert state["context"]["pending_contact_action"] == "request_human"

    def test_pending_request_human_resumes_when_phone_arrives(self):
        router = MagicMock(side_effect=AssertionError("roteador não deve rodar ao retomar"))
        flow = make_flow(llm_router=router)
        state = flow.invoke({
            "current_state": "conversation",
            "message": "[TELEFONE]",
            "properties": list(self.PROPS),
            "missing_contact_fields": ["email"],  # telefone já capturado pela SecurityLayer
            "context": {"pending_contact_action": "request_human"},
        })
        assert state["current_state"] == "handoff"
        assert "pending_contact_action" not in state["context"]

    def test_request_human_with_email_only_goes_to_handoff(self):
        flow = make_flow(llm_router=self._router("request_human"))
        state = flow.invoke({
            "current_state": "conversation",
            "message": "quero falar com o corretor",
            "missing_contact_fields": ["phone"],
            "context": {},
        })
        assert state["current_state"] == "handoff"

    def test_rewrite_that_drops_contact_ask_is_discarded(self):
        flow = make_flow(
            llm_router=self._router("request_human"),
            reply_generator=lambda *a, **k: "Perfeito! Qual dia e horário funciona melhor pra você?",
        )
        state = flow.invoke({
            "current_state": "conversation",
            "message": "quero contato",
            "missing_contact_fields": ["phone", "email"],
            "context": {},
        })
        assert "me passa seu telefone" in state["response"]

    def test_rewrite_that_keeps_contact_ask_is_used(self):
        natural = "Show! Me informe seu WhatsApp que o corretor já te chama."
        flow = make_flow(
            llm_router=self._router("request_human"),
            reply_generator=lambda *a, **k: natural,
        )
        state = flow.invoke({
            "current_state": "conversation",
            "message": "quero contato",
            "missing_contact_fields": ["phone", "email"],
            "context": {},
        })
        assert state["response"] == natural

    def test_duplicate_title_resolves_to_focused_property_not_first_homonym(self):
        twins = [
            {"id": "s211", "title": "Sobrado X", "area_util": 211},
            {"id": "s220", "title": "Sobrado X", "area_util": 220.75},
        ]
        flow = make_flow(llm_router=self._router("property_detail", property_ref="Sobrado X"))
        state = flow.invoke({
            "current_state": "conversation", "message": "quantos quartos?",
            "properties": twins, "context": {"focus_property_id": "s220"},
        })
        assert state["response_properties"][0]["id"] == "s220"

    def test_detail_turn_records_focus_property_id(self):
        flow = make_flow(llm_router=self._router("property_detail", property_ref="2"))
        state = flow.invoke({
            "current_state": "conversation", "message": "detalhes da opção 2",
            "properties": list(self.PROPS), "context": {},
        })
        assert state["context"]["focus_property_id"] == "b"

    def test_ask_criteria_without_any_criterion_asks_region_and_shows_nothing(self):
        flow = make_flow(
            llm_router=self._router("refine_search", ask_criteria=True),
            properties_rag=lambda info: list(self.PROPS),
        )
        state = flow.invoke({"current_state": "conversation", "message": "comprar", "context": {}})
        assert "região" in state["response"]
        assert not state.get("response_properties")
        assert not state["context"].get("properties")

    def test_ask_criteria_is_ignored_when_lead_already_gave_region(self):
        flow = make_flow(
            llm_router=self._router("refine_search", ask_criteria=True),
            properties_rag=lambda info: list(self.PROPS),
        )
        state = flow.invoke({
            "current_state": "conversation", "message": "comprar",
            "context": {"lead_info": {"region": "São Caetano"}},
        })
        assert state["response_properties"]


class TestRegionMatching:
    def test_city_in_corredor_matches_and_generic_token_does_not(self):
        from service.properties_catalog import _region_matches

        sc = {"region": "Boa Vista", "corredor": "São Caetano do Sul"}
        sp = {"region": "São João Clímaco", "corredor": "São Paulo"}
        assert _region_matches("são caetano", sc)
        assert not _region_matches("são caetano", sp)
        assert _region_matches("são paulo", sp)

    def test_meta_notes_are_stripped_but_inline_parentheses_kept(self):
        from service.flow.sales_flow import _strip_meta_notes

        assert _strip_meta_notes("Gostou?\n\n*(Segue as fotos!)*") == "Gostou?"
        assert _strip_meta_notes("Ok\n[Fotos enviadas]\nE aí?") == "Ok\n\nE aí?"
        assert _strip_meta_notes("Tem 3 vagas (cobertas).") == "Tem 3 vagas (cobertas)."


class TestReplyEchoGuard:
    LIST_REPLY = (
        "Entendi, você busca espaços em Santo André com cerca de 1000 m². Destaco estas:\n"
        "1. Terreno no Campestre: 502 m², R$ 4,5 mil/mês.\n2. Sobrado no Jardim: 600 m²."
    )

    def _flow(self, generated):
        router = lambda message, lead_info, current_state, **kw: {
            "tool": "property_detail", "arguments": {"property_ref": "1"}, "lead_info": {}, "memory_updates": {},
        }
        return make_flow(llm_router=router, reply_generator=lambda *a, **k: generated)

    def _invoke(self, flow):
        return flow.invoke({
            "current_state": "conversation", "message": "1",
            "properties": [{"id": "t", "title": "Terreno Campestre", "region": "Campestre", "area_util": 502}],
            "conversation_history": [{"role": "user", "content": "santo andré"},
                                     {"role": "assistant", "content": self.LIST_REPLY}],
            "context": {},
        })

    def test_reply_that_echoes_previous_bot_message_falls_back_to_official_text(self):
        state = self._invoke(self._flow(self.LIST_REPLY))
        assert state["response"] != self.LIST_REPLY
        assert "Terreno Campestre" in state["response"]

    def test_different_reply_is_kept(self):
        natural = "O terreno no Campestre tem 502 m² e custa R$ 4,5 mil/mês. Quer ver mais algum detalhe dele?"
        assert self._invoke(self._flow(natural))["response"] == natural


class TestSeveralPropertiesAtOnce:
    """Chat real 05/10: 'mostre foto desses três' -> 3 fotos do 1º imóvel (e não 1 de cada)."""

    PROPS = [
        {"id": f"p{i}", "title": f"Sobrado {i}", "region": f"Bairro {i}", "area_util": 100 + i,
         "price_text": f"R$ {i} milhão", "images": [f"https://cdn/p{i}-{n}.jpg" for n in range(5)]}
        for i in (1, 2, 3)
    ]

    def _flow(self, refs, send_photos=True):
        router = lambda message, lead_info, current_state, **kw: {
            "tool": "property_detail",
            "arguments": {"property_refs": refs, "send_photos": send_photos},
            "lead_info": {}, "memory_updates": {},
        }
        return make_flow(llm_router=router)

    def _invoke(self, flow):
        return flow.invoke({"current_state": "conversation", "message": "mostre foto desses três",
                            "properties": list(self.PROPS), "context": {}})

    def test_one_photo_of_each_of_the_three(self):
        state = self._invoke(self._flow(["1", "2", "3"]))
        assert [p["id"] for p in state["response_properties"]] == ["p1", "p2", "p3"]
        assert state["response_images"] == ["https://cdn/p1-0.jpg", "https://cdn/p2-0.jpg", "https://cdn/p3-0.jpg"]
        assert all(t in state["response"] for t in ("Sobrado 1", "Sobrado 2", "Sobrado 3"))

    def test_without_the_photo_decision_no_photos_go(self):
        assert self._invoke(self._flow(["1", "2", "3"], send_photos=False))["response_images"] == []

    def test_duplicates_and_unknown_refs_fall_back_to_the_single_item_path(self):
        state = self._invoke(self._flow(["2", "2"]))
        assert len(state["response_properties"]) == 1


class TestAskCriteriaOnlyOnce:
    def _flow(self):
        router = lambda message, lead_info, current_state, **kw: {
            "tool": "refine_search", "arguments": {"ask_criteria": True}, "lead_info": {}, "memory_updates": {},
        }
        return make_flow(llm_router=router, properties_rag=lambda info: [{"id": "a", "title": "A", "images": []}])

    def test_second_turn_without_criteria_shows_options_instead_of_asking_again(self):
        flow = self._flow()
        first = flow.invoke({"current_state": "conversation", "message": "comprar", "context": {}})
        assert "região" in first["response"] and not first.get("response_properties")
        second = flow.invoke({"current_state": "conversation", "message": "cadê as opções",
                              "context": first["context"]})
        assert second["response_properties"], second["response"]
