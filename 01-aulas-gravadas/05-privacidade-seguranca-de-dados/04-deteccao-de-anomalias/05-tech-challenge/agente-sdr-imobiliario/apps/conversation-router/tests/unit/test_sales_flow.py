from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow


def make_flow(**kw):
    return SalesFlow(lead_qualifier=LeadQualifier(), **kw)


class TestSalesFlow:
    def test_greeting_presents_consent(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "greeting", "message": "olá"})
        assert state["consent_recorded"] is True
        assert state["current_state"] == "elicitation"
        assert "LGPD" in state["response"]

    def test_consent_refusal_encloses(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "greeting", "message": "não"})
        assert state["consent_recorded"] is False
        assert state["current_state"] == "followup"
        assert "/start" in state["response"]

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

    def test_qualification_low_score_routes_followup(self):
        flow = make_flow()
        state = flow.invoke({"current_state": "qualification", "message": "ok", "lead_info": {}})
        assert state["current_state"] == "followup"
        assert state["lead_qualified"] is False

    def test_recommendation_with_results_lists_top3(self):
        rag = lambda info: [{"title": f"Imóvel {i}", "region": "B", "area_m2": 100} for i in range(5)]
        flow = make_flow(properties_rag=rag)
        state = flow.invoke({"current_state": "recommendation", "message": "quero ver"})
        assert len(state["properties"]) == 3

    def test_recommendation_empty_explains(self):
        flow = make_flow(properties_rag=lambda info: [])
        state = flow.invoke({"current_state": "recommendation", "message": "quero ver"})
        assert "Não encontramos imóveis" in state["response"]
