"""Na fase de consentimento/intenção a humanização não recebe imóveis antigos (chat de 09/10: "ele nem oferece
locação, venda ou investimento mais")."""
import pytest

from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow

STORED = [{"id": str(i), "title": f"Imóvel {i}", "region": f"Bairro {i}", "images": []} for i in range(9)]


def flow_with_spy():
    seen = {}

    def reply(message, canned, lead_info, properties, **kwargs):
        seen["properties"] = properties
        seen["canned"] = canned
        return canned

    return SalesFlow(lead_qualifier=LeadQualifier(), reply_generator=reply, bot_name_provider=lambda: "Cecília"), seen


@pytest.mark.parametrize("phase", ["greeting", "elicitation", "intent"])
def test_stored_properties_are_not_sent_while_still_asking_consent_or_intent(phase):
    assert SalesFlow._properties_for_reply({"current_state": phase, "properties": STORED}) == []


@pytest.mark.parametrize("phase", ["conversation", "qualification", "recommendation", "scheduling", "handoff", "followup"])
def test_stored_properties_are_still_available_after_the_recommendation_phase(phase):
    assert SalesFlow._properties_for_reply({"current_state": phase, "properties": STORED}) == STORED


def test_properties_of_the_turn_always_win_even_in_the_intent_phase():
    shown = STORED[:2]
    state = {"current_state": "intent", "properties": STORED, "response_properties": shown}
    assert SalesFlow._properties_for_reply(state) == shown


def test_consent_answer_with_stored_properties_asks_intent_without_offering_listings():
    flow, seen = flow_with_spy()
    out = flow.invoke({"current_state": "elicitation", "message": "sim", "context": {"properties": STORED},
                       "properties": STORED, "consent_recorded": False})
    assert out["current_state"] == "intent"
    assert seen["properties"] == []
    assert "compra, locação ou investimento" in seen["canned"]


def test_replies_after_recommendation_still_get_the_stored_properties():
    flow, seen = flow_with_spy()
    flow.invoke({"current_state": "conversation", "message": "e o terceiro?", "context": {"properties": STORED},
                 "properties": STORED, "lead_info": {"intent": "purchase"}})
    assert seen["properties"] == STORED
