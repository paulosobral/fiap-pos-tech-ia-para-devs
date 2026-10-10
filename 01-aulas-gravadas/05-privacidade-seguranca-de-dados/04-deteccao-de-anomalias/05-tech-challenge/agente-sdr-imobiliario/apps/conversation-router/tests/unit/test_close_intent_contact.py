"""Lead que quer fechar sem ter passado contato: o bot pede o WhatsApp/e-mail DELE (não oferece o do corretor)."""
from service import llm
from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow

SHOWN = [{"id": str(i), "title": f"Imóvel {i}", "region": f"Bairro {i}", "images": []} for i in range(1, 4)]


def router(tool, ref="3"):
    return lambda message, lead_info, current_state, **kw: {
        "tool": tool, "arguments": {"property_ref": ref}, "lead_info": {}, "memory_updates": {}, "thought": "x"}


def invoke(flow, message, missing, context=None):
    ctx = context if context is not None else {"properties": list(SHOWN)}
    return flow.invoke({"current_state": "conversation", "message": message, "context": ctx, "properties": list(SHOWN),
                        "lead_info": {"intent": "purchase"}, "missing_contact_fields": missing})


def make(tool="express_visit_interest"):
    return SalesFlow(lead_qualifier=LeadQualifier(), llm_router=router(tool), bot_name_provider=lambda: "Cecília")


def test_wanting_to_close_without_contact_asks_for_the_leads_whatsapp_or_email():
    out = invoke(make(), "quero fechar o terceiro", ["name", "phone", "email"])
    assert "telefone (WhatsApp) ou seu e-mail" in out["response"]
    assert out["_contact_request"] is True
    assert out["context"]["pending_contact_action"] == "request_human"


def test_when_the_contact_arrives_the_lead_goes_to_the_broker():
    flow = make()
    first = invoke(flow, "quero fechar o terceiro", ["name", "phone", "email"])
    second = invoke(flow, "11979918262", ["name", "email"], context=first["context"])
    assert second["current_state"] == "handoff"
    assert "pending_contact_action" not in second["context"]


def test_with_contact_already_given_it_keeps_the_normal_answer():
    out = invoke(make(), "quero fechar o terceiro", ["name"])
    assert out["_contact_request"] is False
    assert "visita" in out["response"]


def test_humanizer_is_told_never_to_offer_the_brokers_contact():
    prompt = llm._REPLY_SYSTEM_PROMPT
    assert "NUNCA ofereça nem passe ao lead telefone, WhatsApp ou e-mail do corretor" in prompt


def test_router_prompt_routes_a_closing_decision_to_the_broker_and_saves_the_choice():
    prompt = llm._ROUTER_SYSTEM_PROMPT
    assert "DECISÃO DE FECHAR" in prompt and "request_human" in prompt
    assert "memory_updates.favorite_property" in prompt


def test_crm_message_carries_the_chosen_property_and_region():
    import json
    from tests.integration.fixtures import make_router, telegram_update as update

    router, sqs, _ = make_router(with_pii=True)
    router.handle(update("oi", update_id=1))
    router.handle(update("sim", update_id=2))
    router.handle(update("meu telefone é 11979918262", update_id=3))

    def fake_flow(state):
        ctx = dict(state.get("context") or {})
        ctx["favorite_property"] = "Apartamento à venda, Boa Vista - São Caetano do Sul/SP"
        ctx["lead_info"] = {"intent": "purchase", "region": "São Caetano do Sul"}
        return {**state, "context": ctx, "current_state": "handoff", "response": "ok"}

    router.flow.invoke = fake_flow
    router.handle(update("quero fechar o terceiro", update_id=4))
    bodies = [json.loads(c.kwargs["MessageBody"]) for c in sqs.send_message.call_args_list if "lead_data" in c.kwargs.get("MessageBody", "")]
    data = bodies[-1]["lead_data"]
    assert data["property"] == "Apartamento à venda, Boa Vista - São Caetano do Sul/SP"
    assert data["region"] == "São Caetano do Sul"


def test_rental_price_is_never_presented_as_iptu_rule_in_the_prompt():
    prompt = llm._REPLY_SYSTEM_PROMPT
    assert "NUNCA é IPTU, condomínio ou outra taxa" in prompt and "sem citar nenhum valor" in prompt


def test_crm_message_carries_the_chosen_property_price_and_budget_is_not_filled_with_it():
    import json
    from tests.integration.fixtures import make_router, telegram_update as update

    router, sqs, _ = make_router(with_pii=True)
    router.handle(update("oi", update_id=1))
    router.handle(update("sim", update_id=2))
    router.handle(update("meu telefone é 11979918262", update_id=3))

    def fake_flow(state):
        ctx = dict(state.get("context") or {})
        ctx["favorite_property"] = "Apartamento Boa Vista"
        ctx["properties"] = [{"title": "Apartamento Boa Vista", "price": 850000, "mode": "purchase", "price_text": "R$ 0.8 milhão"}]
        ctx["lead_info"] = {"intent": "purchase"}
        return {**state, "context": ctx, "current_state": "handoff", "response": "ok"}

    router.flow.invoke = fake_flow
    router.handle(update("quero fechar", update_id=4))
    data = [json.loads(c.kwargs["MessageBody"]) for c in sqs.send_message.call_args_list if "lead_data" in c.kwargs.get("MessageBody", "")][-1]["lead_data"]
    assert data["property_price"] == "R$ 850.000" and data["budget"] is None
