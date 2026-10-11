"""Handoff encerra o atendimento; a mensagem seguinte abre uma conversa nova (lead e sessão novos)."""
import json

from handler import CLOSING_NOTE
from service import llm
from tests.integration.fixtures import make_router, telegram_update as update


def _state(router, user=42):
    lead, conversation = router.store.get_by_telegram_user(user)
    return lead, conversation


def _handoff(router):
    router.flow.invoke = lambda s: {**s, "current_state": "handoff", "response": "Prontinho, o corretor te chama."}


def _texts(telegram):
    return [c.args[1] for c in telegram.send_message.call_args_list]


def test_closing_note_is_added_once_when_the_conversation_reaches_handoff():
    router, _, telegram = make_router()
    router.handle(update("oi", update_id=1))
    _handoff(router)
    router.handle(update("meu whatsapp 11979918262", update_id=2))
    last = _texts(telegram)[-1]
    assert "Prontinho, o corretor te chama." in last and CLOSING_NOTE in last
    assert last.count("atendimento está encerrado") == 1


def test_the_next_message_after_handoff_opens_a_brand_new_conversation():
    router, _, telegram = make_router()
    router.handle(update("oi", update_id=1))
    _handoff(router)
    router.handle(update("meu whatsapp 11979918262", update_id=2))
    closed_lead, closed_conv = _state(router)
    assert closed_conv.current_state == "handoff"

    del router.flow.invoke  # fluxo real de volta
    router.handle(update("oi de novo", update_id=3))
    lead, conversation = _state(router)
    assert lead.lead_id != closed_lead.lead_id and conversation.session_id != closed_conv.session_id
    assert conversation.consent_recorded is False and conversation.current_state == "elicitation"
    assert _texts(telegram)[-1].startswith("Olá! Meu nome é Cecília")  # recomeça pela apresentação
    assert CLOSING_NOTE not in _texts(telegram)[-1]


def test_the_closed_lead_stays_in_the_store_untouched():
    router, _, _ = make_router()
    router.handle(update("oi", update_id=1))
    _handoff(router)
    router.handle(update("pode fechar", update_id=2))
    closed_lead, closed_conv = _state(router)
    del router.flow.invoke
    router.handle(update("oi de novo", update_id=3))
    old_lead = router.store.get_lead(closed_lead.lead_id)
    old_conv = router.store.get_conversation(closed_lead.lead_id, closed_conv.session_id)
    assert old_lead is not None and old_conv.current_state == "handoff"


def test_a_conversation_that_is_not_in_handoff_keeps_going():
    router, _, _ = make_router()
    router.handle(update("oi", update_id=1))
    lead1, conv1 = _state(router)
    router.handle(update("sim", update_id=2))
    lead2, conv2 = _state(router)
    assert lead2.lead_id == lead1.lead_id and conv2.session_id == conv1.session_id


def test_a_declined_conversation_is_not_treated_as_closed_by_handoff():
    router, _, _ = make_router()
    router.handle(update("oi", update_id=1))
    router.handle(update("não", update_id=2))  # recusou o consentimento -> followup
    lead1, conv1 = _state(router)
    assert conv1.current_state == "followup"
    router.handle(update("oi", update_id=3))
    lead2, _ = _state(router)
    assert lead2.lead_id == lead1.lead_id


def test_the_user_always_comes_back_to_the_newest_lead():
    router, _, _ = make_router()
    ids = []
    for round_ in range(3):
        router.handle(update("oi", update_id=10 * round_ + 1))
        _handoff(router)
        router.handle(update("pode fechar", update_id=10 * round_ + 2))
        ids.append(_state(router)[0].lead_id)
        del router.flow.invoke
    assert len(set(ids)) == 3  # três atendimentos, três leads
    router.handle(update("oi", update_id=99))
    assert _state(router)[0].lead_id not in ids  # o 4º também é novo


def test_prompt_makes_the_handoff_reply_a_plain_goodbye():
    rule = llm._REPLY_SYSTEM_PROMPT
    assert "ESTÁGIO handoff" in rule and "NÃO faça perguntas" in rule and "NÃO ofereça fotos" in rule
