import pytest

import handler as handler_module
from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow, _claims_photos, _plain_text
from service.security_layer import CONSENT_REASK_MESSAGE, REFUSAL_MESSAGE


def make_flow(**kw):
    return SalesFlow(lead_qualifier=LeadQualifier(), **kw)


class TestExplicitConsent:
    @pytest.mark.parametrize(
        "answer",
        ["sim", "Sim!", "s", "ok", "pode", "podemos sim", "claro", "aceito", "concordo", "Pode continuar",
         "sim, quero comprar uma sala", "beleza", "com certeza", "tudo bem", "  SIM  "],
    )
    def test_clear_yes_records_consent(self, answer):
        state = make_flow().invoke({"current_state": "elicitation", "message": answer})
        assert state["consent_recorded"] is True
        assert state["current_state"] == "intent"

    @pytest.mark.parametrize("answer", ["compra", "sala comercial em santo andré", "quero alugar", "oi", "?", "quanto custa"])
    def test_anything_else_is_not_consent(self, answer):
        state = make_flow().invoke({"current_state": "elicitation", "message": answer})
        assert state["consent_recorded"] is False
        assert state["current_state"] == "elicitation"  # continua esperando o aceite
        assert state["response"] == CONSENT_REASK_MESSAGE

    @pytest.mark.parametrize("answer", ["não", "nao", "Não quero", "nao quero."])
    def test_refusal_still_closes(self, answer):
        state = make_flow().invoke({"current_state": "elicitation", "message": answer})
        assert state["consent_recorded"] is False
        assert state["current_state"] == "followup"
        assert state["response"] == REFUSAL_MESSAGE

    def test_reask_is_not_rewritten_by_llm(self):
        calls = []
        flow = make_flow(reply_generator=lambda *a, **k: calls.append(1) or "reescrito")
        state = flow.invoke({"current_state": "elicitation", "message": "compra"})
        assert state["response"] == CONSENT_REASK_MESSAGE
        assert not calls

    def test_after_reask_a_yes_proceeds(self):
        flow = make_flow()
        first = flow.invoke({"current_state": "elicitation", "message": "compra"})
        second = flow.invoke({"current_state": first["current_state"], "message": "sim"})
        assert second["consent_recorded"] is True and second["current_state"] == "intent"


class TestPlainText:
    def test_strips_bold_and_headings(self):
        assert _plain_text("**Apartamento 64 m²** - R$ 375 mil") == "Apartamento 64 m² - R$ 375 mil"
        assert _plain_text("__ok__ e **ok**") == "ok e ok"
        assert _plain_text("## Opções\n1. um") == "Opções\n1. um"

    def test_keeps_regular_text(self):
        text = "Área 64 m² * garagem e 2 quartos - ok"
        assert _plain_text(text) == text

    def test_flow_applies_it_to_llm_reply(self):
        flow = make_flow(reply_generator=lambda *a, **k: "**Ótimo!** Vamos lá")
        state = flow.invoke({"current_state": "intent", "message": "quero alugar agora"})
        assert state["response"] == "Ótimo! Vamos lá"


class TestPhotoPromiseGuard:
    @pytest.mark.parametrize(
        "text",
        ["Aqui estão as fotos dos imóveis que mostrei", "Seguem as fotos!", "Estou enviando as imagens agora",
         "Te mando as fotos já", "Segue a foto do apartamento"],
    )
    def test_detects_false_promises(self, text):
        assert _claims_photos(text)

    @pytest.mark.parametrize(
        "text",
        ["Esse imóvel tem 64 m² e custa R$ 375 mil", "Quer que eu envie fotos?", "Posso mandar fotos se quiser",
         "O corretor entra em contato"],
    )
    def test_ignores_offers_and_normal_text(self, text):
        assert not _claims_photos(text)

    def test_promise_without_photos_keeps_official_text(self):
        flow = make_flow(reply_generator=lambda *a, **k: "Aqui estão as fotos dos imóveis!")
        state = flow.invoke({"current_state": "intent", "message": "quero alugar agora"})
        assert "fotos" not in state["response"].lower()
        assert not state["response_images"]

    def test_promise_with_photos_is_kept(self):
        flow = make_flow(reply_generator=lambda *a, **k: "Aqui estão as fotos dos imóveis!")
        state = {"current_state": "intent", "message": "quero alugar agora"}
        # sem roteador LLM o envio é automático; força imagens como se o turno tivesse fotos
        flow._decide_photos = lambda s: s.__setitem__("response_images", ["http://x/1.jpg"])
        out = flow.invoke(state)
        assert out["response"] == "Aqui estão as fotos dos imóveis!"


class TestContactConfirmation:
    @pytest.mark.parametrize(
        "raw,expected",
        [("11979918262", "(11) 97991-8262"), ("(11)979918262", "(11) 97991-8262"),
         ("+55 11 91234-5678", "(11) 91234-5678"), ("1133334444", "(11) 3333-4444"), ("123", "123")],
    )
    def test_format_phone(self, raw, expected):
        assert handler_module._format_phone(raw) == expected

    def test_confirms_only_new_contact(self):
        before = {"TELEFONE": ["11911112222"]}
        after = {"TELEFONE": ["11911112222", "11979918262"], "EMAIL": ["a@b.com"]}
        text = handler_module._contact_confirmation(before, after)
        assert "(11) 97991-8262" in text and "a@b.com" in text and "11911112222" not in text

    def test_nothing_new_means_no_confirmation(self):
        data = {"TELEFONE": ["11979918262"]}
        assert handler_module._contact_confirmation(data, dict(data)) == ""
        assert handler_module._contact_confirmation({}, {}) == ""
