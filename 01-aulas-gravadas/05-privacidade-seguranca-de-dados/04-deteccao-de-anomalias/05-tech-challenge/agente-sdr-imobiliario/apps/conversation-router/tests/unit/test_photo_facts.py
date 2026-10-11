"""O bot sabe quantas fotos o imóvel tem e quantas já enviou (chat de 10/10: imóvel com 1 foto, "mais fotos")."""
import pytest

from service import llm
from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow, _claims_photos


def prop(images, **extra):
    return {"id": "gi-917", "title": "Apartamento à venda, Parque das Nações", "region": "Parque das Nações", "area_util": 154,
            "price_text": "R$ 850 mil", "mode": "purchase", "disponibilidade": "disponível", "images": images, **extra}


class TestPhotoCounts:
    def test_detail_view_gets_how_many_photos_were_sent(self):
        out = SalesFlow._with_photo_counts([prop(["a"])], {"context": {"photos_sent": {"gi-917": 1}}})
        assert out[0]["_fotos_enviadas"] == 1

    def test_unsent_property_counts_zero(self):
        assert SalesFlow._with_photo_counts([prop(["a", "b"])], {"context": {}})[0]["_fotos_enviadas"] == 0

    def test_lists_of_three_or_more_are_left_untouched(self):
        three = [prop(["a"]) for _ in range(3)]
        assert SalesFlow._with_photo_counts(three, {"context": {"photos_sent": {"gi-917": 1}}}) == three
        assert SalesFlow._with_photo_counts([], {"context": {}}) == []

    def test_does_not_mutate_the_stored_properties(self):
        original = prop(["a"])
        SalesFlow._with_photo_counts([original], {"context": {"photos_sent": {"gi-917": 1}}})
        assert "_fotos_enviadas" not in original


class TestPayloadFacts:
    def _payload(self, monkeypatch, props):
        captured = {}
        monkeypatch.setattr(llm, "_completion_with_fallback", lambda messages, **k: captured.setdefault("user", messages[-1]["content"]) and "ok")
        llm.generate_reply("tem mais fotos?", "Ficha.", {}, props, api_key="k", last_tool="property_detail", photos_sending=0)
        return captured["user"]

    def test_sends_total_and_remaining_photos(self, monkeypatch):
        text = self._payload(monkeypatch, [prop(["a"], _fotos_enviadas=1)])
        assert '"fotos_total": 1' in text and '"fotos_restantes": 0' in text

    def test_remaining_counts_what_was_not_sent_yet(self, monkeypatch):
        text = self._payload(monkeypatch, [prop(["a", "b", "c", "d", "e"], _fotos_enviadas=3)])
        assert '"fotos_total": 5' in text and '"fotos_restantes": 2' in text

    def test_no_counts_when_the_flow_did_not_provide_them(self, monkeypatch):
        assert "fotos_restantes" not in self._payload(monkeypatch, [prop(["a"])])

    def test_urls_never_reach_the_payload(self, monkeypatch):
        assert "http" not in self._payload(monkeypatch, [prop(["http://x/1.jpg"], _fotos_enviadas=1)])

    def test_prompt_tells_it_to_say_these_are_all_the_photos(self):
        rule = llm._REPLY_SYSTEM_PROMPT
        assert "fotos_restantes=0" in rule and "todas as fotos disponíveis" in rule and "nunca prometa enviar mais" in rule


class TestPromiseGuard:
    @pytest.mark.parametrize("text", ["Sim, envio agora mais fotos desse apartamento.", "Mando já as fotos", "Envio as fotos em seguida",
                                      "Aqui estão as fotos", "Segue a foto"])
    def test_promises_are_detected(self, text):
        assert _claims_photos(text)

    @pytest.mark.parametrize("text", ["Posso enviar mais fotos se quiser?", "Quer que eu envie fotos?", "Essas são todas as fotos disponíveis desse imóvel.",
                                      "Não tenho mais fotos desse imóvel", "O corretor envia o contrato"])
    def test_offers_and_denials_are_not_promises(self, text):
        assert not _claims_photos(text)


def test_flow_hands_the_counts_to_the_reply_generator():
    seen = {}

    def reply(message, canned, lead_info, properties, **kw):
        seen["properties"] = properties
        return canned

    def router(message, lead_info, current_state, **kw):
        return {"tool": "property_detail", "arguments": {"property_ref": "1", "send_photos": True}, "lead_info": {}, "memory_updates": {}, "thought": "x"}

    shown = [prop(["only-photo.jpg"])]
    flow = SalesFlow(lead_qualifier=LeadQualifier(), llm_router=router, reply_generator=reply, bot_name_provider=lambda: "Cecília")
    ctx = {"properties": shown, "photos_sent": {"gi-917": 1}}
    flow.invoke({"current_state": "conversation", "message": "tem mais fotos desse?", "context": ctx, "properties": shown,
                 "lead_info": {"intent": "purchase"}})
    assert seen["properties"][0]["_fotos_enviadas"] == 1


def test_prompt_forbids_offering_financing_and_sends_questions_to_the_broker():
    rule = llm._REPLY_SYSTEM_PROMPT
    assert "NUNCA ofereça esse assunto" in rule and "o corretor explica" in rule and "sem citar números" in rule
