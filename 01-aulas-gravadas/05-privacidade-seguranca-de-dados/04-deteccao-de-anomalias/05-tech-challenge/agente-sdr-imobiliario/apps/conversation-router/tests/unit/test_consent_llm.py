"""Consentimento LGPD decidido pela LLM lendo a conversa (ADR-027); palavras só como plano B."""
import pytest

from service import llm
from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow
from service.security_layer import CONSENT_REASK_MESSAGE, REFUSAL_MESSAGE

HISTORY = [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "Olá! ... Podemos continuar?"}]


def make_flow(classifier):
    return SalesFlow(lead_qualifier=LeadQualifier(), consent_classifier=classifier, bot_name_provider=lambda: "Cecília")


def answer(flow, message):
    return flow.invoke({"current_state": "elicitation", "message": message, "conversation_history": HISTORY})


class TestLlmDecides:
    def test_accept_in_any_wording_goes_to_intent(self):
        out = answer(make_flow(lambda m, **k: "accepted"), "tô de acordo, manda ver")
        assert out["consent_recorded"] is True and out["current_state"] == "intent"

    def test_refusal_closes(self):
        out = answer(make_flow(lambda m, **k: "refused"), "prefiro não passar meus dados")
        assert out["consent_recorded"] is False and out["current_state"] == "followup"
        assert out["response"] == REFUSAL_MESSAGE

    def test_unclear_asks_again_without_recording(self):
        out = answer(make_flow(lambda m, **k: "unclear"), "compra")
        assert out["consent_recorded"] is False and out["current_state"] == "elicitation"
        assert out["response"] == CONSENT_REASK_MESSAGE

    def test_llm_wins_over_the_word_list(self):
        # "sim" está na lista de palavras, mas quem decide é a LLM
        out = answer(make_flow(lambda m, **k: "unclear"), "sim?? pra que vocês querem isso?")
        assert out["consent_recorded"] is False

    def test_classifier_receives_the_whole_conversation(self):
        seen = {}

        def classifier(message, **kwargs):
            seen.update(message=message, **kwargs)
            return "accepted"

        answer(make_flow(classifier), "pode seguir")
        assert seen["message"] == "pode seguir" and seen["conversation_history"] == HISTORY


class TestFallbackWithoutLlm:
    @pytest.mark.parametrize("bad", [lambda m, **k: (_ for _ in ()).throw(RuntimeError("429")), lambda m, **k: "talvez"])
    def test_llm_failure_or_invalid_answer_uses_the_word_fallback(self, bad):
        assert answer(make_flow(bad), "sim")["consent_recorded"] is True
        assert answer(make_flow(bad), "não")["current_state"] == "followup"
        assert answer(make_flow(bad), "compra")["current_state"] == "elicitation"

    def test_no_classifier_keeps_the_previous_behavior(self):
        assert answer(make_flow(None), "claro")["consent_recorded"] is True


class TestClassifyConsent:
    def _with_reply(self, monkeypatch, raw):
        captured = {}

        def fake(messages, **kwargs):
            captured["messages"] = messages
            return raw

        monkeypatch.setattr(llm, "_completion_with_fallback", fake)
        return captured

    @pytest.mark.parametrize("decision", ["accepted", "refused", "unclear"])
    def test_parses_each_decision(self, monkeypatch, decision):
        self._with_reply(monkeypatch, f'{{"decision": "{decision}", "reason": "x"}}')
        assert llm.classify_consent("x", api_key="k", conversation_history=HISTORY) == decision

    def test_accepts_json_in_a_markdown_fence(self, monkeypatch):
        self._with_reply(monkeypatch, '```json\n{"decision": "accepted"}\n```')
        assert llm.classify_consent("sim", api_key="k") == "accepted"

    @pytest.mark.parametrize("raw", ['{"decision": "maybe"}', '{}', "não é json", ""])
    def test_invalid_answer_raises_so_the_flow_falls_back(self, monkeypatch, raw):
        self._with_reply(monkeypatch, raw)
        with pytest.raises(Exception):
            llm.classify_consent("sim", api_key="k")

    def test_sends_the_conversation_and_the_last_message(self, monkeypatch):
        captured = self._with_reply(monkeypatch, '{"decision": "accepted"}')
        llm.classify_consent("manda ver", api_key="k", conversation_history=HISTORY)
        roles = [m["role"] for m in captured["messages"]]
        assert roles[0] == "system" and roles[1:3] == ["user", "assistant"]
        assert "manda ver" in captured["messages"][-1]["content"]
        assert "na dúvida, responda unclear" in captured["messages"][0]["content"]
