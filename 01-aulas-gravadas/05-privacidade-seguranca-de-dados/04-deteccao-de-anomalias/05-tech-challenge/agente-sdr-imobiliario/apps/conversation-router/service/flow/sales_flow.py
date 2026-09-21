from __future__ import annotations

from typing import Any, Callable

CONFIDENCE_THRESHOLD = 0.85
INTENT_CONFIRM_QUESTION = "Você está buscando compra, locação ou investimento?"


class SalesFlow:
    def __init__(
        self,
        lead_qualifier: Any,
        properties_rag: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
        scheduler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        handoff_builder: Callable[[dict[str, Any]], str] | None = None,
        llm_classify_intent: Callable[[str], tuple[str, float]] | None = None,
    ) -> None:
        self.qualifier = lead_qualifier
        self.properties_rag = properties_rag
        self.scheduler = scheduler
        self.handoff_builder = handoff_builder
        self._llm_classify = llm_classify_intent or self._default_classify

    @staticmethod
    def _default_classify(message: str) -> tuple[str, float]:
        lowered = message.lower()
        if any(w in lowered for w in ("alugar", "locação", "locacao", "locar")):
            return "rent", 0.9
        if any(w in lowered for w in ("investimento", "investir", "renda")):
            return "investment", 0.9
        if any(w in lowered for w in ("comprar", "compra", "adquirir")):
            return "purchase", 0.9
        return "unknown", 0.3

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        current = state.get("current_state", "greeting")
        message = state.get("message", "")
        method = getattr(self, f"_handle_{current}", None)
        if method is None:
            state["response"] = "Como posso ajudar?"
            return state
        return method(state, message)

    def _handle_greeting(self, state: dict[str, Any], message: str) -> dict[str, Any]:
        from service.security_layer import CONSENT_MESSAGE

        if message.strip().lower() in ("não", "nao", "não quero", "nao quero"):
            state["response"] = "Entendido. Se mudar de ideia, envie /start novamente"
            state["current_state"] = "followup"
            state["consent_recorded"] = False
            return state
        state["response"] = CONSENT_MESSAGE
        state["current_state"] = "elicitation"
        state["consent_recorded"] = True
        return state

    def _handle_elicitation(self, state: dict[str, Any], message: str) -> dict[str, Any]:
        state["current_state"] = "intent"
        state.setdefault("context", {})
        state["response"] = "Para indicar as melhores opções, você busca compra, locação ou investimento?"
        return state

    def _handle_intent(self, state: dict[str, Any], message: str) -> dict[str, Any]:
        intent, confidence = self._llm_classify(message)
        if confidence < CONFIDENCE_THRESHOLD:
            state["response"] = INTENT_CONFIRM_QUESTION
            return state
        state["intent"] = intent
        state["current_state"] = "qualification"
        state["response"] = "Ótimo! Qual a metragem desejada, região e orçamento?"
        return state

    def _handle_qualification(self, state: dict[str, Any], message: str) -> dict[str, Any]:
        info = state.get("lead_info", {})
        result = self.qualifier.calculate_score(info)
        state["score"] = result["score"]
        state["score_factors"] = result["factors"]
        if self.qualifier.is_qualified(result["score"]):
            state["current_state"] = "recommendation"
            state["lead_qualified"] = True
            state["response"] = self.qualifier.explain(result)
        else:
            state["current_state"] = "followup"
            state["lead_qualified"] = False
            state["response"] = (
                self.qualifier.explain(result)
                + " Ainda precisamos de algumas informações: "
                + ", ".join(result["missing"])
                + "."
            )
        return state

    def _handle_recommendation(self, state: dict[str, Any], message: str) -> dict[str, Any]:
        if self.properties_rag is None:
            state["current_state"] = "scheduling"
            state["response"] = "Podemos agendar uma visita?"
            return state
        properties = self.properties_rag(state.get("lead_info", {}))
        if not properties:
            state["response"] = (
                "Não encontramos imóveis com esses filtros. Sugerimos ajustar metragem, região ou orçamento."
            )
            return state
        state["properties"] = properties[:3]
        state["current_state"] = "scheduling"
        listed = "\n".join(
            f"{i + 1}. {p.get('title', 'Imóvel')} — {p.get('region', '')}, {p.get('area_m2', '')} m²"
            for i, p in enumerate(state["properties"])
        )
        state["response"] = f"Encontramos estas opções:\n{listed}\nGostaria de agendar uma visita?"
        return state

    def _handle_scheduling(self, state: dict[str, Any], message: str) -> dict[str, Any]:
        if self.scheduler is None:
            state["current_state"] = "handoff"
            state["response"] = "Agendamento registrado. Vou repassar seu perfil ao corretor."
            return state
        result = self.scheduler({"lead_info": state.get("lead_info", {}), "when": message})
        if not result.get("confirmed"):
            state["response"] = "Data/hora indisponível. Podemos tentar outro horário?"
            return state
        state["appointment"] = result
        state["current_state"] = "handoff"
        state["response"] = "Agendamento confirmado! Enviarei o convite e o resumo ao corretor."
        return state

    def _handle_handoff(self, state: dict[str, Any], message: str) -> dict[str, Any]:
        if self.handoff_builder is not None:
            state["handoff_summary"] = self.handoff_builder(state)
        state["current_state"] = "handoff"
        state["done"] = True
        state.setdefault("response", "Resumo enviado ao corretor. Obrigado pelo contato!")
        return state

    def _handle_followup(self, state: dict[str, Any], message: str) -> dict[str, Any]:
        state["current_state"] = "followup"
        state["done"] = True
        state.setdefault("response", "Anotado! Retomaremos o contato em breve.")
        return state
