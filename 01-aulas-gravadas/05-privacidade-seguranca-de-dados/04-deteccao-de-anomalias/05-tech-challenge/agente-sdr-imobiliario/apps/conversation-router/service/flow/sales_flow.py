from __future__ import annotations

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.85
INTENT_CONFIRM_QUESTION = "Você está buscando compra, locação ou investimento?"

# Extração sobre texto JÁ MASCARADO (placeholders [NOME]/[EMAIL] não colidem com os padrões).
# Número: separadores de milhar múltiplos (\d{1,3}(\.\d{3})+, ex. 1.500.000) OU decimal pt-BR (1,5).
# Unidade: "milhões"/"milhão"/"milhao" (singular "milhão" tem Ã — sem ele cairia em "mil", erro ×1000), "mil", "k".
_BUDGET_NUMBER_RE = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?"
_BUDGET_UNIT_RE_TEXT = r"mil(?:h[õo]es|h[ãa]o)?|k"
_AREA_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:m²|m2|metros quadrados|metro quadrado)", re.IGNORECASE)
_BUDGET_PREFIXED_RE = re.compile(
    rf"(?:r\$\s*|orçamento\s*(?:de|:)?\s*)({_BUDGET_NUMBER_RE})\s*({_BUDGET_UNIT_RE_TEXT})?",
    re.IGNORECASE,
)
_BUDGET_UNIT_RE = re.compile(
    rf"\b({_BUDGET_NUMBER_RE})\s*({_BUDGET_UNIT_RE_TEXT})\b",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(r"(?:prazo\s*(?:de|:)?\s*)?(\d+)\s*(m[êe]s(?:es)?|semanas?|dias?)", re.IGNORECASE)
_PEOPLE_RE = re.compile(r"(\d+)\s*(?:pessoas|colaboradores|usuários|usuarios|funcionários|funcionarios)", re.IGNORECASE)
_REGION_RE = re.compile(
    r"(?:regi[ãa]o|bairro|zona)\s+(?:d[oa]s?|de|em)?\s*([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)*)", re.IGNORECASE
)
_DECISOR_YES_RE = re.compile(r"sou\s+(?:o\s+)?decisor|eu\s+(?:que\s+)?decido", re.IGNORECASE)
_DECISOR_NO_RE = re.compile(r"n[ãa]o\s+(?:sou\s+(?:o\s+)?decisor|decido)", re.IGNORECASE)
_REGION_STOP_WORDS = ("com", "e", "para", "pra", "no", "na", "do", "da", "até", "por", "ou")


def extract_lead_structure(message: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if area := _AREA_RE.search(message):
        info["area"] = area.group(0)
    if budget := (_BUDGET_PREFIXED_RE.search(message) or _BUDGET_UNIT_RE.search(message)):
        info["budget"] = budget.group(0)
    if deadline := _DEADLINE_RE.search(message):
        info["deadline"] = deadline.group(0)
    if people := _PEOPLE_RE.search(message):
        info["people_count"] = int(people.group(1))
    if region := _REGION_RE.search(message):
        tokens = region.group(1).split()
        while tokens and tokens[-1].lower() in _REGION_STOP_WORDS:
            tokens.pop()
        if tokens:
            info["region"] = " ".join(tokens)
    if _DECISOR_NO_RE.search(message):
        info["decision_maker"] = "no"
    elif _DECISOR_YES_RE.search(message):
        info["decision_maker"] = "yes"
    return info


class SalesFlow:
    def __init__(
        self,
        lead_qualifier: Any,
        properties_rag: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
        scheduler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        handoff_builder: Callable[[dict[str, Any]], str] | None = None,
        llm_classify_intent: Callable[[str], tuple[str, float]] | None = None,
        specialist_rotation: list[str] | None = None,
        specialist_fallback: str = "diretor",
        restriction_check: Callable[[str], bool] | None = None,
    ) -> None:
        self.qualifier = lead_qualifier
        self.properties_rag = properties_rag
        self.scheduler = scheduler
        self.handoff_builder = handoff_builder
        self._llm_classify = llm_classify_intent or self._default_classify
        self.specialist_rotation = specialist_rotation or []
        self.specialist_fallback = specialist_fallback
        # FR9.4: checker de restrição de agendamento (alertas de anomalia — U5, is_restricted(lead_id)).
        self.restriction_check = restriction_check

    def _is_restricted(self, lead_id: str | None) -> bool:
        """Consulta o checker injetável; fail-open = NÃO restrito em ausência/erro."""
        if self.restriction_check is None or not lead_id:
            return False
        try:
            return bool(self.restriction_check(lead_id))
        except Exception:
            logger.warning("restriction check failed; treating as unrestricted (fail-open)")
            return False

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
        context = state.setdefault("context", {})
        extracted = extract_lead_structure(message)
        if extracted:
            stored = dict(context.get("lead_info") or {})
            stored.update(extracted)
            context["lead_info"] = stored
        stored_info = context.get("lead_info") or {}
        seeded = state.get("lead_info") or {}
        state["lead_info"] = {**stored_info, **seeded}
        method = getattr(self, f"_handle_{current}", None)
        if method is None:
            state["response"] = "Como posso ajudar?"
            return state
        return method(state, message)

    def _handle_greeting(self, state: dict[str, Any], message: str) -> dict[str, Any]:
        from service.security_layer import CONSENT_MESSAGE, REFUSAL_MESSAGE

        if message.strip().lower() in ("não", "nao", "não quero", "nao quero"):
            state["response"] = REFUSAL_MESSAGE
            state["current_state"] = "followup"
            state["consent_recorded"] = False
            return state
        state["response"] = CONSENT_MESSAGE
        state["current_state"] = "elicitation"
        return state

    def _handle_elicitation(self, state: dict[str, Any], message: str) -> dict[str, Any]:
        from service.security_layer import REFUSAL_MESSAGE

        if message.strip().lower() in ("não", "nao", "não quero", "nao quero"):
            state["response"] = REFUSAL_MESSAGE
            state["current_state"] = "followup"
            state["consent_recorded"] = False
            return state
        state["current_state"] = "intent"
        state["consent_recorded"] = True
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
            state["route"] = self.qualifier.route(
                self._area_m2(info.get("area")), self.specialist_rotation, self.specialist_fallback
            )
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

    @staticmethod
    def _area_m2(area: Any) -> float | None:
        if not area:
            return None
        match = re.search(r"(\d+(?:[.,]\d+)?)", str(area))
        return float(match.group(1).replace(",", ".")) if match else None

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
        if self._is_restricted(state.get("lead_id")):
            # FR9.4: lead sinalizado pela U5 — agendamento autônomo adiado; corretor assume a confirmação.
            state["scheduling_restricted"] = True
            state["current_state"] = "handoff"
            state["response"] = "O agendamento da visita precisa de confirmação do corretor; em breve ele fará contato."
            return state
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
        if self._is_restricted(state.get("lead_id")):
            # FR9.4: sem outreach autônomo para lead sinalizado — follow-up fica sob responsabilidade do corretor.
            state["followup_deferred"] = True
            state["current_state"] = "followup"
            state["done"] = True
            state["response"] = "Anotado! Seu atendimento seguirá com o corretor responsável."
            return state
        state["current_state"] = "followup"
        state["done"] = True
        state.setdefault("response", "Anotado! Retomaremos o contato em breve.")
        return state
