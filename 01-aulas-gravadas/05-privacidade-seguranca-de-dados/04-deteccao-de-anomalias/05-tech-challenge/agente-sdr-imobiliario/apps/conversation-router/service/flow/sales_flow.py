"""Sales-flow com LangGraph (PRD §8.1: orquestração via LangGraph).

Substitui o FSM manual por StateGraph do LangGraph: cada estado do fluxo
(saudação → elicitação → intenção → qualificação → recomendação → agendamento →
handoff/followup) é um nó do grafo, com areistas condicionais determinadas
pelo campo `current_state`. A extração de lead_info (regex sobre texto
mascarado) roda em um nó de pré-processamento; a geração de resposta humanizada
via LLM roda em um nó de pós-processamento (FR-02).

Fallback: se LangGraph não estiver instalado (ex.: ambiente de teste sem deps),
o dispatch manual por `current_state` é usado — mantém o mesmo contrato e os
mesmos testes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, TypedDict

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.85
INTENT_CONFIRM_QUESTION = "Você está buscando compra, locação ou investimento?"

# Extração sobre texto JÁ MASCARADO (placeholders [NOME]/[EMAIL] não colidem com os padrões).
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
_BUDGET_CEILING_RE = re.compile(
    rf"(?:at[ée]|até\s+r\$?)\s*({_BUDGET_NUMBER_RE})\s*(?:reais|r\$)?",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(r"(?:prazo\s*(?:de|:)?\s*)?(\d+)\s*(m[êe]s(?:es)?|semanas?|dias?)", re.IGNORECASE)
_PEOPLE_RE = re.compile(r"(\d+)\s*(?:pessoas|colaboradores|usuários|usuarios|funcionários|funcionarios)", re.IGNORECASE)
_REGION_RE = re.compile(
    r"(?:regi[ãa]o|bairro|zona)\s+(?:d[oa]s?|de|em)?\s*([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)*)", re.IGNORECASE
)
_REGION_CONTEXT_RE = re.compile(r"\b(?:em|na|no)\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)?)", re.IGNORECASE)
_DECISOR_YES_RE = re.compile(
    r"sou\s+(?:o\s+)?(?:decisor|propriet[aá]rio)|eu\s+(?:que\s+)?decido|quem\s+decide\s+sou\s+eu",
    re.IGNORECASE,
)
_DECISOR_NO_RE = re.compile(r"n[ãa]o\s+(?:sou\s+(?:o\s+)?decisor|decido)", re.IGNORECASE)
_OPTIONS_REQUEST_RE = re.compile(r"\b(?:opç(?:ão|ões)|imóveis|imoveis|mostrar|mostre|cad[êe])\b", re.IGNORECASE)
_REGION_STOP_WORDS = ("com", "e", "para", "pra", "no", "na", "do", "da", "até", "por", "ou")

_FAVORITE_LIKE_RE = re.compile(
    r"(?:gostei\s+da?s?\s+|quero\s+a\s+|prefiro\s+a\s+)(\d+|[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 \-]+)",
    re.IGNORECASE,
)
_REJECT_LIKE_RE = re.compile(
    r"(?:n[ãa]o\s+quero\s+(?:a\s+|as\s+)?|descart[ae]\s+(?:a\s+|as\s+)?)(\d+|[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 \-]+)",
    re.IGNORECASE,
)


class FlowState(TypedDict, total=False):
    """Estado do grafo LangGraph — um turno por invocação."""
    current_state: str
    message: str
    context: dict[str, Any]
    lead_info: dict[str, Any]
    intent: str
    score: Any
    score_factors: Any
    lead_qualified: bool
    route: str
    properties: list[dict[str, Any]]
    appointment: dict[str, Any]
    handoff_summary: str
    response: str
    consent_recorded: bool
    done: bool
    lead_id: str
    scheduling_restricted: bool
    followup_deferred: bool
    ics_invite: str
    shown_properties_count: int
    favorite_property: str
    visit_interest: bool
    rejected_properties: list[str]
    interests: list[str]
    _router_action: str | None  # legacy ADR-011 enum action (compat)
    _router_tool: str | None  # tool-agent contract (spec 2026-09-23)
    _tool_result: Any  # ToolResult from execute_tool
    _last_tool: str | None
    _legacy_subnode: str


def extract_lead_structure(message: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if area := _AREA_RE.search(message):
        info["area"] = area.group(0)
    if budget := (_BUDGET_PREFIXED_RE.search(message) or _BUDGET_UNIT_RE.search(message) or _BUDGET_CEILING_RE.search(message)):
        info["budget"] = budget.group(0)
    if deadline := _DEADLINE_RE.search(message):
        info["deadline"] = deadline.group(0)
    if people := _PEOPLE_RE.search(message):
        info["people_count"] = int(people.group(1))
    region = _REGION_RE.search(message) or _REGION_CONTEXT_RE.search(message)
    if region:
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


# --- LangGraph (import opcional; fallback FSM manual se ausente) -------------

try:
    from langgraph.graph import END, StateGraph

    _HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    END = "END"  # type: ignore[assignment,misc]
    StateGraph = None  # type: ignore[assignment]
    _HAS_LANGGRAPH = False


class SalesFlow:
    def __init__(
        self,
        lead_qualifier: Any,
        properties_rag: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
        scheduler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        handoff_builder: Callable[[dict[str, Any]], str] | None = None,
        llm_classify_intent: Callable[[str], tuple[str, float]] | None = None,
        reply_generator: Callable[..., str] | None = None,
        specialist_rotation: list[str] | None = None,
        specialist_fallback: str = "diretor",
        restriction_check: Callable[[str], bool] | None = None,
        llm_router: Callable[[str, dict[str, Any], str], dict[str, Any]] | None = None,
    ) -> None:
        self.qualifier = lead_qualifier
        self.properties_rag = properties_rag
        self.scheduler = scheduler
        self.handoff_builder = handoff_builder
        self._llm_classify = llm_classify_intent or self._default_classify
        self.reply_generator = reply_generator
        self.specialist_rotation = specialist_rotation or []
        self.specialist_fallback = specialist_fallback
        self.restriction_check = restriction_check
        self.llm_router = llm_router
        if _HAS_LANGGRAPH:
            self._graph = self._build_graph()
        else:
            self._graph = None

    # --- Graph construction (LangGraph) ---

    def _build_graph(self):
        g: StateGraph = StateGraph(FlowState)
        g.add_node("preprocess", self._node_preprocess)
        g.add_node("greeting", self._node_greeting)
        g.add_node("elicitation", self._node_elicitation)
        g.add_node("conversation", self._node_conversation)
        g.add_node("scheduling", self._node_scheduling)
        g.add_node("handoff", self._node_handoff)
        g.add_node("followup", self._node_followup)
        g.add_node("postprocess", self._node_postprocess)

        g.set_entry_point("preprocess")
        g.add_conditional_edges("preprocess", self._route_state)
        for node in ("greeting", "elicitation", "conversation", "scheduling",
                      "handoff", "followup"):
            g.add_edge(node, "postprocess")
        g.add_edge("postprocess", END)
        return g.compile()

    def _route_state(self, state: FlowState) -> str:
        current = state.get("current_state", "greeting")
        if current in ("greeting", "elicitation"):
            return current  # LGPD: consentimento sempre determinístico, nunca via LLM
        tool = state.get("_router_tool")
        if tool is not None:
            hint = None
            tr = state.get("_tool_result")
            if tr is not None:
                hint = getattr(tr, "current_state_hint", None)
            if tool == "request_human":
                return hint or "handoff"
            if tool == "decline":
                return "followup"
            if tool == "request_schedule":
                return hint or "conversation"
            return "conversation"
        # Legacy ADR-011 action path — destinations only; conversation derives subnode itself
        # (LangGraph may drop mutations made during conditional-edge routing).
        action = state.get("_router_action")
        if action == "request_human" and _OPTIONS_REQUEST_RE.search(state.get("message", "")):
            return "conversation"
        if action == "request_human":
            return "handoff"
        if action == "decline":
            return "followup"
        if (
            action in (None, "provide_info", "unclear")
            and current in ("recommendation", "discovery")
            and re.search(r"\b(?:estacionamento|vagas?|andar|pre[çc]o|valor|quanto|detalhes?|tem|possui)\b", state.get("message", ""), re.IGNORECASE)
        ):
            return "conversation"
        if action == "visit_interest":
            if self.ready_for_scheduling(state):
                shown = max(state.get("shown_properties_count", 0) or 0, len(state.get("properties") or []))
                if shown >= 3:
                    return "scheduling"
            return "conversation"
        if action in ("refine_search", "compare_properties"):
            return "conversation"
        if current in ("intent", "qualification", "discovery", "recommendation", "conversation"):
            return "conversation"
        return current

    def _legacy_subnode_for(self, state: FlowState) -> str:
        """Deriva sub-nó legado dentro do conversation (não confiar em mutação do router)."""
        current = state.get("current_state") or "recommendation"
        action = state.get("_router_action")
        message = state.get("message", "")
        if action == "refine_search" or action == "compare_properties":
            return "recommendation"
        if (
            action in (None, "provide_info", "unclear")
            and current in ("recommendation", "discovery")
            and re.search(r"\b(?:estacionamento|vagas?|andar|pre[çc]o|valor|quanto|detalhes?|tem|possui)\b", message, re.IGNORECASE)
        ):
            return "discovery"
        if action == "request_human" and _OPTIONS_REQUEST_RE.search(message):
            return current if current in (
                "intent", "qualification", "discovery", "recommendation", "conversation"
            ) else "recommendation"
        if action == "visit_interest":
            return current if current in (
                "intent", "qualification", "discovery", "recommendation", "conversation"
            ) else "recommendation"
        if current in ("intent", "qualification", "discovery", "recommendation", "conversation"):
            return current
        return "recommendation"

    def _wants_options(self, state: FlowState) -> bool:
        """Tool-agent: confia em _router_tool; ADR-011: confia em _router_action;
        senão cai no regex (rede de segurança)."""
        tool = state.get("_router_tool")
        if tool is not None:
            return tool == "request_options"
        action = state.get("_router_action")
        if action == "request_human" and _OPTIONS_REQUEST_RE.search(state.get("message", "")):
            return True
        if action is not None:
            return action == "request_options"
        return bool(_OPTIONS_REQUEST_RE.search(state.get("message", "")))

    @staticmethod
    def ready_for_scheduling(state: FlowState) -> bool:
        return bool(
            state.get("favorite_property")
            or state.get("visit_interest")
            or (state.get("lead_info") or {}).get("deadline")
        )

    # --- Nodes (cada um processa UM turno e retorna state atualizado) ---

    def _node_preprocess(self, state: FlowState) -> FlowState:
        message = state.get("message", "")
        context = state.setdefault("context", {})
        extracted = extract_lead_structure(message)  # baseline regex — rede de segurança
        if extracted:
            stored = dict(context.get("lead_info") or {})
            stored.update(extracted)
            context["lead_info"] = stored
        stored_info = context.get("lead_info") or {}
        seeded = state.get("lead_info") or {}
        merged = {**stored_info, **seeded}

        if context.get("favorite_property"):
            state["favorite_property"] = context["favorite_property"]
        if context.get("visit_interest"):
            state["visit_interest"] = True
        if context.get("rejected_properties"):
            state["rejected_properties"] = list(context["rejected_properties"])
        if context.get("interests"):
            state["interests"] = list(context["interests"])
        # Imóveis já exibidos vivem só no context (handler não reenvia properties).
        if context.get("properties") and not state.get("properties"):
            state["properties"] = list(context["properties"])
            state["shown_properties_count"] = int(
                context.get("shown_properties_count") or len(state["properties"])
            )

        state["_router_action"] = None
        state["_router_tool"] = None
        state["_tool_result"] = None
        state["_last_tool"] = None
        if self.llm_router is not None:
            try:
                result = self.llm_router(message, merged, state.get("current_state", "greeting"))
                if isinstance(result, dict) and "tool" in result:
                    from service.tools import execute_tool
                    from service.validation import validate_router_output

                    validated = validate_router_output(result, state, message=message)
                    state["_router_tool"] = validated["tool"]
                    state["_last_tool"] = validated["tool"]
                    llm_deltas = validated.get("lead_info") or {}
                    merged = {**merged, **llm_deltas}
                    context["lead_info"] = {**(context.get("lead_info") or {}), **llm_deltas}
                    mem = validated.get("memory_updates") or {}
                    if mem.get("favorite_property"):
                        state["favorite_property"] = mem["favorite_property"]
                        context["favorite_property"] = mem["favorite_property"]
                    if mem.get("visit_interest"):
                        state["visit_interest"] = True
                        context["visit_interest"] = True
                    args = validated.get("arguments") or {}

                    def _search_fn(info, top_k=9):
                        if self.properties_rag is None:
                            return []
                        scope = args.get("list_scope", "filtered")
                        try:
                            from service.properties_catalog import search_properties

                            return search_properties(info, top_k=top_k, list_scope=scope)
                        except Exception:
                            props = self.properties_rag(info) or []
                            return list(props)[:top_k]

                    tr = execute_tool(
                        validated["tool"],
                        args,
                        state,
                        search_fn=_search_fn if self.properties_rag is not None else None,
                        memory_updates=mem,
                        message=message,
                    )
                    state["_tool_result"] = tr
                    if tr.properties:
                        state["properties"] = tr.properties
                        self._remember_shown(state)
                else:
                    # Legacy action contract (ADR-011)
                    llm_deltas = (result or {}).get("lead_info") or {}
                    merged = {**merged, **llm_deltas}
                    context["lead_info"] = {**(context.get("lead_info") or {}), **llm_deltas}
                    state["_router_action"] = (result or {}).get("action")
                    if state["_router_action"] == "visit_interest":
                        state["visit_interest"] = True
                        context["visit_interest"] = True
            except Exception:
                logger.warning(
                    "llm_router falhou; usando extração regex + FSM determinístico (fallback ADR-011)",
                    exc_info=True,
                )

        state["lead_info"] = merged
        self._apply_commercial_memory(state)
        return state

    def _apply_commercial_memory(self, state: FlowState) -> None:
        message = state.get("message", "")
        properties = state.get("properties") or []
        context = state.setdefault("context", {})

        fav_match = _FAVORITE_LIKE_RE.search(message)
        if fav_match:
            token = fav_match.group(1).strip()
            resolved = self._match_property_token(token, properties)
            if resolved:
                state["favorite_property"] = resolved
                context["favorite_property"] = resolved
                state["visit_interest"] = True
                context["visit_interest"] = True

        rej_match = _REJECT_LIKE_RE.search(message)
        if rej_match:
            token = rej_match.group(1).strip()
            resolved = self._match_property_token(token, properties)
            if resolved:
                rejected = list(state.get("rejected_properties") or [])
                if resolved not in rejected:
                    rejected.append(resolved)
                state["rejected_properties"] = rejected
                context["rejected_properties"] = rejected

    @staticmethod
    def _match_property_token(token: str, properties: list[dict[str, Any]]) -> str | None:
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(properties):
                return properties[idx].get("title") or f"Imóvel {token}"
        token_l = token.lower()
        for p in properties:
            title = str(p.get("title") or "")
            if title and title.lower() in token_l:
                return title
            if token_l and token_l in title.lower():
                return title
        return None

    def _node_conversation(self, state: FlowState) -> FlowState:
        """Dispatcher: tool-agent path OR legacy sub-node (intent/qualification/discovery/recommendation)."""
        if state.get("_router_tool") is not None:
            return self._conversation_from_tool(state)
        sub = state.get("_legacy_subnode") or self._legacy_subnode_for(state)
        if sub == "intent":
            return self._node_intent(state)
        if sub == "qualification":
            return self._node_qualification(state)
        if sub == "discovery":
            return self._node_discovery(state)
        return self._node_recommendation(state)

    def _conversation_from_tool(self, state: FlowState) -> FlowState:
        tool = state.get("_router_tool")
        tr = state.get("_tool_result")
        state["current_state"] = "conversation"
        if tool == "property_detail":
            if tr is not None and tr.needs_clarification:
                state["response"] = (
                    "Qual imóvel específico você quer que eu detalhe? "
                    "Posso comparar as opções que já mostrei."
                )
            elif tr is not None and tr.detail:
                p = tr.detail
                price = p.get("price_text") or p.get("price") or "sob consulta"
                state["response"] = (
                    f"{p.get('title', 'Imóvel')} — {p.get('region', '')}, "
                    f"{p.get('area_util', '')} m², {price}. "
                    "Quer que eu compare com outra opção?"
                )
            else:
                state["response"] = "Qual imóvel específico você quer que eu detalhe?"
            return state
        if tool == "compare_properties":
            if tr is not None and tr.comparison:
                a = tr.comparison.get("a") or {}
                b = tr.comparison.get("b") or {}
                lines = [
                    f"- {p.get('title', 'Imóvel')}: {p.get('region', '')}, "
                    f"{p.get('area_util', '')} m², {p.get('price_text') or p.get('price', 'sob consulta')}"
                    for p in (a, b)
                ]
                state["response"] = (
                    "Comparativo das opções:\n" + "\n".join(lines) + "\n\n"
                    "Quer que eu detalhe alguma ou ajuste algum critério?"
                )
            elif tr is not None and tr.needs_clarification:
                state["response"] = (
                    "Preciso de dois imóveis já mostrados para comparar. "
                    "Quer que eu mostre mais opções?"
                )
            else:
                state["response"] = "Preciso de dois imóveis já mostrados para comparar."
            return state
        if tool in ("request_options", "refine_search"):
            if tr is not None and tr.properties:
                state["properties"] = tr.properties
                if not state.get("shown_properties_count"):
                    state["shown_properties_count"] = len(tr.properties)
                self._remember_shown(state)
                listed = "\n".join(
                    f"{i + 1}. {p.get('title', 'Imóvel')} — {p.get('region', '')}, {p.get('area_util', '')} m²"
                    for i, p in enumerate(state["properties"][:9] if tool == "request_options" else state["properties"][:3])
                )
                state["response"] = (
                    f"Separei algumas opções:\n{listed}\n\n"
                    "Alguma chamou atenção? Posso refinar por metragem, orçamento ou região."
                )
                return state
            if self.properties_rag is not None:
                return self._node_recommendation(state)
            state["response"] = (
                "Não encontrei opções com esses filtros agora. "
                "Posso ajudar com outra busca?"
            )
            return state
        if tool == "express_visit_interest":
            state["response"] = (
                "Ótimo! Fico à vontade pra ajudar a agendar uma visita "
                "quando você quiser."
            )
            return state
        if tool == "request_schedule":
            if tr is not None and not tr.ok:
                state["current_state"] = "conversation"
                state["response"] = (
                    "Antes de agendar, me diga qual imóvel mais te interessou "
                    "ou se quer refinar a busca. Assim consigo preparar a visita ideal."
                )
                return state
            state["response"] = "Vamos escolher um horário pra visita?"
            return state
        if tool == "unclear":
            state["response"] = (
                "Não entendi bem. Pode reformular? "
                "Também posso mostrar opções ou comparar imóveis já vistos."
            )
            return state
        # provide_info / fallback
        return self._node_recommendation(state)

    def _node_greeting(self, state: FlowState) -> FlowState:
        from service.security_layer import CONSENT_MESSAGE, REFUSAL_MESSAGE

        message = state.get("message", "")
        if message.strip().lower() in ("não", "nao", "não quero", "nao quero"):
            state["response"] = REFUSAL_MESSAGE
            state["current_state"] = "followup"
            state["consent_recorded"] = False
            return state
        state["response"] = CONSENT_MESSAGE
        state["current_state"] = "elicitation"
        return state

    def _node_elicitation(self, state: FlowState) -> FlowState:
        from service.security_layer import REFUSAL_MESSAGE

        message = state.get("message", "")
        if message.strip().lower() in ("não", "nao", "não quero", "nao quero"):
            state["response"] = REFUSAL_MESSAGE
            state["current_state"] = "followup"
            state["consent_recorded"] = False
            return state
        state["current_state"] = "intent"
        state["consent_recorded"] = True
        state["response"] = "Para indicar as melhores opções, você busca compra, locação ou investimento?"
        return state

    def _node_intent(self, state: FlowState) -> FlowState:
        message = state.get("message", "")
        intent, confidence = self._llm_classify(message)
        if confidence < CONFIDENCE_THRESHOLD:
            state["response"] = INTENT_CONFIRM_QUESTION
            return state
        state["intent"] = intent
        state["current_state"] = "qualification"
        info = state.get("lead_info", {})
        if self.properties_rag is not None and (info.get("region") or info.get("area")):
            state["current_state"] = "recommendation"
            region = info.get("region", "")
            state["response"] = self._build_early_recommendation(state, region)
        else:
            state["response"] = "Ótimo! Qual a metragem desejada, região e orçamento?"
        return state

    def _node_qualification(self, state: FlowState) -> FlowState:
        info = state.get("lead_info", {})
        if self._wants_options(state) and self.properties_rag is not None:
            return self._node_recommendation(state)
        if self.properties_rag is not None and (info.get("region") or info.get("area")):
            state["current_state"] = "recommendation"
            properties = self.properties_rag(info)
            if properties:
                state["properties"] = properties[:3]
                state["shown_properties_count"] = len(properties[:3])
                listed = "\n".join(
                    f"{i + 1}. {p.get('title', 'Imóvel')} — {p.get('region', '')}, {p.get('area_util', '')} m²"
                    for i, p in enumerate(state["properties"])
                )
                state["response"] = (
                    f"Já tenho uma ideia do que você procura. Separei algumas opções:\n{listed}\n\n"
                    "Alguma chamou atenção? Posso refinar por metragem, orçamento ou região."
                )
                self._remember_shown(state)
                return state
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
            state["current_state"] = "qualification"
            state["lead_qualified"] = False
            missing = result["missing"]
            if self.properties_rag and (info.get("region") or info.get("area")):
                state["response"] = (
                    "Quase lá! "
                    + ", ".join(missing)
                    + " pra eu te mostrar as melhores opções."
                )
            else:
                state["response"] = (
                    self.qualifier.explain(result)
                    + " Ainda precisamos de algumas informações: "
                    + ", ".join(missing)
                    + "."
                )
        return state

    @staticmethod
    def _area_m2(area: Any) -> float | None:
        if not area:
            return None
        match = re.search(r"(\d+(?:[.,]\d+)?)", str(area))
        return float(match.group(1).replace(",", ".")) if match else None

    def _build_early_recommendation(self, state: FlowState, region: str) -> str:
        """Mostra imóveis imediatamente quando a intenção é clara — SDR consultivo."""
        if self.properties_rag is None:
            return f"Perfeito! Tenho opções em {region}. Posso te mostrar agora?" if region else "Perfeito! Posso te mostrar as opções."
        properties = self.properties_rag(state.get("lead_info", {}))
        if not properties:
            return "Perfeito! Vou buscar as melhores opções pra você."
        top = properties[:3]
        listed = "\n".join(
            f"{i + 1}. {p.get('title', 'Imóvel')} — {p.get('region', '')}, {p.get('area_util', '')} m²"
            for i, p in enumerate(top)
        )
        state["properties"] = top
        state["shown_properties_count"] = len(top)
        self._remember_shown(state)
        return (
            f"Tenho algumas opções pra você:\n{listed}\n"
            "Alguma chamou atenção? Posso refinar por metragem, orçamento ou localização."
        )

    def _node_discovery(self, state: FlowState) -> FlowState:
        if self._wants_options(state) and self.properties_rag is not None:
            return self._show_more_options(state)
        focus = state.get("favorite_property") or (
            (state.get("properties") or [{}])[0].get("title") if state.get("properties") else None
        )
        message = state.get("message", "").lower()
        props = state.get("properties") or []
        focus_prop = next(
            (p for p in props if focus and str(p.get("title", "")).lower() == str(focus).lower()),
            props[0] if props else None,
        )
        if focus_prop:
            detail_bits = []
            if "estacionamento" in message or "vaga" in message:
                vagas = focus_prop.get("vagas")
                detail_bits.append(
                    f"estacionamento com {vagas} vaga(s)" if vagas is not None else "estacionamento sob consulta"
                )
            if "andar" in message:
                detail_bits.append(f"andar: {focus_prop.get('andar') or 'sob consulta'}")
            if "preço" in message or "valor" in message or "quanto" in message:
                detail_bits.append(f"valor: {focus_prop.get('price_text') or focus_prop.get('price') or 'sob consulta'}")
            detail = "; ".join(detail_bits) if detail_bits else (
                f"{focus_prop.get('title')} — {focus_prop.get('region', '')}, {focus_prop.get('area_util', '')} m²"
            )
            state["response"] = (
                f"Sobre {focus_prop.get('title', 'o imóvel')}: {detail}. "
                "Quer que eu compare com outra opção ou ajuste algum critério?"
            )
        else:
            state["response"] = (
                "Qual imóvel específico você quer que eu detalhe? "
                "Posso comparar as opções que já mostrei."
            )
        state["current_state"] = "discovery"
        return state

    def _node_recommendation(self, state: FlowState) -> FlowState:
        tool = state.get("_router_tool")
        action = state.get("_router_action")
        if (tool == "compare_properties" or action == "compare_properties") and state.get("properties"):
            props = state["properties"][:3]
            lines = [
                f"- {p.get('title', 'Imóvel')}: {p.get('region', '')}, "
                f"{p.get('area_util', '')} m², {p.get('price_text') or p.get('price', 'sob consulta')}"
                for p in props
            ]
            state["response"] = (
                "Comparativo das opções:\n" + "\n".join(lines) + "\n\n"
                "Quer que eu detalhe alguma ou ajuste algum critério?"
            )
            return state
        if self._wants_options(state) and self.properties_rag is not None:
            return self._show_more_options(state)
        if self.properties_rag is None:
            state["response"] = (
                "Sem imóveis disponíveis agora. "
                "Posso mostrar opções assim que tivermos mais dados, ou ajudar com outra coisa?"
            )
            return state
        properties = self.properties_rag(state.get("lead_info", {}))
        if not properties:
            state["response"] = (
                "Não encontramos imóveis com esses filtros. Sugerimos ajustar metragem, região ou orçamento."
            )
            return state
        state["properties"] = properties[:3]
        if not state.get("shown_properties_count"):
            state["shown_properties_count"] = len(state["properties"])
        listed = "\n".join(
            f"{i + 1}. {p.get('title', 'Imóvel')} — {p.get('region', '')}, {p.get('area_util', '')} m²"
            for i, p in enumerate(state["properties"])
        )
        state["response"] = (
            f"Separei algumas opções que parecem próximas do que você procura:\n{listed}\n\n"
            "Alguma delas chamou atenção? "
            "Posso também ajustar por metragem, orçamento ou localização."
        )
        self._remember_shown(state)
        return state

    def _node_scheduling(self, state: FlowState) -> FlowState:
        message = state.get("message", "")
        if self._wants_options(state) and self.properties_rag is not None:
            return self._show_more_options(state)
        if not self.ready_for_scheduling(state):
            state["current_state"] = "recommendation"
            state["response"] = (
                "Antes de agendar, me diga qual imóvel mais te interessou "
                "ou se quer refinar a busca. Assim consigo preparar a visita ideal."
            )
            return state
        shown_count = max(state.get("shown_properties_count", 0) or 0, len(state.get("properties") or []))
        if shown_count < 3 and not state.get("lead_id"):
            state["current_state"] = "recommendation"
            state["response"] = (
                "Antes de pensar em agendar, vou te mostrar mais opções. "
                "Qualquer uma dessas te interessou?"
            )
            return self._show_more_options(state) if self.properties_rag else state
        if self._is_restricted(state.get("lead_id")):
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
        ics = _build_ics(result)
        if ics:
            state["ics_invite"] = ics
        state["response"] = "Agendamento confirmado! Enviarei o convite e o resumo ao corretor."
        return state

    def _show_more_options(self, state: FlowState) -> FlowState:
        """Pedido de mais opções: reconsulta RAG excluindo o que já foi exibido."""
        shown = {p.get("title") for p in (state.get("properties") or [])}
        properties = self.properties_rag(state.get("lead_info", {}))
        fresh = [p for p in properties if p.get("title") not in shown][:3]
        if not fresh:
            state["response"] = (
                "Essas são todas as opções que atendem aos seus critérios. "
                "Posso mostrar outras ou ajudar com mais alguma coisa?"
            )
            return state
        state["properties"] = (state.get("properties") or []) + fresh
        state["shown_properties_count"] = len(state["properties"])
        listed = "\n".join(
            f"- {p.get('title', 'Imóvel')} — {p.get('region', '')}, {p.get('area_util', '')} m²"
            for p in fresh
        )
        state["response"] = (
            f"Separei mais algumas opções que parecem próximas do que você procura:\n{listed}\n\n"
            "Alguma chamou atenção? Posso também refinar por metragem, orçamento ou localização."
        )
        self._remember_shown(state)
        return state

    def _remember_shown(self, state: FlowState) -> None:
        """Persiste imóveis exibidos no context entre turnos (handler não reenvia properties)."""
        props = state.get("properties") or []
        if not props:
            return
        ctx = state.setdefault("context", {})
        ctx["properties"] = list(props)
        ctx["shown_properties_count"] = int(state.get("shown_properties_count") or len(props))

    def _node_handoff(self, state: FlowState) -> FlowState:
        if self.handoff_builder is not None:
            state["handoff_summary"] = self.handoff_builder(state)
        state["current_state"] = "handoff"
        state["done"] = True
        state.setdefault("response", "Resumo enviado ao corretor. Obrigado pelo contato!")
        return state

    def _node_followup(self, state: FlowState) -> FlowState:
        if self._wants_options(state) and self.properties_rag is not None:
            return self._node_recommendation(state)
        if self._is_restricted(state.get("lead_id")):
            state["followup_deferred"] = True
            state["current_state"] = "followup"
            state["done"] = True
            state["response"] = "Anotado! Seu atendimento seguirá com o corretor responsável."
            return state
        state["current_state"] = "followup"
        state["done"] = True
        state.setdefault("response", "Anotado! Retomaremos o contato em breve.")
        return state

    def _node_postprocess(self, state: FlowState) -> FlowState:
        """FR-02: geração de resposta humanizada via LLM (exceto LGPD/recusa)."""
        if self.reply_generator is None:
            return state
        from service.security_layer import CONSENT_MESSAGE, REFUSAL_MESSAGE

        canned = state.get("response")
        if not canned or canned in (CONSENT_MESSAGE, REFUSAL_MESSAGE):
            return state
        try:
            generated = self.reply_generator(
                state.get("message", ""),
                canned,
                state.get("lead_info") or {},
                state.get("properties") or [],
                favorite_property=state.get("favorite_property"),
                conversation_stage=state.get("current_state"),
                shown_properties_count=state.get("shown_properties_count", 0),
            )
            if generated and generated.strip():
                state["response"] = generated.strip()
        except Exception:
            logger.warning("LLM reply falhou; mantendo resposta oficial (fallback)", exc_info=True)
        return state

    # --- Invocação ---

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        if self._graph is not None:
            result = self._graph.invoke(state)
            return dict(result)
        return self._invoke_fsm(state)

    def _invoke_fsm(self, state: dict[str, Any]) -> dict[str, Any]:
        """Fallback manual (sem LangGraph) — mesmo contrato, mesmo router tool-agent/ADR-011."""
        state = self._node_preprocess(state)
        resolved = self._route_state(state)
        message = state.get("message", "")
        if resolved == "conversation":
            method = self._node_conversation
        else:
            method = getattr(self, f"_handle_{resolved}", None)
        if method is None:
            state["response"] = "Como posso ajudar?"
            return state
        method(state, message)
        return self._polish_reply(state, message)

    # --- Handlers antigos (fallback FSM) ---

    def _handle_greeting(self, state, message):
        return self._node_greeting(state)

    def _handle_elicitation(self, state, message):
        return self._node_elicitation(state)

    def _handle_conversation(self, state, message):
        return self._node_conversation(state)

    def _handle_intent(self, state, message):
        return self._node_conversation(state)

    def _handle_qualification(self, state, message):
        return self._node_conversation(state)

    def _handle_discovery(self, state, message):
        return self._node_conversation(state)

    def _handle_recommendation(self, state, message):
        return self._node_conversation(state)

    def _handle_scheduling(self, state, message):
        return self._node_scheduling(state)

    def _handle_handoff(self, state, message):
        return self._node_handoff(state)

    def _handle_followup(self, state, message):
        return self._node_followup(state)

    def _polish_reply(self, state, message):
        return self._node_postprocess(state)

    # --- Helpers ---

    def _is_restricted(self, lead_id: str | None) -> bool:
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


# --- ICS convite (FR-05) -----------------------------------------------------

def _build_ics(appointment: dict[str, Any]) -> str | None:
    """Gera convite .ics (RFC 5545) a partir do dict de agendamento."""
    if not appointment or not appointment.get("confirmed"):
        return None
    when = appointment.get("when", "")
    summary = appointment.get("summary", "Visita — W Levitt SDR")
    location = appointment.get("location", "")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//W Levitt//SDR//PT-BR",
        "BEGIN:VEVENT",
        f"SUMMARY:{summary}",
    ]
    if location:
        lines.append(f"LOCATION:{location}")
    if when:
        lines.append(f"DTSTART:{when}")
        lines.append(f"DTEND:{when}")
    lines.extend(["END:VEVENT", "END:VCALENDAR"])
    return "\r\n".join(lines)
