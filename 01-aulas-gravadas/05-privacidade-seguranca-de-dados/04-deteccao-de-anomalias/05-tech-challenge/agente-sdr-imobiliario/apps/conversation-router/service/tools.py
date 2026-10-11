from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

from service.validation import _ORDINAL_MAP

# _ORDINAL_MAP vem de service.validation (fonte canonica, ver import no topo) —
# nao duplicar vocabulario de ordinais aqui.


@dataclass
class ToolResult:
    ok: bool = True
    tool: str = ""
    properties: list[dict[str, Any]] = field(default_factory=list)
    detail: dict[str, Any] | None = None
    details: list[dict[str, Any]] = field(default_factory=list)  # lead apontou 2-3 imóveis de uma vez
    comparison: dict[str, Any] | None = None
    memory_updates: dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False
    refusal: str | None = None
    current_state_hint: str | None = None
    raw_arguments: dict[str, Any] = field(default_factory=dict)


def _resolve_in_shown(
    ref: str | None, shown: list[dict[str, Any]], prefer_id: str | None = None
) -> dict[str, Any] | None:
    if not ref:
        return None
    target = re.sub(r"[^a-z0-9]+", " ", ref.lower()).strip()
    # Títulos se repetem no catálogo (ex.: 2 "Sobrado à venda, São João Clímaco");
    # o imóvel em foco (id) desempata — senão o 1º homônimo vencia.
    exact = [
        p for p in shown
        if target == re.sub(r"[^a-z0-9]+", " ", str(p.get("title") or "").lower()).strip()
    ]
    if len(exact) > 1 and prefer_id:
        exact = [p for p in exact if str(p.get("id")) == str(prefer_id)] or exact
    if exact:
        return exact[0]
    # 1. Match exato de titulo primeiro — cobre o caso comum de property_ref ja
    # vir com o titulo exato (instrucao do _ROUTER_SYSTEM_PROMPT). Precisa vir
    # ANTES do loop de ordinais: titulos com numero embutido (ex. "Faria Lima 02")
    # colidiriam com a chave ordinal "2" (substring nao ancorada) e resolveriam
    # errado se checados so depois.
    for p in shown:
        if target == re.sub(r"[^a-z0-9]+", " ", str(p.get("title") or "").lower()).strip():
            return p
    # 2. Resolver referência ordinal ("segunda opção", "2", "o segundo")
    ordinal_idx = _ORDINAL_MAP.get(target)
    if ordinal_idx is not None and ordinal_idx < len(shown):
        return shown[ordinal_idx]
    for key, val in _ORDINAL_MAP.items():
        if key in target and val < len(shown):
            return shown[val]
    # 3. Fuzzy match por similaridade de título
    best, best_score = None, 0.0
    for p in shown:
        title = re.sub(r"[^a-z0-9]+", " ", str(p.get("title") or "").lower()).strip()
        if not title:
            continue
        score = SequenceMatcher(None, target, title).ratio()
        if score > best_score:
            best, best_score = p, score
    return best if best is not None and best_score >= 0.55 else None


def _explicit_shown_reference(
    message: str, shown: list[dict[str, Any]]
) -> dict[str, Any] | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", message.lower()).strip()
    for key, index in _ORDINAL_MAP.items():
        if key.isdigit():
            continue
        normalized_key = re.sub(r"[^a-z0-9]+", " ", key.lower()).strip()
        if normalized_key and re.search(rf"\b{re.escape(normalized_key)}\b", normalized):
            if index < len(shown):
                return shown[index]

    option = re.search(
        r"\b(?:(?:op[çc][ãa]o|im[óo]vel|da|do)\s*)(?:n[ºo]\s*)?(\d+)\b",
        message,
        re.IGNORECASE,
    )
    if option:
        index = int(option.group(1)) - 1
        if 0 <= index < len(shown):
            return shown[index]

    matches = []
    for prop in shown:
        title = str(prop.get("title") or "").split("—", 1)[0]
        words = re.sub(r"[^a-z0-9]+", " ", title.lower()).split()
        phrases = {
            " ".join(words[start : start + size])
            for size in range(2, min(4, len(words)) + 1)
            for start in range(len(words) - size + 1)
        }
        if any(re.search(rf"\b{re.escape(phrase)}\b", normalized) for phrase in phrases):
            matches.append(prop)
    return matches[0] if len(matches) == 1 else None


def contact_unreachable(state: dict[str, Any]) -> bool:
    """Sem nenhum canal pro corretor retornar (nem telefone/WhatsApp nem e-mail).
    Basta um dos dois — nome vem do perfil do Telegram e não bloqueia."""
    missing = set(state.get("missing_contact_fields") or [])
    return {"phone", "email"} <= missing


def _missing_contact(result: ToolResult, state: dict[str, Any]) -> ToolResult:
    result.ok = False
    result.refusal = "missing_contact"
    result.raw_arguments["missing_contact_fields"] = [
        f for f in (state.get("missing_contact_fields") or []) if f in ("phone", "email")
    ]
    result.current_state_hint = "conversation"
    return result


def execute_tool(
    tool: str,
    arguments: dict[str, Any],
    state: dict[str, Any],
    search_fn: Callable[..., list[dict[str, Any]]] | None = None,
    memory_updates: dict[str, Any] | None = None,
    message: str = "",
) -> ToolResult:
    """Pure tool execution. No LLM. Guardrails in code."""
    shown = list(state.get("properties") or [])
    if not shown:
        ctx = state.get("context") or {}
        shown = list(ctx.get("properties") or [])
    result = ToolResult(
        tool=tool,
        raw_arguments=dict(arguments or {}),
        memory_updates=dict(memory_updates or {}),
    )

    if tool == "request_options":
        scope = (arguments or {}).get("list_scope", "filtered")
        top_k = 50 if scope == "all" else 9
        if search_fn is None:
            result.ok = False
            result.refusal = "search_unavailable"
            return result
        lead = state.get("lead_info") or {}
        result.properties = search_fn(lead, top_k=top_k)
        return result

    if tool == "property_detail":
        # Prioridade: confia primeiro na referencia que a LLM ja resolveu
        # (property_ref, normalmente o titulo exato vindo do prompt do roteador);
        # o regex sobre a mensagem crua (_explicit_shown_reference) e so rede de
        # seguranca para quando nao ha LLM configurada ou ela nao extraiu nada —
        # evita ter que ensinar Python a reconhecer cada forma nova de falar.
        focus_id = (state.get("context") or {}).get("focus_property_id")
        # Lead referiu VÁRIOS imóveis já mostrados ("desses três", "as duas primeiras"): a LLM
        # devolve property_refs (números/títulos) e o código só resolve cada um contra a lista.
        refs = (arguments or {}).get("property_refs")
        if isinstance(refs, list) and len(refs) > 1:
            hits: list[dict[str, Any]] = []
            for r in refs[:4]:
                found = _resolve_in_shown(str(r), shown, prefer_id=focus_id)
                if found is not None and all(found is not h for h in hits):
                    hits.append(found)
            if hits:
                result.detail = hits[0]
                if len(hits) > 1:
                    result.details = hits
                return result
        ref = (arguments or {}).get("property_ref")
        focus = state.get("favorite_property")
        hit = (
            _resolve_in_shown(ref, shown, prefer_id=focus_id)
            or (_resolve_in_shown(focus, shown, prefer_id=focus_id) if focus else None)
            or _explicit_shown_reference(message, shown)
        )
        if hit is None and len(shown) == 1:
            hit = shown[0]
        if hit is None:
            result.ok = False
            result.needs_clarification = True
            result.refusal = "property_not_in_shown"
            result.current_state_hint = "conversation"
            return result
        result.detail = hit
        return result

    if tool == "compare_properties":
        if len(shown) < 2:
            result.ok = False
            result.needs_clarification = True
            result.refusal = "need_two_shown"
            result.current_state_hint = "conversation"
            return result
        a = _resolve_in_shown((arguments or {}).get("property_a"), shown) or shown[0]
        b = _resolve_in_shown((arguments or {}).get("property_b"), shown) or shown[1]
        if a is b and len(shown) >= 2:
            b = next(p for p in shown if p is not a)
        result.comparison = {"a": a, "b": b}
        return result

    if tool == "refine_search":
        if search_fn is None:
            result.ok = False
            result.refusal = "search_unavailable"
            return result
        lead = state.get("lead_info") or {}
        result.properties = search_fn(lead, top_k=9)
        return result

    if tool == "express_visit_interest":
        return result

    if tool == "request_schedule":
        shown_n = max(int(state.get("shown_properties_count") or 0), len(shown))
        visit = bool(
            state.get("visit_interest") or (memory_updates or {}).get("visit_interest")
        )
        favorite = bool(state.get("favorite_property"))
        ready = bool(
            favorite
            or state.get("visit_interest")
            or (state.get("lead_info") or {}).get("deadline")
            or visit
        )
        # Sinal comercial forte (pedido explícito + imóvel já indicado) dispensa o
        # mínimo de 3 imóveis mostrados, mas nunca dispensa ter mostrado pelo menos 1.
        min_shown = 1 if (visit and favorite) else 3
        if shown_n < min_shown or not visit or not ready:
            result.ok = False
            result.refusal = "scheduling_gate"
            result.current_state_hint = "conversation"
            return result
        if contact_unreachable(state):
            return _missing_contact(result, state)
        result.current_state_hint = "scheduling"
        return result

    if tool == "request_human":
        # A decisão de quem quer falar com humano é da LLM (prompt do roteador): nada de
        # palavra-chave na mensagem desfazendo isso ("falar com o corretor sobre essa opção").
        if contact_unreachable(state):
            return _missing_contact(result, state)
        result.current_state_hint = "handoff"
        return result

    if tool == "decline":
        result.current_state_hint = "followup"
        return result

    if tool in ("provide_info", "unclear"):
        result.current_state_hint = "conversation"
        return result

    result.ok = False
    result.refusal = "unknown_tool"
    result.current_state_hint = "conversation"
    return result
