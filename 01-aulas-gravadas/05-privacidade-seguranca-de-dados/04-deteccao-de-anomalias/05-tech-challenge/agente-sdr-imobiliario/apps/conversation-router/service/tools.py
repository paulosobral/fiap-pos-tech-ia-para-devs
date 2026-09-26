from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

_OPTIONS_REQUEST_RE = re.compile(
    r"\b(?:op[çc][ãa]o|op[çc][õo]es|mais im[óo]veis|mostrar|lista|tudo|complet[oa])\b",
    re.IGNORECASE,
)

# Ordinais em PT-BR → índice (0-based). Partilhado com validation.py para resolução de referências.
_ORDINAL_MAP = {
    "primeiro": 0,
    "primeira": 0,
    "1": 0,
    "primeira opcao": 0,
    "primeira opçao": 0,
    "segundo": 1,
    "segunda": 1,
    "2": 1,
    "segunda opcao": 1,
    "segunda opçao": 1,
    "terceiro": 2,
    "terceira": 2,
    "3": 2,
    "terceira opcao": 2,
    "terceira opçao": 2,
    "quarto": 3,
    "quarta": 3,
    "4": 3,
    "quinto": 4,
    "quinta": 4,
    "5": 4,
    "sexto": 5,
    "sexta": 5,
    "6": 5,
    "setimo": 6,
    "setima": 6,
    "sétimo": 6,
    "sétima": 6,
    "7": 6,
    "oitavo": 7,
    "oitava": 7,
    "8": 7,
    "nono": 8,
    "nona": 8,
    "9": 8,
    "decimo": 9,
    "decima": 9,
    "décimo": 9,
    "décima": 9,
    "10": 9,
}


@dataclass
class ToolResult:
    ok: bool = True
    tool: str = ""
    properties: list[dict[str, Any]] = field(default_factory=list)
    detail: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    memory_updates: dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False
    refusal: str | None = None
    current_state_hint: str | None = None
    raw_arguments: dict[str, Any] = field(default_factory=dict)


def _resolve_in_shown(
    ref: str | None, shown: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not ref:
        return None
    target = re.sub(r"[^a-z0-9]+", " ", ref.lower()).strip()
    # 1. Resolver referência ordinal ("segunda opção", "2", "o segundo")
    ordinal_idx = _ORDINAL_MAP.get(target)
    if ordinal_idx is not None and ordinal_idx < len(shown):
        return shown[ordinal_idx]
    for key, val in _ORDINAL_MAP.items():
        if key in target and val < len(shown):
            return shown[val]
    # 2. Fuzzy match por similaridade de título
    best, best_score = None, 0.0
    for p in shown:
        title = re.sub(r"[^a-z0-9]+", " ", str(p.get("title") or "").lower()).strip()
        if not title:
            continue
        score = SequenceMatcher(None, target, title).ratio()
        if score > best_score:
            best, best_score = p, score
    return best if best is not None and best_score >= 0.55 else None


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
        ref = (arguments or {}).get("property_ref")
        focus = state.get("favorite_property")
        hit = _resolve_in_shown(ref, shown) or (
            _resolve_in_shown(focus, shown) if focus else None
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
        ready = bool(
            state.get("favorite_property")
            or state.get("visit_interest")
            or (state.get("lead_info") or {}).get("deadline")
            or visit
        )
        if shown_n < 3 or not visit or not ready:
            result.ok = False
            result.refusal = "scheduling_gate"
            result.current_state_hint = "conversation"
            return result
        result.current_state_hint = "scheduling"
        return result

    if tool == "request_human":
        if message and _OPTIONS_REQUEST_RE.search(message):
            result.current_state_hint = "conversation"
            return result
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
