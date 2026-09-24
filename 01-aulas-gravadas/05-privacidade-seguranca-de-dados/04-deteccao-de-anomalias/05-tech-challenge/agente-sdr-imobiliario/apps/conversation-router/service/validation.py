from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_VISIT_EVIDENCE_RE = re.compile(
    r"\b(?:quero visitar|vamos marcar|posso conhecer|tem agenda|quero marcar|"
    r"agendar|marcar visita|conhecer o espa[çc]o)\b",
    re.IGNORECASE,
)
_FUZZY_THRESHOLD = 0.55
_MEMORY_KEYS = ("favorite_property", "visit_interest")
_KNOWN_LEAD_FIELDS = ("area", "region", "budget", "deadline", "people_count", "decision_maker")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _fuzzy_shown(name: str | None, shown: list[dict[str, Any]]) -> str | None:
    if not name:
        return None
    target = _norm(name)
    if not target:
        return None
    best, best_score = None, 0.0
    for p in shown:
        title = _norm(str(p.get("title") or ""))
        if not title:
            continue
        score = SequenceMatcher(None, target, title).ratio()
        tset, title_set = set(target.split()), set(title.split())
        if tset and title_set and tset & title_set:
            score = max(score, len(tset & title_set) / len(tset))
        if score > best_score:
            best, best_score = str(p.get("title") or ""), score
    return best if best is not None and best_score >= _FUZZY_THRESHOLD else None


def validate_router_output(
    raw: dict[str, Any],
    state: dict[str, Any],
    message: str = "",
) -> dict[str, Any]:
    """Validate single-step tool-call output. LLM never writes state directly.

    favorite_property: fuzzy match ONLY against shown_properties (never catalog).
    visit_interest: persists only with explicit visit evidence in message.
    thought is log-only; unknown memory keys dropped.
    """
    shown = list(state.get("properties") or [])
    if not shown:
        ctx = state.get("context") or {}
        shown = list(ctx.get("properties") or [])

    mem = raw.get("memory_updates")
    if not isinstance(mem, dict):
        mem = {}
    clean_mem: dict[str, Any] = {}
    fp = mem.get("favorite_property")
    clean_mem["favorite_property"] = _fuzzy_shown(
        str(fp) if fp not in (None, "") else None, shown
    )
    want_visit = bool(mem.get("visit_interest"))
    has_evidence = bool(_VISIT_EVIDENCE_RE.search(message or ""))
    clean_mem["visit_interest"] = bool(want_visit and has_evidence)
    clean_mem = {k: clean_mem[k] for k in _MEMORY_KEYS if k in clean_mem}

    args = raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {}
    lead = raw.get("lead_info") if isinstance(raw.get("lead_info"), dict) else {}
    clean_lead = {
        k: v for k, v in lead.items() if k in _KNOWN_LEAD_FIELDS and v not in (None, "")
    }
    tool = raw.get("tool")
    return {
        "thought": str(raw.get("thought") or ""),
        "tool": tool,
        "arguments": args,
        "lead_info": clean_lead,
        "memory_updates": clean_mem,
    }
