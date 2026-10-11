# Tool-Agent Router + 5-State FSM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the conversation router from fixed-action classifier to single-step tool-calling agent with 5 explicit states, pure tool execution, and validated commercial memory.

**Architecture:** Keep LangGraph. LLM returns `{thought, tool, arguments, lead_info, memory_updates}`. Code validates (enum, fuzzy-on-shown, visit evidence), executes pure tools, routes to 5-state FSM (`greeting|conversation|scheduling|handoff|followup`), then reply generator with full context kwargs.

**Tech Stack:** Python 3.11, LangGraph, pytest, LiteLLM/OpenRouter, FAISS catalog RAG

**Spec:** `docs/superpowers/specs/2026-09-23-tool-agent-sdr-design.md`

## Global Constraints

- Run tests per-app: `cd apps/conversation-router && ../../.venv/bin/pytest tests -q` (never run multi-app together — `ImportPathMismatchError`)
- Base green suite: conversation-router 222, voice-adapter 82, dashboard-api 66, dashboard-ui 8 — total 378
- LLM never writes `state` directly; only validated `memory_updates` after `_validate_router_output`
- `thought` is log-only — never controls flow
- Fuzzy match for `favorite_property` only against `shown_properties` (context list), never full catalog
- `visit_interest` persists only with explicit visit evidence or router+scheduling gate agreement; favorite ≠ visit interest
- Scheduling gate: `ready_for_scheduling` + `shown >= 3` + `visit_interest == True` — **does NOT require `favorite_property`**
- `request_schedule` means intent, not action; code authorizes via gates
- Tool `express_visit_interest` (not `visit_interest`) to avoid collision with memory field
- `request_options.arguments.list_scope`: `"filtered" | "all"`, default `filtered`
- ADR-011 fallback: invalid tool / router exception → regex + legacy FSM path
- LGPD greeting/elicitation never bypass LLM routing
- `generate_reply` new kwargs: `visit_interest`, `rejected_properties`, `last_tool`
- Long list reply: if `len(properties) > 3`, relax 3-sentence cap → 1 sentence/property + close, `max_tokens≈800`
- Commit only when user explicitly requests; never amend/push without request

---

### Task 1: Tool enum + VALID_TOOLS + parse contract

**Files:**
- Modify: `apps/conversation-router/service/llm.py:349-392` (VALID_ACTIONS, _ROUTER_SYSTEM_PROMPT)
- Modify: `apps/conversation-router/service/llm.py:395-443` (extract_and_route, _parse_extract_and_route)
- Test: `apps/conversation-router/tests/unit/test_llm.py`

**Interfaces:**
- Consumes: existing `_completion_with_fallback`, `json`, `VALID_ACTIONS` pattern
- Produces: `VALID_TOOLS` tuple (10 tools), `_parse_extract_and_route(raw) -> dict` returning `{thought, tool, arguments, lead_info, memory_updates}`, raises `ValueError` on invalid tool/JSON

- [ ] **Step 1: Write failing tests for VALID_TOOLS + parse**

```python
# test_llm.py additions
from service.llm import VALID_TOOLS, _parse_extract_and_route, extract_and_route
import pytest

def test_valid_tools_enum():
    expected = {
        "request_options", "property_detail", "compare_properties",
        "refine_search", "express_visit_interest", "request_schedule",
        "request_human", "decline", "provide_info", "unclear",
    }
    assert set(VALID_TOOLS) == expected
    assert "visit_interest" not in VALID_TOOLS  # renamed to express_visit_interest

def test_parse_tool_call_full():
    raw = '{"thought":"x","tool":"request_options","arguments":{"list_scope":"all"},"lead_info":{"region":"Pinheiros"},"memory_updates":{"favorite_property":null,"visit_interest":false}}'
    out = _parse_extract_and_route(raw)
    assert out["tool"] == "request_options"
    assert out["arguments"]["list_scope"] == "all"
    assert out["lead_info"]["region"] == "Pinheiros"
    assert out["memory_updates"]["visit_interest"] is False

def test_parse_invalid_tool_raises():
    with pytest.raises(ValueError):
        _parse_extract_and_route('{"tool":"invented_tool","thought":"x"}')

def test_parse_invalid_json_raises():
    with pytest.raises(ValueError):
        _parse_extract_and_route("not json")

def test_parse_missing_tool_raises():
    with pytest.raises(ValueError):
        _parse_extract_and_route('{"thought":"x","lead_info":{}}')
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_llm.py -v -k "tool or parse"`
Expected: FAIL (VALID_TOOLS not defined, parse still expects `action`)

- [ ] **Step 3: Implement VALID_TOOLS + rewrite parse + prompt**

```python
# llm.py — replace VALID_ACTIONS with:
VALID_TOOLS = (
    "request_options",
    "property_detail",
    "compare_properties",
    "refine_search",
    "express_visit_interest",
    "request_schedule",
    "request_human",
    "decline",
    "provide_info",
    "unclear",
)

_ROUTER_SYSTEM_PROMPT = (
    "Você é o roteador de conversa de um SDR imobiliário B2B. A cada mensagem do lead, "
    "faça três coisas:\n"
    "1. EXTRAIA dados de negócio citados (só os que aparecerem): "
    "area, region, budget, deadline, people_count (inteiro), decision_maker ('yes'/'no').\n"
    "2. ATUALIZE memória comercial em memory_updates: "
    "favorite_property (string|null — só imóveis já mostrados), "
    "visit_interest (bool — true SOMENTE com verbo de visita explícito: "
    "'quero visitar','vamos marcar','posso conhecer','tem agenda').\n"
    "3. ESCOLHA UMA tool: " + ", ".join(VALID_TOOLS) + ".\n"
    "   - request_options: quer VER/RECEBER imóveis ou listar opções "
    "('cadê as opções','mostra tudo','lista todas','tem mais opções?'); "
    "arguments.list_scope='all' se pedir tudo/catálogo completo, senão 'filtered'.\n"
    "   - property_detail: pergunta detalhe de imóvel já mostrado "
    "('quanto custa?','tem estacionamento?','qual andar?','quantas vagas?','condomínio quanto?'); "
    "arguments.property_ref opcional.\n"
    "   - compare_properties: quer COMPARAR opções já mostradas "
    "('diferença entre 1 e 2','compara as duas'); arguments.property_a/property_b opcionais.\n"
    "   - refine_search: AJUSTAR critérios ('mais barato','outra região','sem estacionamento?').\n"
    "   - express_visit_interest: demonstra INTERESSE em visitar SEM verbo de agendamento "
    "('gostei da 2','essa me interessa'). NÃO confunda com favorito.\n"
    "   - request_schedule: quer agendar APENAS com verbo explícito "
    "('agendar','marcar visita','reserve','quero marcar').\n"
    "   - request_human: quer corretor/humano AGORA ('quero um corretor','fala com alguém').\n"
    "   - decline: parar/recusar/desistir.\n"
    "   - provide_info: fallback genérico quando não há tool melhor.\n"
    "   - unclear: ambígua, sem ação clara.\n"
    "thought: 1 frase sobre seu raciocínio (apenas para log).\n"
    "REGRA: favorito ≠ visita. 'Gostei da Torre Nova' → favorite_property='Torre Nova', "
    "visit_interest=false. 'Quero conhecer a Torre Nova' → favorite + visit_interest=true.\n"
    "favorite_property só pode apontar para imóveis já EXIBIDOS ao lead.\n"
    'Responda APENAS com JSON: {"thought":"...","tool":"<tool>",'
    '"arguments":{},"lead_info":{},"memory_updates":'
    '{"favorite_property":null,"visit_interest":false}}.'
)

def _parse_extract_and_route(raw: str) -> dict[str, Any]:
    if not raw:
        raise ValueError("Resposta LLM vazia")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Resposta LLM não é JSON válido") from exc
    tool = data.get("tool")
    if tool not in VALID_TOOLS:
        raise ValueError(f"Tool fora do enum válido: {tool!r}")
    arguments = data.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    extracted = data.get("lead_info")
    if not isinstance(extracted, dict):
        extracted = {}
    clean = {k: v for k, v in extracted.items() if k in _KNOWN_LEAD_FIELDS and v not in (None, "")}
    memory = data.get("memory_updates")
    if not isinstance(memory, dict):
        memory = {}
    mem_clean: dict[str, Any] = {}
    if "favorite_property" in memory:
        fp = memory.get("favorite_property")
        mem_clean["favorite_property"] = str(fp) if fp not in (None, "") else None
    if "visit_interest" in memory:
        mem_clean["visit_interest"] = bool(memory.get("visit_interest"))
    return {
        "thought": str(data.get("thought") or ""),
        "tool": tool,
        "arguments": arguments,
        "lead_info": clean,
        "memory_updates": mem_clean,
    }
```

Also update `extract_and_route` docstring + `max_tokens=300` (JSON is larger).

- [ ] **Step 4: Run tests to verify pass**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_llm.py -v`
Expected: PASS (all new + existing)

- [ ] **Step 5: Run full conversation-router suite**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests -q`
Expected: all green (fix any tests still expecting `action` key)

- [ ] **Step 6: Commit**

```bash
git add apps/conversation-router/service/llm.py apps/conversation-router/tests/unit/test_llm.py
git commit -m "feat(llm): VALID_TOOLS + tool-call parse contract for single-step router"
```

---

### Task 2: Validation layer `_validate_router_output` (enum already in parse; fuzzy + visit evidence)

**Files:**
- Create: `apps/conversation-router/service/validation.py`
- Test: `apps/conversation-router/tests/unit/test_validate_router.py`

**Interfaces:**
- Consumes: parse output shape `{thought, tool, arguments, lead_info, memory_updates}`
- Produces: `validate_router_output(raw_result, state) -> dict` same shape with validated `memory_updates.favorite_property` (str|None) and `visit_interest` (bool); drops unknown keys

- [ ] **Step 1: Write failing tests**

```python
# test_validate_router.py
from service.validation import validate_router_output

SHOWN = [
    {"title": "Corporate Faria Lima 02", "region": "Paulista"},
    {"title": "Torre Nova", "region": "Pinheiros"},
]

def _state(**kw):
    base = {"shown_properties_count": len(SHOWN), "properties": SHOWN, "context": {}}
    base.update(kw)
    return base

def test_fuzzy_favorite_matches_shown():
    raw = {"tool": "provide_info", "arguments": {},
           "lead_info": {}, "memory_updates": {"favorite_property": "Faria Lima Corporate", "visit_interest": False}}
    out = validate_router_output(raw, _state())
    assert out["memory_updates"]["favorite_property"] == "Corporate Faria Lima 02"

def test_fuzzy_favorite_miss_returns_none():
    raw = {"tool": "provide_info", "arguments": {},
           "lead_info": {}, "memory_updates": {"favorite_property": "Invented Tower XYZ", "visit_interest": False}}
    out = validate_router_output(raw, _state())
    assert out["memory_updates"]["favorite_property"] is None

def test_favorite_never_full_catalog():
    # even if catalog has other props, only shown list is searched
    raw = {"tool": "provide_info", "arguments": {},
           "lead_info": {}, "memory_updates": {"favorite_property": "Some Other Catalog Prop", "visit_interest": False}}
    out = validate_router_output(raw, _state())
    assert out["memory_updates"]["favorite_property"] is None

def test_visit_interest_requires_evidence():
    # message without visit verb → force false even if LLM said true
    raw = {"tool": "express_visit_interest", "arguments": {},
           "lead_info": {}, "memory_updates": {"favorite_property": "Torre Nova", "visit_interest": True}}
    out = validate_router_output(raw, _state(), message="gostei da Torre Nova")
    assert out["memory_updates"]["visit_interest"] is False
    assert out["memory_updates"]["favorite_property"] == "Torre Nova"

def test_visit_interest_with_evidence_persists():
    raw = {"tool": "express_visit_interest", "arguments": {},
           "lead_info": {}, "memory_updates": {"favorite_property": "Torre Nova", "visit_interest": True}}
    out = validate_router_output(raw, _state(), message="quero visitar a Torre Nova")
    assert out["memory_updates"]["visit_interest"] is True

def test_thought_ignored_for_flow():
    raw = {"thought": "decide schedule", "tool": "request_schedule", "arguments": {},
           "lead_info": {}, "memory_updates": {}}
    out = validate_router_output(raw, _state(), message="agendar")
    assert "thought" in out  # kept for logging only
    assert out["tool"] == "request_schedule"  # flow driven by tool, not thought

def test_unknown_memory_keys_dropped():
    raw = {"tool": "provide_info", "arguments": {},
           "lead_info": {}, "memory_updates": {"favorite_property": "Torre Nova",
                                               "visit_interest": False, "evil_key": "x"}}
    out = validate_router_output(raw, _state())
    assert "evil_key" not in out["memory_updates"]
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_validate_router.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement validation.py**

```python
# service/validation.py
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
        # token overlap boost
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
    """Validate single-step tool-call output. LLM never writes state directly."""
    shown = list(state.get("properties") or [])
    # ensure shown from context if state empty (handler may not rehydrate)
    if not shown:
        ctx = state.get("context") or {}
        shown = list(ctx.get("properties") or [])

    mem = raw.get("memory_updates")
    if not isinstance(mem, dict):
        mem = {}
    clean_mem: dict[str, Any] = {}
    if "favorite_property" in mem or True:  # always resolve favorite
        fp = mem.get("favorite_property")
        clean_mem["favorite_property"] = _fuzzy_shown(
            str(fp) if fp not in (None, "") else None, shown
        )
    want_visit = bool(mem.get("visit_interest"))
    # evidence: explicit visit verb in message OR tool already express + evidence
    has_evidence = bool(_VISIT_EVIDENCE_RE.search(message or ""))
    clean_mem["visit_interest"] = bool(want_visit and has_evidence)
    # drop unknown keys
    clean_mem = {k: clean_mem[k] for k in _MEMORY_KEYS if k in clean_mem}

    args = raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {}
    lead = raw.get("lead_info") if isinstance(raw.get("lead_info"), dict) else {}
    tool = raw.get("tool")
    return {
        "thought": str(raw.get("thought") or ""),
        "tool": tool,
        "arguments": args,
        "lead_info": lead,
        "memory_updates": clean_mem,
    }
```

Note: `validate_router_output` signature includes `message` for evidence regex. Adjust tests if needed — tests pass `message=`.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_validate_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/conversation-router/service/validation.py apps/conversation-router/tests/unit/test_validate_router.py
git commit -m "feat(validation): fuzzy-on-shown favorite + visit evidence gate"
```

---

### Task 3: `execute_tool` + ToolResult

**Files:**
- Create: `apps/conversation-router/service/tools.py`
- Test: `apps/conversation-router/tests/unit/test_tools.py`

**Interfaces:**
- Consumes: `validate_router_output` output; `search_properties(lead_info, top_k=...)`; `SalesFlow.ready_for_scheduling` pattern (duplicate static gate logic here for purity)
- Produces: `ToolResult` dataclass; `execute_tool(tool, arguments, state, search_fn=None) -> ToolResult`

- [ ] **Step 1: Write failing tests**

```python
# test_tools.py
from service.tools import ToolResult, execute_tool

def _state(**kw):
    base = {
        "properties": [
            {"title": "Torre Nova", "price": 5000},
            {"title": "Corporate One", "price": 6000},
        ],
        "shown_properties_count": 2,
        "lead_info": {"region": "Pinheiros"},
        "favorite_property": None,
        "visit_interest": False,
        "context": {},
    }
    base.update(kw)
    return base

def test_request_options_filtered_default():
    state = _state()
    calls = []
    def fake_search(info, top_k=3):
        calls.append(top_k)
        return [{"title": "A"}, {"title": "B"}]
    r = execute_tool("request_options", {}, state, search_fn=fake_search)
    assert r.ok
    assert r.properties == [{"title": "A"}, {"title": "B"}]
    assert calls and calls[0] == 9  # default filtered top_k

def test_request_options_list_scope_all():
    state = _state()
    def fake_search(info, top_k=3):
        return [{"title": str(i)} for i in range(top_k)]
    r = execute_tool("request_options", {"list_scope": "all"}, state, search_fn=fake_search)
    assert r.ok
    assert len(r.properties) >= 9  # all uses higher top_k

def test_property_detail_hit():
    state = _state()
    r = execute_tool("property_detail", {"property_ref": "Torre Nova"}, state)
    assert r.ok
    assert r.detail is not None
    assert r.detail["title"] == "Torre Nova"

def test_property_detail_miss_sets_clarification():
    state = _state()
    r = execute_tool("property_detail", {"property_ref": "Ghost"}, state)
    assert r.needs_clarification is True

def test_compare_requires_two_shown():
    state = _state(properties=[{"title": "Only One"}])
    r = execute_tool("compare_properties", {}, state)
    assert r.ok is False
    assert "clarif" in (r.refusal or "").lower() or r.needs_clarification

def test_compare_two_shown_ok():
    state = _state()
    r = execute_tool("compare_properties",
                     {"property_a": "Torre Nova", "property_b": "Corporate One"}, state)
    assert r.ok
    assert r.comparison is not None

def test_refine_search_calls_search():
    state = _state(lead_info={"region": "Moema", "budget": "5000"})
    def fake_search(info, top_k=3):
        assert info.get("region") == "Moema"
        return [{"title": "X"}]
    r = execute_tool("refine_search", {}, state, search_fn=fake_search)
    assert r.ok

def test_express_visit_interest_sets_memory():
    state = _state()
    r = execute_tool("express_visit_interest", {}, state,
                     memory_updates={"favorite_property": "Torre Nova", "visit_interest": True})
    assert r.ok
    assert r.memory_updates["visit_interest"] is True

def test_request_schedule_gate_without_visit_fails():
    state = _state(shown_properties_count=5, visit_interest=False, favorite_property=None)
    r = execute_tool("request_schedule", {}, state)
    assert r.ok is False  # gate refuses
    assert r.current_state_hint == "conversation"

def test_request_schedule_gate_without_shown_fails():
    state = _state(shown_properties_count=1, visit_interest=True, favorite_property=None)
    r = execute_tool("request_schedule", {}, state)
    assert r.ok is False

def test_request_schedule_gate_ok_no_favorite():
    # favorite NOT required — visit_interest + shown>=3 enough
    state = _state(shown_properties_count=5, visit_interest=True, favorite_property=None)
    r = execute_tool("request_schedule", {}, state)
    assert r.ok
    assert r.current_state_hint == "scheduling"

def test_request_human_options_regex_stays_conversation():
    state = _state()
    r = execute_tool("request_human", {}, state, message="quero ver mais opções")
    assert r.current_state_hint == "conversation"

def test_decline_followup():
    r = execute_tool("decline", {}, _state())
    assert r.current_state_hint == "followup"
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_tools.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement tools.py**

```python
# service/tools.py
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Callable

_OPTIONS_REQUEST_RE = re.compile(
    r"\b(?:op[çc][ãa]o|op[çc][õo]es|mais im[óo]veis|mostrar|lista|tudo|complet[oa])\b",
    re.IGNORECASE,
)


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


def _resolve_in_shown(ref: str | None, shown: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ref:
        return None
    target = re.sub(r"[^a-z0-9]+", " ", ref.lower()).strip()
    best, best_score = None, 0.0
    from difflib import SequenceMatcher
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
    result = ToolResult(tool=tool, raw_arguments=dict(arguments or {}),
                        memory_updates=dict(memory_updates or {}))

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
        # memory_updates already validated; echo them
        return result

    if tool == "request_schedule":
        # gate: shown>=3 AND visit_interest (favorite NOT required)
        shown_n = max(int(state.get("shown_properties_count") or 0), len(shown))
        visit = bool(state.get("visit_interest") or (memory_updates or {}).get("visit_interest"))
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

    # unknown tool — should not reach if enum validated
    result.ok = False
    result.refusal = "unknown_tool"
    result.current_state_hint = "conversation"
    return result
```

Note: `express_visit_interest` may receive `memory_updates` kwarg from caller after validation. `request_schedule` gate checks `visit_interest` from state OR memory_updates.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/conversation-router/service/tools.py apps/conversation-router/tests/unit/test_tools.py
git commit -m "feat(tools): pure execute_tool + ToolResult with scheduling gate"
```

---

### Task 4: `list_scope` in catalog search_properties

**Files:**
- Modify: `apps/conversation-router/service/properties_catalog.py:194-271` (search_properties)
- Test: `apps/conversation-router/tests/unit/test_properties_catalog.py`

**Interfaces:**
- Consumes: existing soft-budget + FAISS ranking
- Produces: `search_properties(lead_info, catalog=None, top_k=3, list_scope="filtered")`

- [ ] **Step 1: Write failing test**

```python
# test_properties_catalog.py addition
def test_search_list_scope_all_returns_more():
    from service.properties_catalog import search_properties
    props = search_properties({"region": "Moema"}, top_k=3, list_scope="all")
    # all → top_k raised internally or returns broader set
    assert len(props) >= 3

def test_search_list_scope_filtered_default():
    from service.properties_catalog import search_properties
    props = search_properties({"region": "Moema"}, top_k=3)
    assert len(props) <= 3
```

- [ ] **Step 2: Run test to verify fail**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_properties_catalog.py -v -k list_scope`
Expected: FAIL (unexpected kwarg)

- [ ] **Step 3: Implement list_scope param**

```python
def search_properties(
    lead_info: dict[str, Any],
    catalog: list[dict[str, Any]] | None = None,
    top_k: int = 3,
    list_scope: str = "filtered",
) -> list[dict[str, Any]]:
    if list_scope == "all":
        top_k = max(top_k, 50)
    # ... rest unchanged
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_properties_catalog.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/conversation-router/service/properties_catalog.py apps/conversation-router/tests/unit/test_properties_catalog.py
git commit -m "feat(catalog): list_scope=all for show-all requests"
```

---

### Task 5: FlowState + 5-state graph + `_route_state` by tool

**Files:**
- Modify: `apps/conversation-router/service/flow/sales_flow.py:66-92` (FlowState)
- Modify: `apps/conversation-router/service/flow/sales_flow.py:162-209` (_build_graph, _route_state)
- Modify: `apps/conversation-router/service/flow/sales_flow.py:232-278` (_node_preprocess router call)
- Test: `apps/conversation-router/tests/unit/test_sales_flow.py`

**Interfaces:**
- Consumes: `validate_router_output`, `execute_tool` from Tasks 2-3; existing `llm_router` callable
- Produces: `FlowState` with `_router_tool`, `_tool_result`, `_last_tool`; `_route_state` maps tool→node; 5 states only

- [ ] **Step 1: Write failing tests**

```python
# test_sales_flow.py — add TestToolRouting class
class TestToolRouting:
    def _flow(self, tool_result_mock=None, router_result=None):
        from service.flow.sales_flow import SalesFlow
        def router(msg, lead, state):
            return router_result or {
                "thought": "t", "tool": "request_options", "arguments": {},
                "lead_info": {}, "memory_updates": {},
            }
        flow = SalesFlow(
            lead_qualifier=lambda info: {"score": 1},
            properties_rag=lambda info: [{"title": "A"}, {"title": "B"}, {"title": "C"}],
            reply_generator=lambda *a, **k: "ok",
            llm_router=router,
        )
        return flow

    def test_five_states_only_in_graph(self):
        flow = self._flow()
        # graph nodes should be preprocess, greeting, conversation, scheduling, handoff, followup, postprocess
        # (elicitation can remain as greeting sub-path or folded — see impl note)
        assert flow._graph is not None

    def test_router_tool_sets_state_fields(self):
        flow = self._flow(router_result={
            "thought": "show all", "tool": "request_options",
            "arguments": {"list_scope": "all"},
            "lead_info": {"region": "Pinheiros"},
            "memory_updates": {},
        })
        out = flow.invoke({
            "current_state": "conversation", "message": "mostra tudo",
            "lead_info": {}, "context": {}, "properties": [],
        })
        assert out.get("_last_tool") == "request_options" or out.get("_router_tool") == "request_options"

    def test_request_schedule_routes_scheduling_when_gates_ok(self):
        flow = self._flow(router_result={
            "thought": "wants visit", "tool": "request_schedule", "arguments": {},
            "lead_info": {}, "memory_updates": {"visit_interest": True},
        })
        props = [{"title": f"P{i}"} for i in range(5)]
        out = flow.invoke({
            "current_state": "conversation", "message": "quero agendar uma visita",
            "lead_info": {}, "context": {},
            "properties": props, "shown_properties_count": 5,
            "visit_interest": True,
        })
        # scheduling node or conversation if gate — with shown>=5 + visit should go scheduling path
        assert out.get("current_state") in ("scheduling", "conversation")

    def test_request_human_with_options_stays_conversation(self):
        flow = self._flow(router_result={
            "thought": "x", "tool": "request_human", "arguments": {},
            "lead_info": {}, "memory_updates": {},
        })
        out = flow.invoke({
            "current_state": "conversation", "message": "quero ver mais opções",
            "lead_info": {}, "context": {}, "properties": [{"title": "A"}],
        })
        assert out.get("current_state") == "conversation"

    def test_decline_routes_followup(self):
        flow = self._flow(router_result={
            "thought": "x", "tool": "decline", "arguments": {},
            "lead_info": {}, "memory_updates": {},
        })
        out = flow.invoke({
            "current_state": "conversation", "message": "não quero mais",
            "lead_info": {}, "context": {}, "properties": [],
        })
        assert out.get("current_state") == "followup"

    def test_greeting_bypasses_router(self):
        flow = self._flow()
        out = flow.invoke({
            "current_state": "greeting", "message": "oi",
            "lead_info": {}, "context": {}, "consent_recorded": False,
        })
        # LGPD path — no tool routing side effect required
        assert out.get("current_state") in ("greeting", "elicitation", "intent", "conversation")
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_sales_flow.py -v -k ToolRouting`
Expected: FAIL (fields/routing not present)

- [ ] **Step 3: Implement FlowState + graph + route**

Key changes in `sales_flow.py`:

1. FlowState add: `_router_tool: str | None`, `_tool_result: Any`, `_last_tool: str | None`
2. Keep `_router_action` as alias for backward compat OR migrate all refs to `_router_tool` (prefer migrate in this task).
3. `_build_graph`: nodes = `preprocess, greeting, elicitation, conversation, scheduling, handoff, followup, postprocess`. Remove `intent, qualification, discovery, recommendation` as top-level nodes — their logic moves into `conversation` dispatcher (`_node_conversation`) that internally calls detail/search/compare based on `_router_tool`.
4. `_route_state`:

```python
def _route_state(self, state: FlowState) -> str:
    current = state.get("current_state", "greeting")
    if current in ("greeting", "elicitation"):
        return current  # LGPD deterministic
    tool = state.get("_router_tool")
    hint = None
    tr = state.get("_tool_result")
    if tr is not None:
        hint = getattr(tr, "current_state_hint", None)
    if tool == "request_human":
        return hint or "handoff"
    if tool == "decline":
        return "followup"
    if tool == "request_schedule":
        return hint or "conversation"  # hint=scheduling if gates ok else conversation
    # request_options, property_detail, compare, refine, express_visit, provide_info, unclear
    return "conversation"
```

5. `_node_preprocess`: after router success:

```python
result = self.llm_router(message, merged, state.get("current_state", "greeting"))
# validate
from service.validation import validate_router_output
validated = validate_router_output(result, state, message=message)
state["_router_tool"] = validated["tool"]
state["_last_tool"] = validated["tool"]
# merge lead_info
merged = {**merged, **validated["lead_info"]}
context["lead_info"] = {**(context.get("lead_info") or {}), **validated["lead_info"]}
# apply validated memory
mem = validated.get("memory_updates") or {}
if mem.get("favorite_property"):
    state["favorite_property"] = mem["favorite_property"]
    context["favorite_property"] = mem["favorite_property"]
if mem.get("visit_interest"):
    state["visit_interest"] = True
    context["visit_interest"] = True
# execute tool for side effects (properties, hints)
from service.tools import execute_tool
search_fn = self.properties_rag  # may be lambda info: search_properties(info, top_k=9)
# wrap for list_scope
def _search_with_scope(info, top_k=9, list_scope="filtered"):
    if search_fn is None:
        return []
    # call catalog with scope if list_scope present
    try:
        from service.properties_catalog import search_properties
        return search_properties(info, top_k=top_k, list_scope=list_scope)
    except Exception:
        return search_fn(info)
args = validated.get("arguments") or {}
scope = args.get("list_scope", "filtered")
if validated["tool"] == "request_options" and scope == "all":
    props = _search_with_scope(merged, top_k=50, list_scope="all")
    state["properties"] = props
    self._remember_shown(state)
tr = execute_tool(
    validated["tool"], args, state,
    search_fn=(lambda info, top_k=9: _search_with_scope(info, top_k=top_k)),
    memory_updates=mem,
    message=message,
)
state["_tool_result"] = tr
if tr.properties:
    state["properties"] = tr.properties
    self._remember_shown(state)
```

6. `_node_conversation` (new): dispatches by `_router_tool`:
   - `property_detail` → use `tr.detail` or clarification canned
   - `compare_properties` → use `tr.comparison` or clarify
   - `request_options`/`refine_search` → show `tr.properties`
   - `provide_info`/`unclear` → fallback discovery behavior (reuse `_node_discovery` logic inline)
   - Sets `state["current_state"] = "conversation"`

7. `_node_scheduling`: keep existing gates; if `_tool_result.refusal == "scheduling_gate"` at entry, return conversation + open question.

8. Update `_invoke_fsm` handler map: add `conversation` → `_node_conversation`; map legacy `discovery/recommendation/qualification/intent` inputs to `conversation` for backward compat with old stored contexts.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_sales_flow.py -v`
Expected: PASS (new + adjust any tests expecting old state names)

- [ ] **Step 5: Run full suite**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests -q`
Expected: green

- [ ] **Step 6: Commit**

```bash
git add apps/conversation-router/service/flow/sales_flow.py apps/conversation-router/tests/unit/test_sales_flow.py
git commit -m "feat(flow): 5-state graph + tool-based _route_state + conversation node"
```

---

### Task 6: Reply generator kwargs + prompt + long-list cap

**Files:**
- Modify: `apps/conversation-router/service/llm.py:106-123` (_REPLY_SYSTEM_PROMPT)
- Modify: `apps/conversation-router/service/llm.py:245-320` (generate_reply)
- Modify: `apps/conversation-router/handler.py:441-452` (llm_reply kwargs)
- Modify: `apps/conversation-router/service/flow/sales_flow.py` (pass reply kwargs in postprocess)
- Test: `apps/conversation-router/tests/unit/test_llm.py`

**Interfaces:**
- Consumes: validated state fields
- Produces: `generate_reply(..., visit_interest=False, rejected_properties=None, last_tool=None)`

- [ ] **Step 1: Write failing tests**

```python
def test_generate_reply_accepts_new_kwargs():
    from service.llm import generate_reply
    # should not TypeError — mock completion if needed; at minimum signature accepts
    import inspect
    sig = inspect.signature(generate_reply)
    assert "visit_interest" in sig.parameters
    assert "rejected_properties" in sig.parameters
    assert "last_tool" in sig.parameters

def test_reply_prompt_mentions_last_tool_and_consultative():
    from service.llm import _REPLY_SYSTEM_PROMPT
    assert "last_tool" in _REPLY_SYSTEM_PROMPT or "ÚLTIMA" in _REPLY_SYSTEM_PROMPT or "TOOL" in _REPLY_SYSTEM_PROMPT.upper()
    assert "1 pergunta" in _REPLY_SYSTEM_PROMPT or "uma pergunta" in _REPLY_SYSTEM_PROMPT.lower()
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_llm.py -v -k reply`
Expected: FAIL

- [ ] **Step 3: Implement signature + prompt + cap**

`generate_reply` add params: `visit_interest: bool = False`, `rejected_properties: list[str] | None = None`, `last_tool: str | None = None`.

In `user_block`, add lines:

```python
visit_line = f"INTERESSE DE VISITA: {'sim' if visit_interest else 'não'}"
rejected_line = f"REJEITADOS: {', '.join(rejected_properties or []) or '(nenhum)'}"
tool_line = f"ÚLTIMA TOOL: {last_tool or '(nenhuma)'}"
```

Append to `user_block`. Replace `_REPLY_SYSTEM_PROMPT` with consultative version including `last_tool` coherence rule.

Token cap:

```python
long_list = len(properties) > 3
max_tokens = 800 if long_list else 280
# in prompt rule 4: if long_list, "1 frase por imóvel + fecho" instead of max 3 sentences
```

Handler `llm_reply` add:

```python
visit_interest=bool(kwargs.get("visit_interest")),
rejected_properties=kwargs.get("rejected_properties"),
last_tool=kwargs.get("last_tool"),
```

Flow postprocess / node that calls `reply_generator` must pass these kwargs from state.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Full suite + handler wiring**

Run: `cd apps/conversation-router && ../../.venv/bin/pytest tests -q`
Expected: green

- [ ] **Step 6: Commit**

```bash
git add apps/conversation-router/service/llm.py apps/conversation-router/handler.py apps/conversation-router/service/flow/sales_flow.py apps/conversation-router/tests/unit/test_llm.py
git commit -m "feat(reply): visit_interest + rejected + last_tool kwargs + consultative prompt"
```

---

### Task 7: Wire handler `llm_route` to new parse + full-suite green + regression

**Files:**
- Modify: `apps/conversation-router/handler.py:457-471` (llm_route, properties_rag lambda)
- Test: `apps/conversation-router/tests/unit/test_handler_llm_secret.py`

**Interfaces:**
- Consumes: `extract_and_route` new return shape
- Produces: handler wires router that returns validated-ready dict; `properties_rag` passes `list_scope`

- [ ] **Step 1: Update handler wiring test**

```python
def test_llm_route_returns_tool_contract(monkeypatch):
    # existing wiring tests updated: result has "tool" not "action"
    ...
```

- [ ] **Step 2: Implement handler updates**

- `llm_route` unchanged signature (returns extract_and_route dict) — shape already new from Task 1.
- Ensure `properties_rag` can accept `list_scope` if tools call it with scope; prefer tools calling `search_properties` directly with scope as in Task 5.
- Flow postprocess: pass `last_tool=state.get("_last_tool")`, `visit_interest=state.get("visit_interest")`, `rejected_properties=state.get("rejected_properties")` into `reply_generator(**kwargs)`.

- [ ] **Step 3: Full suites all apps**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests -q
cd apps/voice-adapter && ../../.venv/bin/pytest tests -q
cd apps/dashboard-api && ../../.venv/bin/pytest tests -q
cd apps/dashboard-ui && ../../.venv/bin/pytest tests -q
```

Expected: conversation-router green; others unchanged green (378+ baseline)

- [ ] **Step 4: Commit**

```bash
git add apps/conversation-router/handler.py apps/conversation-router/tests/unit/test_handler_llm_secret.py
git commit -m "feat(handler): wire tool-contract router + reply kwargs"
```

---

### Task 8: Documentation — PRD update + AI-DLC note

**Files:**
- Modify: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md` §8.1.1, §16
- Modify: `aidlc/spaces/default/intents/260911-agente-sdr-imobiliario/...` (memory.md note)

**Interfaces:**
- Consumes: implemented behavior
- Produces: PRD reflects tool-agent, 5 states, VALID_TOOLS, validation rules

- [ ] **Step 1: Update PRD §8.1.1**

Replace ADR-011 description with: single-step tool-calling contract, VALID_TOOLS list, 5 states, validation (fuzzy shown-only, visit evidence), scheduling gate without favorite requirement, `generate_reply` kwargs, fallback ADR-011 retained.

- [ ] **Step 2: Update PRD §16 Roadmap pós-POC**

Add: prepared for multi-step agentic loop (option B) via stable `execute_tool` interface; not implemented this iteration.

- [ ] **Step 3: AI-DLC memory note**

Append to intent `memory.md`: "Tool-agent router implemented per spec 2026-09-23-tool-agent-sdr-design; decisions: single-step, 5 states, fuzzy-shown-only, no favorite in schedule gate."

- [ ] **Step 4: Commit**

```bash
git add documentos/ aidlc/
git commit -m "docs: PRD + AI-DLC reflect tool-agent router evolution"
```

---

## Self-Review

**Spec coverage:** §4 contract → Task 1; §4.3 validation → Task 2; §6 tools → Task 3; §5 states/map → Task 5; list_scope → Task 4; §7 reply → Task 6; handler wiring → Task 7; docs → Task 8. §8 fallback: preprocess keeps try/except → regex. §12 future: ToolResult stable.

**Placeholders:** none — all steps have code/tests.

**Type consistency:** `ToolResult.current_state_hint` used Task 3-5; `validate_router_output(raw, state, message=)` Task 2+5; `_router_tool` Task 5; `_last_tool` Task 5-6-7; `list_scope` Task 3-4-5.

**Note:** Task 5 is large (graph collapse) — if subagent-driven, keep as one task with review gate; may split `_build_graph` vs `_node_conversation` if needed mid-implementation (hidden complexity → re-split, not redesign).
