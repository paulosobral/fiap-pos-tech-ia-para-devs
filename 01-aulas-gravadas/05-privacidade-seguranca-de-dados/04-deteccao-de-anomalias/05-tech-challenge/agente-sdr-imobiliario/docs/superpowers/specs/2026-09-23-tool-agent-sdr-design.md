# Design: Tool-Agent Router + FSM Leve (Evolution)

**Date:** 2026-09-23
**Status:** Approved (user) — Section 3 review: scheduling gate does not require `favorite_property`
**Path:** Architectural (brainstorming)
**Supersedes in part:** `2026-09-23-consultative-sdr-design.md` (consultative behaviors remain; routing contract and state model change here)

## 1. Goal

Migrate the conversation router from a fixed action classifier (LLM as menu router) to a single-step tool-calling agent:

```txt
70% code / 30% LLM  →  target ~30% code / 70% LLM
```

Code keeps only critical guardrails (LGPD, scheduling gates, handoff, ICS, persistence, access). LLM decides commercial strategy per turn. Architecture is prepared for a future multi-step agentic loop (option B) without implementing it now.

## 2. Decisions (user-approved)

| Topic | Decision |
|---|---|
| Agent style | **(A) Single-step** — one LLM call per turn: `{thought, tool, arguments, lead_info, memory_updates}`; code executes tool + reply. No multi-tool loop. |
| States | **(B) Partial collapse** — five explicit states: `greeting`, `conversation`, `scheduling`, `handoff`, `followup`. |
| Memory extraction | **(B) Router + code validation** — fuzzy match only against `shown_properties`; never full catalog. Regex = safety net only. |
| Overall shape | **Approach 1 — Router-Aware FSM** — keep LangGraph; add tool contract + validation + `execute_tool`. |

## 3. Architecture

```txt
User message
  → LLM Router (single call)
      {thought, tool, arguments, lead_info, memory_updates}
  → _validate_router_output()   # enum, fuzzy favorite, visit evidence, lead_info merge
  → execute_tool(tool, arguments, state)  # pure, guardrails
  → LangGraph node (5-state FSM)
  → Reply Generator (generate_reply + full context kwargs)
  → final message
```

**Invariants**

1. LLM never writes `state` directly.
2. `thought` is log-only (debug/observability); never controls flow.
3. `tool` + `arguments` + validated `memory_updates` control behavior.
4. On invalid tool / router exception → ADR-011 fallback (regex + legacy FSM path).
5. Deterministic domains never route via LLM: LGPD greeting/elicitation, decline→followup confirmation, scheduling gates, ICS, CRM, handoff payload, rate limits, PII masking.

## 4. Router contract

### 4.1 Output JSON

```json
{
  "thought": "string, log-only",
  "tool": "<VALID_TOOLS>",
  "arguments": {},
  "lead_info": {},
  "memory_updates": {
    "favorite_property": "string|null",
    "visit_interest": "bool"
  }
}
```

### 4.2 VALID_TOOLS

```txt
request_options
property_detail
compare_properties
refine_search
express_visit_interest
request_schedule
request_human
decline
provide_info
unclear
```

Notes:

- Tool name is `express_visit_interest` (not `visit_interest`) to avoid collision with memory field `visit_interest`.
- `request_schedule` means “user wants to schedule”, not “schedule now”; code still runs `ready_for_scheduling`.
- Scheduling gate does **not** require `favorite_property != None`. Legitimate cases: “Quero marcar uma visita”, “Gostei das opções”, “Podemos conversar amanhã?” — `visit_interest=True` without a named favorite. `favorite_property` only enriches handoff/ICS when present.
- New tool vs roadmap: `property_detail` replaces overloaded `provide_info` for property questions (price, parking, floor, spaces, generator, etc.).
- `request_options.arguments.list_scope`: `"filtered" | "all"` (optional; default `filtered`).

### 4.3 Validation (`_validate_router_output`)

| Rule | Behavior |
|---|---|
| Tool not in enum | Raise → ADR-011 fallback (regex/FSM) |
| `favorite_property` | Fuzzy match **only** against `shown_properties` (context list). No match → `None` + optional `needs_clarification=true`. Never full catalog. |
| `visit_interest` | Persist only with explicit visit evidence in message (e.g. “quero visitar”, “vamos marcar”, “tem agenda”, “posso conhecer”) **or** router + scheduling gate agreement. Favorite ≠ visit interest. |
| `lead_info` | Merge as today (`context` + LLM deltas + regex baseline) |
| `memory_updates` | Applied only after validation; unknown keys dropped |

## 5. States and tool → node mapping

### 5.1 States

| State | Responsibility | Who decides |
|---|---|---|
| `greeting` | LGPD consent, opt-in/out | 100% deterministic (never LLM) |
| `conversation` | search, detail, compare, refine, memory, exploration | LLM chooses tool; code executes |
| `scheduling` | schedule validation, ICS, calendar | LLM may express intent; code authorizes |
| `handoff` | broker routing, executive summary | `request_human` + operational rules |
| `followup` | decline, opt-out, close | deterministic |

**Removed as top-level states:** `intent`, `qualification`, `discovery`, `recommendation`, `elicitation`, `visit_interest`. They become internal context/phases inside `conversation`.

### 5.2 Mapping

| tool | node / behavior | Pre-gate |
|---|---|---|
| `request_options` | conversation: recommendation / `_show_more_options` | none |
| `property_detail` | conversation: discovery/detail | valid ref in `shown` or `favorite`; else `needs_clarification` |
| `compare_properties` | conversation: compare | `len(shown) >= 2`; else clarify |
| `refine_search` | conversation: new RAG query | none |
| `express_visit_interest` | conversation: set memory only | evidence rule |
| `request_schedule` | **scheduling** | `ready_for_scheduling` + `shown >= 3` + `visit_interest == True`. **`favorite_property` NOT required** — only enriches handoff/ICS when present. Fail → stay conversation + explain |
| `request_human` | **handoff** | if message also asks for options → conversation |
| `decline` | **followup** | none |
| `provide_info` / `unclear` | conversation fallback | discovery regex if detail-like |

## 6. Tool execution layer

New module: `apps/conversation-router/service/tools.py`

```python
def execute_tool(tool: str, arguments: dict, state: FlowState) -> ToolResult: ...
```

- Pure functions; no LLM calls inside tools (search uses existing RAG service).
- `ToolResult` carries: updated candidate properties, clarification flag, gate refusal reason, or terminal action hint for the graph node.
- Guardrails stay in code: soft budget, `list_scope` top_k, dedup, show-count persistence (from prior fix), scheduling gate.

## 7. Reply generator

### 7.1 Signature additions

```python
generate_reply(
  ...,
  favorite_property=...,
  conversation_stage=...,
  shown_properties_count=...,
  visit_interest=...,          # new
  rejected_properties=...,     # new
  last_tool=...,               # new
)
```

Handler must pass all of the above from validated state.

### 7.2 Prompt

Replace `_REPLY_SYSTEM_PROMPT` with consultative corporate broker prompt (roadmap):

- One question per reply; never form-like
- If properties available: discuss them; explore preferences before visit
- Offer visit only on explicit interest / clear intent to advance
- Never invent attributes; only `IMÓVEIS RECOMENDADOS` list
- Coherence by `last_tool` (e.g. `property_detail` → talk about that property; `compare_properties` → comparison framing)
- Professional, consultive, objective tone

### 7.3 Long lists

If `len(properties) > 3`: relax “max 3 sentences” → one sentence per property + closing; `max_tokens` ≈ 800 for that turn only.

## 8. Fallback (ADR-011 retained)

- Router exception or invalid tool → regex extraction + deterministic FSM path (existing safety net).
- `request_human` + options-request regex still redirects to conversation.
- Greeting/elicitation never bypassed by LLM.

## 9. Files touched

| File | Change |
|---|---|
| `service/llm.py` | New router prompt + `VALID_TOOLS` parse; `generate_reply` kwargs + prompt + token cap |
| `service/flow/sales_flow.py` | 5-state graph; `_route_state` by tool; validation hook; keep LGPD/gates |
| `service/tools.py` | **New** — `execute_tool` + tool result types |
| `service/properties_catalog.py` | Support `list_scope` (reuse soft-budget path) |
| `handler.py` | Wire new reply kwargs; pass `last_tool` |
| tests | New unit tests for tools/validation; update flow/handler tests |

## 10. Testing strategy (TDD)

1. `test_tools.py` — each tool + refusals (schedule gate, compare `<2`, detail miss, list_scope).
2. `test_validate_router.py` — enum, fuzzy-on-shown-only, visit evidence, lead_info merge, thought ignored.
3. `test_sales_flow.py` — 5 states, tool→node map, conversation absorbs discovery/recommendation paths, fallback still works.
4. `handler` wiring — reply kwargs.
5. Full suites green per app (base ≥378 across apps).

## 11. Non-goals (this iteration)

- Multi-step tool loop (option B) — contract only, no loop.
- Replacing LangGraph.
- Full-catalog fuzzy for favorites.
- Changing LGPD, CRM, ICS, anomaly, voice, dashboard behavior.
- Committing/pushing without user request.

## 12. Future (B) readiness

`VALID_TOOLS` + `execute_tool(tool, arguments, state) -> ToolResult` is the stable interface. A later loop can call `execute_tool` repeatedly and feed results back into the same LLM without changing guardrails or tools.
