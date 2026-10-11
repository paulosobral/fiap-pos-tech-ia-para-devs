# Consultative SDR Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the SDR bot into a consultative real-estate advisor that never pushes scheduling before showing options, with commercial memory, richer router actions, and a discovery state.

**Architecture:** Four sliced stages on the existing LangGraph `SalesFlow`: (1) hard gate `ready_for_scheduling` in `_node_scheduling`; (2) expand `VALID_ACTIONS` with deterministic mapping in `_route_state`/nodes; (3) commercial memory fields on `FlowState` mirrored into `context`; (4) new `discovery` node + richer `generate_reply` kwargs and consultative prompt. Each stage ships with tests green before the next starts.

**Tech Stack:** Python 3.14, LangGraph StateGraph, LiteLLM/OpenRouter, pytest, TypedDict FlowState.

**Spec:** `docs/superpowers/specs/2026-09-23-consultative-sdr-design.md`

## Global Constraints

- Never let LLM choose graph nodes (ADR-011): only classify into `VALID_ACTIONS`; code maps action → node.
- `request_schedule` only with explicit scheduling verbs (agendar/marcar/reserve/agende/quero marcar).
- No implicit "quer agendar?" loop: without `favorite_property` OR `visit_interest` OR `lead_info.deadline`, never call `scheduler`.
- Reply safety (unaltered): only cite listed properties; never invent price/area/neighborhood; max 3 sentences; final question must not force scheduling.
- LGPD greeting/elicitation stay deterministic (never via LLM router).
- Run test suites separately (import-path collision when run together — pre-existing):
  - `cd apps/conversation-router && ../../.venv/bin/pytest tests -q`
  - `cd apps/voice-adapter && ../../.venv/bin/pytest tests -q`
  - `cd apps/dashboard-api && ../../.venv/bin/pytest tests -q` and same for `dashboard-ui`
- Repo root for relative paths: `agente-sdr-imobiliario/`.
- Baseline green before Task 1: 192 conversation-router + 82 voice-adapter + 74 dashboard = 348 pass, 2 skipped.
- Commits: Conventional Commits, one per task (or sub-step if asked); do not amend shared history unless requested.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `apps/conversation-router/service/flow/sales_flow.py` | FlowState, graph nodes, gates, routing, postprocess |
| `apps/conversation-router/service/llm.py` | VALID_ACTIONS, prompts, generate_reply, extract_and_route |
| `apps/conversation-router/service/entities.py` | VALID_STATES persistence whitelist |
| `apps/conversation-router/handler.py` | wiring llm_reply / llm_router |
| `apps/conversation-router/tests/unit/test_sales_flow.py` | flow + router unit tests |
| `apps/conversation-router/tests/unit/test_prd_gaps.py` | ICS / langgraph tests |
| `apps/conversation-router/tests/unit/test_restriction.py` | restriction integration with flow |
| `apps/conversation-router/tests/unit/test_llm.py` | LLM parse/reply tests |
| `apps/conversation-router/tests/unit/test_entities.py` | VALID_STATES |
| `apps/conversation-router/tests/unit/test_handler_llm_secret.py` | handler wiring smoke |

---

## Task 1: Gate `ready_for_scheduling`

**Files:**
- Modify: `apps/conversation-router/service/flow/sales_flow.py` (`FlowState`, `SalesFlow.ready_for_scheduling`, `_node_scheduling`)
- Test: `apps/conversation-router/tests/unit/test_sales_flow.py`
- Test: `apps/conversation-router/tests/unit/test_prd_gaps.py`
- Test: `apps/conversation-router/tests/unit/test_restriction.py`

**Interfaces:**
- Consumes: existing `_node_scheduling` (order: `_wants_options` → shown_count gate → restriction → scheduler); `FlowState.lead_info`, `shown_properties_count`.
- Produces: `SalesFlow.ready_for_scheduling(state: FlowState) -> bool` (OR of `favorite_property`, `visit_interest`, `lead_info.deadline`); FlowState fields `favorite_property: str`, `visit_interest: bool` (declared now so LangGraph does not strip them; populated fully in Task 3).

- [ ] **Step 1: Write failing tests for gate**

Add to `test_sales_flow.py` (new class after `TestSchedulingRestriction` or inside it):

```python
class TestReadyForSchedulingGate:
    def scheduler(self):
        return MagicMock(return_value={"confirmed": True, "when": "amanhã 10h"})

    def test_gate_blocks_without_signal(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
        })
        scheduler.assert_not_called()
        assert state["current_state"] == "recommendation"
        assert "agendar" not in state["response"].lower() or "opções" in state["response"].lower()

    def test_gate_allows_visit_interest(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
            "visit_interest": True,
        })
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"

    def test_gate_allows_favorite_property(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
            "favorite_property": "Torre Nova",
        })
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"

    def test_gate_allows_deadline_in_lead_info(self):
        scheduler = self.scheduler()
        flow = make_flow(scheduler=scheduler)
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
            "lead_info": {"deadline": "6 meses"},
        })
        scheduler.assert_called_once()
        assert state["current_state"] == "handoff"
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_sales_flow.py::TestReadyForSchedulingGate -v
```

Expected: FAIL — `ready_for_scheduling` missing or AttributeError; or gate not applied so `test_gate_blocks_without_signal` fails (scheduler called).

- [ ] **Step 3: Declare FlowState fields + implement gate**

In `sales_flow.py` `FlowState`, after `shown_properties_count: int`:

```python
    favorite_property: str
    visit_interest: bool
```

In `SalesFlow` (near `_wants_options` or before `_node_scheduling`):

```python
    @staticmethod
    def ready_for_scheduling(state: FlowState) -> bool:
        return bool(
            state.get("favorite_property")
            or state.get("visit_interest")
            or (state.get("lead_info") or {}).get("deadline")
        )
```

In `_node_scheduling`, insert **after** the `_wants_options` block and **before** the `shown_count` block:

```python
        if not self.ready_for_scheduling(state):
            state["current_state"] = "recommendation"
            state["response"] = (
                "Antes de agendar, me diga qual imóvel mais te interessou "
                "ou se quer refinar a busca. Assim consigo preparar a visita ideal."
            )
            return state
```

- [ ] **Step 4: Run gate tests to verify they pass**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_sales_flow.py::TestReadyForSchedulingGate -v
```

Expected: 4 passed.

- [ ] **Step 5: Fix existing scheduling tests (add signal)**

For each of these invocations, add `"visit_interest": True` (or a `deadline` in `lead_info`) to the state dict so they still reach restriction/scheduler:

- `test_sales_flow.py`: `test_restricted_scheduling_defers_action`, `test_unrestricted_scheduling_calls_scheduler`, `test_scheduling_without_checker_runs_normally`, `test_restriction_check_failure_is_fail_open`, `test_restricted_lead_without_lead_id_is_not_restricted` (this last already has `shown_properties_count: 3` — still needs a gate signal).
- `test_prd_gaps.py::test_sales_flow_ics_generated_on_scheduling` — add `"visit_interest": True`.
- `test_restriction.py::test_flow_uses_checker_result` — add `"visit_interest": True`.

Example edit:

```python
state = flow.invoke({
    "current_state": "scheduling",
    "message": "amanhã 10h",
    "lead_id": "L1",
    "visit_interest": True,
})
```

Also check `test_sales_flow.py` `TestAgenticRouter.test_request_options_action_in_scheduling_shows_more`: second invoke has `current_state: "scheduling"` but hits `_wants_options` first (stays scheduling) — if it fails after the gate, ensure first branch still wins (it should: options before gate). If gate runs first on some path, add `"visit_interest": True` only if options path is not taken.

- [ ] **Step 6: Run full conversation-router suite**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests -q
```

Expected: all pass (192+ new).

- [ ] **Step 7: Run voice-adapter suite (regression)**

```bash
cd apps/voice-adapter && ../../.venv/bin/pytest tests -q
```

Expected: 82 pass.

- [ ] **Step 8: Commit**

```bash
git add apps/conversation-router/service/flow/sales_flow.py \
  apps/conversation-router/tests/unit/test_sales_flow.py \
  apps/conversation-router/tests/unit/test_prd_gaps.py \
  apps/conversation-router/tests/unit/test_restriction.py
git commit -m "feat(sales-flow): gate scheduling behind ready_for_scheduling signals"
```

---

## Task 2: Router actions `refine_search`, `compare_properties`, `visit_interest`

**Files:**
- Modify: `apps/conversation-router/service/llm.py` (`VALID_ACTIONS`, `_ROUTER_SYSTEM_PROMPT`)
- Modify: `apps/conversation-router/service/flow/sales_flow.py` (`_route_state`, `_node_preprocess` / recommendation helpers)
- Test: `apps/conversation-router/tests/unit/test_llm.py`
- Test: `apps/conversation-router/tests/unit/test_sales_flow.py`

**Interfaces:**
- Consumes: Task 1 `ready_for_scheduling`; existing `extract_and_route` / `_parse_extract_and_route`.
- Produces: extended `VALID_ACTIONS`; router may return `{"lead_info", "action"}` with new actions; `_route_state` maps `visit_interest` → scheduling only if gate + shown_count; `refine_search`/`compare_properties` stay on current node then recommendation re-runs RAG.

- [ ] **Step 1: Write failing parser tests**

In `test_llm.py` `TestExtractAndRoute`:

```python
    def test_accepts_refine_search_action(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion('{"lead_info": {"budget": "R$ 80 mil"}, "action": "refine_search"}'),
        )
        result = lm.extract_and_route("tem algo mais barato?", {}, "recommendation", api_key="k")
        assert result["action"] == "refine_search"

    def test_accepts_compare_properties_action(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion('{"lead_info": {}, "action": "compare_properties"}'),
        )
        result = lm.extract_and_route("qual a diferença entre 1 e 2?", {}, "recommendation", api_key="k")
        assert result["action"] == "compare_properties"

    def test_accepts_visit_interest_action(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion('{"lead_info": {}, "action": "visit_interest"}'),
        )
        result = lm.extract_and_route("quero visitar a Torre Nova", {}, "recommendation", api_key="k")
        assert result["action"] == "visit_interest"
```

Also assert prompt includes the new actions (can check `_ROUTER_SYSTEM_PROMPT` content):

```python
def test_router_prompt_documents_new_actions():
    assert "refine_search" in lm._ROUTER_SYSTEM_PROMPT
    assert "compare_properties" in lm._ROUTER_SYSTEM_PROMPT
    assert "visit_interest" in lm._ROUTER_SYSTEM_PROMPT
```

- [ ] **Step 2: Run tests to verify fail**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_llm.py::TestExtractAndRoute -k "refine or compare or visit" -v
cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_llm.py::test_router_prompt_documents_new_actions -v
```

Expected: FAIL — action outside enum / prompt missing keys.

- [ ] **Step 3: Extend VALID_ACTIONS + prompt**

In `llm.py` replace `VALID_ACTIONS` with:

```python
VALID_ACTIONS = (
    "provide_info",
    "request_options",
    "refine_search",
    "compare_properties",
    "visit_interest",
    "request_schedule",
    "request_human",
    "decline",
    "unclear",
)
```

Extend `_ROUTER_SYSTEM_PROMPT` action list (after `request_options` bullet, before `request_schedule`):

```python
    "   - refine_search: quer AJUSTAR critérios da busca ('mais barato', 'menor', 'outra região', "
    "'sem estacionamento?', filtros novos)\n"
    "   - compare_properties: quer COMPARAR opções já mostradas ('diferença entre 1 e 2', "
    "'compara as duas', 'qual é melhor')\n"
    "   - visit_interest: demonstra INTERESSE em visitar ou em um imóvel específico "
    "('quero visitar', 'gostei da 2', 'essa me interessa') SEM verbo de agendamento explícito\n"
```

Update REGRA DE OURO line to mention: pedir mais detalhes = `request_options`; ajustar filtros = `refine_search`; visitar/gostei = `visit_interest`; agendar = verbo explícito.

- [ ] **Step 4: Run llm tests to verify pass**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_llm.py -q
```

Expected: all pass.

- [ ] **Step 5: Write failing flow routing tests**

In `test_sales_flow.py` `TestAgenticRouter`:

```python
    def test_refine_search_reruns_rag_with_new_lead_info(self):
        captured = {}
        def rag(info):
            captured.update(info)
            return [{"title": "Opção Barata", "region": "Pinheiros", "area_util": 80}]
        router = lambda message, lead_info, current_state: {
            "lead_info": {"budget": "R$ 80 mil"},
            "action": "refine_search",
        }
        flow = make_flow(properties_rag=rag, llm_router=router)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "tem algo mais barato?",
            "lead_info": {"budget": "R$ 200 mil"},
            "shown_properties_count": 3,
        })
        assert captured.get("budget") == "R$ 80 mil"
        assert "Opção Barata" in state["response"]

    def test_compare_properties_shows_listed_options(self):
        catalog = [
            {"title": "A", "region": "Pinheiros", "area_util": 100},
            {"title": "B", "region": "Pinheiros", "area_util": 150},
        ]
        router = lambda message, lead_info, current_state: {
            "lead_info": {},
            "action": "compare_properties",
        }
        flow = make_flow(properties_rag=lambda info: catalog, llm_router=router)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "qual a diferença entre A e B?",
            "lead_info": {},
            "properties": catalog,
            "shown_properties_count": 2,
        })
        assert state["current_state"] == "recommendation"
        assert "A" in state["response"] and "B" in state["response"]

    def test_visit_interest_goes_to_scheduling_when_ready(self):
        scheduler = MagicMock(return_value={"confirmed": True, "when": "amanhã 10h"})
        router = lambda message, lead_info, current_state: {
            "lead_info": {},
            "action": "visit_interest",
        }
        flow = make_flow(scheduler=scheduler, llm_router=router, properties_rag=lambda i: [{"title": "X"}])
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "quero visitar a Torre Nova",
            "lead_info": {},
            "shown_properties_count": 3,
            "properties": [{"title": "X"}],
        })
        assert state.get("visit_interest") is True
        scheduler.assert_called_once()

    def test_visit_interest_stays_if_not_enough_shown(self):
        scheduler = MagicMock(return_value={"confirmed": True})
        router = lambda message, lead_info, current_state: {
            "lead_info": {},
            "action": "visit_interest",
        }
        flow = make_flow(scheduler=scheduler, llm_router=router)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "quero visitar",
            "lead_info": {},
            "shown_properties_count": 1,
            "properties": [],
        })
        scheduler.assert_not_called()
        assert state.get("visit_interest") is True
```

- [ ] **Step 6: Run flow tests to verify fail**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_sales_flow.py::TestAgenticRouter -k "refine or compare or visit" -v
```

Expected: FAIL — actions not mapped (stay on default node / no visit_interest flag).

- [ ] **Step 7: Map actions in sales_flow**

**`_node_preprocess`:** after setting `_router_action`, if action is `visit_interest`, set `state["visit_interest"] = True` (and mirror to `context` in Task 3 — can set context here too if preferred; Task 3 will formalize).

```python
                state["_router_action"] = result.get("action")
                if state["_router_action"] == "visit_interest":
                    state["visit_interest"] = True
```

**`_route_state`:** after existing `request_human`/`decline` handling, before `return current`:

```python
        if action == "visit_interest":
            if self.ready_for_scheduling(state):
                shown = state.get("shown_properties_count", 0) + len(state.get("properties", []))
                if shown >= 3 or state.get("lead_id"):
                    return "scheduling"
            return current  # keep current node; preprocess already flagged interest
        if action in ("refine_search", "compare_properties"):
            return "recommendation"
```

Note: `ready_for_scheduling` after setting `visit_interest` in preprocess will be True (visit_interest alone ORs). Spec: `visit_interest` → scheduling if gate + shown_count. Gate is satisfied by `visit_interest` itself; still require shown >= 3 (or lead_id for legacy).

**`_node_recommendation`:** at top, if `_router_action == "compare_properties"` and `state["properties"]`, build comparison response listing titles/area/price (do not re-fetch if properties present; else fetch):

```python
        if state.get("_router_action") == "compare_properties" and state.get("properties"):
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
```

If `refine_search` or default: existing RAG path runs (lead_info already merged in preprocess).

- [ ] **Step 8: Run flow + full conversation-router suite**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_sales_flow.py -q
cd apps/conversation-router && ../../.venv/bin/pytest tests -q
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add apps/conversation-router/service/llm.py \
  apps/conversation-router/service/flow/sales_flow.py \
  apps/conversation-router/tests/unit/test_llm.py \
  apps/conversation-router/tests/unit/test_sales_flow.py
git commit -m "feat(router): add refine_search, compare_properties, visit_interest actions"
```

---

## Task 3: Commercial memory (favorite / rejected / interests)

**Files:**
- Modify: `apps/conversation-router/service/flow/sales_flow.py` (`FlowState`, `_node_preprocess`, `_node_recommendation`)
- Modify: `apps/conversation-router/service/llm.py` (optional: extract `favorite_property` in router — keep minimal: regex first)
- Test: `apps/conversation-router/tests/unit/test_sales_flow.py`

**Interfaces:**
- Consumes: Task 1 gate reads `favorite_property`; Task 2 sets `visit_interest`.
- Produces: `FlowState.favorite_property`, `rejected_properties`, `visit_interest`, `interests`; persisted under `state["context"]`; seed from context each preprocess.

- [ ] **Step 1: Write failing memory tests**

```python
class TestCommercialMemory:
    def test_detects_favorite_by_list_number(self):
        props = [
            {"title": "Torre Nova", "region": "Pinheiros", "area_util": 100},
            {"title": "Torre Antiga", "region": "Pinheiros", "area_util": 80},
        ]
        flow = make_flow(properties_rag=lambda info: props)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "gostei da 2",
            "lead_info": {},
            "properties": props,
            "shown_properties_count": 2,
        })
        assert state.get("favorite_property") == "Torre Antiga"
        assert state.get("context", {}).get("favorite_property") == "Torre Antiga"

    def test_detects_favorite_by_title_substring(self):
        props = [{"title": "Torre Nova", "region": "Pinheiros", "area_util": 100}]
        flow = make_flow(properties_rag=lambda info: props)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "gostei da Torre Nova",
            "lead_info": {},
            "properties": props,
            "shown_properties_count": 1,
        })
        assert state.get("favorite_property") == "Torre Nova"

    def test_detects_rejection(self):
        props = [
            {"title": "Torre Nova", "region": "Pinheiros", "area_util": 100},
            {"title": "Torre Antiga", "region": "Pinheiros", "area_util": 80},
        ]
        flow = make_flow(properties_rag=lambda info: props)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "não quero a 1",
            "lead_info": {},
            "properties": props,
            "shown_properties_count": 2,
        })
        assert "Torre Nova" in (state.get("rejected_properties") or [])
        assert state.get("context", {}).get("rejected_properties") is not None

    def test_context_seed_restores_favorite_across_invokes(self):
        props = [{"title": "Torre Nova", "region": "Pinheiros", "area_util": 100}]
        flow = make_flow(scheduler=MagicMock(return_value={"confirmed": True, "when": "x"}))
        # simulate prior turn persisted context
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
            "context": {"favorite_property": "Torre Nova"},
        })
        assert state["current_state"] == "handoff"  # gate opens via context seed

    def test_visit_interest_flag_from_context_seed(self):
        flow = make_flow(scheduler=MagicMock(return_value={"confirmed": True, "when": "x"}))
        state = flow.invoke({
            "current_state": "scheduling",
            "message": "amanhã 10h",
            "lead_id": "L1",
            "shown_properties_count": 3,
            "context": {"visit_interest": True},
        })
        assert state["current_state"] == "handoff"
```

- [ ] **Step 2: Run to verify fail**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_sales_flow.py::TestCommercialMemory -v
```

Expected: FAIL — no detection / no context seed.

- [ ] **Step 3: Implement FlowState fields, seed, detection**

**FlowState** — add after `visit_interest`/`favorite_property` (if not already in Task 1):

```python
    rejected_properties: list[str]
    interests: list[str]
```

**Module-level regexes** near other `_RE`:

```python
_FAVORITE_LIKE_RE = re.compile(
    r"(?:gostei\s+da?s?\s+|quero\s+a\s+|prefiro\s+a\s+)(\d+|[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 \-]+)",
    re.IGNORECASE,
)
_REJECT_LIKE_RE = re.compile(
    r"(?:n[ãa]o\s+quero\s+(?:a\s+|as\s+)?|descart[ae]\s+(?:a\s+|as\s+)?)(\d+|[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 \-]+)",
    re.IGNORECASE,
)
```

**`_node_preprocess`** — after `context = state.setdefault("context", {})` and lead_info merge, seed:

```python
        if context.get("favorite_property"):
            state["favorite_property"] = context["favorite_property"]
        if context.get("visit_interest"):
            state["visit_interest"] = True
        if context.get("rejected_properties"):
            state["rejected_properties"] = list(context["rejected_properties"])
        if context.get("interests"):
            state["interests"] = list(context["interests"])
```

**New helper** on SalesFlow:

```python
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
```

Call `_apply_commercial_memory(state)` at end of `_node_preprocess` (before return), after router merge (so LLM `favorite_property` can override later if added; for now regex only).

Also: when `_router_action == "visit_interest"`, ensure `context["visit_interest"] = True`.

- [ ] **Step 4: Run memory tests**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_sales_flow.py::TestCommercialMemory -v
```

Expected: pass.

- [ ] **Step 5: Full conversation-router suite**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add apps/conversation-router/service/flow/sales_flow.py \
  apps/conversation-router/tests/unit/test_sales_flow.py
git commit -m "feat(sales-flow): commercial memory favorite/rejected/visit_interest with context persistence"
```

---

## Task 4: Discovery state + rich reply_generator

**Files:**
- Modify: `apps/conversation-router/service/flow/sales_flow.py` (`_build_graph`, `_route_state`, `_node_discovery`, `_node_postprocess`)
- Modify: `apps/conversation-router/service/llm.py` (`generate_reply` kwargs, `_REPLY_SYSTEM_PROMPT`, user_block)
- Modify: `apps/conversation-router/service/entities.py` (`VALID_STATES`)
- Modify: `apps/conversation-router/handler.py` (`llm_reply` passes kwargs)
- Test: `apps/conversation-router/tests/unit/test_sales_flow.py`
- Test: `apps/conversation-router/tests/unit/test_llm.py`
- Test: `apps/conversation-router/tests/unit/test_entities.py`
- Test: `apps/conversation-router/tests/unit/test_handler_llm_secret.py`

**Interfaces:**
- Consumes: Tasks 1–3 (`favorite_property`, `shown_properties_count`, `visit_interest`).
- Produces: node `"discovery"`; `VALID_STATES` includes `"discovery"`; `generate_reply(..., favorite_property=None, conversation_stage=None, shown_properties_count=None, **)`; `_node_postprocess` passes those kwargs; `handler.llm_reply` accepts `**kwargs` and forwards.

- [ ] **Step 1: Write failing entities + discovery + reply tests**

`test_entities.py` update:

```python
    def test_valid_states_include_all_flow_states(self):
        assert set(VALID_STATES) == {
            "greeting", "elicitation", "intent", "qualification",
            "discovery", "recommendation", "scheduling", "handoff", "followup",
        }
```

`test_sales_flow.py`:

```python
class TestDiscoveryState:
    def test_discovery_answers_about_shown_property_without_changing_stage(self):
        props = [{"title": "Torre Nova", "region": "Pinheiros", "area_util": 100, "vagas": 2}]
        router = lambda message, lead_info, current_state: {
            "lead_info": {},
            "action": "provide_info",
        }
        flow = make_flow(properties_rag=lambda info: props, llm_router=router)
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "a Torre Nova tem estacionamento?",
            "lead_info": {},
            "properties": props,
            "favorite_property": "Torre Nova",
            "shown_properties_count": 1,
        })
        # Either discovery node or recommendation kept stage; must mention property context
        assert state["current_state"] in ("discovery", "recommendation")
        assert state.get("favorite_property") == "Torre Nova"
        assert "Torre Nova" in state["response"] or "estacionamento" in state["response"].lower() or "opções" in state["response"].lower()

    def test_postprocess_passes_rich_kwargs(self):
        captured = {}
        def fake_reply(message, canned, lead_info, properties, **kwargs):
            captured.update(kwargs)
            return canned + " (llm)"
        flow = make_flow(reply_generator=fake_reply, properties_rag=lambda i: [{"title": "X"}])
        state = flow.invoke({
            "current_state": "recommendation",
            "message": "ok",
            "lead_info": {},
            "properties": [{"title": "X"}],
            "favorite_property": "X",
            "shown_properties_count": 3,
        })
        assert captured.get("favorite_property") == "X"
        assert captured.get("conversation_stage") == "recommendation"
        assert captured.get("shown_properties_count") == 3
        assert state["response"].endswith("(llm)")
```

`test_llm.py`:

```python
class TestGenerateReplyRichContext:
    def test_prompt_is_consultative_and_user_block_has_stage(self, monkeypatch: pytest.MonkeyPatch):
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _make_completion("Certo! Sobre a Torre Nova...")

        monkeypatch.setattr(lm.litellm, "completion", fake_completion)
        out = lm.generate_reply(
            message="tem estacionamento?",
            canned_response="A Torre Nova tem 2 vagas.",
            lead_info={"region": "Pinheiros"},
            properties=[{"title": "Torre Nova", "area_util": 100}],
            api_key="k",
            favorite_property="Torre Nova",
            conversation_stage="discovery",
            shown_properties_count=3,
        )
        system = captured["messages"][0]["content"]
        assert "consultor" in system.lower()
        assert "1 pergunta" in system or "uma pergunta" in system.lower()
        user = captured["messages"][1]["content"]
        assert "ESTÁGIO DA CONVERSA: discovery" in user
        assert "Torre Nova" in user
        assert "IMÓVEIS JÁ EXIBIDOS: 3" in user
        assert out == "Certo! Sobre a Torre Nova..."

    def test_backward_compat_positional_call(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            lm.litellm, "completion",
            lambda **kw: _make_completion("ok"),
        )
        assert lm.generate_reply("oi", "resposta", {}, [], api_key="k") == "ok"
```

- [ ] **Step 2: Run to verify fail**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest \
  tests/unit/test_entities.py::TestConversation::test_valid_states_include_all_flow_states \
  tests/unit/test_sales_flow.py::TestDiscoveryState \
  tests/unit/test_llm.py::TestGenerateReplyRichContext -v
```

Expected: FAIL (discovery missing, kwargs unexpected TypeError, prompt/content assertions).

- [ ] **Step 3: Implement entities + generate_reply + prompt**

**`entities.py`** `VALID_STATES`:

```python
VALID_STATES = (
    "greeting",
    "elicitation",
    "intent",
    "qualification",
    "discovery",
    "recommendation",
    "scheduling",
    "handoff",
    "followup",
)
```

**`llm.py` `generate_reply`** — add keyword-only params after `api_key`:

```python
def generate_reply(
    message: str,
    canned_response: str,
    lead_info: dict[str, Any],
    properties: list[dict[str, Any]],
    api_key: str,
    model: str | None = None,
    url: str = "",
    timeout: float | None = None,
    force_complex: bool = False,
    favorite_property: str | None = None,
    conversation_stage: str | None = None,
    shown_properties_count: int | None = None,
) -> str:
```

Extend `user_block` after IMÓVEIS block:

```python
    stage_line = f"ESTÁGIO DA CONVERSA: {conversation_stage}" if conversation_stage else "ESTÁGIO DA CONVERSA: (desconhecido)"
    fav_line = f"IMÓVEL FAVORITO DO LEAD: {favorite_property}" if favorite_property else "IMÓVEL FAVORITO DO LEAD: (nenhum)"
    shown_line = (
        f"IMÓVEIS JÁ EXIBIDOS: {shown_properties_count}"
        if shown_properties_count is not None
        else "IMÓVEIS JÁ EXIBIDOS: 0"
    )
    user_block = (
        "RESPOSTA OFICIAL DO SISTEMA (transmita o conteúdo, pode melhorar o tom):\n"
        f"{canned_response}\n\n"
        f"{stage_line}\n{fav_line}\n{shown_line}\n\n"
        f"DADOS DO LEAD (estruturados, já mascarados):\n{lead or '(vazio)'}\n\n"
        f"IMÓVEIS RECOMENDADOS (SÓ estes podem ser citados):\n{props or '(nenhum)'}\n\n"
        f"ÚLTIMA MENSAGEM DO LEAD:\n{message[:500]}"
    )
```

**`_REPLY_SYSTEM_PROMPT`** — replace with consultative version (keep safety rules 1–4 semantics):

```python
_REPLY_SYSTEM_PROMPT = (
    "Você é um consultor imobiliário corporativo da W Levitt. Atenda leads do Telegram em português.\n"
    "NUNCA pareça um formulário. Faça apenas 1 pergunta por vez.\n"
    "Se houver imóveis disponíveis: converse sobre eles, explore preferências, "
    "não tente agendar imediatamente.\n"
    "Só ofereça visita quando o usuário demonstrar interesse explícito OU mencionar uma opção específica.\n"
    "Tom: consultivo, profissional, objetivo.\n\n"
    "Regras rígidas:\n"
    "1. A RESPOSTA OFICIAL contém o que DEVE ser dito. Você APENAS melhora o tom, "
    "NUNCA altera o significado nem adiciona informações novas.\n"
    "2. IMÓVEIS RECOMENDADOS: SÓ cite imóveis que apareçam nesta lista. "
    "Se a lista for '(nenhum)' ou vazia, NUNCA mencione imóvel, preço, metragem, "
    "bairro ou valor — apenas reescreva a resposta oficial.\n"
    "3. NUNCA invente: preço, metragem, bairro, nome de empreendimento, "
    "disponibilidade, ou prazo.\n"
    "4. Máximo 3 frases. A pergunta final DEVE SER coerente com a ação do lead; "
    "NUNCA force agendamento quando o lead só quer ver propriedades ou conversar sobre elas."
)
```

- [ ] **Step 4: Run llm + entities tests**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests/unit/test_llm.py tests/unit/test_entities.py -q
```

Expected: pass (rich + backward-compat + VALID_STATES).

- [ ] **Step 5: Implement discovery node + graph + route**

**`_build_graph`:** add

```python
        g.add_node("discovery", self._node_discovery)
```

Add `"discovery"` to the edge list node tuple.

**`_node_discovery`:**

```python
    def _node_discovery(self, state: FlowState) -> FlowState:
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
```

**`_route_state`:** after `decline` branch, before options/handoff — route detail questions:

```python
        if (
            action in (None, "provide_info", "unclear")
            and current in ("recommendation", "discovery")
            and re.search(r"\b(?:estacionamento|vagas?|andar|pre[çc]o|valor|quanto|detalhes?|tem|possui)\b", state.get("message", ""), re.IGNORECASE)
        ):
            return "discovery"
```

Keep LGPD branches first.

**`_node_recommendation` / other nodes:** unchanged except discovery is terminal for that turn (edges to postprocess).

**FSM fallback:** add `_handle_discovery` next to other handlers if `_invoke_fsm` used:

```python
    def _handle_discovery(self, state, message):
        return self._node_discovery(state)
```

- [ ] **Step 6: Wire postprocess kwargs**

**`_node_postprocess`:**

```python
            generated = self.reply_generator(
                state.get("message", ""),
                canned,
                state.get("lead_info") or {},
                state.get("properties") or [],
                favorite_property=state.get("favorite_property"),
                conversation_stage=state.get("current_state"),
                shown_properties_count=state.get("shown_properties_count", 0),
            )
```

**Constructor type hint** (optional but recommended):

```python
        reply_generator: Callable[..., str] | None = None,
```

**Existing mocks:** `test_sales_flow.py` already uses `lambda *_:` and `fake_reply(message, canned, lead_info, properties)` — must become `(..., **kwargs)` in:
- `test_reply_generator_failure_keeps_canned` — already `lambda *_:` OK if kwargs passed as keyword — **`lambda *_:` does NOT accept kwargs**. Change all reply_generator test doubles to `lambda *a, **k:` or `def fake(..., **kwargs)`.

Grep and fix:

```bash
rg -n "reply_generator=" apps/conversation-router/tests
```

Update every mock to accept `**kwargs`.

**`handler.py` `llm_reply`:**

```python
        def llm_reply(
            message: str, canned: str, lead_info: dict[str, Any], properties: list[dict[str, Any]], **kwargs: Any
        ) -> str:
            try:
                return _llm_generate_reply(
                    message, canned, lead_info, properties,
                    api_key=llm_key,
                    favorite_property=kwargs.get("favorite_property"),
                    conversation_stage=kwargs.get("conversation_stage"),
                    shown_properties_count=kwargs.get("shown_properties_count"),
                )
            except Exception:
                logger.warning("LLM reply falhou; usando resposta oficial como fallback", exc_info=True)
                return canned
```

- [ ] **Step 7: Handler smoke — extend test if needed**

If `test_handler_llm_secret` only asserts `reply_generator is not None`, no change. Optional smoke:

```python
def test_llm_reply_accepts_kwargs(monkeypatch, ...):
    # after wiring, call captured reply_generator with extra kwargs and ensure no TypeError
```

Only add if cheap with existing fixtures; otherwise skip (covered by sales_flow + llm tests).

- [ ] **Step 8: Run all suites separately**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests -q
cd apps/voice-adapter && ../../.venv/bin/pytest tests -q
cd apps/dashboard-api && ../../.venv/bin/pytest tests -q
cd apps/dashboard-ui && ../../.venv/bin/pytest tests -q
```

Expected: conversation-router all pass; voice 82; dashboard 74+2 skipped.

- [ ] **Step 9: Commit**

```bash
git add apps/conversation-router/service/flow/sales_flow.py \
  apps/conversation-router/service/llm.py \
  apps/conversation-router/service/entities.py \
  apps/conversation-router/handler.py \
  apps/conversation-router/tests/unit/test_sales_flow.py \
  apps/conversation-router/tests/unit/test_llm.py \
  apps/conversation-router/tests/unit/test_entities.py \
  apps/conversation-router/tests/unit/test_handler_llm_secret.py
git commit -m "feat(sdr): discovery state and consultative reply_generator with commercial context"
```

---

## Task 5: Final verification (acceptance)

**Files:** none (verification only)

- [ ] **Step 1: Run all suites**

```bash
cd apps/conversation-router && ../../.venv/bin/pytest tests -q
cd apps/voice-adapter && ../../.venv/bin/pytest tests -q
cd apps/dashboard-api && ../../.venv/bin/pytest tests -q
cd apps/dashboard-ui && ../../.venv/bin/pytest tests -q
```

Expected: ≥348 total pass (may be higher with new tests), 0 fail.

- [ ] **Step 2: Confirm acceptance criteria**

Manual checks via pytest already encoded:
- No scheduler call without favorite/visit_interest/deadline (Task 1 tests).
- New router actions accepted (Task 2).
- Favorite persisted to context (Task 3).
- Reply prompt consultative + kwargs (Task 4).

- [ ] **Step 3: Report summary**

List commit SHAs per task and final test counts. Do not push unless asked.

---

## Self-Review Notes (author)

**Spec coverage:**
- Etapa 1 gate → Task 1 ✓
- Etapa 2 VALID_ACTIONS + mapping → Task 2 ✓
- Etapa 3 memory + context → Task 3 ✓
- Etapa 4 discovery + generate_reply + VALID_STATES + handler → Task 4 ✓
- Acceptance / suites → Task 5 ✓

**Placeholders:** none intended; all steps have concrete code.

**Type consistency:** `ready_for_scheduling(state) -> bool`; FlowState fields `favorite_property`, `visit_interest`, `rejected_properties`, `interests`; generate_reply kwargs names match postprocess and handler.
