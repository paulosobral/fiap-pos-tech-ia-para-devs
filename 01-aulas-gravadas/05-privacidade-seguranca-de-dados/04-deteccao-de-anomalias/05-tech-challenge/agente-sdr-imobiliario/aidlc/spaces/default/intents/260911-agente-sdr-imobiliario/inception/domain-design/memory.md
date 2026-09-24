# Domain design — memory

Tool-agent router implemented per spec 2026-09-23-tool-agent-sdr-design; decisions: single-step, 5 states, fuzzy-shown-only, no favorite in schedule gate.

## Session Work (2026-09-23)

### Tool-agent router (spec 2026-09-23-tool-agent-sdr) — COMPLETE
- `llm.py`: `VALID_TOOLS` (10), `_ROUTER_SYSTEM_PROMPT` tool-agent, `_parse_extract_and_route` contract `{thought, tool, arguments, lead_info, memory_updates}`, `max_tokens=300`.
- `validation.py`: `validate_router_output` — fuzzy favorite shown-only (0.55), visit evidence gate, drop unknown memory/lead keys, `thought` log-only.
- `tools.py`: pure `execute_tool` + `ToolResult` with `current_state_hint`; scheduling gate `shown>=3` + visit + ready (no favorite required).
- `properties_catalog.py`: `search_properties(..., list_scope)` — `all` raises top_k to >=50.
- `sales_flow.py`: 5-state graph (`greeting|conversation|scheduling|handoff|followup` + elicitation/preprocess/postprocess); `_route_state` by tool; conversation dispatcher; legacy `_router_action` path retained for compat.
- `llm.py` `generate_reply`: kwargs `visit_interest`, `rejected_properties`, `last_tool`; long-list cap 800/280 tokens.
- `handler.py`: wires tool-contract router + reply kwargs.
- PRD §8.1.1/§16 updated. Suites: conversation-router 265, voice 82, dashboard-api 66, dashboard-ui 8+2skip.
