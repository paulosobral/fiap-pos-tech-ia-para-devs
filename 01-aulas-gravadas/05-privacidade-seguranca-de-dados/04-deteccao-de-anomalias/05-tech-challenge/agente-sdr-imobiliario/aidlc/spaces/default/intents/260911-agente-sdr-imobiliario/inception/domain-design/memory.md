# Domain Design memory

Tool-agent router implemented per spec 2026-09-23-tool-agent-sdr-design; decisions: single-step, 5 states, fuzzy-shown-only, no favorite in schedule gate.

- Single-step contract `{thought, tool, arguments, lead_info, memory_updates}` replaces ADR-011 enum as primary path; ADR-011 regex/FSM retained as fallback.
- VALID_TOOLS (10): request_options, property_detail, compare_properties, refine_search, express_visit_interest, request_schedule, request_human, decline, provide_info, unclear.
- 5 states: greeting, conversation, scheduling, handoff, followup (+ elicitation/preprocess/postprocess).
- validate_router_output: fuzzy favorite shown-only (threshold 0.55), visit evidence gate, drop unknown keys, thought log-only.
- Scheduling gate: shown>=3 + visit_interest + ready; does NOT require favorite_property.
- generate_reply kwargs: visit_interest, rejected_properties, last_tool; long-list cap 800 tokens vs 280.
- Spec: docs/superpowers/specs/2026-09-23-tool-agent-sdr-design.md
