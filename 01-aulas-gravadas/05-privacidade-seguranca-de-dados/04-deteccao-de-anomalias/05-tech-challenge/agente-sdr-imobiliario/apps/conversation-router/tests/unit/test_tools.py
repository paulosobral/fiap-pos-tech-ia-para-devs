from __future__ import annotations

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
    r = execute_tool(
        "compare_properties",
        {"property_a": "Torre Nova", "property_b": "Corporate One"},
        state,
    )
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
    r = execute_tool(
        "express_visit_interest",
        {},
        state,
        memory_updates={"favorite_property": "Torre Nova", "visit_interest": True},
    )
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
