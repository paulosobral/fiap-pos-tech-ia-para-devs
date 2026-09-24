from __future__ import annotations

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
    raw = {
        "tool": "provide_info",
        "arguments": {},
        "lead_info": {},
        "memory_updates": {"favorite_property": "Faria Lima Corporate", "visit_interest": False},
    }
    out = validate_router_output(raw, _state())
    assert out["memory_updates"]["favorite_property"] == "Corporate Faria Lima 02"


def test_fuzzy_favorite_miss_returns_none():
    raw = {
        "tool": "provide_info",
        "arguments": {},
        "lead_info": {},
        "memory_updates": {"favorite_property": "Invented Tower XYZ", "visit_interest": False},
    }
    out = validate_router_output(raw, _state())
    assert out["memory_updates"]["favorite_property"] is None


def test_favorite_never_full_catalog():
    # even if catalog has other props, only shown list is searched
    raw = {
        "tool": "provide_info",
        "arguments": {},
        "lead_info": {},
        "memory_updates": {"favorite_property": "Some Other Catalog Prop", "visit_interest": False},
    }
    out = validate_router_output(raw, _state())
    assert out["memory_updates"]["favorite_property"] is None


def test_visit_interest_requires_evidence():
    # message without visit verb → force false even if LLM said true
    raw = {
        "tool": "express_visit_interest",
        "arguments": {},
        "lead_info": {},
        "memory_updates": {"favorite_property": "Torre Nova", "visit_interest": True},
    }
    out = validate_router_output(raw, _state(), message="gostei da Torre Nova")
    assert out["memory_updates"]["visit_interest"] is False
    assert out["memory_updates"]["favorite_property"] == "Torre Nova"


def test_visit_interest_with_evidence_persists():
    raw = {
        "tool": "express_visit_interest",
        "arguments": {},
        "lead_info": {},
        "memory_updates": {"favorite_property": "Torre Nova", "visit_interest": True},
    }
    out = validate_router_output(raw, _state(), message="quero visitar a Torre Nova")
    assert out["memory_updates"]["visit_interest"] is True


def test_thought_ignored_for_flow():
    raw = {
        "thought": "decide schedule",
        "tool": "request_schedule",
        "arguments": {},
        "lead_info": {},
        "memory_updates": {},
    }
    out = validate_router_output(raw, _state(), message="agendar")
    assert "thought" in out  # kept for logging only
    assert out["tool"] == "request_schedule"  # flow driven by tool, not thought


def test_unknown_memory_keys_dropped():
    raw = {
        "tool": "provide_info",
        "arguments": {},
        "lead_info": {},
        "memory_updates": {
            "favorite_property": "Torre Nova",
            "visit_interest": False,
            "evil_key": "x",
        },
    }
    out = validate_router_output(raw, _state())
    assert "evil_key" not in out["memory_updates"]
