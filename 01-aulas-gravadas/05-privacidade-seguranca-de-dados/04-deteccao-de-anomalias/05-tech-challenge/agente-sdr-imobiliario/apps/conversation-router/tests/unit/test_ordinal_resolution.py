from typing import Any
import pytest

from service.validation import _fuzzy_shown, _resolve_ordinal, validate_router_output
from service.tools import execute_tool, _resolve_in_shown
from service.flow.sales_flow import SalesFlow


_PROPERTIES = [
    {
        "id": "pr-002",
        "title": "Corporate Faria Lima 02 — laje corporativa em Paulista",
        "region": "Paulista",
        "area_util": 80,
        "price_text": "R$ 4.9 mil/mês",
        "disponibilidade": "reservado",
    },
    {
        "id": "pr-075",
        "title": "Vila Olímpia Executive 75 — conjunto comercial em Berrini",
        "region": "Berrini",
        "area_util": 80,
        "price_text": "R$ 6.4 mil/mês",
        "disponibilidade": "reservado",
    },
    {
        "id": "pr-107",
        "title": "Centro Empresarial Pinheiros 107 — escritório em Moema",
        "region": "Moema",
        "area_util": 60,
        "price_text": "R$ 7.2 mil/mês",
        "disponibilidade": "disponível",
    },
]


def test_resolve_ordinal_exact_and_phrasal():
    assert _resolve_ordinal("segunda opção", _PROPERTIES) == _PROPERTIES[1]["title"]
    assert _resolve_ordinal("segundo", _PROPERTIES) == _PROPERTIES[1]["title"]
    assert _resolve_ordinal("2", _PROPERTIES) == _PROPERTIES[1]["title"]
    assert _resolve_ordinal("primeira opcao", _PROPERTIES) == _PROPERTIES[0]["title"]
    assert _resolve_ordinal("terceiro imóvel", _PROPERTIES) == _PROPERTIES[2]["title"]


def test_fuzzy_shown_resolves_ordinal_references():
    assert _fuzzy_shown("segunda opção", _PROPERTIES) == _PROPERTIES[1]["title"]
    assert _fuzzy_shown("o segundo", _PROPERTIES) == _PROPERTIES[1]["title"]
    assert _fuzzy_shown("2", _PROPERTIES) == _PROPERTIES[1]["title"]


def test_validate_router_output_persists_ordinal_favorite():
    raw = {
        "thought": "lead escolheu a segunda opção",
        "tool": "express_visit_interest",
        "arguments": {},
        "lead_info": {},
        "memory_updates": {
            "favorite_property": "segunda opção",
            "visit_interest": False,
        },
    }
    state = {"properties": _PROPERTIES}
    validated = validate_router_output(raw, state, message="gostei da segunda opção")
    assert validated["memory_updates"]["favorite_property"] == _PROPERTIES[1]["title"]


def test_tools_resolve_in_shown_ordinal():
    resolved = _resolve_in_shown("segunda opção", _PROPERTIES)
    assert resolved is not None
    assert resolved["title"] == _PROPERTIES[1]["title"]

    resolved_digit = _resolve_in_shown("2", _PROPERTIES)
    assert resolved_digit is not None
    assert resolved_digit["title"] == _PROPERTIES[1]["title"]


def test_property_detail_tool_and_response_contain_disponibilidade():
    from unittest.mock import MagicMock

    target = _PROPERTIES[1]
    state = {"properties": _PROPERTIES, "favorite_property": target["title"]}
    tr = execute_tool(
        "property_detail",
        {"property_ref": target["title"]},
        state,
        message="checa a disponibilidade dele",
    )
    assert tr.ok is True
    assert tr.detail is not None
    assert tr.detail.get("disponibilidade") == "reservado"

    flow = SalesFlow(lead_qualifier=MagicMock())
    flow_state = {
        "current_state": "conversation",
        "message": "checa a disponibilidade dele",
        "properties": _PROPERTIES,
        "favorite_property": target["title"],
        "_router_tool": "property_detail",
        "_tool_result": tr,
    }
    res = flow._conversation_from_tool(flow_state)
    assert "Disponibilidade: reservado" in res["response"]


def test_sales_flow_match_property_token_resolves_ordinals():
    matched = SalesFlow._match_property_token("segunda opção", _PROPERTIES)
    assert matched == _PROPERTIES[1]["title"]

    matched_num = SalesFlow._match_property_token("2", _PROPERTIES)
    assert matched_num == _PROPERTIES[1]["title"]
