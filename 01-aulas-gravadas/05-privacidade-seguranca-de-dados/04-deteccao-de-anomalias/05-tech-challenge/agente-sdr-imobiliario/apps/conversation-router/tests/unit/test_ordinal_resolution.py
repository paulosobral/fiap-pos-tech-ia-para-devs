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


def test_property_detail_generic_followup_uses_favorite_over_model_reference():
    """Mensagem generica, sem referencia nenhuma: o roteador nao tem nada pra
    resolver em property_ref (fica vazio/ausente) — o favorito salvo e quem
    decide, nao um property_ref hipotetico e desalinhado da mensagem atual."""
    favorite = _PROPERTIES[1]
    state = {"properties": _PROPERTIES, "favorite_property": favorite["title"]}

    result = execute_tool(
        "property_detail",
        {},
        state,
        message="quero mais detalhes",
    )

    assert result.detail == favorite


def test_explicit_ordinal_overrides_saved_favorite():
    """Mensagem com referencia explicita ('do primeiro'): o roteador resolve
    property_ref pro titulo exato do item referenciado na mensagem atual (nao
    pro favorito salvo) — e esse property_ref, ja resolvido, e quem manda,
    vencendo o favorito antigo."""
    state = {"properties": _PROPERTIES, "favorite_property": _PROPERTIES[1]["title"]}

    result = execute_tool(
        "property_detail",
        {"property_ref": _PROPERTIES[0]["title"]},
        state,
        message="quero mais detalhes do primeiro",
    )

    assert result.detail == _PROPERTIES[0]


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


def test_property_detail_trusts_llm_resolved_ref_for_any_phrasing():
    """A resolução de referência não depende de Python reconhecer a palavra usada
    na mensagem (ordinal, posição, apelido) — o roteador já resolve isso pro título
    exato em property_ref; o Python só precisa casar esse título contra a lista.
    Aqui a mensagem usa uma forma ("mostra o penúltimo") que NÃO existe em nenhum
    dict de ordinais do projeto, mas funciona porque o teste simula o property_ref
    já resolvido pela LLM (não um regex Python adivinhando "penúltimo")."""
    from service.tools import execute_tool

    state = {"properties": _PROPERTIES}
    result = execute_tool(
        "property_detail",
        {"property_ref": _PROPERTIES[-2]["title"]},
        state,
        message="mostra o penúltimo",
    )
    assert result.detail == _PROPERTIES[-2]


def test_property_detail_exact_title_ref_wins_over_digit_substring_collision():
    """Regressão: títulos com número embutido (ex. 'Faria Lima 02') não devem
    colidir com a chave ordinal '2' do _ORDINAL_MAP quando property_ref já é o
    título exato — o match exato de título precisa vir antes do loop ordinal."""
    from service.tools import _resolve_in_shown

    props = [
        {"title": "Corporate Faria Lima 02 — laje corporativa em Paulista"},
        {"title": "Vila Olímpia Executive 75 — conjunto comercial em Berrini"},
    ]
    resolved = _resolve_in_shown(props[0]["title"], props)
    assert resolved is props[0]


def test_explicit_shown_reference_is_last_resort_fallback():
    """Quando property_ref vem vazio (LLM não extraiu nada) e não há favorito,
    o regex sobre a mensagem crua (_explicit_shown_reference) ainda serve de
    rede de segurança — mas só é consultado nesse caso, nunca quando já há
    uma referência resolvida."""
    from service.tools import execute_tool

    state = {"properties": _PROPERTIES}
    result = execute_tool(
        "property_detail",
        {},
        state,
        message="quero saber mais da segunda opção",
    )
    assert result.detail == _PROPERTIES[1]
