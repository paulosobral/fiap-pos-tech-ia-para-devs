"""Teste de qualidade como código (TaaC) — plano "naturalidade do bot SDR", seção 5.

Diferente da suíte `tests/unit`, este arquivo chama a LLM REAL (via
`extract_and_route`/`generate_reply` de `service/llm.py`), passando pelo mesmo
cliente tiered/fallback de produção (`_completion_with_fallback`,
`resolve_model(TIER_PRIMARY/TIER_FALLBACK/...)`) — não mocka `litellm.completion`
nem usa um modelo fixo só para teste. Isso custa crédito real no OpenRouter
(poucas chamadas, todas Tier 1), então só roda quando há uma chave configurada.

Gate de execução: pula a suíte inteira se não houver `OPENROUTER_API_KEY` nem
`LLM_API_KEY` no ambiente — dev local sem chave e CI de PR sem secret não
quebram; roda de verdade no deploy/CI que tiver a credencial.

Asserções são soltas/semânticas (saída de LLM real varia a cada chamada):
checam pertencimento a um conjunto de tools válidas, presença/ausência de
substrings-chave, ou que o título resolvido está entre os imóveis exibidos —
nunca comparam a frase completa da resposta.
"""
from __future__ import annotations

import os

import pytest

from service import llm

API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY") or ""

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="Teste de qualidade com LLM real — requer OPENROUTER_API_KEY/LLM_API_KEY no ambiente.",
)

_SHOWN = [
    {"title": "Sala comercial 30m2 - Sao Judas", "region": "Sao Judas", "area_util": 30},
    {"title": "Apartamento 36m2 - Parque Sao Rafael", "region": "Parque Sao Rafael", "area_util": 36},
    {"title": "Terreno Eldorado 19m2", "region": "Eldorado", "area_util": 19},
]
_TITLES = {p["title"] for p in _SHOWN}


def _route(message: str, **kwargs) -> dict:
    kwargs.setdefault("shown_properties", _SHOWN)
    return llm.extract_and_route(
        message,
        lead_info={},
        current_state="conversation",
        api_key=API_KEY,
        **kwargs,
    )


@pytest.mark.parametrize(
    "message",
    [
        "mostra o um",
        "mostra o terceiro",
        "quero ver o penúltimo",
        "mostra esse aí",
        "me manda o mais barato",
        "quero o do meio",
    ],
)
def test_reference_to_specific_item_resolves_to_exact_shown_title(message: str):
    """Core do bug original: verbo de exibição + referência a item específico
    (ordinal, posição, apelido) deve virar property_detail com property_ref
    batendo um título real já exibido — nenhuma dessas formas está em um dict
    Python; é a LLM resolvendo a lista que ela já recebeu no prompt."""
    from service.tools import _resolve_in_shown

    result = _route(message)
    assert result["tool"] == "property_detail", (message, result)
    ref = (result.get("arguments") or {}).get("property_ref")
    # título exato ou número do item (preferido quando há títulos repetidos)
    assert _resolve_in_shown(str(ref), _SHOWN) is not None, (message, ref)


@pytest.mark.parametrize("message", ["volta pra lista anterior", "repete a lista"])
def test_navigation_request_routes_to_unclear_not_provide_info(message: str):
    result = _route(message)
    assert result["tool"] == "unclear", (message, result)


@pytest.mark.parametrize("message", ["mostra tudo", "cadê as opções", "tem mais opções?"])
def test_new_list_request_routes_to_request_options(message: str):
    result = _route(message)
    assert result["tool"] == "request_options", (message, result)


def test_reply_confirms_photos_instead_of_denying_them():
    canned = "A sala em São Judas tem 30m², R$ 200.000, com 1 vaga de garagem."
    reply = llm.generate_reply(
        message="cadê a foto?",
        canned_response=canned,
        lead_info={},
        properties=[{**_SHOWN[0], "images": [f"https://cdn/{i}.jpg" for i in range(9)]}],
        api_key=API_KEY,
    )
    lowered = reply.lower()
    denial_phrases = ("não tenho foto", "nao tenho foto", "não há foto", "nao ha foto", "sem foto")
    assert not any(phrase in lowered for phrase in denial_phrases), reply


def test_follow_up_for_more_photos_on_single_focused_property_stays_property_detail():
    """'mais fotos' com 1 único imóvel em foco (estado realista: favorito setado
    pelo detalhe anterior) deve continuar property_detail, não cair em
    unclear/provide_info — combinado com `_response_images` (sales_flow.py),
    isso fecha o caso de uso de 'mais fotos' sem precisar de tool nova."""
    focused = [_SHOWN[0]]
    history = [
        {"role": "user", "content": "mostra o um"},
        {"role": "assistant", "content": "Sala comercial 30m² em São Judas, R$ 200.000, 1 vaga."},
    ]
    result = _route(
        "mais fotos",
        shown_properties=focused,
        favorite_property=focused[0]["title"],
        conversation_history=history,
    )
    assert result["tool"] == "property_detail", result


def test_bare_intent_asks_for_criteria_before_listing():
    result = llm.extract_and_route(
        "comprar", lead_info={}, current_state="conversation", api_key=API_KEY,
    )
    assert result["tool"] in ("refine_search", "request_options"), result
    assert (result.get("arguments") or {}).get("ask_criteria") is True, result


def test_list_request_does_not_send_photos_but_explicit_ask_does():
    listing = _route("tem apartamentos em são caetano?", shown_properties=[])
    assert not (listing.get("arguments") or {}).get("send_photos"), listing
    asked = _route("manda as fotos do primeiro")
    assert asked["tool"] == "property_detail", asked
    assert (asked.get("arguments") or {}).get("send_photos") is True, asked


def test_contact_request_reply_asks_for_phone_or_email_instead_of_promising():
    from service.flow.sales_flow import _asks_for_contact

    reply = llm.generate_reply(
        message="quero entrar em contato por esse imóvel",
        canned_response="Claro! Pro corretor falar com você, me passa seu telefone (WhatsApp) ou seu e-mail?",
        lead_info={},
        properties=[_SHOWN[0]],
        api_key=API_KEY,
        last_tool="request_human",
        contact_channels={"telefone": False, "email": False},
        contact_request=True,
    )
    assert _asks_for_contact(reply), reply


def test_detail_reply_uses_listing_description_instead_of_inventing():
    prop = {
        **_SHOWN[0],
        "description": "Sala comercial com 2 banheiros, copa e 1 vaga. IPTU R$ 90.",
        "images": ["https://cdn/x.jpg"],
    }
    reply = llm.generate_reply(
        message="quantos banheiros tem?",
        canned_response=f"{prop['title']} — {prop['region']}, 30 m².",
        lead_info={},
        properties=[prop],
        api_key=API_KEY,
        last_tool="property_detail",
    )
    assert "2 banheiros" in reply.lower() or "dois banheiros" in reply.lower(), reply
