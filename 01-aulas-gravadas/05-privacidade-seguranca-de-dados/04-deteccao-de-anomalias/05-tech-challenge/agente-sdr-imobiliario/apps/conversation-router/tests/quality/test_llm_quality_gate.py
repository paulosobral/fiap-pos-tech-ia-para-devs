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
    {"title": "Sala comercial 30m2 - Sao Judas", "region": "Sao Judas", "area_util": 30, "price_text": "R$ 200 mil"},
    {"title": "Apartamento 36m2 - Parque Sao Rafael", "region": "Parque Sao Rafael", "area_util": 36, "price_text": "R$ 300 mil"},
    {"title": "Terreno Eldorado 19m2", "region": "Eldorado", "area_util": 19, "price_text": "R$ 11 milhões"},
]
_TITLES = {p["title"] for p in _SHOWN}


def _route(message: str, **kwargs) -> dict:
    from service.properties_catalog import known_places

    kwargs.setdefault("shown_properties", _SHOWN)
    kwargs.setdefault("places", known_places())  # igual à produção (handler.llm_route)
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


def test_deictic_reference_resolves_to_item_in_focus():
    """'esse aí' só é resolvível com algo em foco (favorito/último detalhado) —
    sem foco e com 3 itens na tela, pedir esclarecimento também seria correto."""
    from service.tools import _resolve_in_shown

    result = _route("mostra esse aí", favorite_property=_SHOWN[1]["title"])
    assert result["tool"] == "property_detail", result
    ref = (result.get("arguments") or {}).get("property_ref")
    assert _resolve_in_shown(str(ref), _SHOWN) is _SHOWN[1], ref


def test_specific_property_request_delivers_it_instead_of_search_mismatch():
    """Chat real 05/10: lead busca 1000 m², pede 'a primeira opção' (terreno de 502 m²) e o
    bot respondia 'abaixo da metragem que você busca / não tenho 1000 m²' em vez de apresentar
    o imóvel pedido. A resposta tem que trazer o imóvel (metragem e preço)."""
    prop = {
        "id": "t", "title": "Terreno para alugar, Campestre - Santo André/SP", "region": "Campestre",
        "area_util": 502, "price_text": "R$ 4.5 mil/mês", "images": [],
        "description": "Espaço amplo para uso comercial em Santo André (Campestre), com 502 m² de terreno.",
    }
    for message in ("1", "eu falei que quero ver a primeira opção"):
        reply = llm.generate_reply(
            message=message,
            canned_response="Terreno para alugar, Campestre - Santo André/SP — Campestre, 502.0 m², "
                            "R$ 4.5 mil/mês. Disponibilidade: disponível. Quer que eu compare com outra opção?",
            lead_info={"intent": "rent", "area": "1000 metros quadrados", "region": "Santo André"},
            properties=[prop],
            api_key=API_KEY,
            last_tool="property_detail",
            conversation_history=[
                {"role": "user", "content": "santo andré, 1000 metros quadrados"},
                {"role": "assistant", "content": "Destaco: 1. Terreno no Campestre, 502 m², R$ 4,5 mil/mês. "
                                                 "2. Sobrado no Jardim, 600 m², R$ 26 mil/mês."},
            ],
        )
        low = reply.lower()
        assert "502" in low and "4,5" in low, (message, reply)
        assert "no momento só tenho" not in low and "não tenho disponíve" not in low, (message, reply)


# --- Interpretação 100% pela LLM: fala/transcrição, lugar, metragem, detalhes --------------------

import re  # noqa: E402

from service.properties_catalog import known_places  # noqa: E402


def _num(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d[\d.,]*", str(value or ""))
    if not m:
        return None
    raw = m.group(0).rstrip(".,")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", raw):
        raw = raw.replace(".", "")
    return float(raw.replace(",", "."))


def _extract(message: str, lead_info: dict | None = None) -> dict:
    return llm.extract_and_route(
        message, lead_info=lead_info or {"intent": "rent"}, current_state="conversation",
        api_key=API_KEY, places=known_places(),
    )["lead_info"]


@pytest.mark.parametrize(
    "message, area, city",
    [
        ("santo andré, 1000 metros quadrados", 1000, "Santo André"),
        ("mil metros quadrados em santo andre", 1000, "Santo André"),
        ("uns mil e duzentos metros lá no scs", 1200, "São Caetano do Sul"),
        ("quero algo em sao bernardo com uns 500 metros", 500, "São Bernardo do Campo"),
    ],
)
def test_spoken_area_and_place_are_interpreted_by_the_llm(message, area, city):
    info = _extract(message)
    assert _num(info.get("area")) == area, (message, info)
    assert info.get("region") == city, (message, info)


@pytest.mark.parametrize(
    "message",
    ["pode ser em qualquer momento", "quero ver os detalhes no terreno", "tem vaga na garagem?", "sim, por favor"],
)
def test_words_that_are_not_places_do_not_become_region(message):
    """O regex antigo gravava 'qualquer momento', 'terreno', 'garagem' como região."""
    assert not _extract(message).get("region"), message


def test_spoken_budget_is_interpreted():
    info = _extract("uns quinze mil por mês")
    assert _num(info.get("budget")) == 15000, info


def test_visit_intent_in_free_words_is_accepted_with_a_real_quote():
    from service.validation import validate_router_output

    message = "eu queria dar uma olhada no local pessoalmente, sabe"
    raw = _route(message, favorite_property=_SHOWN[0]["title"])
    out = validate_router_output(raw, {"properties": _SHOWN}, message=message)
    assert out["memory_updates"].get("visit_interest") is True, raw


def test_human_request_mentioning_an_option_is_not_turned_into_a_list():
    result = _route("quero falar com o corretor sobre essa opção", favorite_property=_SHOWN[0]["title"])
    assert result["tool"] == "request_human", result


_FICHA = {
    "id": "x", "title": "Sala comercial - Centro", "region": "Centro", "area_util": 45, "vagas": 1,
    "andar": 7, "elevadores": 2, "entrega": "imediata", "price_text": "R$ 3 mil/mês", "mode": "rent",
    "disponibilidade": "reservado", "images": [],
}


def _detail_reply(question: str, **prop_overrides) -> str:
    prop = {**_FICHA, **prop_overrides}
    return llm.generate_reply(
        message=question,
        canned_response=f"{prop['title']} — {prop['region']}, {prop['area_util']} m², {prop['price_text']}.",
        lead_info={}, properties=[prop], api_key=API_KEY, last_tool="property_detail",
    ).lower()


def test_any_detail_in_the_sheet_is_answerable_without_keyword_mapping():
    reply = _detail_reply("em que andar fica e tem elevador?")
    assert "7" in reply and ("elevador" in reply or "2" in reply), reply


def test_detail_missing_from_the_sheet_is_not_invented():
    reply = _detail_reply("quanto é o IPTU?")
    assert "corretor" in reply or "confirm" in reply, reply


def test_reserved_for_whom_is_not_invented():
    reply = _detail_reply("está reservado pra quem?")
    assert "corretor" in reply or "não" in reply, reply
