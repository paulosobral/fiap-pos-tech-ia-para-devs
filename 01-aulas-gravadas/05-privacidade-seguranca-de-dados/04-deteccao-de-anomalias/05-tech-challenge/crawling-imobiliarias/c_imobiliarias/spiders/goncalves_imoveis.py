"""Spider para www.goncalvesimoveis.com.br (plataforma ImoView/UniSoft).

O site é uma SPA: a listagem de imóveis não vem no HTML, é carregada via POST
AJAX para o próprio ImovelController. Em vez de renderizar JS, falamos
diretamente com esses dois endpoints (descobertos nos assets públicos
assets/js/busca/index.js e assets/js/busca/busca.js do próprio site):

    POST /retornar-imoveis-disponiveis   -> página de resultados da busca
    POST /retornar-imoveis-codigo        -> ficha completa de 1 imóvel

Filtro usado: cidade = São Paulo (codigocidade=3 no catálogo de cidades do
site) e finalidade = aluguel (1) ou venda (2) — "Mais Filtros" do site.

Nota: a Gonçalves Imóveis atua principalmente em São Bernardo do Campo/ABC;
o estoque na cidade de São Paulo (capital) é pequeno, em especial para
aluguel (poucas unidades). Isso é uma característica real do site, não um
bug do spider.
"""

from __future__ import annotations

import re
import unicodedata

import scrapy

BASE_URL = "https://www.goncalvesimoveis.com.br/"

# finalidade: 1 = aluguel (rent), 2 = venda (purchase) — confirmado comparando
# os valores de "valor" retornados (ex.: aluguel R$ 4.500 x venda R$ 1.437.000).
FINALIDADE_MODE = {1: "rent", 2: "purchase"}

# codigocidade=3 == "São Paulo" em /retornar-cidades-disponiveis.
CODIGO_CIDADE_SAO_PAULO = 3

TYPE_SLUGS = {
    "apartamento": "apartamento",
    "apartamento área privativa": "apartamento",
    "apartamento cobertura duplex": "cobertura",
    "casa": "casa",
    "chácara": "chacara",
    "flat": "flat",
    "galpão": "galpao",
    "prédio comercial": "predio_comercial",
    "sala": "sala_comercial",
    "sobrado": "sobrado",
    "terreno": "terreno",
    "conjunto comercial": "conjunto",
    "laje corporativa": "laje",
    "escritório": "escritorio",
}

SITUACAO_SLUGS = {
    "vago/disponível": "disponível",
    "disponível": "disponível",
    "reservado": "reservado",
    "vendido": "vendido",
    "alugado": "alugado",
}


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return text or "imovel"


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


class GoncalvesImoveisSpider(scrapy.Spider):
    """Imóveis de compra e aluguel na cidade de São Paulo (goncalvesimoveis.com.br)."""

    name = "goncalves_imoveis"
    allowed_domains = ["goncalvesimoveis.com.br"]

    custom_settings = {
        # Garante que o header AJAX exigido pelo backend vá em toda request.
        "DEFAULT_REQUEST_HEADERS": {
            "X-Requested-With": "XMLHttpRequest",
        },
    }

    def __init__(self, max_per_mode: int = 15, codigo_cidade: int = CODIGO_CIDADE_SAO_PAULO, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_per_mode = int(max_per_mode)
        self.codigo_cidade = int(codigo_cidade)

    async def start(self):
        # Scrapy >= 2.13 chama spider.start() (async) em vez de fazer bridge
        # automático para start_requests(); delegamos explicitamente para
        # manter start_requests() como ponto único da lógica de entrada.
        for request in self.start_requests():
            yield request

    def start_requests(self):
        for finalidade in FINALIDADE_MODE:
            yield self._search_request(finalidade=finalidade, pagina=1)

    def _search_request(self, finalidade: int, pagina: int) -> scrapy.FormRequest:
        formdata = {
            "finalidade": str(finalidade),
            "codigocidade": str(self.codigo_cidade),
            "numeroregistros": str(self.max_per_mode),
            "numeropagina": str(pagina),
        }
        return scrapy.FormRequest(
            url=BASE_URL + "retornar-imoveis-disponiveis",
            formdata=formdata,
            callback=self.parse_search,
            cb_kwargs={"finalidade": finalidade, "pagina": pagina},
        )

    def parse_search(self, response: scrapy.http.Response, finalidade: int, pagina: int):
        data = response.json()
        listagem = data.get("lista") or []
        quantidade = data.get("quantidade", 0)
        self.logger.info(
            "finalidade=%s (%s) pagina=%s: %d/%d imóveis",
            finalidade, FINALIDADE_MODE[finalidade], pagina, len(listagem), quantidade,
        )

        for listing in listagem:
            yield scrapy.FormRequest(
                url=BASE_URL + "retornar-imoveis-codigo",
                formdata={"codigo": str(listing["codigo"]), "pagina": "1"},
                callback=self.parse_detail,
                cb_kwargs={"listing": listing, "finalidade": finalidade},
            )

        already_fetched = pagina * self.max_per_mode
        if listagem and already_fetched < min(quantidade, self.max_per_mode):
            yield self._search_request(finalidade=finalidade, pagina=pagina + 1)

    def parse_detail(self, response: scrapy.http.Response, listing: dict, finalidade: int):
        data = response.json()
        detail = (data.get("lista") or [{}])[0]
        if not detail:
            self.logger.warning("sem detalhe para codigo=%s", listing.get("codigo"))
            return
        yield self._build_property(listing, detail, finalidade)

    def _build_property(self, listing: dict, detail: dict, finalidade: int) -> dict:
        codigo = detail.get("codigo") or listing.get("codigo")
        mode = FINALIDADE_MODE[finalidade]

        tipo_raw = (detail.get("tipo") or listing.get("tipo") or "").strip()
        type_slug = TYPE_SLUGS.get(tipo_raw.lower(), _slugify(tipo_raw))

        area_util = round(_to_float(detail.get("areainternatratado")) / 100, 2)
        if not area_util:
            area_util = round(_to_float(detail.get("areaprincipaltratado")) / 100, 2)
        area_externa = round(_to_float(detail.get("areaexternatratado")) / 100, 2)
        area_bruta = round(area_util + area_externa, 2) if area_externa else area_util

        vagas = _to_int(detail.get("numerovagastratado"))
        andar = _to_int(detail.get("numeroandar"))
        elevadores = _to_int(detail.get("numeroelevadortratado"))

        valor_condominio = _to_float(detail.get("valorcondominio"))
        condominio_m2 = round(valor_condominio / area_util) if area_util else 0

        price = _to_int(detail.get("valortratado") or listing.get("valortratado"))
        if mode == "rent":
            price_text = f"R$ {price / 1000:.1f} mil/mês"
        else:
            price_text = f"R$ {price / 1_000_000:.1f} milhão"

        situacao = (detail.get("situacao") or "").strip()
        disponibilidade = SITUACAO_SLUGS.get(situacao.lower(), _slugify(situacao) or "disponível")

        tipolancamento = (detail.get("tipolancamentonome") or "").strip()
        entrega = tipolancamento or "imediata"

        bairro = detail.get("bairro") or listing.get("bairro") or ""
        cidade = detail.get("cidade") or listing.get("cidade") or ""

        description = (detail.get("descricao") or "").strip() or detail.get("metadescription", "")

        url_amigavel = detail.get("url_amigavel") or listing.get("url_amigavel") or ""
        source_url = f"{BASE_URL}imovel/{url_amigavel}/{codigo}" if url_amigavel else BASE_URL

        images = [
            foto["url"]
            for foto in (detail.get("fotos") or listing.get("fotos") or [])
            if isinstance(foto, dict) and foto.get("url")
        ]
        if not images:
            main_photo = detail.get("urlfotoprincipal") or listing.get("urlfotoprincipal")
            if main_photo:
                images = [main_photo]

        return {
            "id": f"gi-{codigo}",
            "title": detail.get("titulo") or listing.get("titulo") or "",
            "type": type_slug,
            "class": None,
            "region": bairro,
            "regions": [],
            "corredor": cidade,
            "area_util": area_util,
            "area_bruta": area_bruta,
            "laje": "laje" in tipo_raw.lower(),
            "condominio_m2": condominio_m2,
            "vagas": vagas,
            "andar": andar,
            "elevadores": elevadores,
            "entrega": entrega,
            "cep": detail.get("cep") or "",
            "disponibilidade": disponibilidade,
            "mode": mode,
            "price": price,
            "price_text": price_text,
            "description": description,
            "source_url": source_url,
            "images": images,
        }
