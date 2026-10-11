# crawling-imobiliarias

Spider [Scrapy](https://github.com/scrapy/scrapy) que coleta imóveis de compra e
aluguel a partir do site da **Gonçalves Imóveis**
(`https://www.goncalvesimoveis.com.br/`) e grava o resultado no mesmo formato
JSON consumido pelo catálogo RAG do `agente-sdr-imobiliario`.

## Como o site funciona

O site é uma SPA (ImoView/UniSoft): a listagem e a ficha de cada imóvel não
vêm no HTML, são carregadas via POST AJAX direto no `ImovelController`. O
spider fala com esses endpoints em vez de renderizar JS:

- `POST /retornar-imoveis-disponiveis` — página de resultados da busca
  (`finalidade`: `1`=aluguel, `2`=venda; `codigocidade`: ver tabela abaixo)
- `POST /retornar-imoveis-codigo` — ficha completa de 1 imóvel

Códigos de cidade usados neste projeto (via `/retornar-cidades-disponiveis`):

| Cidade                  | `codigocidade` |
|--------------------------|----------------|
| São Paulo (capital)      | `3`            |
| São Bernardo do Campo    | `2`            |
| Santo André              | `4`            |
| São Caetano do Sul       | `7`            |

## Limitação real do site (não é bug do spider)

A Gonçalves Imóveis atua principalmente no ABC Paulista (São Bernardo do
Campo). O estoque disponível na cidade de São Paulo (capital) é pequeno:

- **Venda**: 84 imóveis disponíveis
- **Aluguel**: apenas 1 imóvel disponível

Ou seja, filtrando só por São Paulo capital o teto real é **85 imóveis**, não
dá pra chegar a algo como 250 só com esse filtro. Por isso o dataset atual
combina São Paulo (85) com as 3 cidades do ABC onde a imobiliária realmente
tem grande estoque (Santo André, São Bernardo do Campo e São Caetano do Sul),
28 por finalidade em cada uma, totalizando **253 imóveis únicos**:

| Cidade                | aluguel | venda | total |
|-----------------------|---------|-------|-------|
| São Paulo             | 1       | 84    | 85    |
| Santo André           | 28      | 28    | 56    |
| São Bernardo do Campo | 28      | 28    | 56    |
| São Caetano do Sul    | 28      | 28    | 56    |
| **Total**             |         |       | **253** |

O estoque também é majoritariamente residencial (apartamentos), diferente do
nicho comercial (laje corporativa/conjunto/escritório/galpão) dos dados
sintéticos gerados por `scripts/seed_properties.py` no projeto
`agente-sdr-imobiliario`.

Também foi observado que o campo de preço estruturado (`valor`/`valortratado`,
usado pelo spider) às vezes diverge do valor citado na descrição em texto
livre do próprio site — a descrição escrita pelo corretor fica desatualizada
em relação ao preço vigente. O spider usa o campo estruturado por ser a fonte
autoritativa.

## Instalação

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Rodando para uma cidade

```bash
# coleta padrão (até 15 imóveis por finalidade, cidade = São Paulo capital)
.venv/bin/scrapy crawl goncalves_imoveis

# parâmetros customizáveis (ex.: São Bernardo do Campo, até 28 por finalidade)
.venv/bin/scrapy crawl goncalves_imoveis -a max_per_mode=28 -a codigo_cidade=2 \
  -s OUTPUT_PATH=output/abc/sao_bernardo.json
```

Cada execução grava em `output/properties.json` por padrão (ou em
`OUTPUT_PATH`, se sobrescrito via `-s`). Rode uma cidade por vez — não em
paralelo — pra não multiplicar a carga sobre o site (cada processo já aplica
seu próprio rate limit educado via `DOWNLOAD_DELAY`/`AUTOTHROTTLE`).

## Rodando para várias cidades e mesclando

Reproduz o dataset de 253 imóveis (São Paulo + ABC):

```bash
.venv/bin/scrapy crawl goncalves_imoveis -a max_per_mode=150 -a codigo_cidade=3 \
  -s OUTPUT_PATH=output/properties.json          # São Paulo: pega tudo (teto real = 85)

mkdir -p output/abc
.venv/bin/scrapy crawl goncalves_imoveis -a max_per_mode=28 -a codigo_cidade=4 \
  -s OUTPUT_PATH=output/abc/santo_andre.json
.venv/bin/scrapy crawl goncalves_imoveis -a max_per_mode=28 -a codigo_cidade=2 \
  -s OUTPUT_PATH=output/abc/sao_bernardo.json
.venv/bin/scrapy crawl goncalves_imoveis -a max_per_mode=28 -a codigo_cidade=7 \
  -s OUTPUT_PATH=output/abc/sao_caetano.json

# mescla tudo, removendo duplicados por "id"
.venv/bin/python3 scripts/merge_properties.py \
  output/properties.json output/abc/santo_andre.json output/abc/sao_bernardo.json output/abc/sao_caetano.json \
  -o output/properties_merged.json
```

Depois do merge, **remova as fotos genéricas** antes de entregar ao bot (ver
seção abaixo) e só então popule o arquivo consumido pelo `agente-sdr-imobiliario`:

```bash
.venv/bin/python3 scripts/drop_generic_images.py output/properties_merged.json \
  -o output/properties_clean.json
cp output/properties_clean.json \
  ../agente-sdr-imobiliario/apps/conversation-router/data/properties.json
```

## Fotos genéricas da imobiliária (por que existe `drop_generic_images.py`)

O CMS anexa a ~1/3 dos anúncios uma **colagem das fachadas da própria Gonçalves
Imóveis** — às vezes como a única foto (imóveis sem foto), às vezes no meio/fim
das fotos reais — e, sem foto nenhuma, devolve o ícone cinza `house1.png`. Sem
tratamento, o bot mandava "a foto do imóvel" e chegava o prédio da imobiliária.

- O ícone é descartado pelo spider (`PLACEHOLDER_IMAGE_MARKERS`).
- A colagem **não dá pra filtrar por URL** (é servida de `/Imoveis/<codigo>/` como
  qualquer foto, em ~9 formatos/recortes). `scripts/drop_generic_images.py` compara a
  miniatura de cada foto (cinza 32x32, sem margens brancas) com
  `scripts/generic_image_signature.json` por correlação; o limiar 0,65 fica numa
  lacuna medida: colagem ≥ 0,73, foto real mais parecida 0,52. Foto que não baixa é
  mantida; 404/410 é removida; imóvel que fica sem foto sai com `images: []`
  (o bot passa a dizer que não há foto em vez de mandar a errada).
- Resultados ficam em cache (`output/.generic_image_cache.json`): re-rodar só baixa
  as fotos novas. Se aparecer outra imagem genérica, gere uma nova assinatura.

(O pipeline também suporta `-s POPULATE_TARGET_PATH=<caminho>` para gravar
direto numa cidade só, sem mesclagem — útil só quando não há mais de um
arquivo pra combinar.)

## Atenção: `properties.json` é artefato de build

Em `agente-sdr-imobiliario`, `apps/conversation-router/data/properties.json`
é **gitignored** e é **regenerado em toda execução de `start.sh`** pela fase 2
(`python scripts/seed_properties.py > apps/conversation-router/data/properties.json`),
antes dos gates de teste, e depois carregado na tabela DynamoDB
`sdr-properties` pela fase 5d.5 (`scripts/load_properties_dynamodb.py`).

O `start.sh` só gera os imóveis sintéticos de `seed_properties.py` quando
`data/properties.json` **não existe**; um arquivo já populado com os dados deste
spider (passo acima) é preservado e é o que vai para o DynamoDB.
