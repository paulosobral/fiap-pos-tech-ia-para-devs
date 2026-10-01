# crawling-imobiliarias

Spider [Scrapy](https://github.com/scrapy/scrapy) que coleta imóveis de compra e
aluguel na cidade de São Paulo a partir do site da **Gonçalves Imóveis**
(`https://www.goncalvesimoveis.com.br/`) e grava o resultado no mesmo formato
JSON consumido pelo catálogo RAG do `agente-sdr-imobiliario`.

## Como o site funciona

O site é uma SPA (ImoView/UniSoft): a listagem e a ficha de cada imóvel não
vêm no HTML, são carregadas via POST AJAX direto no `ImovelController`. O
spider fala com esses endpoints em vez de renderizar JS:

- `POST /retornar-imoveis-disponiveis` — página de resultados da busca
  (`finalidade`: `1`=aluguel, `2`=venda; `codigocidade`: `3`=São Paulo capital)
- `POST /retornar-imoveis-codigo` — ficha completa de 1 imóvel

## Limitação real do site (não é bug do spider)

A Gonçalves Imóveis atua principalmente no ABC Paulista (São Bernardo do
Campo). O estoque filtrado para a cidade de São Paulo (capital) é:

- **Venda**: 84 imóveis disponíveis
- **Aluguel**: apenas 1 imóvel disponível

Por isso uma coleta com `max_per_mode=15` (padrão) resulta em 16 imóveis (1
aluguel + 15 venda), e não 30. O estoque também é majoritariamente
residencial (apartamentos), diferente do nicho comercial (laje
corporativa/conjunto/escritório/galpão) dos dados sintéticos gerados por
`scripts/seed_properties.py` no projeto `agente-sdr-imobiliario`.

Também foi observado que o campo de preço estruturado (`valor`/`valortratado`,
usado pelo spider) às vezes diverge do valor citado na descrição em texto
livre do próprio site — a descrição escrita pelo corretor fica desatualizada
em relação ao preço vigente. O spider usa o campo estruturado por ser a fonte
autoritativa.

## Uso

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# coleta padrão (até 15 imóveis por finalidade, cidade = São Paulo capital)
.venv/bin/scrapy crawl goncalves_imoveis

# parâmetros customizáveis
.venv/bin/scrapy crawl goncalves_imoveis -a max_per_mode=30 -a codigo_cidade=3
```

Resultado em `output/properties.json`. Para gravar também diretamente no
arquivo consumido pelo `agente-sdr-imobiliario`, defina `POPULATE_TARGET_PATH`:

```bash
.venv/bin/scrapy crawl goncalves_imoveis \
  -s POPULATE_TARGET_PATH=../agente-sdr-imobiliario/apps/conversation-router/data/properties.json
```

## Atenção: `properties.json` é artefato de build

Em `agente-sdr-imobiliario`, `apps/conversation-router/data/properties.json`
é **gitignored** e é **regenerado em toda execução de `start.sh`** pela fase 2
(`python scripts/seed_properties.py > apps/conversation-router/data/properties.json`),
antes dos gates de teste, e depois carregado na tabela DynamoDB
`sdr-properties` pela fase 5d.5 (`scripts/load_properties_dynamodb.py`).

Ou seja: popular o arquivo com os dados deste spider é válido para uso/teste
local imediato, mas **qualquer novo `start.sh` vai sobrescrever com os 120
imóveis sintéticos de `seed_properties.py`**, a menos que o pipeline de build
seja ajustado para chamar este spider em vez do gerador sintético (não feito
aqui — decisão que cabe ao dono do projeto).
