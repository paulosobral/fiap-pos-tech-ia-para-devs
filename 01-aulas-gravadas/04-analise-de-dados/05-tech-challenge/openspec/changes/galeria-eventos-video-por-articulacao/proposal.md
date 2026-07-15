## Why

Após o redesenho da aba Vídeo (`redesenho-aba-video-postura`), o relatório visual lista TODOS os eventos irregulares um a um, cada um com sua imagem anotada. Em um vídeo real isso explodiu: o usuário viu "500 evento(s) irregular(es) detectado(s)" — 500 imagens renderizadas em sequência travam/inundam a página, e não há como navegar por articulação. O conteúdo retornado fica enorme e não-navegável.

## What Changes

- **Agrupar os eventos por articulação/tipo na saída**, em vez de uma lista plana única. Cada articulação (Cotovelo direito, Joelho esquerdo, ...), mais Velocidade e Zona crítica, vira uma seção própria.
- Cada seção é um **painel colapsável** (`st.expander`) com título "Articulação — N eventos", fechado por padrão (a página abre limpa; o usuário expande só o que quer ver). As seções são ordenadas da mais afetada (mais eventos) para a menos.
- Dentro de cada seção, mostrar apenas os **top 10 eventos mais graves** daquela articulação (maior |z-score|), em uma **grade de colunas** (thumbnails lado a lado, legenda com o intervalo de tempo). Quando houver mais que 10, exibir "mostrando 10 de N".
- O resumo no topo (número total de eventos + articulação mais afetada) é mantido.
- **BREAKING** (relativo à saída anterior da aba, não a specs externas): a apresentação dos eventos muda de "lista plana de todos os eventos" para "seções colapsáveis por articulação, top-N por seção em grade".

## Capabilities

### New Capabilities
(nenhuma)

### Modified Capabilities
- `video-motion-analysis`: o requirement do relatório visual passa a especificar que os eventos são agrupados por articulação em seções colapsáveis, com os top-N mais graves por seção exibidos em galeria, em vez de uma lista plana de todos os eventos.

## Impact

- **Código alterado**: `app.py` (aba Vídeo — renderização do relatório: expanders por articulação + grade de colunas + limite top-N); `video/analysis.py` (novo helper de agrupamento/seleção top-N, testável, para manter `app.py` fino).
- **Testes**: novos para o helper de agrupamento (agrupa por articulação, ordena por gravidade, aplica top-N, ordena seções por contagem).
- **Sem impacto** em `video/pose.py`, `video/draw.py`, nas outras 3 abas, no feed de alertas (continua 1 alerta por evento — o agrupamento é só de apresentação, não muda quantos alertas são gerados), nem no módulo `anomaly/`.
- **Sem dependências novas** — `st.expander`/`st.columns` são nativos do Streamlit.
