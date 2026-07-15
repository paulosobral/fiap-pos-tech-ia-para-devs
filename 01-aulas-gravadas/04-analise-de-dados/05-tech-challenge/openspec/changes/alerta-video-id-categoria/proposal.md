## Why

Os alertas de vídeo no feed unificado (ex.: "Cotovelo direito irregular entre 5.2s e 6.4s") não têm nenhuma ligação explícita com a imagem do momento correspondente na galeria da aba Vídeo. Como a galeria agrupa por articulação e reordena por gravidade, enquanto o feed lista os alertas em ordem cronológica, o usuário não consegue casar um alerta com a foto que o ilustra. Além disso, o texto do alerta não indica de forma rápida a região do corpo afetada.

## What Changes

- **ID curto e único por evento de vídeo** (ex.: `#V01`, `#V02`, ...), atribuído de forma determinística no `analyze`. O mesmo ID aparece no texto do alerta E como legenda da imagem correspondente na galeria — permitindo casar alerta ↔ foto mesmo com ordens de exibição diferentes (feed cronológico vs. galeria por gravidade).
- **Categoria/região do corpo no alerta**: cada evento é classificado numa região (Cabeça, Braços, Tronco, Pernas, Corpo para velocidade, Zona de risco para zona) e essa categoria entra no texto do alerta.
- A descrição do alerta passa de "Cotovelo direito irregular entre 5.2s e 6.4s." para "#V03 [Braço] Cotovelo direito irregular entre 5.2s e 6.4s.".
- A legenda de cada foto na galeria passa a incluir o mesmo ID (ex.: "#V03 — 5.2s a 6.4s").

## Capabilities

### New Capabilities
(nenhuma)

### Modified Capabilities
- `video-motion-analysis`: os requirements de geração de alerta por evento e do relatório visual passam a incluir um identificador curto e único por evento (compartilhado entre o alerta e a imagem na galeria) e a categoria/região do corpo no texto do alerta.

## Impact

- **Código alterado**: `video/analysis.py` (schema do evento ganha `event_id`; novo mapa articulação→categoria e helper `event_category`; `_event_description` inclui ID + categoria; IDs atribuídos deterministicamente no `analyze`); `app.py` (legenda da galeria mostra o `event_id`).
- **Sem mudança** em `alerts/feed.py` — o ID e a categoria vão embutidos no texto `description` do alerta, então o dataclass `Alert` e o feed compartilhado das outras 3 abas não mudam.
- **Testes**: novos para atribuição de ID (único, determinístico, presente em todos os eventos), mapeamento de categoria por articulação/tipo, e o formato da descrição do alerta.
- **Sem impacto** em `video/pose.py`, `video/draw.py`, nas outras 3 abas, nem no módulo `anomaly/`.
- **Sem dependências novas.**
