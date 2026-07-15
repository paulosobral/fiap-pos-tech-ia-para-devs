## Why

A aba Vídeo hoje é confusa e pouco útil na prática (feedback direto do usuário testando com vídeos reais):

- Só rastreia **um** ponto do corpo (ângulo do cotovelo direito + velocidade do punho direito). Se a irregularidade postural está em qualquer outra parte (joelho, quadril, tronco, cabeça, lado esquerdo), o sistema não vê nada — o usuário subiu um vídeo com postura visivelmente irregular e não pegou.
- O resultado é uma lista de texto crua com jargão (`valor=141.63 (|z-score| > 2.29...)`), sem nenhuma imagem, sem mostrar ONDE no corpo nem COMO era a postura. O usuário não entende nada do que foi feito.
- A detecção de zona crítica gera **um alerta por frame** (centenas de linhas repetidas: `t=0.00s`, `t=0.03s`, `t=0.07s`...), poluindo o feed unificado.
- Os controles (slider de sensibilidade, sliders de zona) não têm nenhum feedback visual — o usuário mexe e não vê o efeito, não entende o que os números 0-1 da zona significam no quadro.

## What Changes

- **Rastreamento multi-articulação** (substitui o ponto único): o sistema passa a calcular ângulos de várias articulações por frame — cotovelos (E/D), joelhos (E/D), quadris/tronco (E/D) e pescoço/cabeça — e aponta automaticamente qual estava irregular, em vez de só cotovelo direito.
- **Resultado visual em vez de lista crua**: para cada momento irregular, o sistema mostra a **imagem daquele frame com o esqueleto de pose desenhado por cima**, destacando a articulação problemática, mais um rótulo em linguagem simples (ex.: "Joelho direito irregular, 5.2s–6.4s"). Substitui a tabela/lista de texto atual.
- **Agrupamento em eventos**: frames irregulares consecutivos da mesma articulação viram um único "evento" (com início, fim e o frame-pior do grupo), em vez de um alerta por frame. Gera 1 `Alert` por evento no feed unificado — corrige o flood.
- **Resumo no topo**: contadores e diagnóstico geral (ex.: "N eventos irregulares; articulação mais afetada: joelho direito").
- **Controles auto-explicativos**:
  - Sensibilidade: legenda dinâmica mostrando, no valor atual do slider, aproximadamente quantos % dos frames do vídeo carregado seriam marcados como irregulares.
  - Zona de risco: desativada por padrão (checkbox); quando ligada, desenha uma **prévia do retângulo da zona sobre o primeiro frame do vídeo**, para o usuário ver exatamente onde a zona cai antes de processar. Alertas de zona também agrupados em eventos (não 1 por frame).
- **BREAKING** (relativo ao comportamento anterior da aba, não a specs externas): a saída da aba Vídeo muda de "lista de texto por frame" para "eventos visuais com esqueleto"; o rastreamento deixa de ser de ponto único.

## Capabilities

### New Capabilities
(nenhuma)

### Modified Capabilities
- `video-motion-analysis`: os requirements de extração de série, detecção de anomalia postural, detecção de zona crítica e relatório de desvios são reformulados para: (1) múltiplas articulações incluindo cabeça, (2) agrupamento em eventos em vez de por-frame, (3) saída visual (frame anotado com esqueleto) em vez de lista de texto, (4) zona crítica opcional com prévia visual.

## Impact

- **Código alterado**: `video/pose.py` (múltiplas articulações + guardar keypoints por frame), `video/analysis.py` (agrupamento em eventos, sensibilidade multi-articulação, zona opcional), `app.py` (aba Vídeo: resumo, frames anotados, controles com feedback visual), novo módulo `video/draw.py` (desenho do esqueleto sobre o frame via `cv2`).
- **Testes**: novos/ajustados para cálculo multi-articulação, agrupamento em eventos, e a lógica pura de desenho (mockando frames/keypoints).
- **Sem impacto** nas outras 3 abas (Sinais Vitais, Áudio, Prescrições) nem no módulo compartilhado `anomaly/zscore.py` / `alerts/feed.py` (a interface de `add_alert` continua a mesma; só muda quantos alertas a aba Vídeo gera).
- **Dependências**: nenhuma nova — `cv2` (opencv-python) e `numpy` já estão no projeto.
