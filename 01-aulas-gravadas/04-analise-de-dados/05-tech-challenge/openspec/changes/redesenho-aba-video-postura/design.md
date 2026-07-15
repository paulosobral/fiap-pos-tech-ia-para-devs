## Context

A aba Vídeo foi implementada na change `monitoramento-multimodal-pacientes` (Task 4) e ajustada por `sugestao-automatica-sensibilidade-video` e `fix-video-cache-replay-error`. Estrutura atual:

- `video/pose.py`: `extract_frame_series(model, frames, fps, on_frame_processed)` roda YOLOv8-pose frame a frame e retorna, por frame, um dict `{timestamp_s, has_pose, angle, velocity, detections}` — onde `angle` é só o ângulo do cotovelo direito (`RIGHT_ELBOW_TRIPLET`) e `velocity` é o deslocamento do punho direito. Os keypoints brutos (x,y de cada frame) são descartados após o cálculo.
- `video/analysis.py`: `analyze(frame_series, threshold, window, zone, ...)` aplica `anomaly.zscore.detect_anomalies` sobre as séries de ângulo e velocidade, gera um `Alert` por frame anômalo e um por interseção de zona, e monta um `deviation_report` (lista de dicts por alerta). Também tem `suggest_sensitivity_threshold`.
- `app.py` aba Vídeo: extrai a pose no upload (cacheado), mostra sliders (sensibilidade + zona x/y), botão "Processar vídeo", e renderiza o `deviation_report` como `st.dataframe` + alertas como `st.warning`.

O feedback do usuário (testando com vídeos reais) motivou este redesenho: ponto único não pega irregularidade em outras partes do corpo; a saída em texto é incompreensível; a zona gera um alerta por frame; os controles não têm feedback visual.

Decisões de escopo fechadas com o usuário no brainstorming: rastrear **corpo todo incluindo cabeça** (sistema aponta a articulação afetada); saída = **frames-chave anotados com esqueleto** (não vídeo re-renderizado, não só gráfico); agrupar frames consecutivos num **evento com o frame-pior**; zona crítica **mantida mas agrupada em eventos e desativada por padrão**; controles com **feedback visual** (prévia da zona sobre o frame, legenda dinâmica da sensibilidade).

## Goals / Non-Goals

**Goals:**
- Detectar postura/movimento irregular em múltiplas articulações do corpo (cotovelos, joelhos, quadris/tronco, pescoço/cabeça — ambos os lados), apontando qual foi afetada.
- Apresentar cada momento irregular como um evento agrupado com a imagem do frame-pior e o esqueleto desenhado, em linguagem simples — não uma lista de números.
- Tornar os controles auto-explicativos com feedback visual no próprio vídeo carregado.
- Reduzir o ruído no feed unificado: 1 `Alert` por evento, não por frame.

**Non-Goals:**
- Não re-renderiza o vídeo inteiro com esqueleto (custo/tempo alto) — só os frames-chave dos eventos.
- Não adiciona seleção manual de articulação pelo usuário (o sistema decide/aponta) — foi a opção "corpo todo, sistema decide".
- Não muda a interface de `anomaly/zscore.py` nem de `alerts/feed.py`.
- Não toca nas outras 3 abas.
- Não faz avaliação clínica de "certo/errado" absoluto — a detecção continua sendo estatística (desvio do padrão do próprio vídeo via rolling z-score), agora por articulação; "irregular" = "estatisticamente fora do padrão deste vídeo", documentado como tal.

## Decisions

### D1: Articulações rastreadas e como (`video/pose.py`)
Calcular, por frame, um dicionário de ângulos nomeados a partir dos keypoints COCO já disponíveis:

| Articulação (chave) | Triplet (a, vértice, b) |
|---|---|
| `cotovelo_esquerdo` | left_shoulder, left_elbow, left_wrist |
| `cotovelo_direito` | right_shoulder, right_elbow, right_wrist |
| `joelho_esquerdo` | left_hip, left_knee, left_ankle |
| `joelho_direito` | right_hip, right_knee, right_ankle |
| `quadril_esquerdo` (tronco) | left_shoulder, left_hip, left_knee |
| `quadril_direito` (tronco) | right_shoulder, right_hip, right_knee |
| `pescoco` (cabeça) | ombro-médio (base), nariz, ... |

Para o pescoço/cabeça: usar o ponto médio dos dois ombros como vértice e medir o ângulo entre a vertical do tronco (ombro-médio → quadril-médio) e a direção ombro-médio → nariz — captura cabeça caída/inclinada. Se keypoints necessários faltarem (não detectados, `_MISSING_KEYPOINT`), aquela articulação fica `None` naquele frame (não interrompe as demais).

`compute_joint_angle` já existe e é genérico (recebe qualquer triplet) — reusar. Adicionar só um `JOINTS` (dict nome→triplet) e um helper para o ângulo de pescoço (que precisa de pontos médios calculados, não é um triplet puro de keypoints nomeados).

Velocidade: manter o conceito, mas generalizar para a velocidade do **centro de massa aproximado** (média dos keypoints detectados) em vez de só o punho direito — assim captura movimento brusco geral do corpo, não só do braço. (Alternativa: velocidade por articulação — rejeitada por multiplicar séries sem ganho claro; um único sinal de "movimento brusco global" é suficiente e mais interpretável.)

`extract_frame_series` passa a retornar, por frame: `{timestamp_s, has_pose, angles: {nome: float|None}, velocity: float|None, keypoints_xy: list|None, detections}`. **Novo**: `keypoints_xy` (os 17 pontos do frame) é guardado, pois é necessário para desenhar o esqueleto depois. Custo de memória trivial (17 pontos × ~1000 frames).

### D2: Detecção por articulação + agrupamento em eventos (`video/analysis.py`)
Para cada articulação, montar a série temporal dos seus ângulos e aplicar `detect_anomalies` (reuso). Idem para a série de velocidade global. Um frame é "anômalo para a articulação X" se o z-score do ângulo de X excede o threshold.

**Agrupamento**: para cada articulação (e para velocidade, e para zona), varrer os frames em ordem e agrupar índices anômalos **consecutivos** (tolerância de gap configurável pequena, ex.: 1-2 frames, para não fragmentar por um frame isolado sem pose) num evento:
```
{
  tipo: "postura" | "velocidade" | "zona_critica",
  articulacao: str | None,       # ex.: "joelho_direito"; None p/ velocidade/zona
  t_inicio: float, t_fim: float,
  frame_index_pior: int,         # frame de maior |z-score| (ou maior interseção p/ zona)
  valor_pior: float,
  z_pior: float,
}
```
Gerar **1 `Alert` por evento** (origem "Vídeo", descrição em linguagem simples: "Joelho direito irregular entre 5.2s e 6.4s"). O `deviation_report` passa a ser a lista de eventos, não de frames.

Zona crítica: mesma lógica de agrupamento; só é executada se `zone is not None` (mantido) — e a UI passa `zone=None` por padrão (checkbox desligado), então nenhum evento de zona é gerado a menos que o usuário ligue.

### D3: Desenho do esqueleto (`video/draw.py`, novo)
`draw_pose_on_frame(frame_bgr, keypoints_xy, highlight_joint=None) -> frame_bgr` — desenha as conexões do esqueleto COCO (pares de keypoints ligados por linhas) sobre uma cópia do frame, pontos como círculos; se `highlight_joint` for dado, desenha a(s) linha(s)/ponto(s) daquela articulação em vermelho e o resto em cor neutra. Usa só `cv2` (linhas/círculos) — nenhuma dependência nova. Retorna a imagem BGR (Streamlit exibe com `channels="BGR"` ou converte).

`draw_zone_on_frame(frame_bgr, zone_rel) -> frame_bgr` — desenha o retângulo da zona (coordenadas relativas 0-1 convertidas para pixels) sobre o frame, para a prévia da zona na UI.

Módulo separado (não em `pose.py` nem `analysis.py`) porque é responsabilidade distinta (renderização), testável isoladamente com um frame sintético + keypoints mock, e sem dependência de `ultralytics`.

### D4: UI da aba Vídeo (`app.py`)
- Após extração (no upload, cacheado): mostrar **resumo** — total de eventos, e a articulação com mais eventos ("articulação mais afetada").
- **Sensibilidade**: slider + `st.caption` dinâmico calculado do vídeo real ("Neste valor, ~X% dos frames seriam marcados como irregulares") — reaproveita a distribuição de z-score já computável.
- **Zona de risco**: `st.checkbox("Analisar zona de risco", value=False)`. Só quando marcado: mostrar os sliders x/y **e** uma prévia — `st.image(draw_zone_on_frame(primeiro_frame, zona_atual))` — atualizada conforme os sliders. `analyze` recebe `zone=zona` só nesse caso; senão `zone=None`.
- Botão "Processar vídeo" → roda `analyze` → para cada evento: `st.image(draw_pose_on_frame(frame_do_evento, keypoints_do_evento, highlight=articulacao))` + legenda simples. Sem `st.dataframe` de texto cru.
- Manter o cache de extração de pose e o padrão de botão (não reprocessar em rerun) já estabelecidos.

## Risks / Trade-offs

- **[Trade-off] Guardar keypoints de todos os frames aumenta a memória da sessão** → trivial (17 pontos × 2 floats × ~1000 frames ≈ dezenas de KB); aceito.
- **[Risco] Ângulo de pescoço/cabeça depende de nariz + ombros + quadris detectados; em enquadramentos ruins pode faltar** → degradação graciosa (articulação `None` no frame, não entra na série), consistente com o tratamento atual de keypoint faltante.
- **[Risco] Velocidade de "centro de massa" muda a semântica do sinal atual (punho direito)** → aceito e documentado; o objetivo do redesenho é justamente sair do foco em um único ponto do braço.
- **[Trade-off] Rolling z-score continua sendo "desvio do padrão do próprio vídeo", não um padrão clínico absoluto** → mantido conscientemente (mesma técnica compartilhada do projeto); a melhoria é cobrir o corpo todo e apresentar visualmente, não trocar o método estatístico.
- **[Risco] Desenhar esqueleto exige o frame original (imagem), que hoje é decodificado e usado só na extração** → o frame do índice-pior de cada evento precisa estar acessível na hora de renderizar; a extração cacheada já tem os frames decodificados — passar/guardar os frames necessários (ou re-decodificar sob demanda a partir dos bytes cacheados) é detalhe de implementação a resolver mantendo o cache existente.

## Migration Plan

Não aplicável (mudança de comportamento de UI em código já implementado, sem dados persistidos). O modelo YOLOv8-pose e o cache continuam iguais.

## Open Questions

Nenhuma pendente — escopo fechado no brainstorming.
