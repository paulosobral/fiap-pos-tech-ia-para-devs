## 1. Extração multi-articulação (`video/pose.py`)

- [x] 1.1 Definir `JOINTS`: dict de nome de articulação → triplet de keypoints, para cotovelos (E/D), joelhos (E/D) e quadris/tronco (E/D)
- [x] 1.2 Implementar cálculo do ângulo de pescoço/cabeça a partir de pontos médios (ombro-médio como vértice, ângulo entre a vertical do tronco ombro-médio→quadril-médio e a direção ombro-médio→nariz), tratando keypoints faltantes como articulação sem valor no frame
- [x] 1.3 Generalizar a velocidade para o deslocamento do centro de massa aproximado (média dos keypoints detectados) entre frames consecutivos, em vez de só o punho direito
- [x] 1.4 Alterar `extract_frame_series` para retornar por frame `{timestamp_s, has_pose, angles: {nome: float|None}, velocity: float|None, keypoints_xy: list|None, detections}` (guardando os keypoints do frame)
- [x] 1.5 Testes: ângulos multi-articulação a partir de keypoints mock (incluindo pescoço e casos de keypoint faltante por articulação), velocidade de centro de massa, e a estrutura de retorno de `extract_frame_series` com um modelo mockado

## 2. Detecção por articulação + agrupamento em eventos (`video/analysis.py`)

- [x] 2.1 Aplicar `detect_anomalies` (reuso) por série de articulação e sobre a série de velocidade, marcando frames irregulares por articulação/velocidade
- [x] 2.2 Implementar agrupamento de índices anômalos consecutivos (com tolerância pequena de gap) em eventos `{tipo, articulacao, t_inicio, t_fim, frame_index_pior, valor_pior, z_pior}`, com o frame-pior = maior |z-score| do grupo
- [x] 2.3 Gerar 1 `Alert` por evento (origem "Vídeo", descrição em linguagem simples com articulação e intervalo), substituindo o alerta-por-frame
- [x] 2.4 Zona crítica: manter o parâmetro `zone` (default `None` = não executa) e agrupar as interseções de zona em eventos com a mesma lógica
- [x] 2.5 Substituir `deviation_report` (lista por frame) por lista de eventos; adicionar um resumo (contagem de eventos, articulação mais afetada)
- [x] 2.6 Atualizar `suggest_sensitivity_threshold` para operar sobre a distribuição combinada de z-scores de todas as articulações (não só um ângulo) e expor a estimativa de "% do vídeo marcado" no valor atual do threshold
- [x] 2.7 Testes: detecção por articulação, agrupamento em eventos (frames consecutivos → 1 evento; frame-pior correto), 1 alerta por evento, resumo (articulação mais afetada), zona agrupada em evento

## 3. Desenho do esqueleto (`video/draw.py`, novo)

- [x] 3.1 Implementar `draw_pose_on_frame(frame_bgr, keypoints_xy, highlight_joint=None)` — desenha conexões e pontos do esqueleto COCO via `cv2`, com a articulação destacada em vermelho quando informada; keypoints faltantes (0,0) não são desenhados
- [x] 3.2 Implementar `draw_zone_on_frame(frame_bgr, zone_rel)` — desenha o retângulo da zona (coords relativas → pixels) sobre o frame
- [x] 3.3 Testes: desenho não altera o frame original (opera sobre cópia), keypoints faltantes são ignorados, e o highlight afeta só a articulação alvo — usando um frame sintético (`np.zeros`) e keypoints mock, verificando que pixels mudam nas regiões esperadas

## 4. UI da aba Vídeo (`app.py`)

- [x] 4.1 Substituir a renderização atual (lista/`st.dataframe` de texto) pelo relatório visual: resumo no topo (nº de eventos + articulação mais afetada) e, por evento, `st.image(draw_pose_on_frame(frame_do_evento, keypoints, highlight=articulacao))` + legenda simples
- [x] 4.2 Sensibilidade: manter o slider e adicionar `st.caption` dinâmico com a estimativa de "% do vídeo marcado como irregular" no valor atual (de 2.6)
- [x] 4.3 Zona de risco: `st.checkbox` desativado por padrão; quando marcado, mostrar os sliders x/y e a prévia `st.image(draw_zone_on_frame(primeiro_frame, zona_atual))`; passar `zone=None` quando desmarcado
- [x] 4.4 Garantir acesso ao frame-imagem do índice-pior de cada evento na hora de renderizar, reaproveitando o cache de extração de pose existente (guardar os frames decodificados ou re-decodificar sob demanda dos bytes já cacheados), sem reprocessar YOLO
- [x] 4.5 Preservar: cache de extração de pose, gate por botão "Processar vídeo", e não tocar nas outras 3 abas

## 5. Verificação e documentação

- [ ] 5.1 Rodar a suíte completa (`venv/bin/python -m pytest tests/ -q`) — tudo verde
- [ ] 5.2 Validar o fluxo real via `AppTest` (upload de `data/demo_pose_walk.mp4` → "Processar vídeo") sem crash, e boot real via `streamlit run app.py`
- [ ] 5.3 Atualizar `README.md` (descrição da aba Vídeo: multi-articulação, eventos visuais, zona opcional) e `RELATORIO_TECNICO.md` (seção de vídeo: articulações rastreadas, agrupamento em eventos, saída visual)
