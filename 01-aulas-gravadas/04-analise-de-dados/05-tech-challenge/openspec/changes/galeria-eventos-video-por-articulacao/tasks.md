## 1. Helper de agrupamento (`video/analysis.py`)

- [x] 1.1 Implementar `group_events_for_display(events, top_n=10) -> List[Dict]`: agrupa eventos por `articulacao` (tipo "postura"), mais uma seção para `velocidade` e uma para `zona_critica`; cada seção `{chave, label, total, eventos}`
- [x] 1.2 Ordenar os eventos dentro de cada seção por gravidade — `abs(z_pior)` desc para postura/velocidade; para zona (`z_pior` NaN) ordenar por `valor_pior` desc (área de interseção), sem quebrar com NaN — desempate por `t_inicio`
- [x] 1.3 Cortar cada seção a `top_n` eventos, preservando `total` (contagem antes do corte) para a UI
- [x] 1.4 Ordenar as seções por `total` desc (mais afetada primeiro); desempate por ordem canônica de `JOINT_LABELS`, com velocidade/zona ao final; rótulos via `joint_label`/labels de apresentação
- [x] 1.5 Testes: agrupa por articulação, ordena por gravidade dentro da seção, aplica top-N mantendo `total`, ordena seções por contagem, seção de zona com `z_pior` NaN não quebra, resultado vazio quando não há eventos

## 2. Renderização em galeria (`app.py`, aba Vídeo)

- [x] 2.1 Substituir o laço plano de `st.image` por evento por: para cada seção de `group_events_for_display(events, top_n=10)`, um `st.expander(f"{label} — {total} evento(s)", expanded=False)`
- [x] 2.2 Dentro do expander, quando `total > len(eventos)`, um `st.caption("mostrando N de M")`; renderizar os eventos numa grade `st.columns(K)` (K=3 ou 4), cada célula com o frame anotado (`draw_pose_on_frame`/`draw_zone_on_frame` conforme o tipo, mesmo highlight e guard de índice/keypoints do redesign atual) + legenda com o intervalo de tempo
- [x] 2.3 Preservar: resumo no topo, gate do botão "Processar vídeo", cache de decode de frames (sem reprocessar YOLO), guard `0 <= idx < len(frames)`, e não tocar nas outras 3 abas nem no feed
- [x] 2.4 Manter caption/label derivados do tipo do evento (postura/velocidade/zona), consistente com a correção anterior (não assumir articulação em velocidade/zona)

## 3. Verificação e documentação

- [x] 3.1 Rodar a suíte completa (`venv/bin/python -m pytest tests/ -q`) — tudo verde
- [x] 3.2 Validar via `AppTest` (upload `data/demo_pose_walk.mp4` → "Processar vídeo") sem exceção; confirmar que a saída agora são expanders por articulação (não centenas de imagens soltas); boot real via `streamlit run app.py` sem traceback
- [x] 3.3 Atualizar `README.md` e `RELATORIO_TECNICO.md` (descrição da saída da aba Vídeo: seções colapsáveis por articulação, top-N em galeria)
