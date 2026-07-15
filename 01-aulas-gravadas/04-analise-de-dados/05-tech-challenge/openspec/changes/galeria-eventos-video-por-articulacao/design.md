## Context

A change `redesenho-aba-video-postura` fez a aba Vídeo produzir uma lista de eventos (`analyze` retorna `events`, cada um `{tipo, articulacao, t_inicio, t_fim, frame_index_pior, valor_pior, z_pior}`, mais `summary` e `alerts`). O `app.py` renderiza HOJE um `st.image` por evento numa lista plana, na ordem dos eventos. Em um vídeo real isso gerou ~500 eventos → ~500 imagens, inviável de navegar. Decisões fechadas com o usuário no brainstorming: agrupar por articulação em `st.expander` colapsável (fechado por padrão, ordenado da mais afetada para a menos), top 10 piores por seção, galeria em grade de colunas. O item "desenho da pose ruim" levantado pelo usuário foi descartado por ele (era o conteúdo do vídeo demo, não o código).

## Goals / Non-Goals

**Goals:**
- Tornar a saída navegável e leve mesmo com centenas de eventos: agrupar por articulação, colapsar por padrão, limitar a top-N por seção.
- Manter `app.py` fino: a lógica de agrupamento/ordenação/top-N vive em `video/analysis.py` (testável), `app.py` só renderiza.

**Non-Goals:**
- Não muda a detecção nem a contagem de eventos/alertas (o feed continua 1 alerta por evento; o agrupamento é só de apresentação).
- Não muda `video/pose.py` nem `video/draw.py` (o desenho da pose permanece como está).
- Não adiciona paginação "ver mais" (o usuário escolheu top-N fixo, não paginação).

## Decisions

### D1: Helper de agrupamento em `video/analysis.py`
Nova função pura, ex.:
```
group_events_for_display(events, top_n=10) -> List[Dict]
```
Retorna uma lista de seções já ordenadas para exibição, cada seção:
```
{
  chave: str,            # "cotovelo_direito" | "velocidade" | "zona_critica"
  label: str,            # "Cotovelo direito" | "Movimento brusco (corpo todo)" | "Zona de risco"
  total: int,            # total de eventos na seção (antes do corte)
  eventos: List[dict],   # top_n eventos, ordenados por |z_pior| desc (desempate por t_inicio)
}
```
- Agrupa por `articulacao` para `tipo == "postura"`; eventos `velocidade` viram uma seção `chave="velocidade"`; eventos `zona_critica` viram `chave="zona_critica"`.
- Ordena os eventos dentro da seção por `abs(z_pior)` desc. Zona tem `z_pior = NaN` (do redesign anterior) — ordenar zona por `valor_pior` (área de interseção) desc, para não quebrar com NaN. Documentar esse ramo.
- Corta a `top_n` (mantém `total` para a UI mostrar "10 de N").
- Ordena as seções por `total` desc (mais afetada primeiro); desempate por ordem canônica de `JOINT_LABELS` e depois velocidade/zona por último.
- Reusa `JOINT_LABELS`/`joint_label` já existentes para os rótulos.
**Alternativa considerada**: fazer tudo em `app.py` — rejeitada; deixa a UI grossa e a lógica de ordenação/top-N sem teste.

### D2: Renderização em `app.py` — expanders + grade
Substituir o laço plano `for event in events: st.image(...)` por:
- Resumo no topo (mantido).
- `for secao in group_events_for_display(events, top_n=10):` → `with st.expander(f"{secao['label']} — {secao['total']} evento(s)", expanded=False):` e, se `total > len(eventos)`, um `st.caption("mostrando N de M")`.
- Dentro do expander, uma grade: iterar os eventos em blocos de K (ex.: 3 ou 4) via `st.columns(K)`, cada coluna com `st.image(frame_anotado, channels="BGR", caption=intervalo)`. O frame anotado é montado como hoje (decode cacheado + `draw_pose_on_frame`/`draw_zone_on_frame` conforme o tipo, com o mesmo highlight/guarda de índice e keypoints do redesign atual).
- Manter o gate do botão "Processar vídeo", o cache de decode, e o guard `0 <= idx < len(frames)`.

### D3: Rótulos de seção
Reusar `joint_label` para articulações. Para `velocidade` → "Movimento brusco (corpo todo)"; para `zona_critica` → "Zona de risco". Constantes/labels definidas junto ao helper em `analysis.py` para a vocabulário de apresentação ficar num lugar só.

## Risks / Trade-offs

- **[Trade-off] Top-N esconde eventos além dos 10 piores por seção** → aceito e sinalizado na UI ("mostrando 10 de N"); o objetivo é navegabilidade, e os N piores por gravidade são o que importa. O total real continua visível no título da seção e no resumo.
- **[Risco] Ordenar zona por `valor_pior` (não `z_pior=NaN`)** → tratado explicitamente no helper; um teste cobre que a seção de zona não quebra com NaN.
- **[Trade-off] Ainda renderiza até top_n × nº_de_seções imagens se o usuário abrir todos os expanders** → limitado e sob controle do usuário (expanders fechados por padrão; só decodifica/desenha ao expandir? — nota: Streamlit executa o corpo do expander mesmo fechado, então as imagens dos top-N SÃO montadas; com top_n=10 e ~7 seções isso é ~70 imagens no pior caso, ordem de magnitude menor que 500 e aceitável). Documentado.

## Migration Plan

Não aplicável (mudança de apresentação de UI, sem dados persistidos).

## Open Questions

Nenhuma — layout (expander), top-N (10) e formato (grade de colunas) fechados no brainstorming.
