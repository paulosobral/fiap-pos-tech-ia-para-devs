## Context

Estado atual (após `redesenho-aba-video-postura` + `galeria-eventos-video-por-articulacao`):
- `video/analysis.py::analyze` produz `events` (cada um `{tipo, articulacao, t_inicio, t_fim, frame_index_pior, valor_pior, z_pior}`), gera 1 `Alert` por evento via `add_alert(origin="Vídeo", description=_event_description(event))`, e retorna `events`/`summary`/`alerts`.
- `_event_description(event)` monta o texto: "{joint_label} irregular entre Xs e Ys." (ou velocidade/zona).
- `app.py` renderiza a galeria via `group_events_for_display(events, top_n=10)` — seções por articulação, ordenadas por gravidade, cada foto com legenda de intervalo de tempo.
- O feed unificado da sidebar (`alerts/feed.py`) lista os `Alert` em ordem cronológica (mais recente primeiro). `Alert` = `{origin, description, timestamp}`.

Problema: alerta (feed, ordem cronológica) e foto (galeria, ordem por gravidade) não têm nada em comum que o usuário use para casá-los, e o alerta não indica a região do corpo.

## Goals / Non-Goals

**Goals:**
- Dar a cada evento de vídeo um ID curto único (`#V01`...), estável, exibido tanto no alerta quanto na legenda da foto, para casar os dois independentemente da ordem de exibição de cada um.
- Incluir a categoria/região do corpo (Cabeça/Braços/Tronco/Pernas/Corpo/Zona de risco) no texto do alerta.

**Non-Goals:**
- Não altera o dataclass `Alert` nem `alerts/feed.py` (ID e categoria vão embutidos no `description`) — evita mexer no contrato compartilhado pelas 4 abas.
- Não muda a detecção, o agrupamento, a contagem de eventos/alertas, nem `pose.py`/`draw.py`.
- Não adiciona navegação clicável alerta→foto (só o ID textual compartilhado; clicar seria escopo bem maior em Streamlit).

## Decisions

### D1: `event_id` no schema do evento, atribuído no `analyze`
Após `analyze` montar e ordenar a lista final de `events` (já ordenada por `t_inicio`), atribuir sequencialmente `event["event_id"] = f"#V{i:02d}"` (i a partir de 1). A ordem de atribuição é a ordem cronológica final dos eventos — determinística e estável. Como o ID é gravado no dict do evento ANTES de gerar os alertas e ANTES de `group_events_for_display` ser chamado na UI, tanto o alerta quanto a galeria leem o mesmo `event_id` do mesmo objeto de evento. Isso resolve o descasamento de ordem (feed cronológico vs. galeria por gravidade) — o ID é a chave comum.
Formato `#V01` com zero-padding de 2 dígitos; se houver 100+ eventos o padding cresce naturalmente (`f"#V{i:02d}"` já exibe `#V100`). Documentar que o "V" marca origem Vídeo (distinto de futuros prefixos de outras abas, se algum dia quiserem IDs lá).

### D2: Mapa articulação/tipo → categoria e helper `event_category(event)`
Novo dict em `analysis.py`:
```
JOINT_CATEGORY = {
  "pescoco": "Cabeça",
  "cotovelo_esquerdo": "Braços", "cotovelo_direito": "Braços",
  "quadril_esquerdo": "Tronco", "quadril_direito": "Tronco",
  "joelho_esquerdo": "Pernas", "joelho_direito": "Pernas",
}
```
`event_category(event)`: para `tipo=="postura"` → `JOINT_CATEGORY[articulacao]`; `tipo=="velocidade"` → "Corpo"; `tipo=="zona_critica"` → "Zona de risco". Colocado junto de `JOINT_LABELS`/`joint_label` (vocabulário de apresentação num lugar só).

### D3: `_event_description` inclui ID + categoria
Novo formato: `"{event_id} [{categoria}] {texto_atual}"`, ex.: `"#V03 [Braço] Cotovelo direito irregular entre 5.2s e 6.4s."`. Nota: a categoria é exibida no singular no colchete conforme o exemplo do usuário ("[Braço]") — usar a forma que ficar natural; manter consistência entre alerta e (se aplicável) qualquer outro uso. Decisão: usar exatamente as chaves de `JOINT_CATEGORY` ("Braços", "Pernas") para não ter dois vocabulários; o colchete fica "[Braços]" — consistente e sem mapa extra. (O exemplo "[Braço]" do usuário era ilustrativo; priorizar um único vocabulário.)

### D4: Galeria (`app.py`) mostra o `event_id` na legenda
A legenda de cada foto passa de "{intervalo}" para "{event_id} — {intervalo}" (ex.: "#V03 — 5.2s a 6.4s"). Nenhuma outra mudança na galeria.

## Risks / Trade-offs

- **[Trade-off] ID e categoria embutidos no texto do `description` em vez de campos estruturados no `Alert`** → aceito; mantém o contrato do feed intacto para as 4 abas e é suficiente para o objetivo (casar visualmente). Se um dia o feed precisar filtrar por categoria programaticamente, aí valeria estender o dataclass — fora de escopo agora (YAGNI).
- **[Risco] Se algum código externo faz parsing do texto atual do alerta de vídeo** → improvável (o feed só exibe `description`); a mudança é aditiva (prefixo), não remove o texto existente.
- **[Trade-off] IDs são por-execução (reatribuídos a cada processamento de vídeo)** → aceito; são identificadores de sessão/relatório, não persistentes — coerente com o feed viver em `st.session_state`.

## Migration Plan

Não aplicável (mudança de apresentação/So schema em memória, sem dados persistidos).

## Open Questions

Nenhuma — ID (#V01, no description), categoria (Cabeça/Braços/Tronco/Pernas/Corpo/Zona de risco) e local de exibição (texto do alerta + legenda da galeria) fechados no brainstorming.
