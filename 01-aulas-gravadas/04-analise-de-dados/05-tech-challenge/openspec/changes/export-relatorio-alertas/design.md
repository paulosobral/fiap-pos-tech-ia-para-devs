## Context

`alerts/feed.py` guarda `Alert{origin, description, timestamp, alert_id, category, level}` em `st.session_state` e `get_alerts()` retorna newest-first. `app.py::_render_alert_feed` renderiza esse feed na sidebar (com ícone de nível, da change anterior). Falta um caminho de saída: exportar todos os alertas da sessão como relatório.

A change anterior (`alertas-estruturados-e-vinculo-sinais-vitais`) já preencheu os campos estruturados nas 4 abas, então o export pode ser tabular e limpo (não precisa parsear texto).

## Goals / Non-Goals

**Goals:**
- Exportar todos os alertas da sessão (todas as abas) como CSV, via botão na sidebar junto do feed.
- Mostrar um resumo (contagem por aba e por nível) junto do botão.
- Lógica de montagem em helper puro testável; `app.py` só chama e oferece o download.

**Non-Goals:**
- Não persiste em disco no servidor — o CSV é gerado em memória e baixado pelo `st.download_button`.
- Não adiciona formato Markdown (o usuário escolheu CSV + resumo; Markdown ficou fora).
- Não muda o feed nem a detecção.

## Decisions

### D1: Helper puro `build_alerts_report(alerts) -> (csv_text: str, summary: dict)` em `alerts/feed.py`
Recebe uma lista de `Alert` (a mesma de `get_alerts()`) e retorna:
- `csv_text`: CSV (via `csv`/`io.StringIO`) com cabeçalho e uma linha por alerta. Colunas fixas: `id, origem, timestamp, categoria, nivel, descricao`. Campos ausentes (`alert_id`/`category`/`level` None) viram célula vazia. `timestamp` formatado ISO (`%Y-%m-%d %H:%M:%S`). `nivel` pode ser exibido pelo rótulo amigável quando conhecido — decisão: exportar o valor cru do `level` (ex.: `alta_confianca`) numa coluna `nivel`, mais uma coluna `nivel_label` com o rótulo amigável quando houver mapeamento, senão vazio; assim o CSV serve tanto pra máquina quanto pra leitura. (Manter simples: se preferir uma coluna só, exportar o rótulo amigável e cair no valor cru quando não mapeado — implementar a variante de duas colunas por ser mais útil e ainda simples.)
- `summary`: `{total, por_origem: {aba: n}, por_nivel: {nivel: n}}` — determinístico, para a UI exibir contadores.
Função pura (não lê `session_state`, recebe a lista) → testável direto. Fica em `alerts/feed.py` junto do modelo de alerta.
**Alternativa considerada**: montar o CSV com pandas `to_csv` — funcionaria, mas o `csv` da stdlib evita depender do pandas aqui e não tem risco do escape/encoding surpreender; a lista de alertas é pequena.

### D2: UI — `st.download_button` + resumo na sidebar
Em `_render_alert_feed` (ou logo após), quando houver alertas: chamar `build_alerts_report(get_alerts())`, exibir o `summary` (uma linha de contadores: total, por aba, por nível) e um `st.download_button("Baixar relatório (CSV)", data=csv_text, file_name="relatorio_alertas.csv", mime="text/csv")`. Quando não houver alertas, não mostrar o botão (ou desabilitá-lo) — coerente com a mensagem "nenhum alerta" que já existe.
Nome do arquivo fixo `relatorio_alertas.csv` (sem timestamp no nome, pois `Date.now()`-style é evitado no código do app; a data já está dentro do CSV por alerta).

## Risks / Trade-offs

- **[Trade-off] CSV em vez de também Markdown** → escolha do usuário; CSV abre no Excel e é universal. Markdown pode ser uma change futura se quiser um relatório mais "leitura".
- **[Risco] Encoding/acentos no CSV (categorias e descrições em PT)** → usar UTF-8 (com BOM opcional pra Excel abrir acentos certo — decidir na implementação; UTF-8-SIG é seguro pro Excel). Documentar a escolha.
- **[Risco] `st.download_button` re-renderiza/reexecuta o script** → o helper é barato (lista pequena), e gerar o CSV a cada rerun não tem custo relevante; não há chamada pesada envolvida.

## Migration Plan

Não aplicável — feature aditiva de UI, sem persistência/estado novo.

## Open Questions

Nenhuma — formato (CSV + resumo), local (sidebar, junto do feed) e dependência (campos estruturados já entregues na change anterior) fechados com o usuário.
