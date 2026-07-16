## Why

O feed unificado da sidebar acumula os alertas de todas as 4 abas na sessão, mas não há como levar isso pra fora do app — salvar, compartilhar com a equipe, anexar a um relatório. Como a change anterior (`alertas-estruturados-e-vinculo-sinais-vitais`) tornou ID, categoria e nível campos estruturados do `Alert`, agora dá pra exportar um relatório limpo e tabular de todos os alarmes gerados.

## What Changes

- **Botão "Baixar relatório" na sidebar**, junto do feed unificado, que gera um **CSV com todos os alertas da sessão** (todas as abas). Colunas: identificador (`alert_id`), origem (aba), timestamp, categoria, nível, descrição.
- **Resumo no export/UI**: contagem de alertas por aba e por nível, mostrada junto do botão (ex.: "12 alertas: 5 Vídeo, 4 Sinais Vitais, 3 Prescrições").
- O download usa `st.download_button` (arquivo gerado em memória, sem escrever em disco).
- Quando não há alertas na sessão, o botão fica desabilitado/oculto com uma nota.

## Capabilities

### New Capabilities
(nenhuma)

### Modified Capabilities
- `clinical-alerting`: o feed unificado passa a permitir exportar todos os alertas da sessão como um relatório CSV com colunas estruturadas (id, origem, timestamp, categoria, nível, descrição), além de exibir um resumo por aba/nível.

## Impact

- **Código alterado**: `app.py` (seção do feed na sidebar: botão de download + resumo); um helper puro e testável para montar o relatório a partir da lista de `Alert` (ex.: em `alerts/feed.py` — `build_alerts_report(alerts) -> (csv_text, summary_dict)` ou similar), mantendo `app.py` fino.
- **Testes**: para o helper de montagem do relatório (colunas corretas, uma linha por alerta, campos ausentes tratados, resumo por aba/nível), a partir de uma lista de `Alert` construída à mão.
- **Sem impacto** na detecção nem nas abas individuais — só consome o feed já existente. Depende da change `alertas-estruturados-e-vinculo-sinais-vitais` (campos estruturados no `Alert`), já aplicada.
- **Sem dependências novas** — `csv`/`io` da stdlib + `st.download_button`.
