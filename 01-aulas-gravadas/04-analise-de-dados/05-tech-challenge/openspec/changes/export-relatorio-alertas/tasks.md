## 1. Helper de montagem do relatório (`alerts/feed.py`)

- [x] 1.1 Implementar `build_alerts_report(alerts) -> (csv_text, summary)`: função pura (recebe a lista de `Alert`, não lê `session_state`)
- [x] 1.2 `csv_text`: CSV via `csv`/`io` da stdlib, cabeçalho + uma linha por alerta; colunas `id, origem, timestamp, categoria, nivel, nivel_label, descricao`; campos ausentes → célula vazia; `timestamp` ISO (`%Y-%m-%d %H:%M:%S`); `nivel` = valor cru, `nivel_label` = rótulo amigável quando mapeável (senão vazio); encoding compatível com Excel (UTF-8-SIG)
- [x] 1.3 `summary`: `{total, por_origem: {aba: n}, por_nivel: {nivel: n}}`, determinístico
- [x] 1.4 Testes: uma linha por alerta com colunas corretas; alerta sem id/categoria/nível vira células vazias sem erro; `summary` conta certo por aba e por nível; lista vazia → CSV só com cabeçalho e summary zerado

## 2. UI: botão de download + resumo (`app.py`, sidebar)

- [x] 2.1 No feed da sidebar, quando houver alertas, chamar `build_alerts_report(get_alerts())` e exibir o resumo (total + por aba + por nível)
- [x] 2.2 `st.download_button("Baixar relatório (CSV)", data=csv_text, file_name="relatorio_alertas.csv", mime="text/csv")`
- [x] 2.3 Sem alertas: não oferecer o botão (coerente com a mensagem "nenhum alerta registrado" já existente)

## 3. Verificação e documentação

- [x] 3.1 Rodar a suíte completa (`venv/bin/python -m pytest tests/ -q`) — tudo verde
- [x] 3.2 Validar via boot/AppTest: gerar alertas (ex.: processar `data/vital_signs_sample_anomalias.csv`), confirmar que o resumo aparece e o download gera um CSV com as colunas esperadas; sem alertas, sem botão. Se o segfault conhecido `AppTest`+`st.dataframe` atrapalhar, validar via boot + teste direto do helper
- [x] 3.3 Atualizar `README.md`/`RELATORIO_TECNICO.md` (feed unificado agora exporta relatório CSV de todos os alertas + resumo por aba/nível)
