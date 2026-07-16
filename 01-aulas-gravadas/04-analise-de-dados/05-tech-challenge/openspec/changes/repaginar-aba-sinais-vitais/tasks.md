## 1. Helpers de apresentação (`vital_signs/analysis.py`)

- [x] 1.1 Adicionar `VITAL_SIGN_LABELS` (heart_rate→"Frequência cardíaca", spo2→"Saturação de O₂ (SpO₂)", resp_rate/respiratory_rate→"Frequência respiratória", systolic_bp→"Pressão sistólica", diastolic_bp→"Pressão diastólica", blood_pressure→"Pressão arterial", temperature→"Temperatura") + `vital_sign_label(col)` com fallback para a coluna crua
- [x] 1.2 Adicionar `CONFIDENCE_LEVELS` (para `alta_confianca`/`zscore_only`/`isolation_forest_only`: `{label, icon, short, help}` conforme design D1) + `confidence_level(agreement) -> dict`
- [x] 1.3 Adicionar `build_vitals_summary(combined_report) -> dict`: total de leituras anômalas, contagem por nível, e lista curta de itens legíveis (nível, sinal responsável traduzido — ou "padrão geral" quando só Isolation Forest marcou —, valor, timestamp), priorizando alta confiança. Função pura e determinística
- [x] 1.4 Testes: `vital_sign_label` (conhecidos + fallback), `confidence_level` (3 níveis + comportamento p/ agreement inesperado), `build_vitals_summary` sobre um `combined_report` construído à mão (contagens por nível corretas; item de alta confiança priorizado; caso "padrão geral" quando só IF marca; caso sem anomalias → resumo vazio/zerado)

## 2. UI amigável (`app.py`, aba Sinais Vitais)

- [x] 2.1 Camadas com nomes amigáveis ("Detecção em tempo real" / "Análise do histórico completo") e termo técnico + o que faz no `help`/caption
- [x] 2.2 Controles renomeados ("Sensibilidade" / "Tamanho da janela de comparação") com `help` explicando efeito em linguagem simples e exemplo; manter `zscore_threshold_is_reachable` + aviso de combinação inefetiva
- [x] 2.3 Resumo no topo do resultado via `build_vitals_summary` (frase/bullets em linguagem clara)
- [x] 2.4 Tabela de anomalias com cabeçalhos amigáveis (Horário, Sinal vital, Valor, Nível de confiança com ícone+label) e nomes de sinais traduzidos, substituindo o `st.dataframe` de colunas cruas
- [x] 2.5 Legenda compacta dos 3 níveis de confiança (ícone + frase curta; help longo em tooltip)
- [x] 2.6 Preservar: gráfico da série temporal, botão "Processar", bloco de alertas do feed, e não tocar nas outras abas nem na detecção

## 3. Verificação e documentação

- [x] 3.1 Rodar a suíte completa (`venv/bin/python -m pytest tests/ -q`) — tudo verde
- [x] 3.2 Validar via `AppTest` (upload `data/vital_signs_sample_anomalias.csv` → "Processar sinais vitais") sem exceção; se o segfault conhecido `AppTest`+`st.dataframe` atrapalhar, validar via boot real (`streamlit run app.py`, HTTP 200) + teste direto dos helpers. Confirmar que o `..._normal.csv` mostra a mensagem "nenhuma anomalia"
- [x] 3.3 Atualizar `README.md`/`RELATORIO_TECNICO.md` (descrição da saída amigável da aba Sinais Vitais: nomes amigáveis das camadas, resumo, tabela traduzida, níveis de confiança explicados)
