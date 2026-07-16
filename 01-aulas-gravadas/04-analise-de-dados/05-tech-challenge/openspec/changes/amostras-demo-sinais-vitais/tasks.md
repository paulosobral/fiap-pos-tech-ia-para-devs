## 1. Script gerador

- [x] 1.1 Criar `scripts/gen_vital_signs_demo.py`: carrega `data/vital_signs_sample.csv`, seleciona uma janela contígua estável (ex.: ~120 leituras) e, se necessário, suaviza minimamente para que `vital_signs.analysis.analyze` retorne ZERO anomalias nas duas camadas → grava `data/vital_signs_sample_normal.csv`
- [x] 1.2 A partir do trecho normal, injetar eventos clínicos plausíveis em blocos consecutivos (taquicardia em `heart_rate`, hipoxemia em `spo2`, pico hipertensivo em `systolic_bp`/`diastolic_bp`), grandes o bastante para z-score (`|z|>3.0`, janela padrão 13) e Isolation Forest concordarem → grava `data/vital_signs_sample_anomalias.csv`
- [x] 1.3 Determinismo: seeds fixas para qualquer aleatoriedade; cabeçalho do script documenta trecho escolhido, eventos injetados e parâmetros
- [x] 1.4 Auto-verificação no fim do script: roda `analyze` sobre cada CSV e imprime a contagem de rótulos; falha (exit ≠ 0) se o normal tiver qualquer anomalia ou o anômalo não tiver ao menos uma `alta_confianca`

## 2. Gerar os arquivos e validar

- [x] 2.1 Executar o script; confirmar `data/vital_signs_sample_normal.csv` (0 anomalias nas duas camadas) e `data/vital_signs_sample_anomalias.csv` (≥1 `alta_confianca`, normais ao redor limpos)
- [x] 2.2 Confirmar que ambos carregam sem erro por `load_vital_signs_csv` (colunas reconhecidas, timestamps válidos)
- [x] 2.3 Adicionar os dois CSVs ao git com `git add -f` (regra `.gitignore` abrangente em `data/`, mesmo padrão do `vital_signs_sample.csv`)

## 3. Documentação e verificação final

- [x] 3.1 Atualizar `README.md`: nota na seção Sinais Vitais indicando qual arquivo usar para ver "sem alertas" (`..._normal.csv`) e "com alertas" (`..._anomalias.csv`), e como regerar via o script
- [x] 3.2 Rodar a suíte completa (`venv/bin/python -m pytest tests/ -q`) — nada quebrou (mudança é só dados + script, não toca código do app)
- [x] 3.3 (Opcional) Validar via `AppTest` que o upload de cada CSV na aba Sinais Vitais reflete o esperado (normal sem alertas; anômalo com alertas) — se o segfault conhecido do AppTest com `st.dataframe` atrapalhar, registrar que a verificação foi feita via o script gerador (que chama o mesmo `analyze`)
