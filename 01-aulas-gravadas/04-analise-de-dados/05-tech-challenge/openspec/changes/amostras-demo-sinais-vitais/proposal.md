## Why

Para testar/demonstrar a aba Sinais Vitais o projeto tem só um arquivo (`data/vital_signs_sample.csv`, MIMIC-III real), e ao rodá-lo as duas camadas divergem de forma pouco didática: 0 anomalias pelo rolling z-score ("tempo real") e 24 só pelo Isolation Forest. Não há um exemplo claramente **sem anomalias** (as duas camadas limpas) nem um claramente **com anomalias** (ambas as camadas disparando em eventos clínicos óbvios). Isso dificulta demonstrar o contraste que a aba foi feita para mostrar.

## What Changes

- Adicionar **dois CSVs de demonstração** em `data/`, derivados da amostra MIMIC-III real já existente (mantendo o realismo dos valores):
  - `vital_signs_sample_normal.csv` — um trecho estável, sem anomalias em nenhuma das duas camadas (rolling z-score e Isolation Forest ambos limpos).
  - `vital_signs_sample_anomalias.csv` — o mesmo trecho estável com alguns eventos clínicos plausíveis injetados (ex.: episódio de taquicardia, queda de SpO2, pico hipertensivo) que disparam as duas camadas de detecção.
- Adicionar um **script gerador versionado** que produz esses dois CSVs a partir do `vital_signs_sample.csv`, de forma reproduzível e documentada (que trecho, quais eventos injetados, com que parâmetros).
- Documentar no README qual arquivo usar para ver "sem alertas" vs. "com alertas" na aba Sinais Vitais.

## Capabilities

### New Capabilities
- `demo-data`: conjuntos de dados de demonstração versionados e reproduzíveis para exercitar as abas do app — nesta change, os dois CSVs de sinais vitais (um sem anomalias, um com anomalias) derivados da amostra MIMIC-III real por um script gerador.

### Modified Capabilities
(nenhuma — não altera o comportamento de nenhuma aba nem de nenhum módulo de detecção)

## Impact

- **Novos arquivos**: `data/vital_signs_sample_normal.csv`, `data/vital_signs_sample_anomalias.csv`, e um script gerador (ex.: `scripts/gen_vital_signs_demo.py` ou similar).
- **Documentação**: `README.md` (nota sobre os arquivos de demo da aba Sinais Vitais).
- **Sem impacto** em código de app/detecção (`vital_signs/`, `anomaly/`, `app.py` não mudam) — os CSVs apenas alimentam o upload existente. O `data/vital_signs_sample.csv` original é preservado.
- **Sem dependências novas** — `pandas`/`numpy` já no projeto.
- **Nota**: `data/` tem regra `.gitignore` abrangente; os CSVs de demo precisam ser adicionados com `git add -f` (mesmo padrão já usado para `vital_signs_sample.csv`).
