## Context

A aba Sinais Vitais (`vital_signs/analysis.py`) roda duas camadas sobre o CSV carregado: rolling z-score linha a linha (`anomaly.zscore.detect_anomalies`, `DEFAULT_WINDOW=6`, `DEFAULT_THRESHOLD=3.0`) e Isolation Forest em lote (`vital_signs/isolation_forest.py`), combinando em `combined_report` com rótulos `normal` / `zscore_only` / `isolation_forest_only` / `alta_confianca`. Colunas reconhecidas: `heart_rate, spo2, resp_rate, systolic_bp, diastolic_bp` (+ `timestamp`).

A amostra real `data/vital_signs_sample.csv` (MIMIC-III, 480 leituras horárias) roda com 0 anomalias z-score e 24 isolation-forest-only — não é um contraste didático. Objetivo: dois CSVs derivados dessa base real, um claramente limpo (ambas as camadas sem anomalia) e um claramente anômalo (ambas disparando em eventos clínicos plausíveis).

## Goals / Non-Goals

**Goals:**
- Um CSV "normal": nenhuma anomalia em nenhuma das duas camadas ao ser processado pela aba.
- Um CSV "anomalias": eventos clínicos plausíveis injetados que disparam AMBAS as camadas (z-score e Isolation Forest), visíveis como `alta_confianca` no relatório.
- Reproduzível: um script versionado gera os dois a partir de `vital_signs_sample.csv`, documentando trecho e eventos.
- Realismo preservado (derivar dos valores MIMIC reais, não inventar do zero).

**Non-Goals:**
- Não alterar nenhum código de app/detecção nem os thresholds.
- Não remover/substituir `vital_signs_sample.csv` (preservado).
- Não garantir número exato de anomalias — garantir a PRESENÇA (≥1 de alta confiança) no caso anômalo e a AUSÊNCIA total no caso normal.

## Decisions

### D1: Base = trecho estável do MIMIC real, selecionado programaticamente
O script carrega `vital_signs_sample.csv` e escolhe uma janela contígua de N leituras (ex.: 120 = 5 dias horários) cujo processamento pela aba já resulte em ZERO anomalias nas duas camadas — validado rodando o próprio `analyze` sobre o candidato. Se nenhum trecho contíguo for 100% limpo (Isolation Forest sempre marca uma fração `contamination`), o script suaviza levemente o trecho (ex.: leve winsorização/mediana móvel curta apenas o suficiente para o Isolation Forest treinado NAQUELE trecho não marcar nada) — mantendo os valores dentro de faixas fisiológicas reais. O resultado é `vital_signs_sample_normal.csv`. Documentar no script o critério e a verificação.
**Alternativa considerada**: pegar trecho real sem tratamento e aceitar 1-2 marcações residuais do Isolation Forest — rejeitada porque o objetivo do arquivo "normal" é ser inequivocamente limpo na demo.

### D2: Caso anômalo = trecho normal + eventos clínicos injetados
Partindo do `vital_signs_sample_normal.csv`, injetar alguns episódios curtos e clinicamente plausíveis, cada um em um bloco de leituras consecutivas para que o rolling z-score (janela 6) e o Isolation Forest concordem:
- **Taquicardia**: `heart_rate` sobe para ~150–170 por algumas horas.
- **Queda de SpO2 (hipoxemia)**: `spo2` cai para ~85–88.
- **Pico hipertensivo**: `systolic_bp` ~200+, `diastolic_bp` ~120+.
Os picos são grandes o bastante (vários desvios-padrão acima do trecho estável) para o z-score marcar `|z|>3.0` E o Isolation Forest classificar como outlier — resultando em linhas `alta_confianca`. Resultado: `vital_signs_sample_anomalias.csv`. O script valida rodando `analyze` e conferindo que há ≥1 linha `alta_confianca` e que as leituras normais ao redor continuam `normal`.
**Alternativa considerada**: injetar ruído aleatório disperso — rejeitada por não ser clinicamente interpretável nem garantir concordância das duas camadas.

### D3: Script gerador versionado + validação embutida
Um script (ex.: `scripts/gen_vital_signs_demo.py`) que: lê a base, produz os dois CSVs, e ao final roda `vital_signs.analysis.analyze` sobre cada um imprimindo a contagem de rótulos (`normal`/`alta_confianca`/etc.) como auto-verificação. Determinístico (seeds fixas se houver qualquer aleatoriedade). Documenta no cabeçalho o que faz. Assim os CSVs são reproduzíveis e a garantia "limpo vs. anômalo" é checável rodando o script.
**Alternativa considerada**: gerar os CSVs uma vez e só commitá-los sem script — rejeitada; sem o gerador versionado, a proveniência e a reprodutibilidade se perdem (mesma lição da change de dataset anterior).

## Risks / Trade-offs

- **[Risco] Isolation Forest tem `contamination` fixo — pode marcar algo mesmo num trecho estável** → mitigado por D1 (selecionar/suavizar até `analyze` retornar zero anomalias no arquivo normal, verificado pelo próprio script).
- **[Trade-off] Suavizar o trecho normal afasta-o um pouco do dado bruto** → aceito e documentado; a suavização é mínima e os valores permanecem fisiologicamente reais; o objetivo do arquivo é ser um exemplo didático "sem alertas".
- **[Risco] Eventos injetados podem parecer artificiais** → aceito; são rotulados como sintéticos no script/README, e valores escolhidos dentro de faixas clínicas plausíveis para os respectivos quadros.

## Migration Plan

Não aplicável — adição de dados de demo, sem migração.

## Open Questions

Nenhuma — fonte (derivar do MIMIC real) e fluxo (openspec + script gerador) fechados com o usuário.
