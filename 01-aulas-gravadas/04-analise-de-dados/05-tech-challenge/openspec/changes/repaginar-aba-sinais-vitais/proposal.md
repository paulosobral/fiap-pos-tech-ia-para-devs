## Why

A saída da aba Sinais Vitais é difícil de interpretar para quem não é técnico (feedback do usuário, mesmo problema já resolvido na aba Vídeo). Hoje ela mostra: nomes de camadas em jargão ("rolling z-score", "Isolation Forest"), controles crus ("Janela do rolling z-score", "Threshold do z-score (|z| >)"), e uma tabela com colunas técnicas (`zscore_anomaly`, `isolation_forest_anomaly`, `agreement`, nomes de sinais em inglês como `heart_rate`). O rótulo de concordância ("alta_confianca"/"zscore_only"/"isolation_forest_only") não é explicado — o usuário não sabe o que cada nível significa nem o que fazer.

## What Changes

- **Nomes amigáveis para as duas camadas**, com o termo técnico e o que fazem em tooltip: "Detecção em tempo real" (rolling z-score) e "Análise do histórico completo" (Isolation Forest).
- **Controles renomeados com tooltip explicativo e exemplo** (padrão da aba Vídeo): "Sensibilidade" (threshold) e "Tamanho da janela de comparação" (window). Mantém o aviso já existente de combinação inefetiva.
- **Resumo no topo em linguagem clara**: uma frase/bullets descrevendo as leituras críticas ("3 leituras críticas — frequência cardíaca alta às 14h, saturação baixa às 16h..."), derivada do relatório.
- **Tabela com cabeçalhos amigáveis** (Horário, Sinal vital, Valor, Nível de confiança) em vez das colunas cruas; nomes dos sinais vitais traduzidos (heart_rate → "Frequência cardíaca", spo2 → "Saturação de O₂ (SpO₂)", etc.).
- **Os 3 níveis de confiança explicados** com frase clara + ícone/cor + tooltip: Alta confiança (as duas análises concordam — mais provável ser real), Só tempo real (pico isolado momentâneo), Só histórico (fora do padrão geral, sem pico súbito).
- **BREAKING** (apresentação da aba, não spec externa): a saída deixa de expor jargão/colunas técnicas cruas e passa a linguagem amigável interpretável.

## Capabilities

### New Capabilities
(nenhuma)

### Modified Capabilities
- `vital-signs-monitoring`: o requirement de apresentação do relatório combinado passa a exigir rótulos/colunas em linguagem amigável, um resumo interpretável, explicação dos níveis de confiança e controles auto-explicativos com tooltips — sem mudar a detecção subjacente.

## Impact

- **Código alterado**: `app.py` (aba Sinais Vitais — nomes de camadas/controles amigáveis com tooltips, resumo, tabela renomeada/traduzida, explicação dos níveis); `vital_signs/analysis.py` (helpers de apresentação puros e testáveis: mapa de nome de sinal → rótulo PT, mapa de nível de confiança → rótulo/ícone/descrição, montagem do resumo textual — mantém `app.py` fino, mesmo padrão do `group_events_for_display` da aba Vídeo).
- **Sem mudança** na detecção (`analyze`, `anomaly/zscore.py`, `vital_signs/isolation_forest.py`) nem nos thresholds/defaults — só apresentação.
- **Testes**: novos para os helpers de apresentação (rótulos de sinal, níveis de confiança, resumo textual a partir de um `combined_report` construído à mão).
- **Sem impacto** nas outras 3 abas, no feed, nem em dependências.
