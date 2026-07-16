## Why

Os alertas de Sinais Vitais não têm um identificador que ligue o alerta (no feed unificado, ordem cronológica) à linha correspondente na tabela de anomalias — o mesmo problema que a aba Vídeo já resolveu com IDs `#V01`. Além disso, os alertas de Sinais Vitais no feed são todos `st.warning` amarelo, sem distinguir visualmente o nível de confiança (alta confiança / só tempo real / só histórico) que a tabela e a legenda já exibem. E, olhando adiante, um export unificado de todos os alertas (próxima change) precisa de ID, categoria e nível como dados estruturados — hoje eles só existem embutidos no texto da descrição de cada aba, o que é frágil de parsear.

## What Changes

- **Estender o dataclass `Alert`** (`alerts/feed.py`) com três campos OPCIONAIS, além de `origin`/`description`/`timestamp`: `alert_id`, `category`, `level`. Retrocompatível — abas que não os preenchem continuam funcionando (campos ficam `None`).
- **Sinais Vitais**: cada leitura anômala recebe um ID curto único (`#S01`, `#S02`, ...) que aparece no alerta E na linha correspondente da tabela de anomalias, permitindo casar alerta↔linha (paralelo do `#V01` do vídeo). O alerta também passa a carregar `category` (nome do sinal, ex.: "Frequência cardíaca") e `level` (nível de confiança).
- **Classificação visual dos alertas de Sinais Vitais no feed**: em vez de sempre `st.warning`, usar o ícone/cor do nível de confiança (🔴 alta confiança, 🟠 só tempo real, 🟡 só histórico), consistente com a tabela e a legenda já existentes.
- **Migrar Vídeo, Áudio e Prescrições** para também preencher os campos estruturados (`alert_id`/`category`/`level`) que hoje vão só embutidos no texto, mantendo o texto da descrição — para que o export unificado (próxima change) tenha colunas limpas.
- **BREAKING** (interno): assinatura de `add_alert` ganha parâmetros opcionais; nenhum chamador existente quebra (todos passam pelo menos origin+description).

## Capabilities

### New Capabilities
(nenhuma)

### Modified Capabilities
- `clinical-alerting`: o modelo de alerta compartilhado passa a suportar identificador, categoria e nível estruturados (opcionais), e o feed exibe os alertas com indicador visual de nível quando disponível.
- `vital-signs-monitoring`: os alertas e a tabela de anomalias de sinais vitais passam a compartilhar um identificador único por leitura anômala, e os alertas no feed ganham indicador visual de nível de confiança.

## Impact

- **Código alterado**: `alerts/feed.py` (campos opcionais em `Alert` + `add_alert`); `vital_signs/analysis.py` (atribuir `alert_id` por leitura anômala, preencher category/level nos alertas; helper para expor o ID na tabela); `app.py` (aba Sinais Vitais: ID na tabela, ícone/cor por nível no feed inline; abas Vídeo/Áudio/Prescrições: preencher os campos estruturados ao gerar alertas); `video/analysis.py`, `audio/*`, `prescriptions/bedrock_review.py` (passar os campos estruturados que hoje só vão no texto).
- **Testes**: novos para os campos do `Alert`, atribuição de ID em sinais vitais, e que Vídeo/Áudio/Prescrições preenchem os campos estruturados coerentes com o texto.
- **Sem impacto** na detecção nem em dependências. Habilita a change seguinte (`export-relatorio-alertas`).
