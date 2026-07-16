## Why

As abas Vídeo e Sinais Vitais já atribuem um identificador curto (`#V01`, `#S01`) a cada evento/leitura anômala, casando o alerta no feed compartilhado com o item exibido na própria aba (evento na galeria, linha na tabela). Áudio e Prescrições ainda geram alertas sem esse vínculo — o usuário vê o alerta no feed, mas não tem como localizar rapidamente o trecho da transcrição ou a inconsistência de prescrição que o originou. Além disso, a aba Prescrições só tem um dataset sintético com anomalias já misturadas; falta um dataset "limpo" para demonstrar o caso sem inconsistências, como já existe em Sinais Vitais.

## What Changes

- Áudio: cada alerta (termo crítico, taxa de fala anômala, pausa anômala) passa a carregar um `alert_id` curto e único (ex.: `#A01`), atribuído em ordem cronológica/de detecção, e a UI exibe esse mesmo id junto do trecho/segmento correspondente na aba (contexto do termo crítico, segmento de taxa de fala, gap de pausa).
- Prescrições: cada inconsistência apontada pelo Bedrock passa a carregar um `alert_id` curto e único (ex.: `#P01`), atribuído por ordem de finding, e a UI exibe esse mesmo id junto do card da inconsistência correspondente.
- Novo dataset sintético `data/prescricoes_sinteticas_normal.csv`: mesmos 3 pacientes (estrutura idêntica ao arquivo atual), mas com histórico estável — doses constantes, sem combinação de risco conhecida, sem alteração sem justificativa aparente. O arquivo atual `data/prescricoes_sinteticas.csv` passa a ser referenciado como o dataset "com anomalias" (sem alteração de conteúdo).
- Gerador reprodutível (script), análogo a `scripts/gen_vital_signs_demo.py`, que documenta como o dataset normal foi construído e por que não dispara nenhuma inconsistência ao ser revisado.

## Capabilities

### New Capabilities

(nenhuma)

### Modified Capabilities

- `clinical-alerting`: alertas de Áudio e Prescrições passam a carregar `alert_id`, vinculando o alerta no feed ao item de origem exibido na própria aba (mesmo padrão já coberto para Sinais Vitais/Vídeo).
- `demo-data`: adiciona um dataset sintético de prescrições sem anomalias, complementando o existente (com anomalias), para a aba Prescrições.

## Impact

- `audio/aws_speech.py` (`raise_critical_term_alerts`), `audio/analysis.py` (`analyze`): atribuição e propagação do `alert_id`.
- `prescriptions/bedrock_review.py` (`generate_alerts_for_findings`): atribuição e propagação do `alert_id`.
- `app.py`: exibição do id junto ao trecho/card correspondente nas abas Áudio e Prescrições.
- `data/prescricoes_sinteticas_normal.csv` (novo arquivo) e um script gerador em `scripts/`.
- Testes: `tests/test_audio_aws_speech.py`, `tests/test_audio_analysis.py`, `tests/test_bedrock_review.py`.
