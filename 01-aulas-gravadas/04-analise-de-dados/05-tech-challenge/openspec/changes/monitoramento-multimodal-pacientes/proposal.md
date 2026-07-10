## Why

O Tech Challenge da Fase 4 (8IADT) exige um sistema de monitoramento contínuo de pacientes usando dados multimodais (vídeo, áudio, sinais vitais, prescrições) para identificar sinais precoces de risco, com integração a serviços gerenciados em nuvem e detecção de anomalias em tempo real. Hoje o diretório `05-tech-challenge/` só contém o PDF do desafio — nenhuma solução foi implementada. Esta proposta cobre a entrega obrigatória (90% da nota da fase) dentro do prazo, como projeto individual.

## What Changes

- Nova aplicação Streamlit única com 4 abas de upload de arquivo (sem captura real-time de mic/câmera, por limitação de ambiente WSL sem device passthrough).
- **Aba Vídeo**: processa vídeo de fisioterapia/cirurgia com YOLOv8-pose (substitui OpenPose citado no PDF); extrai ângulo de articulação e velocidade de movimento por frame; detecta anomalia postural via rolling z-score com threshold ajustável por slider na UI; detecta objeto/área crítica via YOLO com alerta por regra de zona; gera relatório automático com timestamps.
- **Aba Áudio**: processa áudio de consulta com AWS Transcribe (fala→texto) e AWS Comprehend (sentimento + entidades/termos críticos); extrai features acústicas (taxa de fala, duração de pausa) dos timestamps do Transcribe; detecta fadiga/disartria via rolling z-score. **Substitui Azure Speech to Text e Azure Text Analytics** (exigidos literalmente no PDF) por equivalentes AWS — decisão documentada no relatório técnico.
- **Aba Sinais Vitais**: processa upload de CSV de série temporal (FC/PA/SpO2, dataset público tipo PhysioNet/MIMIC-III demo ou VitalDB); aplica duas camadas de detecção: rolling z-score linha a linha (alerta em tempo real) e Isolation Forest treinado sobre o dataset carregado (validação batch).
- **Aba Prescrições**: processa upload de CSV/Excel com histórico de prescrições por paciente; usa AWS Bedrock (Claude Sonnet) para apontar inconsistências (mudança abrupta de dose, interação medicamentosa, alteração sem justificativa clínica) via raciocínio de LLM, sem treino de modelo customizado.
- Módulo compartilhado de detecção de anomalia (rolling z-score genérico) reaproveitado pelas abas Vídeo, Áudio e Sinais Vitais.
- Feed de alertas unificado na UI simulando notificação em tempo real à equipe médica, alimentado por todas as 4 abas.
- **BREAKING** (relativo ao PDF, não ao código existente): todos os serviços Azure Cognitive Services citados no PDF (Speech to Text, Text Analytics) são substituídos por serviços AWS equivalentes (Transcribe, Comprehend, Bedrock). Essa troca é assumida conscientemente e documentada no relatório técnico como equivalência funcional.

## Capabilities

### New Capabilities
- `video-motion-analysis`: upload de vídeo clínico, extração de pose via YOLOv8-pose, detecção de anomalia postural (rolling z-score com threshold configurável) e de objeto/área crítica (regra de zona), geração de relatório de desvios.
- `audio-speech-analysis`: upload de áudio de consulta, transcrição e análise de sentimento/entidades via AWS Transcribe + Comprehend, extração de features acústicas e detecção de anomalia de fala (fadiga/disartria) via rolling z-score.
- `vital-signs-monitoring`: upload de série temporal de sinais vitais, detecção de anomalia em tempo real (rolling z-score) e validação batch (Isolation Forest).
- `prescription-review`: upload de histórico de prescrições, análise de inconsistências via AWS Bedrock (Claude Sonnet).
- `clinical-alerting`: módulo compartilhado de detecção de anomalia por rolling z-score e feed unificado de alertas consumido pelas 4 abas.

### Modified Capabilities
(nenhuma — projeto greenfield, não há specs existentes em `openspec/specs/`)

## Impact

- **Novo código**: aplicação Streamlit (`app.py`) e módulos Python por capability (`video/`, `audio/`, `vital_signs/`, `prescriptions/`, `anomaly/`) dentro de `01-aulas-gravadas/04-analise-de-dados/05-tech-challenge/`.
- **Dependências novas**: `streamlit`, `ultralytics` (YOLOv8-pose), `opencv-python`, `boto3` (AWS Transcribe/Comprehend/Bedrock), `scikit-learn` (Isolation Forest), `pandas`, `numpy`.
- **Serviços externos**: conta AWS com acesso a Transcribe, Comprehend e Bedrock (modelo Claude Sonnet habilitado).
- **Datasets**: download de dataset público de sinais vitais (PhysioNet/MIMIC-III demo ou VitalDB); mídia de vídeo/áudio pública livre de direitos para demonstração; dataset sintético de prescrições criado manualmente.
- **Entregáveis adicionais**: relatório técnico (fluxo multimodal, modelos por tipo de dado, exemplos de anomalias) e vídeo de demonstração de até 15 minutos — fora do código, mas dependem da aplicação estar funcional.
