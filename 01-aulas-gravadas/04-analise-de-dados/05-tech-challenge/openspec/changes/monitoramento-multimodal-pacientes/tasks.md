## 1. Setup do projeto

- [x] 1.1 Criar `requirements.txt` com `streamlit`, `ultralytics`, `opencv-python`, `boto3`, `scikit-learn`, `pandas`, `numpy`, `openpyxl`
- [x] 1.2 Criar venv local (`python -m venv venv`) e instalar dependências
- [x] 1.3 Criar estrutura de diretórios: `anomaly/`, `video/`, `audio/`, `vital_signs/`, `prescriptions/`, `alerts/`, `data/`
- [x] 1.4 Configurar credenciais AWS localmente (variáveis de ambiente ou `~/.aws/credentials`) com permissão para Transcribe, Comprehend e Bedrock (já existentes em `~/.aws/credentials`, perfis `default` e `bedrock` — não foi necessário reconfigurar)

## 2. Módulo compartilhado de anomalia (clinical-alerting)

- [x] 2.1 Implementar `anomaly/zscore.py` com função `detect_anomalies(series, window, threshold)` que retorna série booleana de anomalias via rolling z-score
- [x] 2.2 Tratar caso de série menor que a janela configurada (retornar tudo `False`, sem erro)
- [x] 2.3 Implementar `alerts/feed.py` com classe/dataclass `Alert` (origem, timestamp, descrição) e funções `add_alert()` / `get_alerts()` sobre `st.session_state`
- [x] 2.4 Garantir que `get_alerts()` retorna alertas ordenados do mais recente para o mais antigo

## 3. Sinais vitais (vital-signs-monitoring)

- [x] 3.1 Baixar dataset público de sinais vitais (PhysioNet/MIMIC-III demo ou VitalDB) para realizar o reinamento e após o treino com score adequado, salvar amostra CSV em `data/vital_signs_sample.csv` (dados reais do MIMIC-III Clinical Database Demo, PhysioNet — CHARTEVENTS.csv de um icustay real, sem credenciais)
- [x] 3.2 Implementar carregamento e validação de CSV de sinais vitais em `vital_signs/analysis.py` (rejeitar arquivo sem coluna de sinal vital reconhecida)
- [x] 3.3 Aplicar `detect_anomalies()` linha a linha por sinal vital e gerar `Alert` para cada leitura marcada como anômala
- [x] 3.4 Implementar `vital_signs/isolation_forest.py` com `fit_and_predict(df) -> pd.Series[bool]` usando `sklearn.ensemble.IsolationForest`
- [x] 3.5 Combinar resultado das duas camadas (z-score e Isolation Forest) destacando concordância/discordância no relatório da aba
- [x] 3.6 Construir aba "Sinais Vitais" em `app.py`: upload de CSV, gráfico da série temporal, tabela de anomalias das duas camadas

## 4. Vídeo (video-motion-analysis)

- [ ] 4.1 Baixar/salvar vídeo público livre de direitos de fisioterapia/exercício em `data/` para uso como demo
- [ ] 4.2 Implementar `video/pose.py`: carregar `yolov8n-pose.pt` via `ultralytics`, processar vídeo frame a frame extraindo keypoints
- [ ] 4.3 Implementar cálculo de ângulo de articulação e velocidade de movimento entre frames a partir dos keypoints, tratando frames sem pessoa detectada
- [ ] 4.4 Implementar `video/analysis.py`: aplicar `detect_anomalies()` sobre as séries de ângulo e velocidade, com `threshold` recebido como parâmetro
- [ ] 4.5 Implementar detecção de objeto/área crítica via YOLOv8 com regra de zona configurável, gerando `Alert` imediato por interseção de bounding box
- [ ] 4.6 Implementar geração de relatório de desvios (lista ordenada por timestamp) combinando anomalias posturais e alertas de zona crítica
- [ ] 4.7 Construir aba "Vídeo" em `app.py`: upload de vídeo (validando formato mp4/avi/mov), slider de sensibilidade, exibição do relatório de desvios

## 5. Áudio (audio-speech-analysis)

- [ ] 5.1 Baixar/salvar áudio público livre de direitos simulando consulta médica em `data/` para uso como demo
- [ ] 5.2 Implementar `audio/aws_speech.py`: função de transcrição via AWS Transcribe retornando texto e timestamps por segmento
- [ ] 5.3 Implementar chamada AWS Comprehend para sentimento e entidades sobre o texto transcrito
- [ ] 5.4 Implementar detecção de termos críticos pré-definidos (lista configurável, ex.: "dor", "não consigo respirar") no texto transcrito, gerando `Alert` quando encontrados
- [ ] 5.5 Implementar `audio/analysis.py`: derivar séries de taxa de fala e duração de pausa a partir dos timestamps do Transcribe
- [ ] 5.6 Aplicar `detect_anomalies()` sobre as séries acústicas para marcar segmentos compatíveis com fadiga/disartria
- [ ] 5.7 Construir aba "Áudio" em `app.py`: upload de áudio (validando formato mp3/wav), exibição de transcrição, sentimento, termos críticos e anomalias de fala

## 6. Prescrições (prescription-review)

- [ ] 6.1 Criar dataset sintético de prescrições (paciente, medicamento, dose, data) em `data/prescricoes_sinteticas.csv`, incluindo ao menos um caso de mudança abrupta de dose e um caso normal
- [ ] 6.2 Implementar carregamento/validação de CSV ou Excel de prescrições em `prescriptions/bedrock_review.py` (rejeitar arquivo com coluna obrigatória faltante)
- [ ] 6.3 Implementar chamada ao AWS Bedrock (Claude Sonnet) com prompt estruturado pedindo identificação de inconsistências (dose, interação medicamentosa, justificativa clínica)
- [ ] 6.4 Implementar tratamento de erro/timeout na chamada Bedrock, exibindo mensagem sem interromper as demais abas
- [ ] 6.5 Parsear resposta do Bedrock e gerar `Alert` para cada inconsistência apontada
- [ ] 6.6 Construir aba "Prescrições" em `app.py`: upload de CSV/Excel, tabela de histórico por paciente, exibição das inconsistências identificadas pelo Bedrock

## 7. Integração final e feed de alertas

- [ ] 7.1 Adicionar seção de feed unificado de alertas na UI principal de `app.py`, consumindo `alerts/feed.py`
- [ ] 7.2 Verificar que alertas de todas as 4 abas aparecem corretamente no feed com origem, timestamp e descrição
- [ ] 7.3 Testar manualmente o fluxo completo: upload em cada aba → processamento → alerta aparece no feed
- [ ] 7.4 Revisar mensagens de erro de validação de upload em todas as abas (formato inválido, colunas faltantes)

## 8. Documentação e entregáveis da Fase 4

- [ ] 8.1 Escrever `README.md` com instruções de setup (venv, requirements, credenciais AWS, download do modelo YOLOv8-pose)
- [ ] 8.2 Escrever relatório técnico cobrindo: descrição do fluxo multimodal, modelos aplicados em cada tipo de dado (incluindo justificativa AWS no lugar de Azure e YOLOv8-pose no lugar de OpenPose), resultados obtidos e exemplos de anomalias detectadas
- [ ] 8.3 Gravar vídeo de demonstração de até 15 minutos cobrindo: análise de áudio e vídeo, detecção e resposta a anomalias, integração dos serviços AWS, fluxo final de alerta à equipe médica
- [ ] 8.4 Publicar vídeo no YouTube ou Vimeo (público ou não listado) e adicionar link ao README
