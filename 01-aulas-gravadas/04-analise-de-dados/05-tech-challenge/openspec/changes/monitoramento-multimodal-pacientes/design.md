## Context

Projeto individual, prazo curto, ambiente de desenvolvimento WSL (sem passthrough fácil de câmera/microfone). O PDF do Tech Challenge Fase 4 pede monitoramento multimodal (vídeo, áudio, sinais vitais, prescrições) com detecção de anomalia e integração a serviços gerenciados em nuvem (cita Azure Cognitive Services). Não há código-fonte anterior neste diretório — greenfield. Estrutura de referência do monorepo (Fase 3, `03-openia/07-tech-challenge/`) usa Python + venv + requirements.txt + Streamlit-like entrypoint único, e é o padrão a seguir aqui.

## Goals / Non-Goals

**Goals:**
- Cobrir as 3 entregas técnicas obrigatórias do PDF (vídeo, áudio, detecção de anomalias) mais o requisito de anomalia em prescrições, tudo via upload de arquivo numa única app Streamlit.
- Reaproveitar uma única implementação de detecção de anomalia estatística (rolling z-score) nas 3 modalidades que fazem sentido para ela (vídeo, áudio, sinais vitais).
- Demonstrar duas abordagens de IA distintas no relatório: estatística/ML clássico (rolling z-score + Isolation Forest) e reasoning via LLM (Bedrock) — cobrindo bem a exigência do relatório técnico de "modelos aplicados em cada tipo de dado".
- Produzir um feed de alertas único, para satisfazer o requisito "alertar a equipe médica em tempo real" de forma visível na demo.

**Non-Goals:**
- Captura em tempo real via microfone/câmera (streaming ao vivo) — fora de escopo por limitação de ambiente WSL.
- Uso de Azure Cognitive Services — substituído integralmente por AWS (Transcribe, Comprehend, Bedrock); a equivalência é documentada no relatório, não implementada como fallback dual-cloud.
- Treino de modelo customizado (fine-tuning) para qualquer modalidade — todas as detecções usam técnica estatística sem treino, Isolation Forest com fit local simples, ou LLM pré-treinado via API.
- Persistência multi-usuário / autenticação / banco de dados de produção — a app é uma demonstração local, alertas vivem em memória de sessão (Streamlit `session_state`).

## Decisions

### D1: YOLOv8-pose no lugar de OpenPose + YOLOv8 separados
O PDF sugere OpenPose (postura) e YOLOv8 (objetos) como exemplos, não obrigatórios ("modelos como"). OpenPose depende de build Caffe antigo, doloroso de instalar e manter. `ultralytics` YOLOv8-pose (`yolov8n-pose.pt`) fornece keypoints humanos e detecção de objeto no mesmo forward pass, instalável via `pip install ultralytics`. Um único modelo cobre os dois sub-requisitos do PDF ("análise postural" e "detecção de objetos e áreas críticas").
**Alternativa considerada**: YOLOv8 puro + MediaPipe Pose separados — rejeitada por exigir dois pipelines e duas dependências onde um resolve.

### D2: Rolling z-score como técnica de anomalia compartilhada (vídeo, áudio, sinais vitais)
Função genérica `detect_anomalies(series: pd.Series, window: int, threshold: float) -> pd.Series[bool]` computa média/desvio móvel e marca `|z| > threshold` como anomalia. Não exige treino, decisão é interpretável (documentável no relatório com o threshold usado), e funciona igual em qualquer série numérica derivada de qualquer modalidade — ângulo/velocidade de keypoint (vídeo), taxa de fala/duração de pausa (áudio), FC/PA/SpO2 (sinais vitais).
**Alternativa considerada**: Isolation Forest generalizado para as 3 modalidades — rejeitado como default porque exige fit prévio (não cabe bem em upload avulso de vídeo/áudio de tamanho variável) e é menos explicável; mantido apenas como camada extra nos sinais vitais (D3), onde há dataset maior disponível para fit.

### D3: Isolation Forest como segunda camada só em Sinais Vitais
Sinais vitais têm a fonte de dados mais robusta (dataset público real, PhysioNet/MIMIC-III demo ou VitalDB), suficiente para `sklearn.ensemble.IsolationForest.fit()` de forma razoável. Roda em paralelo ao rolling z-score: z-score decide alerta linha a linha (rápido, "tempo real"); Isolation Forest roda em lote sobre a série completa carregada e aparece no relatório da aba como validação cruzada. Vídeo e áudio não recebem essa camada extra porque a mídia de demo é curta e variável, insuficiente para um fit útil.

### D4: Threshold do rolling z-score do vídeo ajustável via slider Streamlit
Decisão do usuário: em vez de threshold fixo no código, a aba Vídeo expõe `st.slider("Sensibilidade", ...)` controlando o `threshold` passado à função de D2. Fica mais interativo para o vídeo de demonstração (mostrar sensibilidade alta vs. baixa ao vivo). Sinais vitais e áudio usam threshold fixo documentado no código/relatório (não há requisito de interatividade ali).

### D5: AWS no lugar de Azure Cognitive Services (Transcribe, Comprehend, Bedrock)
O PDF exige literalmente Azure Speech to Text e Azure Text Analytics. Decisão consciente do usuário: usar AWS Transcribe (fala→texto) e AWS Comprehend (sentimento + entidades) como equivalentes funcionais, e AWS Bedrock (Claude Sonnet) para a análise de prescrições — sem equivalente Azure citado no PDF, mas natural dentro da mesma conta cloud já escolhida. Risco aceito: avaliação pode penalizar por não usar Azure literalmente; mitigado documentando a equivalência explicitamente no relatório técnico (seção "Modelos aplicados em cada tipo de dado").
**Alternativa considerada**: manter Azure só para os dois serviços citados e AWS para o resto — rejeitada pelo usuário para não gerenciar duas contas cloud e dois SDKs diferentes.

### D6: Prescrições via AWS Bedrock (Claude Sonnet) sem treino de modelo
Dataset de prescrições é sintético e pequeno (criado manualmente, não há fonte pública apropriada) — insuficiente para treinar/validar um modelo estatístico ou de ML com confiança. Inconsistência em prescrição (mudança abrupta de dose, interação medicamentosa, alteração sem justificativa clínica) é uma tarefa de raciocínio semântico sobre texto/dados estruturados, adequada a um LLM instruído via prompt, sem necessidade de fine-tuning.
**Alternativa considerada**: regras hard-coded (ex.: variação de dose > X%) — rejeitada por não capturar interação medicamentosa nem justificativa clínica textual, que exigem entendimento de linguagem.

### D7: Estrutura de módulos por capability, não por camada técnica
```
05-tech-challenge/
├── app.py                    # Streamlit entrypoint, 4 abas
├── anomaly/
│   └── zscore.py             # detect_anomalies() genérica (D2)
├── video/
│   ├── pose.py               # YOLOv8-pose: extração de keypoints, ângulo, velocidade
│   └── analysis.py           # pipeline vídeo: anomalia postural + zona crítica + relatório
├── audio/
│   ├── aws_speech.py         # wrappers AWS Transcribe + Comprehend
│   └── analysis.py           # pipeline áudio: features acústicas + anomalia de fala
├── vital_signs/
│   ├── isolation_forest.py   # fit/predict Isolation Forest (D3)
│   └── analysis.py           # pipeline sinais vitais: zscore + isolation forest
├── prescriptions/
│   └── bedrock_review.py     # chamada Bedrock Claude Sonnet + parsing de resposta
├── alerts/
│   └── feed.py                # modelo de Alert + gerenciamento de session_state
├── data/                      # datasets de demo (vitals CSV, prescrições sintéticas, mídia)
├── requirements.txt
└── README.md / relatório técnico
```
Cada módulo de capability expõe uma função `analyze(...)` que a aba correspondente em `app.py` chama e que internamente empurra `Alert`s para o feed compartilhado (`alerts/feed.py`). Isso mantém `app.py` fino (só UI) e cada capability testável isoladamente.

## Risks / Trade-offs

- **[Risco] Avaliação penaliza por não usar Azure literalmente** → Mitigação: seção dedicada no relatório técnico justificando equivalência AWS↔Azure serviço a serviço (D5).
- **[Risco] `ultralytics` YOLOv8-pose baixa pesos na primeira execução (depende de internet)** → Mitigação: documentar no README passo de download antecipado do modelo `yolov8n-pose.pt`, incluir no setup do requirements/instruções.
- **[Risco] Custo de chamadas AWS (Transcribe/Comprehend/Bedrock) durante desenvolvimento/demo** → Mitigação: usar arquivos de mídia curtos para os testes de desenvolvimento; documentar custo estimado no relatório.
- **[Risco] Dataset sintético de prescrições pode não convencer a banca de robustez** → Mitigação: relatório explicita que a ausência de fonte pública apropriada motivou o dataset sintético, e que a validação é feita por reasoning de LLM, não por estatística sobre poucos dados.
- **[Trade-off] Rolling z-score é mais simples que ML mas menos "impressionante"** → Aceito conscientemente (D2); compensado por Isolation Forest nos sinais vitais (D3) para mostrar variedade de técnica no relatório.
- **[Risco] Sem captura real-time, um requisito implícito do PDF ("monitorar continuamente") fica só simulado via upload** → Mitigação: relatório e vídeo de demonstração explicam a limitação de ambiente (WSL) e descrevem como a mesma pipeline se estenderia a streaming real (trabalho futuro).

## Migration Plan

Não aplicável — projeto novo (greenfield), sem sistema em produção a migrar. Ordem de implementação recomendada (detalhada em `tasks.md`): módulo `anomaly/` (base compartilhada) → `vital_signs/` (mais simples, valida a base) → `video/` → `audio/` → `prescriptions/` → `alerts/` + integração final em `app.py`.

## Open Questions

Nenhuma pendente — todas as decisões de arquitetura foram fechadas com o usuário durante o brainstorming (ver proposal.md e histórico da sessão).
