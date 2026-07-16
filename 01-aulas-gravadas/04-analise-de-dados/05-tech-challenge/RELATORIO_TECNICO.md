# Relatório Técnico — Monitoramento Multimodal de Pacientes

Tech Challenge Fase 4 (8IADT), projeto individual. Referências: `openspec/changes/monitoramento-multimodal-pacientes/proposal.md`
e `design.md` (decisões D1–D7), specs em `openspec/changes/monitoramento-multimodal-pacientes/specs/*/spec.md`.

## 1. Visão geral do fluxo multimodal

A aplicação (`app.py`, Streamlit) expõe 4 abas independentes — Sinais Vitais, Vídeo, Áudio e
Prescrições — cada uma recebendo upload de um arquivo (não há captura em tempo real de
câmera/microfone; ver seção "Limitações"). Cada aba delega o processamento a um módulo Python
próprio (`vital_signs/`, `video/`, `audio/`, `prescriptions/`), organizado por *capability* e não
por camada técnica (decisão D7 do design). Três das quatro modalidades (Vídeo, Áudio, Sinais
Vitais) compartilham a mesma função genérica de detecção estatística de anomalia
(`anomaly/zscore.py::detect_anomalies`, decisão D2); a quarta (Prescrições) usa raciocínio de LLM
via AWS Bedrock (decisão D6), sem estatística.

Toda anomalia ou inconsistência detectada, em qualquer aba, gera um `Alert` (origem, timestamp,
descrição) empurrado para um feed compartilhado em `st.session_state`
(`alerts/feed.py`), renderizado na sidebar do `app.py` (`_render_alert_feed()`), do mais recente
para o mais antigo — simulando notificação em tempo real à equipe médica. Essa visão unificada é
adicional à exibição de alertas que cada aba já mostra inline; não a substitui.

O `Alert` carrega, além de origem/timestamp/descrição, três campos estruturados **opcionais** —
`alert_id`, `category` e `level` — preenchidos por cada aba quando fizerem sentido e retrocompatíveis
(campos ficam `None` para abas/alertas que não os informam; nenhum chamador existente quebra). São
dados de primeira classe (não só texto embutido na descrição): a aba Sinais Vitais preenche
`alert_id` (o ID de vínculo `#S01`), `category` (nome do sinal) e `level` (nível de confiança); a
Vídeo, `alert_id=event_id` (`#V01`), `category` (região do corpo) e `level` (tipo do evento); a Áudio,
`alert_id` (`#A01`, encadeado entre termo crítico → taxa de fala → pausa) e `category` ("Termo
crítico" / "Fadiga de fala" / "Disartria"); a Prescrições, `alert_id` (`#P01`, por finding) e
`category` ("Inconsistência de prescrição") e `level` (o tipo da inconsistência). Isso habilita um
export unificado a ler colunas limpas em vez de parsear a descrição.

Sobre esses campos estruturados, o feed unificado oferece um **export do relatório da sessão**:
junto do feed, um resumo mostra o total de alertas e a contagem por aba e por nível, e um botão
"Baixar relatório (CSV)" gera, em memória, um CSV tabular com todos os alertas da sessão (colunas
`id, origem, timestamp, categoria, nivel, nivel_label, descricao`; níveis ganham um rótulo amigável
reaproveitando o vocabulário de confiança de Sinais Vitais e rótulos curtos dos eventos de Vídeo).
A montagem do CSV e do resumo fica num helper puro e testável (`alerts.feed.build_alerts_report`,
que recebe a lista de `Alert` e não lê `session_state`); o `app.py` só chama o helper, exibe o
resumo e oferece o `st.download_button` (codificando o texto em UTF-8-SIG para o Excel abrir os
acentos). Sem alertas na sessão, resumo e botão não aparecem.

Fluxo resumido por aba:

```
Sinais Vitais:  CSV → validação/parse → rolling z-score (linha a linha)
                                       → Isolation Forest (lote)      → relatório combinado + Alert
Vídeo:          vídeo → YOLOv8-pose (frame a frame) → séries de ângulo por articulação + velocidade global → rolling z-score por série
                                                     → agrupamento de frames irregulares em eventos (1 Alert por evento)
                                                     → bounding box da pessoa → regra de zona crítica (opcional) → eventos
                                                                                        → relatório visual (resumo + seções colapsáveis por articulação, galeria top-N de esqueletos anotados)
Áudio:          áudio → AWS Transcribe (texto + timestamps) → AWS Comprehend (sentimento/entidades)
                                                             → busca de termos críticos
                                                             → taxa de fala / duração de pausa → rolling z-score
                                                                                                        → Alert
Prescrições:    CSV/Excel → histórico por paciente → prompt estruturado → AWS Bedrock (Claude Sonnet)
                                                                          → parsing da resposta JSON → Alert
```

## 2. Modelos/técnicas aplicados por tipo de dado, e por quê

| Modalidade | Técnica aplicada | Por quê |
|---|---|---|
| Vídeo | YOLOv8-pose (`yolov8n-pose.pt`, `ultralytics`) + rolling z-score por articulação (cotovelos, joelhos, quadris/tronco, pescoço/cabeça) e sobre a velocidade global, agrupado em eventos + regra de zona | Um único forward pass fornece keypoints humanos **e** bounding boxes no mesmo modelo (ver D1 abaixo) |
| Áudio | AWS Transcribe (fala→texto) + AWS Comprehend (sentimento/entidades) + rolling z-score sobre taxa de fala/pausa | Serviços gerenciados de nuvem, sem necessidade de treinar um ASR ou classificador de sentimento próprio (ver D5 abaixo) |
| Sinais Vitais | Rolling z-score (linha a linha) + Isolation Forest (`sklearn`, lote) | Duas camadas complementares: uma "tempo real"/explicável, outra estatística/ML validando em lote sobre o dataset completo (ver D3) |
| Prescrições | AWS Bedrock, Claude Sonnet (raciocínio de LLM, sem fine-tuning) | Dataset sintético pequeno, tarefa de interpretação semântica de texto/dados estruturados — mais adequada a um LLM instruído por prompt do que a um modelo estatístico treinado em poucos exemplos (ver D6) |

### 2.1 Justificativa: AWS em vez de Azure Cognitive Services (design.md D5)

O PDF do desafio cita literalmente **Azure Speech to Text** e **Azure Text Analytics** como
exemplos de serviço gerenciado para a análise de áudio. Esta implementação usa **AWS Transcribe**
(fala→texto) e **AWS Comprehend** (sentimento + entidades) como equivalentes funcionais, e
adicionalmente **AWS Bedrock** (Claude Sonnet) para a análise de prescrições — capability sem
equivalente citado no PDF, mas natural dentro da mesma conta cloud já escolhida.

Motivo da troca: gerenciar duas contas cloud e dois SDKs diferentes (Azure SDK + boto3) só para
dois serviços específicos, dentro de um projeto individual com prazo curto, foi julgado um custo
de engenharia desproporcional ao benefício. A equivalência funcional é direta serviço a serviço:

- Azure Speech to Text ↔ AWS Transcribe: ambos convertem áudio em texto com timestamps por
  palavra/segmento.
- Azure Text Analytics ↔ AWS Comprehend: ambos oferecem classificação de sentimento e extração
  de entidades sobre texto.

Risco assumido conscientemente: uma avaliação que exija literalmente Azure pode penalizar esta
escolha. Mitigação: esta seção documenta a equivalência explicitamente, e o código (`audio/aws_speech.py`)
comenta a escolha de região/serviço para que a substituição seja auditável.

### 2.2 Justificativa: YOLOv8-pose em vez de OpenPose + YOLOv8 separados (design.md D1)

O PDF sugere OpenPose (para análise postural) e YOLOv8 (para detecção de objetos/áreas críticas)
como exemplos ("modelos como"), não como exigência obrigatória e exclusiva. OpenPose depende de
um build Caffe antigo, difícil de instalar/manter em 2026. Optou-se por
`ultralytics` YOLOv8-pose (`yolov8n-pose.pt`), que no mesmo forward pass fornece:

1. Keypoints humanos (usados para calcular os ângulos de múltiplas articulações — cotovelos,
   joelhos, quadris/tronco e pescoço/cabeça, ambos os lados — e a velocidade de movimento global
   pelo deslocamento do centro de massa — análise postural).
2. Bounding boxes de detecção (usadas para a regra de zona crítica).

Consequência documentada do modelo escolhido: `yolov8n-pose.pt` é de detecção **single-class**
("person", dataset COCO-keypoints) — a mesma passada que extrai a pose só produz caixas de
pessoa, não de objetos genéricos (ex.: instrumento cirúrgico). A "detecção de objeto/área crítica"
implementada é, portanto, "uma **pessoa** entrando em uma zona configurada do quadro", não um
objeto arbitrário — isso exigiria um segundo modelo de propósito geral (`yolov8n.pt`, 80 classes
COCO), o que a decisão D1 evitou deliberadamente para manter um único forward pass. Essa troca de
escopo está registrada em código (`video/analysis.py`, constante `ZONE_CRITICAL_CLASSES = {0}`,
facilmente extensível se um segundo modelo for adicionado no futuro).

Alternativa considerada e rejeitada: YOLOv8 puro + MediaPipe Pose separados — exigiria dois
pipelines e duas dependências onde um modelo já resolve ambos os sub-requisitos do PDF.

## 3. As duas abordagens de IA demonstradas

O `design.md` estabelece como meta cobrir, no relatório, duas abordagens de IA distintas:

1. **Estatística / ML clássico, sem necessidade de treino "pesado":**
   - Rolling z-score (`anomaly/zscore.py`) — média/desvio-padrão móveis sobre uma janela,
     `|z| > threshold` marca anomalia. Reaproveitado, sem modificação, pelas 3 modalidades
     numéricas (ângulo/velocidade no Vídeo; taxa de fala/duração de pausa no Áudio;
     FC/PA/SpO2/temperatura/frequência respiratória em Sinais Vitais). Não exige treino,
     decisão interpretável e documentável (threshold explícito no relatório/UI).
   - Isolation Forest (`sklearn.ensemble.IsolationForest`, `vital_signs/isolation_forest.py`) —
     ajustado (`fit`) sobre o dataset completo de sinais vitais carregado, roda em lote como
     segunda camada de validação cruzada, exclusivamente em Sinais Vitais (única modalidade com
     dataset robusto suficiente para um fit útil — ver D3).
2. **Raciocínio via LLM:**
   - AWS Bedrock (Claude Sonnet), em `prescriptions/bedrock_review.py` — sem treino de modelo
     customizado, análise de inconsistências (mudança abrupta de dose, interação medicamentosa,
     falta de justificativa clínica) feita por raciocínio semântico sobre texto/dados
     estruturados via prompt estruturado em português, respondendo em JSON.

Essa divisão deliberada (estatística onde há dado numérico reaproveitável e volume suficiente;
LLM onde a tarefa é de interpretação semântica sobre pouco dado) é o eixo central da narrativa
técnica do projeto.

## 4. Resultados obtidos e exemplos reais de anomalias

Os exemplos abaixo são resultados reais, obtidos durante a implementação e validação de cada
capability (ver `.superpowers/sdd/task-{3,4,5,6}-report.md` para o detalhamento completo). Nenhum
número foi inventado — onde a validação real foi limitada, isso é indicado explicitamente.

### 4.1 Sinais Vitais — dataset real (MIMIC-III Demo)

Dataset: **MIMIC-III Clinical Database Demo v1.4** (PhysioNet), um estágio de UTI real
(`icustay_id = 250305`), 480 leituras horárias de frequência cardíaca, SpO2, frequência
respiratória e pressão arterial sistólica/diastólica (`data/vital_signs_sample.csv`) — dado
clínico real e de-identificado, não sintético.

Resultado real, com os thresholds padrão do código (`window=13`, `threshold=3.0`):
- **19 leituras marcadas pelo rolling z-score** — anomalias locais reais (picos isolados fora do
  padrão da janela recente), não ruído.
- **5 leituras de alta confiança** (`alta_confianca`) em que z-score e Isolation Forest
  (`contamination=0.05`) concordam — o rótulo que a aba se propõe a evidenciar.
- **14 leituras marcadas exclusivamente pelo Isolation Forest** (`isolation_forest_only`) e outras
  só pelo z-score (`zscore_only`) — a **divergência entre as duas camadas** ainda aparece: o
  Isolation Forest, olhando o padrão multivariado do lote completo, aponta leituras que o z-score
  linha-a-linha (cada sinal isoladamente, janela local) não captura, e vice-versa. É exatamente o
  cenário "Isolation Forest identifica anomalia não capturada pelo z-score" descrito no spec de
  `vital-signs-monitoring`.

> **Nota sobre a janela padrão (`window=13`).** O rolling z-score de `anomaly/zscore.py` usa
> desvio-padrão populacional (ddof=0) sobre uma janela que **inclui o próprio ponto**, o que limita
> o `|z|` máximo alcançável a `sqrt(janela-1)`, independentemente de quão extremo seja o pico. Com
> a antiga janela padrão 6 o teto era `sqrt(5) ≈ 2.24 < 3.0`, então a camada z-score **nunca**
> disparava nos defaults (ficava inerte). A janela padrão foi ajustada para **13** (`sqrt(12) ≈
> 3.46 > 3.0`), dando folga para picos reais cruzarem o threshold. A UI da aba avisa (via
> `st.warning`) quando o usuário escolhe manualmente uma combinação inefetiva
> (`threshold ≥ sqrt(janela-1)`), para não reintroduzir o problema silenciosamente.

**Apresentação amigável (não-técnica).** A saída da aba é apresentada em linguagem interpretável por
quem não é técnico, sem alterar a detecção (só apresentação — mesmo espírito da aba Vídeo). As duas
camadas aparecem como **"Detecção em tempo real"** (o rolling z-score) e **"Análise do histórico
completo"** (o Isolation Forest), com o termo técnico e o que faz no tooltip/caption. Os controles são
**"Sensibilidade"** (o threshold) e **"Tamanho da janela de comparação"** (a janela), cada um com um
`help=` explicando em linguagem simples e com exemplo o efeito de aumentar/diminuir. Após processar, um
**resumo no topo** (helper puro e testado `build_vitals_summary` em `vital_signs/analysis.py`) informa
quantas leituras ficaram fora do padrão e destaca, em bullets, as principais — sinal vital traduzido,
valor e horário. A tabela substitui as colunas cruas (`zscore_anomaly`/`agreement`/`heart_rate`...) por
cabeçalhos amigáveis (**ID, Horário, Sinal vital, Valor, Nível de confiança**) com os nomes dos sinais
traduzidos via `vital_sign_label` (heart_rate → "Frequência cardíaca", spo2 → "Saturação de O₂ (SpO₂)"
etc.). Os três níveis de confiança são explicados com rótulo, ícone e tooltip via `confidence_level`:
🔴 **Alta confiança** (as duas análises concordam — mais provável ser real), 🟠 **Só tempo real** (pico
isolado momentâneo) e 🟡 **Só histórico** (fora do padrão geral, sem pico súbito), com uma legenda
compacta abaixo da tabela. Quando nenhuma leitura é anômala, a aba mostra uma mensagem clara de que
nada fora do padrão foi encontrado.

Cada leitura anômala recebe, no `analyze`, um **identificador de vínculo** curto e determinístico
(`#S01`, `#S02`, ... em ordem de linha, gravado na coluna `id` do `combined_report`) — paralelo do
`#V01` do vídeo. O mesmo ID aparece na coluna **ID** da tabela e no `alert_id` do alerta
correspondente, permitindo casar alerta↔linha mesmo com ordens de exibição diferentes. Como o alerta
de z-score é por (linha, sinal) mas a tabela é por linha, o ID identifica a **linha**: se vários
sinais disparam na mesma leitura, seus alertas compartilham o ID daquela linha (cada um com sua
`category` = nome do sinal). No bloco inline "Alertas gerados", cada alerta passou a ser renderizado
com o **ícone/cor do seu nível** (`confidence_level(alert.level)` — 🔴 alta confiança via `st.error`,
demais via `st.warning`), prefixado pelo ícone e pelo `alert_id`, em vez de um `st.warning` amarelo
uniforme — consistente com a tabela e a legenda.

> **Sinal responsável por uma leitura (escolha documentada).** `analyze` grava só um booleano
> por linha (`zscore_anomaly` = *alguma* coluna disparou), não um flag por sinal. Sem modificar a
> detecção, `build_vitals_summary` deriva o "sinal responsável" como a coluna cujo valor é o **mais
> extremo** em relação à própria distribuição da série (`|valor - média| / desvio`). Quando a leitura
> foi marcada **só** pelo Isolation Forest (sem z-score), não há coluna única responsável e o item é
> rotulado **"padrão geral"**. É um proxy de apresentação: o proxy é **global** (`|valor - média| /
> desvio` sobre toda a série), enquanto o z-score é **local** (compara com a janela recente); portanto
> o sinal apontado pode divergir do que realmente disparou o z-score — não só quando vários sinais
> sobem juntos, mas mesmo numa linha de um único sinal, se outro sinal for globalmente mais extremo
> naquele instante. Nunca altera a detecção.

### 4.2 Prescrições — AWS Bedrock, achados reais sobre o dataset sintético

Dataset sintético (`data/prescricoes_sinteticas.csv`, 3 pacientes, criado manualmente por falta de
fonte pública apropriada — ver seção "Limitações"): Paciente A (Losartana 50mg estável, caso
normal), Paciente B (Metformina 500mg → 2000mg em 7 dias, mudança abrupta de dose), Paciente C
(Warfarina + Aspirina co-prescritas, possível interação medicamentosa). Há também um par "sem
anomalias" (`data/prescricoes_sinteticas_normal.csv`, mesmos 3 pacientes, dose constante e sem a
combinação de risco), reproduzível por `scripts/gen_prescriptions_demo.py`.

Chamada real ao Bedrock (perfil `bedrock`, região `us-east-2`, modelo
`us.anthropic.claude-sonnet-5`) para o **Paciente B** retornou:

```json
[
  {
    "tipo": "mudanca_dose_abrupta",
    "explicacao": "A dose de Metformina passou de 500mg (em 2026-01-17) para 2000mg (em 2026-01-24), um aumento de 4 vezes em apenas 7 dias, sem evidência de titulação gradual..."
  },
  {
    "tipo": "dose_sem_justificativa",
    "explicacao": "Não há registro de exames, sintomas ou avaliação clínica que justifique o aumento da dose de Metformina..."
  }
]
```

Ambos os achados geraram `Alert(origin="Prescrições", ...)` no feed compartilhado. Para o
**Paciente A** (caso normal), a chamada real ao Bedrock retornou `[]` — nenhuma inconsistência —
confirmando também o caminho "sem achados" com uma resposta real do modelo, não apenas com um
mock. O caso do Paciente C (interação medicamentosa) foi validado via testes unitários mockados,
não com uma segunda chamada real, para respeitar o orçamento de chamadas reais definido na
tarefa (ver `.superpowers/sdd/task-6-report.md`).

### 4.3 Vídeo — smoke run real contra o modelo YOLOv8-pose

Vídeo de demonstração: `data/demo_pose_walk.mp4`, recorte de 6s (re-encodado para mp4/h264,
480×360) do vídeo de amostra `vtest.avi` distribuído com o próprio OpenCV (uso livre), mostrando
pedestres caminhando — usado como fallback documentado na ausência de um vídeo de fisioterapia
específico obtido sem credenciais.

Execução real ponta a ponta (via `streamlit.testing.v1.AppTest`, upload real de
`data/demo_pose_walk.mp4` → botão "Processar vídeo", inferência YOLOv8-pose real, sem mocks),
com a sensibilidade sugerida automática (`threshold ≈ 2.0`) e a zona crítica **desativada**
(padrão):

```
frames lidos: 60          fps: 10.0
frames com pose detectada: 47 / 60 (13 sem dados de pose)
eventos irregulares detectados: 10
articulação mais afetada: Joelho direito
alertas gerados: 10        (1 por evento, não por frame)
seções colapsáveis no relatório: 6 (Joelho direito 3, Joelho esquerdo 2, Quadril esquerdo 2,
                                     Cotovelo esquerdo 1, Pescoço 1, Movimento brusco 1)
imagens do relatório visual: 10 (um esqueleto anotado por evento, em galeria por seção)
% do vídeo estimado como irregular no nível atual: ~12%
```

O relatório visual agrupa os eventos **por articulação/tipo em seções colapsáveis**
(`st.expander`, fechadas por padrão), ordenadas da mais afetada (mais eventos) para a menos; nesta
execução foram 6 seções — a mais afetada, "Joelho direito", com 3 eventos. Dentro de cada seção,
os **10 eventos mais graves** (maior |z-score|; para zona crítica, maior área de interseção, já
que `z_pior` é `NaN` por construção) são exibidos numa **galeria em grade de 3 colunas** — cada
célula com a imagem do frame mais representativo (o de maior |z-score| do grupo), o esqueleto COCO
desenhado e a articulação afetada destacada em vermelho, e como legenda o **identificador curto
único do evento** (`#V01`, `#V02`, ...) seguido do intervalo de tempo (ex.: `#V02 — 2.0s a 2.2s`);
eventos de velocidade destacam o corpo todo e, quando a zona crítica é ligada, os eventos de zona
desenham o retângulo configurado. Quando uma seção tem mais eventos que o limite, a UI indica
"Mostrando N de M". Esse agrupamento (change `galeria-eventos-video-por-articulacao`) substitui a
lista plana de uma imagem por evento — que num vídeo real chegou a ~500 imagens e travava a
página — mantendo a saída navegável mesmo com centenas de eventos; a lógica de agrupamento/
ordenação/top-N vive em `video.analysis.group_events_for_display` (testada), e `app.py` apenas
renderiza. A contagem de eventos e de alertas não muda (o agrupamento é só de apresentação: 1
`Alert` por evento). Cada evento recebe, no `analyze` (após a ordenação cronológica final e antes
de gerar os alertas), um **identificador curto único** `event_id` (`#V01`, `#V02`, ...,
determinístico e sequencial) gravado no próprio dict do evento; esse mesmo `event_id` é lido tanto
pelo texto do alerta quanto pela legenda da galeria, então os dois sempre casam mesmo com ordens de
exibição diferentes (feed cronológico vs. galeria por gravidade). O texto do alerta ganha ainda a
**categoria/região do corpo** afetada — via `event_category` (`JOINT_CATEGORY`: pescoço→Cabeça,
cotovelos→Braços, quadris→Tronco, joelhos→Pernas; velocidade→Corpo; zona→Zona de risco) — ficando,
por exemplo, `#V02 [Braços] Cotovelo direito irregular entre 2.0s e 2.2s.` no feed, com a foto
correspondente na galeria legendada `#V02 — 2.0s a 2.2s` (change `alerta-video-id-categoria`; id e
categoria vão embutidos no `description`, sem alterar o dataclass `Alert` nem `alerts/feed.py`). O
**agrupamento de frames irregulares consecutivos em eventos** já substituía
o antigo alerta por frame (que gerava dezenas de alertas quase idênticos). Evidência de que a
pipeline completa (extração de pose multi-articulação → z-score por série → agrupamento em eventos
→ relatório visual agrupado por articulação, com zona opcional) funciona de ponta a ponta contra o
modelo real, e não só contra dados de teste sintéticos.

### 4.4 Áudio — validação real limitada, honestamente reportada

Áudio de demonstração: `data/demo_consulta_audio.mp3`, um clipe curto (~10s) de fala sintética em
pt-BR gerada via TTS (`gTTS`), com um roteiro contendo termos críticos ("dor no peito", "não
consigo respirar", "tontura") — não foi encontrada rapidamente uma fonte pública gratuita de
áudio médico real curto, então o clipe foi gerado como fallback documentado.

Uma chamada real a cada serviço (Transcribe, Comprehend) foi feita durante a validação: a
transcrição reproduziu exatamente o roteiro (18 palavras com timestamp), o sentimento retornado
pelo Comprehend foi `NEGATIVE` (99,75% de confiança) e 3 entidades foram extraídas. A busca de
termos críticos (lógica local, sem chamada AWS) encontrou corretamente "dor", "não consigo
respirar", "dor no peito" e "tontura" no texto transcrito real.

**Honestamente**: por se tratar de um clipe curto e sintético (sem variação natural real de fadiga
vocal), não há um exemplo real de anomalia de taxa de fala/duração de pausa detectada nesse clipe
— a validação real do módulo `audio/analysis.py` (rolling z-score sobre taxa de fala/pausa) foi
feita majoritariamente via testes unitários com séries sintéticas construídas para exercitar o
caminho de detecção (ex.: `test_analyze_flags_anomaly_and_raises_alert_when_pause_is_extreme`), não
contra um áudio real com um evento de fadiga genuíno. Isso é uma limitação da validação real desta
modalidade, não da lógica implementada — a mesma função `detect_anomalies` já validada nas outras
duas modalidades é reutilizada sem alteração.

## 5. Limitações conhecidas

- **Sem captura em tempo real de câmera/microfone.** O requisito implícito do PDF de
  "monitoramento contínuo" é simulado via upload de arquivo, não streaming ao vivo — decisão de
  escopo por limitação do ambiente de desenvolvimento (WSL, sem passthrough fácil de
  câmera/microfone). A mesma pipeline (extração de features → `detect_anomalies` →
  `alerts.feed.add_alert`) se estenderia a um fluxo de streaming real trocando apenas a fonte de
  entrada; isso é trabalho futuro, não implementado aqui.
- **Rastreamento postural dependente de keypoints visíveis.** `video/pose.py` rastreia múltiplas
  articulações (cotovelos, joelhos, quadris/tronco e pescoço/cabeça, ambos os lados) e uma
  velocidade de movimento global, mas cada ângulo só é calculado nos frames em que os keypoints
  necessários daquela articulação foram detectados (keypoints faltantes → articulação sem valor no
  frame). Na prática, enquadramentos parciais (planos superiores de corpo, comuns em demos de
  fisioterapia) produzem séries mais curtas/ruidosas para joelhos e quadris do que para cotovelos e
  pescoço. A detecção degrada de forma graciosa (a articulação simplesmente não contribui naquele
  frame), mas a cobertura efetiva por articulação depende do enquadramento do vídeo.
- **Detecção de "objeto crítico" limitada a pessoas.** Como descrito na seção 2.2, o modelo
  `yolov8n-pose.pt` só detecta a classe "person" no mesmo forward pass que extrai a pose. A regra
  de zona crítica implementada, portanto, sinaliza entrada de **pessoas** em uma área configurada,
  não objetos genéricos (ex.: instrumental). Adicionar um segundo modelo de detecção geral
  (`yolov8n.pt`, 80 classes COCO) resolveria isso, ao custo de duas passadas de inferência por
  frame em vez de uma — trade-off explicitamente evitado pela decisão D1.
- **Dataset sintético de prescrições, pequeno.** `data/prescricoes_sinteticas.csv` tem apenas 3
  pacientes, criados manualmente porque não foi localizada rapidamente uma fonte pública
  apropriada de histórico de prescrições com casos de mudança abrupta de dose/interação
  medicamentosa. Isso é suficiente para demonstrar o raciocínio do Bedrock qualitativamente (ver
  seção 4.2), mas não permite qualquer afirmação estatística sobre taxa de acerto/erro do modelo
  em escala.
- **Bedrock só validado em `us-east-2`, não em `us-east-1`.** Para a conta AWS usada no
  desenvolvimento, `us-east-1` retornou `ResourceNotFoundException` ("Model use case details have
  not been submitted for this account") de forma consistente em ~10 tentativas ao longo de vários
  minutos — um bloqueio de nível de conta/formulário de uso, não um problema transitório ou de
  permissão (`list_foundation_models`/`list_inference_profiles` funcionam normalmente nessa
  região). A região `us-east-2` (e também `us-west-2`, testada) funcionou imediatamente. Contas
  AWS diferentes podem não ter essa restrição.
- **Threshold/janela fixos em Áudio e no rolling z-score de Sinais Vitais (parcialmente).** Por
  decisão de design (D4), apenas o slider de sensibilidade da aba Vídeo (e o par
  window/threshold da aba Sinais Vitais, que também é ajustável na UI) é interativo; o
  threshold/janela da detecção de anomalia de fala em Áudio (`DEFAULT_WINDOW=5`,
  `DEFAULT_THRESHOLD=2.5`) é fixo, documentado em código (`audio/analysis.py`), sem controle na
  UI — não havia requisito de interatividade para essa aba.

## 6. Referências internas

- Decisões de design detalhadas: `openspec/changes/monitoramento-multimodal-pacientes/design.md`
  (D1–D7).
- Requisitos formais por capability: `openspec/changes/monitoramento-multimodal-pacientes/specs/*/spec.md`.
- Relatórios de execução de cada tarefa (evidência de TDD, chamadas AWS reais, bugs encontrados e
  corrigidos): `.superpowers/sdd/task-{1,...,7}-report.md`.
