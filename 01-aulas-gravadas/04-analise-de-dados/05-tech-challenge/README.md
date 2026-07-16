# Tech Challenge Fase 4 — Monitoramento Multimodal de Pacientes

Aplicação Streamlit única que implementa o desafio da Fase 4 (8IADT): monitoramento contínuo
de pacientes a partir de dados multimodais — vídeo, áudio, sinais vitais e prescrições — com
detecção de anomalias em tempo real, integração a serviços gerenciados em nuvem (AWS) e um
feed unificado de alertas simulando notificação à equipe médica.

Projeto individual, entrega via upload de arquivo em 4 abas (sem captura real-time de
câmera/microfone — ver "Limitações conhecidas" no relatório técnico para o motivo).

## Pré-requisitos

- Python 3.11+.
- Conta AWS com acesso aos serviços **Transcribe**, **Comprehend** e **Bedrock** (modelo Claude
  Sonnet habilitado na região utilizada).
- Credenciais AWS configuradas em `~/.aws/credentials` / `~/.aws/config`. Este projeto usa dois
  perfis:
  - `default` — usado como fallback geral (as abas Sinais Vitais e Vídeo não fazem nenhuma
    chamada AWS).
  - `bedrock` — usado pela aba Prescrições (`prescriptions/bedrock_review.py`). **Importante:**
    nesta implementação o Bedrock só funcionou com o perfil `bedrock` na região **`us-east-2`**,
    com o inference profile `us.anthropic.claude-sonnet-5`. A região `us-east-1` retornou
    consistentemente `ResourceNotFoundException` ("Model use case details have not been
    submitted for this account") para a conta usada no desenvolvimento — se você rodar com uma
    conta AWS diferente, talvez seja necessário ajustar `AWS_REGION`/`BEDROCK_MODEL_ID` em
    `prescriptions/bedrock_review.py` conforme o que estiver disponível na sua conta.
  - A aba Áudio (`audio/aws_speech.py`) usa `region_name="us-east-1"` fixo para Transcribe e
    Comprehend (hardcoded no módulo, independente do perfil ativo).
- **Bucket S3 para a aba Áudio**: o AWS Transcribe exige um bucket S3 para entrada/saída do job
  assíncrono de transcrição. Defina a variável de ambiente `AUDIO_TRANSCRIBE_BUCKET` com o nome
  de um bucket já existente e acessível pelas credenciais configuradas, por exemplo:
  ```bash
  export AUDIO_TRANSCRIBE_BUCKET=meu-bucket-transcribe
  ```
  Sem essa variável, a aba Áudio exibe um erro e bloqueia o processamento (não tenta a chamada
  AWS). Não há provisionamento automático do bucket — crie-o manualmente antes de usar a aba,
  por exemplo `aws s3 mb s3://meu-bucket-transcribe --region us-east-1`. O código limpa os
  objetos temporários e o job do Transcribe após cada uso (best-effort), então o bucket não
  acumula arquivos de áudio ao longo do tempo.
- **Modelo YOLOv8-pose (`yolov8n-pose.pt`)**: baixado automaticamente pela biblioteca
  `ultralytics` na primeira execução da aba Vídeo (requer acesso à internet nesse momento). Para
  antecipar o download antes de uma demo:
  ```bash
  venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
  ```

## Setup

```bash
cd 01-aulas-gravadas/04-analise-de-dados/05-tech-challenge/
python -m venv venv
source venv/bin/activate      # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`ultralytics` traz uma árvore de dependências grande (torch/torchvision), então a instalação
pode demorar e ocupar bastante espaço em disco na primeira vez.

`requirements.txt` fixa `pyarrow<25` explicitamente. `pyarrow==25.0.0` tem um bug real de
segfault (`libarrow.so`, dentro de `pyarrow/pandas_compat.py::convert_column`) ao renderizar
`st.dataframe`/`st.table` quando `scikit-learn` está importado no processo — o que este app
sempre faz, para a aba Sinais Vitais — **e** a renderização ocorre num rerun do Streamlit
posterior ao primeiro (ex.: dentro de `if st.button(...):`, como em "Processar vídeo"). Sem
esse pin, uma reinstalação do `venv/` pode resolver `pyarrow` para a série 25.x e a aplicação
trava com `segmentation fault` ao clicar em qualquer botão de processamento que exiba uma
tabela.

## Como executar

```bash
source venv/bin/activate
export AUDIO_TRANSCRIBE_BUCKET=meu-bucket-transcribe   # necessário para a aba Áudio
streamlit run app.py
```

A aplicação abre em `http://localhost:8501` com 4 abas e uma sidebar de alertas.

## As 4 abas + feed de alertas

- **Sinais Vitais** — upload de CSV de série temporal (frequência cardíaca, SpO2, pressão
  arterial, frequência respiratória, temperatura). Aplica duas camadas de detecção de anomalia:
  rolling z-score linha a linha (alerta "em tempo real", threshold/janela ajustáveis na UI;
  padrões `janela=13`, `threshold=3.0`) e Isolation Forest treinado em lote sobre o CSV carregado
  (validação complementar). O relatório combinado destaca leituras em que as duas camadas
  concordam ("alta confiança") ou divergem. O z-score usa desvio-padrão populacional sobre uma
  janela que inclui o próprio ponto, então o `|z|` máximo alcançável é `sqrt(janela-1)`; a janela
  padrão 13 (`sqrt(12) ≈ 3.46 > 3.0`) mantém a camada efetiva, e a UI avisa quando uma combinação
  manual `threshold ≥ sqrt(janela-1)` tornaria a detecção inerte.
  A **saída é apresentada em linguagem amigável** (não-técnica), sem mudar a detecção: as camadas
  aparecem como "Detecção em tempo real" (o z-score) e "Análise do histórico completo" (o Isolation
  Forest), com o termo técnico no tooltip; os controles são "Sensibilidade" (threshold) e "Tamanho da
  janela de comparação" (janela), cada um com tooltip explicando o efeito com exemplo. Ao processar, há
  um **resumo no topo** (quantas leituras fora do padrão + bullets com o sinal, valor e horário das
  principais) e uma **tabela traduzida** (Horário, Sinal vital, Valor, Nível de confiança) com nomes de
  sinais em português (heart_rate → "Frequência cardíaca" etc.) e uma coluna **ID** (`#S01`, `#S02`,
  ...) por leitura anômala. Os três **níveis de confiança** são
  explicados com ícone + frase e tooltip: 🔴 Alta confiança (as duas concordam), 🟠 Só tempo real (pico
  isolado) e 🟡 Só histórico (fora do padrão geral). Cada leitura anômala recebe um **identificador de
  vínculo** curto (`#S01`, `#S02`, ...) que aparece tanto na linha da tabela quanto no alerta
  correspondente — paralelo do `#V01` do vídeo — permitindo casar alerta↔linha mesmo quando as ordens
  diferem; o ID identifica a **linha** (leitura), então múltiplos sinais que disparam na mesma linha
  compartilham o ID dela. No bloco inline "Alertas gerados", cada alerta é renderizado com o **ícone/cor
  do seu nível** (🔴 alta confiança via `st.error`, demais via `st.warning`) em vez de um `st.warning`
  uniforme, consistente com a tabela e a legenda. Sem anomalias, mostra uma mensagem clara de "nada
  fora do padrão". Os helpers de apresentação (`vital_sign_label`, `confidence_level`,
  `build_vitals_summary`) são puros e testados em `vital_signs/analysis.py`, mantendo `app.py` fino.
  Para demonstrar o contraste "sem alertas" vs. "com alertas", há dois CSVs de demo em `data/`,
  derivados da amostra MIMIC-III real: `vital_signs_sample_normal.csv` (nenhuma leitura anômala em
  nenhuma das duas camadas — toda leitura sai `normal`) e `vital_signs_sample_anomalias.csv` (os
  mesmos dados com 3 episódios clínicos de pico injetados — taquicardia, hipoxemia e pico
  hipertensivo — que fazem as duas camadas concordarem em `alta_confianca` nas leituras de pico,
  mantendo as leituras ao redor `normal`). Ambos são reproduzíveis pelo gerador versionado
  `scripts/gen_vital_signs_demo.py` (`venv/bin/python scripts/gen_vital_signs_demo.py`), que ao
  final roda `analyze` sobre cada arquivo nos parâmetros padrão e falha se o normal tiver qualquer
  anomalia ou o anômalo não tiver ao menos uma `alta_confianca`.
- **Vídeo** — upload de vídeo (mp4/avi/mov/mkv) de fisioterapia/exercício ou cirurgia. O
  processamento é disparado pelo botão "Processar vídeo" (a extração de pose fica em cache para
  não reprocessar a cada rerun). Com YOLOv8-pose o sistema rastreia **múltiplas articulações** do
  corpo por frame — cotovelos, joelhos, quadris/tronco e pescoço/cabeça, ambos os lados — além de
  uma velocidade de movimento global (deslocamento do centro de massa) e das detecções de pessoa
  no mesmo forward pass. Cada articulação (e a velocidade) passa por rolling z-score; os frames
  irregulares consecutivos são **agrupados em eventos** (um alerta por evento, não por frame),
  cada evento com um **identificador curto único** (`#V01`, `#V02`, ...), a **categoria/região do
  corpo** afetada (Cabeça, Braços, Tronco, Pernas, Corpo para velocidade, Zona de risco) e o
  intervalo de tempo — o texto do alerta fica, por exemplo,
  `#V02 [Braços] Cotovelo direito irregular entre 2.0s e 2.2s.`. A saída é um **relatório visual**:
  um resumo no topo (número de eventos + articulação mais afetada) e, em seguida, os eventos
  **agrupados por articulação/tipo em seções colapsáveis** (`st.expander`, fechadas por padrão),
  ordenadas da mais afetada (mais eventos) para a menos. Cada seção mostra os **10 eventos mais
  graves** daquela articulação numa **galeria em grade** (3 colunas) — cada célula com a imagem do
  frame mais representativo, esqueleto desenhado e articulação afetada destacada, e como legenda o
  **mesmo identificador** do alerta seguido do intervalo de tempo (ex.: `#V02 — 2.0s a 2.2s`),
  permitindo casar o alerta do feed (cronológico) com sua foto na galeria (ordenada por gravidade);
  quando há mais eventos que o limite, exibe "Mostrando N de M". Esse
  agrupamento mantém a página navegável mesmo em vídeos com centenas de eventos (a lista plana
  anterior renderizava uma imagem por evento). Uma
  sensibilidade sugerida é calculada automaticamente para o vídeo e pré-popula o slider (ponto de
  partida ajustável), com uma legenda auto-explicativa estimando o "~X% do vídeo" que seria
  marcado como irregular no nível atual. A detecção de zona crítica (pessoa entrando em uma área
  configurável do quadro) é **opcional, desativada por padrão** via checkbox; quando ligada,
  mostra os sliders de zona e uma prévia do retângulo sobre o primeiro frame.
- **Áudio** — upload de áudio (mp3/wav) de consulta médica. Transcreve via AWS Transcribe
  (texto + timestamps por palavra), analisa sentimento e entidades via AWS Comprehend, busca
  termos críticos configuráveis (ex.: "dor", "não consigo respirar") no texto transcrito, e
  deriva séries de taxa de fala e duração de pausa dos timestamps, passando-as pelo mesmo
  detector de rolling z-score para sinalizar possível fadiga/disartria.
- **Prescrições** — upload de histórico de prescrições por paciente (CSV/Excel). Para o
  paciente selecionado, envia o histórico ao AWS Bedrock (Claude Sonnet), que aponta
  inconsistências via raciocínio de linguagem: mudança abrupta de dose, possível interação
  medicamentosa ou alteração sem justificativa clínica aparente — sem treino de modelo
  customizado.
- **Feed de Alertas (sidebar)** — acumula, em `st.session_state`, todos os alertas gerados por
  qualquer uma das 4 abas durante a sessão, exibidos do mais recente para o mais antigo com
  origem, timestamp e descrição — simulando notificação em tempo real à equipe médica. É
  adicional à exibição inline de alertas que cada aba já mostra, não a substitui. Além de origem/
  descrição/timestamp, o `Alert` compartilhado (`alerts/feed.py`) carrega três campos estruturados
  **opcionais** — `alert_id` (identificador de vínculo, ex.: `#S01`/`#V03`), `category` (ex.: nome
  do sinal vital, região do corpo, "Termo crítico", "Inconsistência de prescrição") e `level` (ex.:
  o nível de confiança em Sinais Vitais) — preenchidos por cada aba quando aplicável e retrocompatíveis
  (abas que não os fornecem seguem funcionando, com os campos em `None`). Servem de base para um export
  unificado ler colunas limpas em vez de parsear o texto.

Detalhes de arquitetura e decisões de design (incluindo por que AWS no lugar de Azure e
YOLOv8-pose no lugar de OpenPose) estão em
`openspec/changes/monitoramento-multimodal-pacientes/design.md` e no relatório técnico
(`RELATORIO_TECNICO.md`).

## Estrutura de diretórios

```
anomaly/        # detecção de anomalia compartilhada (rolling z-score)
video/          # análise de vídeo (YOLOv8-pose)
audio/          # análise de áudio (AWS Transcribe + Comprehend)
vital_signs/    # sinais vitais (z-score + Isolation Forest)
prescriptions/  # revisão de prescrições (AWS Bedrock)
alerts/         # feed de alertas compartilhado
data/           # datasets/mídia de demonstração
tests/          # suíte de testes (pytest)
```

## Rodando os testes

```bash
venv/bin/python -m pytest tests/ -q
```

Não é necessário definir `PYTHONPATH=.` manualmente — o `pytest`, ao rodar a partir da raiz do
projeto (onde estão os pacotes `anomaly/`, `video/`, `vital_signs/` etc., sem `src/` layout),
já resolve os imports corretamente por conta própria (confirmado executando o comando acima com
`PYTHONPATH` explicitamente vazio). No momento em que esta seção foi escrita, a suíte completa
tem 136 testes e passa integralmente (`136 passed`).

Observação: os testes automatizados cobrem a lógica de cada módulo isoladamente (com mocks para
as chamadas AWS); eles não exercitam a UI do Streamlit ponta a ponta.

## Pendências

Os itens 8.3 e 8.4 do plano de entrega são passos manuais, fora do escopo desta implementação,
e ficam a cargo do usuário:

- **Gravar vídeo de demonstração** (até 15 minutos) cobrindo análise de áudio e vídeo, detecção
  e resposta a anomalias, integração dos serviços AWS e o fluxo final de alerta à equipe médica.
- **Publicar o vídeo** no YouTube ou Vimeo (público ou não listado) e adicionar o link a este
  README.

Nenhum código foi escrito para automatizar esses dois passos — são ações humanas por natureza
(gravar e publicar um vídeo).
