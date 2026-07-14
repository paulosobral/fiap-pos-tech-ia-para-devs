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
  rolling z-score linha a linha (alerta "em tempo real", threshold/janela ajustáveis na UI) e
  Isolation Forest treinado em lote sobre o CSV carregado (validação complementar). O relatório
  combinado destaca leituras em que as duas camadas concordam ("alta confiança") ou divergem.
- **Vídeo** — upload de vídeo (mp4/avi/mov/mkv) de fisioterapia/exercício ou cirurgia. Processa
  frame a frame com YOLOv8-pose, extraindo o ângulo do cotovelo direito e a velocidade do punho
  direito por frame. Duas detecções independentes: anomalia postural via rolling z-score
  (slider de sensibilidade na UI) e alerta de zona crítica (interseção da bounding box da
  pessoa detectada com uma área configurável do quadro, também via sliders). Gera um relatório
  de desvios ordenado por timestamp.
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
  adicional à exibição inline de alertas que cada aba já mostra, não a substitui.

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
tem 98 testes e passa integralmente (`98 passed`).

Observação: os testes automatizados cobrem a lógica de cada módulo isoladamente (com mocks para
as chamadas AWS); eles não exercitam a UI do Streamlit ponta a ponta. Há uma incompatibilidade
conhecida entre `streamlit.testing.v1.AppTest` e `pandas`/`pyarrow` neste ambiente (segfault ao
processar upload de CSV) — ver "Limitações conhecidas" no relatório técnico.

## Pendências

Os itens 8.3 e 8.4 do plano de entrega são passos manuais, fora do escopo desta implementação,
e ficam a cargo do usuário:

- **Gravar vídeo de demonstração** (até 15 minutos) cobrindo análise de áudio e vídeo, detecção
  e resposta a anomalias, integração dos serviços AWS e o fluxo final de alerta à equipe médica.
- **Publicar o vídeo** no YouTube ou Vimeo (público ou não listado) e adicionar o link a este
  README.

Nenhum código foi escrito para automatizar esses dois passos — são ações humanas por natureza
(gravar e publicar um vídeo).
