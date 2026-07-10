## ADDED Requirements

### Requirement: Upload e transcrição de áudio de consulta
O sistema SHALL aceitar upload de um arquivo de áudio de consulta médica através da aba Áudio da aplicação Streamlit e transcrevê-lo usando AWS Transcribe, obtendo texto e timestamps por palavra ou segmento.

#### Scenario: Upload de áudio válido
- **WHEN** o usuário faz upload de um arquivo de áudio em formato suportado (mp3 ou wav)
- **THEN** o sistema envia o áudio para AWS Transcribe e exibe o texto transcrito com os timestamps correspondentes

#### Scenario: Upload de arquivo em formato não suportado
- **WHEN** o usuário faz upload de um arquivo que não é áudio
- **THEN** o sistema rejeita o upload e exibe mensagem de erro indicando os formatos aceitos

### Requirement: Análise de sentimento e termos críticos via AWS Comprehend
O sistema SHALL enviar o texto transcrito para AWS Comprehend para obter classificação de sentimento e extração de entidades, e SHALL destacar termos críticos pré-definidos (ex.: "dor", "não consigo respirar") encontrados no texto.

#### Scenario: Texto transcrito contém termo crítico
- **WHEN** o texto transcrito contém um dos termos críticos configurados
- **THEN** o sistema gera um alerta indicando o termo crítico encontrado e o trecho de contexto correspondente

#### Scenario: AWS Comprehend retorna sentimento negativo
- **WHEN** o AWS Comprehend classifica o sentimento do texto transcrito como negativo com alta confiança
- **THEN** o sistema exibe essa classificação de sentimento junto ao relatório da aba Áudio

### Requirement: Extração de features acústicas e detecção de anomalia de fala
O sistema SHALL derivar, a partir dos timestamps retornados pelo AWS Transcribe, séries de taxa de fala e duração de pausa, e SHALL aplicar rolling z-score sobre essas séries para detectar variações compatíveis com fadiga ou disartria.

#### Scenario: Segmento de fala com taxa anômala
- **WHEN** o z-score da taxa de fala ou da duração de pausa em um segmento excede o threshold configurado
- **THEN** o sistema marca o segmento como possível indicador de fadiga ou disartria e registra seu timestamp

#### Scenario: Áudio sem variação significativa
- **WHEN** nenhum segmento do áudio excede o threshold de anomalia de fala
- **THEN** o sistema exibe relatório indicando ausência de indicadores de fadiga ou disartria detectados
