## ADDED Requirements

### Requirement: Upload de série temporal de sinais vitais
O sistema SHALL aceitar upload de um arquivo CSV contendo série temporal de sinais vitais (frequência cardíaca, pressão arterial, oxigenação) através da aba Sinais Vitais da aplicação Streamlit.

#### Scenario: Upload de CSV válido
- **WHEN** o usuário faz upload de um CSV com colunas de timestamp e ao menos um sinal vital reconhecido
- **THEN** o sistema carrega a série temporal e exibe uma visualização gráfica dos sinais vitais ao longo do tempo

#### Scenario: Upload de CSV com colunas inválidas
- **WHEN** o usuário faz upload de um CSV sem nenhuma coluna de sinal vital reconhecida
- **THEN** o sistema rejeita o arquivo e exibe mensagem de erro indicando as colunas esperadas

### Requirement: Detecção de anomalia em tempo real via rolling z-score
O sistema SHALL aplicar rolling z-score linha a linha sobre cada série de sinal vital carregada, marcando como anômala qualquer leitura cujo z-score absoluto exceda o threshold configurado, simulando alerta em tempo real.

#### Scenario: Leitura de sinal vital fora do padrão
- **WHEN** uma leitura de sinal vital tem z-score absoluto acima do threshold configurado
- **THEN** o sistema marca a leitura como anômala e gera um alerta referenciando o timestamp e o sinal vital afetado

### Requirement: Validação batch via Isolation Forest
O sistema SHALL treinar um modelo Isolation Forest sobre a série temporal completa carregada e usá-lo para identificar leituras anômalas em lote, apresentando o resultado como camada de validação complementar ao rolling z-score.

#### Scenario: Isolation Forest identifica anomalia não capturada pelo z-score
- **WHEN** o Isolation Forest classifica uma leitura como anômala mesmo que seu z-score individual esteja abaixo do threshold
- **THEN** o sistema exibe essa leitura no relatório da aba como anomalia identificada exclusivamente pela camada Isolation Forest

#### Scenario: Concordância entre as duas camadas de detecção
- **WHEN** uma leitura é marcada como anômala tanto pelo rolling z-score quanto pelo Isolation Forest
- **THEN** o sistema exibe essa leitura destacada como anomalia de alta confiança no relatório
