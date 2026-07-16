## ADDED Requirements

### Requirement: Amostras de demonstração de sinais vitais (sem e com anomalias)
O sistema SHALL fornecer dois conjuntos de dados de demonstração de sinais vitais, versionados e derivados da amostra MIMIC-III real do projeto por um script gerador reproduzível: um arquivo sem anomalias (nenhuma das duas camadas de detecção — rolling z-score e Isolation Forest — marca qualquer leitura) e um arquivo com anomalias (eventos clínicos plausíveis injetados que fazem as duas camadas concordarem em ao menos uma leitura de alta confiança). Ambos SHALL usar as colunas reconhecidas pela aba Sinais Vitais e carregar sem erro por ela.

#### Scenario: Arquivo sem anomalias não gera nenhum alerta
- **WHEN** o arquivo de demonstração sem anomalias é processado pela aba Sinais Vitais
- **THEN** nenhuma leitura é marcada como anômala por qualquer uma das duas camadas (rolling z-score e Isolation Forest), e nenhum alerta de sinais vitais é gerado

#### Scenario: Arquivo com anomalias gera alertas de alta confiança
- **WHEN** o arquivo de demonstração com anomalias é processado pela aba Sinais Vitais
- **THEN** ao menos uma leitura é marcada como anômala por AMBAS as camadas (alta confiança), gerando alerta(s), enquanto as leituras normais ao redor permanecem não anômalas

#### Scenario: Amostras reproduzíveis a partir da base real
- **WHEN** o script gerador é executado sobre a amostra MIMIC-III do projeto
- **THEN** ele produz os dois arquivos de demonstração de forma determinística e verifica, ao final, que o arquivo sem anomalias não tem anomalias e que o arquivo com anomalias tem ao menos uma leitura de alta confiança
