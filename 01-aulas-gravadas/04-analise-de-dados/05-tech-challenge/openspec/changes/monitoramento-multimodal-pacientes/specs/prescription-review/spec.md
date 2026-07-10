## ADDED Requirements

### Requirement: Upload de histórico de prescrições
O sistema SHALL aceitar upload de um arquivo CSV ou Excel contendo histórico de prescrições por paciente ao longo do tempo (medicamento, dose, data) através da aba Prescrições da aplicação Streamlit.

#### Scenario: Upload de arquivo de prescrições válido
- **WHEN** o usuário faz upload de um CSV ou Excel com colunas de paciente, medicamento, dose e data
- **THEN** o sistema carrega o histórico e exibe uma tabela com a evolução de prescrições por paciente

#### Scenario: Upload de arquivo com colunas faltantes
- **WHEN** o usuário faz upload de um arquivo sem uma das colunas obrigatórias (paciente, medicamento, dose ou data)
- **THEN** o sistema rejeita o arquivo e exibe mensagem de erro indicando as colunas obrigatórias

### Requirement: Análise de inconsistências via AWS Bedrock
O sistema SHALL enviar o histórico de prescrições de um paciente para o modelo Claude Sonnet via AWS Bedrock, solicitando identificação de inconsistências como mudança abrupta de dose, possível interação medicamentosa ou alteração sem justificativa clínica aparente.

#### Scenario: Bedrock identifica inconsistência na prescrição
- **WHEN** o histórico de prescrições de um paciente contém uma mudança abrupta de dose ou combinação de medicamentos potencialmente conflitante
- **THEN** o sistema exibe a inconsistência apontada pelo Bedrock, incluindo a explicação textual retornada pelo modelo

#### Scenario: Bedrock não identifica inconsistências
- **WHEN** o histórico de prescrições de um paciente não apresenta padrões considerados inconsistentes pelo modelo
- **THEN** o sistema exibe mensagem indicando que nenhuma inconsistência foi identificada para o paciente

#### Scenario: Falha na chamada ao Bedrock
- **WHEN** a chamada à API do AWS Bedrock falha ou expira por timeout
- **THEN** o sistema exibe mensagem de erro ao usuário sem interromper as demais abas da aplicação
