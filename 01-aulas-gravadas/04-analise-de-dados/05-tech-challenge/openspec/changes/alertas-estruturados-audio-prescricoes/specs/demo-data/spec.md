## ADDED Requirements

### Requirement: Amostras de demonstração de prescrições (sem anomalias)
O sistema SHALL fornecer um conjunto de dados de demonstração de prescrições sem inconsistências, versionado e derivado da mesma estrutura de pacientes do dataset sintético existente (com anomalias), por meio de um script gerador reproduzível. O arquivo sem anomalias SHALL conter, para cada paciente, um histórico de dose estável ao longo do tempo e sem combinação de medicamentos presente na lista de risco conhecida usada na revisão via AWS Bedrock, e SHALL carregar sem erro pela aba Prescrições.

#### Scenario: Arquivo sem anomalias é aceito pela aba Prescrições
- **WHEN** o arquivo de demonstração de prescrições sem anomalias é carregado na aba Prescrições
- **THEN** o arquivo é validado com sucesso (colunas `paciente`, `medicamento`, `dose`, `data`) e o histórico de cada paciente é exibido normalmente

#### Scenario: Amostra reproduzível a partir de um gerador
- **WHEN** o script gerador de prescrições é executado
- **THEN** ele produz o arquivo de demonstração sem anomalias de forma determinística, com dose constante por medicamento e sem a combinação de risco conhecida usada no prompt de revisão
