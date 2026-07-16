## ADDED Requirements

### Requirement: Apresentação amigável e interpretável do relatório de sinais vitais
O sistema SHALL apresentar o resultado da análise de sinais vitais em linguagem interpretável por não-técnicos: as duas camadas de detecção SHALL ser exibidas com nomes amigáveis (mantendo o termo técnico acessível via tooltip); os controles de sensibilidade e janela SHALL ter rótulos amigáveis e tooltips explicando seu efeito; o resultado SHALL incluir um resumo em linguagem clara e uma tabela com cabeçalhos amigáveis e nomes de sinais vitais traduzidos; e os níveis de confiança (concordância entre camadas) SHALL ser explicados com rótulo claro, indicador visual e tooltip. A detecção subjacente não é alterada — apenas a apresentação.

#### Scenario: Resumo interpretável das leituras críticas
- **WHEN** a análise detecta ao menos uma leitura anômala
- **THEN** o sistema exibe um resumo em linguagem clara indicando quantas leituras críticas foram encontradas e, para as principais, qual sinal vital e em que momento (sem exigir que o usuário interprete colunas técnicas)

#### Scenario: Tabela com rótulos amigáveis
- **WHEN** o sistema exibe a tabela de leituras anômalas
- **THEN** as colunas usam rótulos amigáveis (ex.: Horário, Sinal vital, Valor, Nível de confiança) e os nomes dos sinais vitais aparecem traduzidos (ex.: "Frequência cardíaca" em vez de "heart_rate")

#### Scenario: Níveis de confiança explicados
- **WHEN** o sistema exibe o nível de confiança de cada leitura anômala
- **THEN** cada nível (as duas camadas concordam / somente detecção em tempo real / somente análise do histórico) é apresentado com um rótulo claro, um indicador visual e uma explicação acessível do que significa

#### Scenario: Controles auto-explicativos
- **WHEN** o usuário vê os controles de ajuste da detecção (sensibilidade e janela de comparação)
- **THEN** cada controle tem um rótulo amigável e um tooltip explicando, em linguagem simples e com exemplo, o efeito de aumentar ou diminuir o valor

#### Scenario: Nenhuma anomalia detectada
- **WHEN** a análise não detecta nenhuma leitura anômala em nenhuma das camadas
- **THEN** o sistema exibe uma mensagem clara indicando que nenhuma anomalia foi encontrada
