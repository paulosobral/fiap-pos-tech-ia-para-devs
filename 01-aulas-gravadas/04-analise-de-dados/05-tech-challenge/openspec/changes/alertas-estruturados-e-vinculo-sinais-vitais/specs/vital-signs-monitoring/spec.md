## ADDED Requirements

### Requirement: Identificador de vínculo e classificação visual nos alertas de sinais vitais
O sistema SHALL atribuir a cada leitura anômala de sinais vitais exibida na tabela um identificador curto único (ex.: `#S01`), e esse mesmo identificador SHALL aparecer no alerta correspondente no feed, permitindo casar o alerta com a linha da tabela mesmo que a ordem de exibição difira. Os alertas de sinais vitais no feed SHALL ser exibidos com um indicador visual (ícone/cor) correspondente ao seu nível de confiança (as duas camadas concordam / somente detecção em tempo real / somente análise do histórico).

#### Scenario: Identificador casa alerta e linha da tabela
- **WHEN** uma leitura anômala de sinais vitais gera um alerta e também aparece na tabela de anomalias
- **THEN** o identificador exibido na linha da tabela é o mesmo exibido no alerta correspondente, permitindo ao usuário localizar a leitura pelo identificador

#### Scenario: Identificadores únicos e determinísticos
- **WHEN** o processamento gera várias leituras anômalas
- **THEN** cada leitura anômala recebe um identificador único e a atribuição é determinística para a mesma entrada

#### Scenario: Alerta de sinais vitais com indicador de nível
- **WHEN** um alerta de sinais vitais é exibido no feed
- **THEN** ele apresenta um indicador visual (ícone/cor) correspondente ao seu nível de confiança, consistente com a tabela e a legenda de níveis
