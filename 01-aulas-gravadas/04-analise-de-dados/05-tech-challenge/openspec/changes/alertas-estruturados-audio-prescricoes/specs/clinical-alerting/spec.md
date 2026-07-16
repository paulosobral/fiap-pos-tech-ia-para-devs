## MODIFIED Requirements

### Requirement: Feed unificado de alertas para a equipe médica
O sistema SHALL manter um feed de alertas único, acumulando alertas gerados pelas abas Vídeo, Áudio, Sinais Vitais e Prescrições, exibido na interface Streamlit como simulação de notificação em tempo real à equipe médica. Cada alerta SHALL suportar, além de origem, timestamp e descrição, campos estruturados opcionais de identificador (`alert_id`), categoria e nível, preenchidos pela aba que gera o alerta quando aplicável, sem quebrar abas que não os fornecem. Quando um alerta tiver um nível associado, o feed SHALL exibi-lo com um indicador visual desse nível. O sistema SHALL permitir exportar todos os alertas da sessão como um relatório CSV (uma linha por alerta, com colunas para identificador, origem, timestamp, categoria, nível e descrição), e SHALL exibir um resumo com a contagem de alertas por aba e por nível. Todas as 4 abas (Vídeo, Sinais Vitais, Áudio e Prescrições) SHALL atribuir um identificador (`alert_id`) curto e único a cada alerta que gerarem, casando-o com o item correspondente exibido na própria aba.

#### Scenario: Alerta gerado em qualquer aba aparece no feed
- **WHEN** qualquer aba (Vídeo, Áudio, Sinais Vitais ou Prescrições) gera um alerta durante o processamento
- **THEN** o alerta é adicionado ao feed unificado, incluindo origem (aba), timestamp e descrição do alerta, além dos campos estruturados (identificador, categoria, nível) que a aba tenha fornecido

#### Scenario: Alerta sem campos estruturados continua válido
- **WHEN** uma aba gera um alerta fornecendo apenas origem e descrição (sem identificador, categoria ou nível)
- **THEN** o alerta é adicionado e exibido normalmente, com os campos estruturados ausentes tratados como não informados (sem erro)

#### Scenario: Indicador visual de nível quando disponível
- **WHEN** um alerta que carrega um nível é exibido no feed
- **THEN** o feed apresenta um indicador visual correspondente a esse nível

#### Scenario: Exportar relatório de todos os alertas da sessão
- **WHEN** há ao menos um alerta na sessão e o usuário aciona a exportação do relatório
- **THEN** o sistema gera um arquivo CSV com uma linha por alerta, contendo identificador, origem, timestamp, categoria, nível e descrição, disponível para download

#### Scenario: Resumo por aba e por nível
- **WHEN** o usuário visualiza a área de exportação do feed com alertas na sessão
- **THEN** o sistema exibe a contagem total de alertas e a contagem por aba de origem e por nível

#### Scenario: Sem alertas para exportar
- **WHEN** não há nenhum alerta na sessão
- **THEN** a opção de exportar o relatório não é oferecida (ou fica indisponível), coerente com a mensagem de que não há alertas registrados

#### Scenario: Feed exibido em ordem cronológica
- **WHEN** o usuário visualiza o feed de alertas
- **THEN** os alertas são exibidos ordenados do mais recente para o mais antigo

#### Scenario: Nenhum alerta gerado na sessão
- **WHEN** o usuário visualiza o feed de alertas antes de qualquer processamento gerar anomalias
- **THEN** o sistema exibe mensagem indicando que não há alertas registrados na sessão atual

#### Scenario: Alerta de Áudio vinculado ao trecho de origem
- **WHEN** a aba Áudio gera um alerta (termo crítico, taxa de fala anômala ou pausa anômala)
- **THEN** o alerta recebe um identificador curto e único, e esse mesmo identificador é exibido junto do trecho/segmento correspondente na aba

#### Scenario: Alerta de Prescrições vinculado à inconsistência de origem
- **WHEN** a aba Prescrições gera um alerta a partir de uma inconsistência identificada pelo AWS Bedrock
- **THEN** o alerta recebe um identificador curto e único, e esse mesmo identificador é exibido junto do card da inconsistência correspondente na aba
