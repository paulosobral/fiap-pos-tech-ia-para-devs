## ADDED Requirements

### Requirement: Função genérica de detecção de anomalia por rolling z-score
O sistema SHALL fornecer uma função compartilhada que recebe uma série numérica, um tamanho de janela e um threshold, e retorna quais pontos da série são anômalos com base no z-score calculado sobre média e desvio padrão móveis. Essa função SHALL ser reutilizada pelas capabilities de vídeo, áudio e sinais vitais.

#### Scenario: Série com ponto fora do threshold
- **WHEN** um ponto da série tem z-score absoluto, calculado sobre a janela móvel configurada, maior que o threshold fornecido
- **THEN** a função retorna esse ponto como anômalo

#### Scenario: Série menor que a janela configurada
- **WHEN** a série fornecida tem menos pontos que o tamanho de janela configurado
- **THEN** a função retorna todos os pontos como não anômalos, sem lançar erro

### Requirement: Feed unificado de alertas para a equipe médica
O sistema SHALL manter um feed de alertas único, acumulando alertas gerados pelas abas Vídeo, Áudio, Sinais Vitais e Prescrições, exibido na interface Streamlit como simulação de notificação em tempo real à equipe médica.

#### Scenario: Alerta gerado em qualquer aba aparece no feed
- **WHEN** qualquer aba (Vídeo, Áudio, Sinais Vitais ou Prescrições) gera um alerta durante o processamento
- **THEN** o alerta é adicionado ao feed unificado, incluindo origem (aba), timestamp e descrição do alerta

#### Scenario: Feed exibido em ordem cronológica
- **WHEN** o usuário visualiza o feed de alertas
- **THEN** os alertas são exibidos ordenados do mais recente para o mais antigo

#### Scenario: Nenhum alerta gerado na sessão
- **WHEN** o usuário visualiza o feed de alertas antes de qualquer processamento gerar anomalias
- **THEN** o sistema exibe mensagem indicando que não há alertas registrados na sessão atual
