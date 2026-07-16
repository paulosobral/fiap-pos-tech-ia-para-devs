## MODIFIED Requirements

### Requirement: Detecção de anomalia em tempo real via rolling z-score
O sistema SHALL aplicar rolling z-score linha a linha sobre cada série de sinal vital carregada, marcando como anômala qualquer leitura cujo z-score absoluto exceda o threshold configurado, simulando alerta em tempo real. Os parâmetros padrão (janela e threshold) da aba SHALL ser efetivos — isto é, a janela padrão SHALL ser grande o suficiente para que o z-score máximo alcançável (`sqrt(janela-1)`) supere o threshold padrão, de modo que anomalias reais sejam de fato detectáveis nos valores padrão. Quando o usuário escolher manualmente uma combinação de janela e threshold em que nenhuma anomalia seja matematicamente detectável (threshold ≥ `sqrt(janela-1)`), o sistema SHALL avisá-lo.

#### Scenario: Leitura de sinal vital fora do padrão
- **WHEN** uma leitura de sinal vital tem z-score absoluto acima do threshold configurado
- **THEN** o sistema marca a leitura como anômala e gera um alerta referenciando o timestamp e o sinal vital afetado

#### Scenario: Pico claro é detectado com os parâmetros padrão
- **WHEN** uma série de sinais vitais com um pico claramente anômalo é processada com a janela e o threshold padrão da aba
- **THEN** a camada de rolling z-score marca ao menos a leitura do pico como anômala (os parâmetros padrão são efetivos, não deixam a camada inerte)

#### Scenario: Combinação janela/threshold inefetiva é sinalizada
- **WHEN** o usuário escolhe manualmente uma janela e um threshold tais que `threshold ≥ sqrt(janela-1)` (nenhum z-score alcançável cruza o threshold)
- **THEN** o sistema exibe um aviso indicando que nenhuma anomalia de z-score é detectável com esses valores e como ajustá-los
