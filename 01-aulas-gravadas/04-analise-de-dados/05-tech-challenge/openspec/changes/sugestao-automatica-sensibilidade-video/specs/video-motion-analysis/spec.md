## MODIFIED Requirements

### Requirement: Detecção de anomalia postural com sensibilidade ajustável
O sistema SHALL detectar anomalias posturais aplicando rolling z-score sobre as séries de ângulo e velocidade, usando um threshold de sensibilidade controlado por um slider na interface Streamlit. Ao carregar um vídeo, o sistema SHALL calcular automaticamente um valor de sensibilidade sugerido a partir da variação real das séries de ângulo/velocidade daquele vídeo, e SHALL pré-popular o slider com esse valor sugerido.

#### Scenario: Frame com desvio postural acima do threshold
- **WHEN** o valor absoluto do z-score de ângulo ou velocidade em um frame excede o threshold selecionado no slider
- **THEN** o sistema marca o frame como anômalo e registra seu timestamp na lista de desvios do vídeo

#### Scenario: Ajuste de sensibilidade pelo usuário
- **WHEN** o usuário move o slider de sensibilidade para um valor mais baixo
- **THEN** o sistema recalcula a detecção de anomalia com o novo threshold e atualiza a lista de desvios exibida

#### Scenario: Sugestão automática de sensibilidade ao carregar um vídeo
- **WHEN** o usuário faz upload de um vídeo válido e o sistema extrai as séries de ângulo/velocidade
- **THEN** o sistema calcula um valor de sensibilidade sugerido a partir da variação dessas séries, pré-popula o slider com esse valor e indica na interface que se trata de uma sugestão para aquele vídeo, permanecendo livremente ajustável pelo usuário

#### Scenario: Vídeo sem nenhuma pessoa detectada em nenhum frame
- **WHEN** o vídeo carregado não tem nenhum frame com pose detectada, tornando impossível calcular uma sugestão a partir da variação de ângulo/velocidade
- **THEN** o sistema usa o valor de sensibilidade padrão existente como ponto de partida do slider, sem erro
