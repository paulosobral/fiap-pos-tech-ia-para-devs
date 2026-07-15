## MODIFIED Requirements

### Requirement: Extração de séries de ângulo de múltiplas articulações e de velocidade de movimento
Para cada frame processado, o sistema SHALL calcular os ângulos de múltiplas articulações do corpo (cotovelos esquerdo/direito, joelhos esquerdo/direito, quadris/tronco esquerdo/direito e pescoço/cabeça) e uma medida de velocidade de movimento entre frames consecutivos, a partir dos keypoints detectados pelo YOLOv8-pose, formando uma série temporal por articulação mais uma série de velocidade por vídeo. O sistema SHALL também reter os keypoints de cada frame para permitir a anotação visual posterior.

#### Scenario: Cálculo das séries a partir de keypoints válidos
- **WHEN** o YOLOv8-pose detecta keypoints de uma pessoa em um frame
- **THEN** o sistema calcula o ângulo de cada articulação cujos keypoints necessários estão presentes e a velocidade de movimento em relação ao frame anterior, adicionando cada valor à série temporal correspondente

#### Scenario: Articulação com keypoints faltantes em um frame
- **WHEN** um ou mais keypoints necessários para uma articulação não são detectados em um frame (mas a pessoa é detectada)
- **THEN** o sistema registra aquela articulação como sem valor nesse frame, sem interromper o cálculo das demais articulações nem o processamento dos frames seguintes

#### Scenario: Frame sem pessoa detectada
- **WHEN** o YOLOv8-pose não detecta nenhuma pessoa em um frame
- **THEN** o sistema registra o frame como sem dados de pose (todas as articulações sem valor), sem interromper o processamento do restante do vídeo

### Requirement: Detecção de anomalia postural por articulação com sensibilidade ajustável e auto-explicativa
O sistema SHALL detectar anomalias posturais aplicando rolling z-score sobre a série de cada articulação e sobre a série de velocidade, usando um threshold de sensibilidade controlado por um slider na interface Streamlit, e SHALL exibir, no valor atual do slider, uma indicação em linguagem simples de aproximadamente qual proporção dos frames do vídeo carregado seria marcada como irregular.

#### Scenario: Frame com desvio postural acima do threshold em alguma articulação
- **WHEN** o valor absoluto do z-score do ângulo de uma articulação (ou da velocidade) em um frame excede o threshold selecionado no slider
- **THEN** o sistema marca aquele frame como irregular para aquela articulação (ou para velocidade) e registra o timestamp e a articulação afetada

#### Scenario: Feedback do efeito da sensibilidade escolhida
- **WHEN** o usuário visualiza o controle de sensibilidade com um vídeo carregado
- **THEN** o sistema exibe uma estimativa, calculada a partir do próprio vídeo, de quanto do vídeo seria marcado como irregular no valor de sensibilidade atual, para que o usuário entenda o efeito do controle sem precisar interpretar o valor numérico do z-score

### Requirement: Agrupamento de frames irregulares consecutivos em eventos
O sistema SHALL agrupar frames irregulares consecutivos de uma mesma articulação (ou de uma mesma origem: velocidade, zona crítica) em um único evento, identificado por instante de início, instante de fim e o frame mais representativo do grupo (o de maior desvio), gerando um único alerta por evento em vez de um alerta por frame.

#### Scenario: Sequência de frames irregulares da mesma articulação vira um evento
- **WHEN** vários frames consecutivos são marcados como irregulares para a mesma articulação
- **THEN** o sistema os agrupa em um único evento com início, fim e o frame de maior desvio, e gera um único alerta referenciando a articulação e o intervalo de tempo

#### Scenario: Feed unificado recebe um alerta por evento, não por frame
- **WHEN** o processamento do vídeo gera eventos irregulares
- **THEN** o feed unificado de alertas recebe um alerta por evento (com origem "Vídeo", intervalo de tempo e articulação/tipo), e não um alerta por frame

### Requirement: Detecção opcional de zona crítica com prévia visual
O sistema SHALL oferecer a detecção de entrada em uma zona/área crítica configurável como um recurso opcional, desativado por padrão; quando ativado, o sistema SHALL exibir uma prévia visual do retângulo da zona sobre um frame do vídeo carregado antes do processamento, e SHALL agrupar as detecções de zona em eventos (não um alerta por frame).

#### Scenario: Zona crítica desativada por padrão
- **WHEN** o usuário carrega um vídeo e não ativa explicitamente a análise de zona crítica
- **THEN** o sistema não executa a detecção de zona e não gera nenhum alerta de zona

#### Scenario: Prévia da zona sobre o frame ao ativar
- **WHEN** o usuário ativa a análise de zona crítica e ajusta a área
- **THEN** o sistema exibe o retângulo da zona desenhado sobre um frame real do vídeo, para que o usuário veja onde a zona cai na imagem antes de processar

#### Scenario: Detecção de zona agrupada em evento
- **WHEN** a pessoa permanece dentro da zona crítica por frames consecutivos
- **THEN** o sistema agrupa esses frames em um único evento de zona (com início e fim) e gera um único alerta, não um por frame

### Requirement: Relatório visual de desvios do vídeo com esqueleto anotado
Ao final do processamento de um vídeo, o sistema SHALL apresentar um relatório visual dos eventos irregulares detectados: um resumo com a contagem de eventos e a articulação mais afetada, e, para cada evento, a imagem do frame mais representativo com o esqueleto de pose desenhado sobre ele destacando a articulação afetada, acompanhada de um rótulo em linguagem simples (articulação e intervalo de tempo). O relatório SHALL substituir a apresentação anterior baseada em lista de texto por frame.

#### Scenario: Vídeo processado com eventos irregulares detectados
- **WHEN** o processamento do vídeo termina e ao menos um evento irregular foi registrado
- **THEN** o sistema exibe um resumo (número de eventos e articulação mais afetada) e, para cada evento, a imagem do frame representativo com o esqueleto desenhado e a articulação afetada destacada, mais um rótulo textual simples com a articulação e o intervalo de tempo

#### Scenario: Vídeo processado sem eventos irregulares
- **WHEN** o processamento do vídeo termina e nenhum evento irregular foi registrado
- **THEN** o sistema exibe uma mensagem indicando que nenhum desvio foi encontrado no vídeo processado
