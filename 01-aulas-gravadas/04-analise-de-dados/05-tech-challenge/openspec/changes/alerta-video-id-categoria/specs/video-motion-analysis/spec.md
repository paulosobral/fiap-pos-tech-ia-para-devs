## MODIFIED Requirements

### Requirement: Agrupamento de frames irregulares consecutivos em eventos identificados
O sistema SHALL agrupar frames irregulares consecutivos de uma mesma articulação (ou de uma mesma origem: velocidade, zona crítica) em um único evento, identificado por instante de início, instante de fim, o frame mais representativo do grupo (o de maior desvio) e um identificador curto único do evento (ex.: `#V01`), atribuído de forma determinística. O sistema SHALL gerar um único alerta por evento, cujo texto inclui esse identificador único e a categoria/região do corpo afetada (Cabeça, Braços, Tronco, Pernas, Corpo ou Zona de risco), além da descrição em linguagem simples com o intervalo de tempo.

#### Scenario: Cada evento recebe um identificador curto único
- **WHEN** o processamento do vídeo gera eventos irregulares
- **THEN** cada evento recebe um identificador curto único e estável dentro daquele processamento, e nenhum identificador se repete entre eventos

#### Scenario: Alerta inclui identificador e categoria do corpo
- **WHEN** um evento gera um alerta no feed unificado
- **THEN** o texto do alerta inclui o identificador único do evento e a categoria/região do corpo afetada, além da articulação e do intervalo de tempo

#### Scenario: Feed unificado recebe um alerta por evento, não por frame
- **WHEN** o processamento do vídeo gera eventos irregulares
- **THEN** o feed unificado de alertas recebe um alerta por evento (com origem "Vídeo", identificador, categoria, intervalo de tempo e articulação/tipo), e não um alerta por frame

### Requirement: Relatório visual de desvios do vídeo agrupado por articulação
Ao final do processamento de um vídeo, o sistema SHALL apresentar um relatório visual dos eventos irregulares agrupado por articulação (e por tipo, para velocidade e zona crítica): um resumo com a contagem total de eventos e a articulação mais afetada, e uma seção colapsável por articulação/tipo — ordenada da mais afetada (mais eventos) para a menos — contendo os eventos mais graves daquela seção exibidos como uma galeria de frames anotados (imagem do frame mais representativo com o esqueleto desenhado destacando a articulação afetada, mais o identificador único do evento e o intervalo de tempo). Cada seção SHALL exibir sua contagem total de eventos e, quando houver mais eventos do que o limite exibido, SHALL indicar quantos estão sendo mostrados de quantos há no total. O identificador exibido na legenda de cada imagem SHALL ser o mesmo que aparece no alerta correspondente no feed, permitindo casar o alerta com a imagem. O relatório SHALL substituir a lista plana de todos os eventos usada anteriormente, para permanecer navegável mesmo com centenas de eventos.

#### Scenario: Legenda da imagem casa com o alerta pelo identificador
- **WHEN** um evento aparece tanto no feed de alertas quanto na galeria da aba Vídeo
- **THEN** o identificador único exibido na legenda da imagem na galeria é o mesmo exibido no texto do alerta correspondente, permitindo ao usuário casar os dois mesmo que a ordem de exibição do feed (cronológica) e da galeria (por gravidade) sejam diferentes

#### Scenario: Vídeo com muitos eventos em várias articulações
- **WHEN** o processamento do vídeo termina e há eventos irregulares distribuídos por várias articulações
- **THEN** o sistema exibe uma seção colapsável por articulação/tipo (fechada por padrão), com o título da seção mostrando a articulação e sua contagem total de eventos, ordenadas da mais afetada para a menos, cada seção contendo uma galeria dos eventos mais graves daquela articulação

#### Scenario: Vídeo processado sem eventos irregulares
- **WHEN** o processamento do vídeo termina e nenhum evento irregular foi registrado
- **THEN** o sistema exibe uma mensagem indicando que nenhum desvio foi encontrado no vídeo processado
