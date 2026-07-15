## MODIFIED Requirements

### Requirement: Relatório visual de desvios do vídeo agrupado por articulação
Ao final do processamento de um vídeo, o sistema SHALL apresentar um relatório visual dos eventos irregulares agrupado por articulação (e por tipo, para velocidade e zona crítica): um resumo com a contagem total de eventos e a articulação mais afetada, e uma seção colapsável por articulação/tipo — ordenada da mais afetada (mais eventos) para a menos — contendo os eventos mais graves daquela seção exibidos como uma galeria de frames anotados (imagem do frame mais representativo com o esqueleto desenhado destacando a articulação afetada, mais o intervalo de tempo). Cada seção SHALL exibir sua contagem total de eventos e, quando houver mais eventos do que o limite exibido, SHALL indicar quantos estão sendo mostrados de quantos há no total. O relatório SHALL substituir a lista plana de todos os eventos usada anteriormente, para permanecer navegável mesmo com centenas de eventos.

#### Scenario: Vídeo com muitos eventos em várias articulações
- **WHEN** o processamento do vídeo termina e há eventos irregulares distribuídos por várias articulações
- **THEN** o sistema exibe uma seção colapsável por articulação/tipo (fechada por padrão), com o título da seção mostrando a articulação e sua contagem total de eventos, ordenadas da mais afetada para a menos, cada seção contendo uma galeria dos eventos mais graves daquela articulação

#### Scenario: Seção com mais eventos do que o limite exibido
- **WHEN** uma articulação tem mais eventos do que o limite de exibição por seção
- **THEN** o sistema mostra apenas os eventos mais graves (maior desvio) até o limite, e indica que está mostrando N de um total maior

#### Scenario: Vídeo processado sem eventos irregulares
- **WHEN** o processamento do vídeo termina e nenhum evento irregular foi registrado
- **THEN** o sistema exibe uma mensagem indicando que nenhum desvio foi encontrado no vídeo processado
