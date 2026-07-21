## ADDED Requirements

### Requirement: Limite de resolução na decodificação de vídeo

O sistema MUST (DEVE) reduzir a resolução de cada frame decodificado de modo que o maior lado não ultrapasse um teto configurável, preservando a proporção original, antes de reter o frame em memória ou executar inferência de pose sobre ele. Frames já dentro do teto MUST (DEVEM) permanecer inalterados. A extração de pose e o desenho do relatório MUST (DEVEM) operar sobre frames na mesma escala reduzida, de forma que os keypoints extraídos permaneçam alinhados aos frames sobre os quais são desenhados.

#### Scenario: Frame acima do teto é reduzido preservando proporção
- **WHEN** um frame decodificado tem o maior lado acima do teto configurado (ex.: 1920×1080 com teto 640)
- **THEN** o frame é redimensionado para que o maior lado fique igual ao teto (ex.: 640×360), mantendo a razão de aspecto

#### Scenario: Frame já pequeno não é alterado
- **WHEN** um frame decodificado já tem o maior lado menor ou igual ao teto
- **THEN** o frame é retido sem redimensionamento

#### Scenario: Keypoints alinhados aos frames desenhados
- **WHEN** o relatório desenha o esqueleto de pose sobre o frame de um evento
- **THEN** os keypoints (extraídos na escala reduzida) coincidem com as posições no frame reduzido, sem desalinhamento

### Requirement: Subamostragem temporal com timestamps preservados

O sistema MUST (DEVE) subamostrar os frames do vídeo durante a decodificação para aproximar uma taxa-alvo de quadros por segundo, mantendo aproximadamente 1 a cada N frames de origem (stride derivado do fps de origem, sempre no mínimo 1). O fps efetivo usado para calcular o timestamp de cada frame retido MUST (DEVE) ser `fps_origem / stride`, de modo que os instantes (em segundos) dos eventos permaneçam corretos apesar da subamostragem.

#### Scenario: Vídeo a 30 fps com alvo de 10 fps
- **WHEN** o vídeo de origem está a 30 fps e a taxa-alvo é 10 fps
- **THEN** o stride é 3 (retém aproximadamente 1 a cada 3 frames) e o fps efetivo dos frames retidos é 10

#### Scenario: Vídeo já abaixo da taxa-alvo
- **WHEN** o fps de origem é menor ou igual à taxa-alvo
- **THEN** o stride é 1 (nenhum frame é descartado) e o fps efetivo é igual ao fps de origem

#### Scenario: Timestamps consistentes após subamostragem
- **WHEN** eventos são reportados a partir da série de frames subamostrada
- **THEN** o instante em segundos de cada evento é calculado com o fps efetivo, refletindo o tempo real no vídeo original

### Requirement: Leitura sob demanda dos frames exibidos no relatório

O relatório visual de desvios MUST NOT (NÃO DEVE) decodificar nem reter todos os frames do vídeo em memória. Ele MUST (DEVE) ler, em uma passada sequencial de decodificação, somente os frames efetivamente exibidos — o frame representativo (`frame_index_pior`) de cada evento apresentado na galeria, além do primeiro frame usado no preview da zona de risco — aplicando o mesmo teto de resolução e o mesmo stride da decodificação principal, retornando apenas esses frames.

#### Scenario: Apenas frames exibidos são retidos
- **WHEN** o relatório precisa desenhar os frames representativos dos eventos exibidos
- **THEN** somente os índices desses frames (mais o frame 0 do preview de zona, quando aplicável) são lidos e retidos, e não o vídeo inteiro

#### Scenario: Índice inexistente é tratado sem falha
- **WHEN** um frame representativo solicitado não pôde ser decodificado
- **THEN** o relatório sinaliza aquele evento de forma degradada (sem imagem) em vez de falhar

#### Scenario: Índices sob demanda referem-se aos mesmos frames da série de pose
- **WHEN** o relatório lê um frame pelo índice subamostrado
- **THEN** esse índice refere-se ao mesmo frame de origem que o índice correspondente na série de pose, porque ambos derivam o mesmo stride do mesmo vídeo
