## ADDED Requirements

### Requirement: Upload e processamento de vídeo clínico
O sistema SHALL aceitar upload de um arquivo de vídeo (fisioterapia ou cirurgia gravada) através da aba Vídeo da aplicação Streamlit e processar o vídeo frame a frame usando um modelo YOLOv8-pose para extrair keypoints posturais e detecções de objeto.

#### Scenario: Upload de vídeo válido
- **WHEN** o usuário faz upload de um arquivo de vídeo em formato suportado (mp4, avi ou mov)
- **THEN** o sistema processa o vídeo frame a frame e exibe progresso do processamento na UI

#### Scenario: Upload de arquivo em formato não suportado
- **WHEN** o usuário faz upload de um arquivo que não é vídeo (ex.: .txt, .csv)
- **THEN** o sistema rejeita o upload e exibe mensagem de erro indicando os formatos aceitos

### Requirement: Extração de série de ângulo e velocidade de movimento
Para cada frame processado, o sistema SHALL calcular o ângulo de articulação relevante e a velocidade de movimento entre frames consecutivos a partir dos keypoints detectados pelo YOLOv8-pose, formando duas séries temporais por vídeo.

#### Scenario: Cálculo de série a partir de keypoints válidos
- **WHEN** o YOLOv8-pose detecta keypoints de uma pessoa em um frame
- **THEN** o sistema calcula o ângulo da articulação configurada e a velocidade de deslocamento em relação ao frame anterior, adicionando ambos os valores às séries temporais do vídeo

#### Scenario: Frame sem pessoa detectada
- **WHEN** o YOLOv8-pose não detecta nenhuma pessoa em um frame
- **THEN** o sistema registra o frame como sem dados de pose, sem interromper o processamento do restante do vídeo

### Requirement: Detecção de anomalia postural com sensibilidade ajustável
O sistema SHALL detectar anomalias posturais aplicando rolling z-score sobre as séries de ângulo e velocidade, usando um threshold de sensibilidade controlado por um slider na interface Streamlit.

#### Scenario: Frame com desvio postural acima do threshold
- **WHEN** o valor absoluto do z-score de ângulo ou velocidade em um frame excede o threshold selecionado no slider
- **THEN** o sistema marca o frame como anômalo e registra seu timestamp na lista de desvios do vídeo

#### Scenario: Ajuste de sensibilidade pelo usuário
- **WHEN** o usuário move o slider de sensibilidade para um valor mais baixo
- **THEN** o sistema recalcula a detecção de anomalia com o novo threshold e atualiza a lista de desvios exibida

### Requirement: Detecção de objeto ou área crítica
O sistema SHALL detectar, via YOLOv8, a presença de objetos ou entrada em áreas críticas pré-configuradas dentro do vídeo, gerando um alerta direto por regra de zona, independente do cálculo de rolling z-score.

#### Scenario: Objeto detectado em área crítica configurada
- **WHEN** o YOLOv8 detecta um objeto de interesse cuja bounding box intersecta uma área crítica configurada no vídeo
- **THEN** o sistema gera um alerta imediato referenciando o timestamp e o tipo de objeto detectado

### Requirement: Relatório automático de desvios do vídeo
Ao final do processamento de um vídeo, o sistema SHALL gerar um relatório automático listando todos os desvios ou falhas identificadas (anomalias posturais e alertas de zona crítica), com seus respectivos timestamps.

#### Scenario: Vídeo processado com anomalias detectadas
- **WHEN** o processamento do vídeo termina e ao menos uma anomalia postural ou alerta de zona crítica foi registrado
- **THEN** o sistema exibe um relatório com a lista ordenada por timestamp de todas as anomalias e alertas encontrados

#### Scenario: Vídeo processado sem anomalias detectadas
- **WHEN** o processamento do vídeo termina e nenhuma anomalia foi registrada
- **THEN** o sistema exibe um relatório indicando que nenhum desvio foi encontrado no vídeo processado
