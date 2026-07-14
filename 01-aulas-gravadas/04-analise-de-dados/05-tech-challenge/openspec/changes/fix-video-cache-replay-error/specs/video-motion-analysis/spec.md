## MODIFIED Requirements

### Requirement: Upload e processamento de vídeo clínico
O sistema SHALL aceitar upload de um arquivo de vídeo (fisioterapia ou cirurgia gravada) através da aba Vídeo da aplicação Streamlit e processar o vídeo frame a frame usando um modelo YOLOv8-pose para extrair keypoints posturais e detecções de objeto.

#### Scenario: Upload de vídeo válido
- **WHEN** o usuário faz upload de um arquivo de vídeo em formato suportado (mp4, avi, mov ou mkv)
- **THEN** o sistema processa o vídeo frame a frame e exibe progresso do processamento na UI

#### Scenario: Upload de arquivo em formato não suportado
- **WHEN** o usuário faz upload de um arquivo que não é vídeo (ex.: .txt, .csv)
- **THEN** o sistema rejeita o upload e exibe mensagem de erro indicando os formatos aceitos

#### Scenario: Reprocessamento do mesmo vídeo ou upload de um segundo vídeo na mesma sessão
- **WHEN** o usuário clica em "Processar vídeo" novamente para o mesmo arquivo (ex.: após ajustar sensibilidade ou zona crítica) ou faz upload de um vídeo diferente na mesma sessão
- **THEN** o sistema processa o novo pedido sem lançar erro de execução, reaproveitando a extração de pose já calculada quando o conteúdo do vídeo for idêntico ao de uma chamada anterior
