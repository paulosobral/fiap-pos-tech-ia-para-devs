## Context

A aba Vídeo (`app.py`) restringe upload a mp4/avi/mov via `VIDEO_ALLOWED_EXTENSIONS`, checada em dois pontos: `st.file_uploader(type=...)` (filtro do widget) e uma validação explícita de extensão pós-upload que gera `st.error` para formatos não aceitos. A decodificação real (`_decode_video_frames`, `app.py`) grava os bytes recebidos em um arquivo temporário e usa `cv2.VideoCapture`, cujo backend nesta instalação é FFmpeg (confirmado via `cv2.getBuildInformation()`) — FFmpeg lê containers mkv nativamente, sem necessidade de dependência adicional.

## Goals / Non-Goals

**Goals:**
- Permitir upload de vídeo em formato mkv na aba Vídeo, com mesma validação e mesmo pipeline de processamento já usado para mp4/avi/mov.
- Manter mensagens de erro/help text consistentes com a nova lista de formatos aceitos.

**Non-Goals:**
- Não há mudança na decodificação, extração de pose, cálculo de ângulo/velocidade ou detecção de zona crítica — esse pipeline já é agnóstico de container.
- Não adiciona suporte a outros formatos além de mkv (webm, flv, etc.) — fora do escopo desta change.

## Decisions

### D1: Apenas ampliar a tupla `VIDEO_ALLOWED_EXTENSIONS`
Adicionar `"mkv"` à tupla existente em `app.py` é suficiente — o mesmo valor já alimenta o filtro do `st.file_uploader`, a validação de extensão e a mensagem de erro (todos leem da mesma constante). Nenhuma outra alteração de código é necessária, pois `_decode_video_frames` recebe apenas bytes + extensão e delega a decodificação ao FFmpeg via `cv2.VideoCapture`, que já suporta mkv.
**Alternativa considerada**: usar `cv2.VideoCapture` para sniffar o formato real do arquivo em vez de confiar na extensão — rejeitada por ser mudança de escopo maior (afetaria a validação de todos os formatos, não só mkv) e sem necessidade concreta neste momento.

## Risks / Trade-offs

- **[Risco] Alguns arquivos `.mkv` usam codecs de vídeo não suportados pelo build FFmpeg local** → Mitigação: `_decode_video_frames` já trata frame count zero / falha de leitura sem crashar (comportamento existente, não alterado por esta change); se um mkv específico falhar, o usuário vê o mesmo tratamento de erro que já existe para vídeo sem frames legíveis.
- **[Trade-off] Validação continua baseada em extensão do nome de arquivo, não no conteúdo real** → Aceito, consistente com o comportamento já existente para mp4/avi/mov (não é uma regressão introduzida por esta change).
