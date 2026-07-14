## Why

A aba Vídeo aceita hoje apenas mp4, avi e mov. Vídeos em mkv (formato comum de gravações de fisioterapia/exercício e de capturas de tela) são rejeitados na validação de upload mesmo sendo decodificáveis pelo backend FFmpeg já usado pelo `cv2.VideoCapture` — não há limitação técnica real, só uma lista de extensões incompleta.

## What Changes

- Adiciona `mkv` à lista de extensões aceitas na aba Vídeo (`VIDEO_ALLOWED_EXTENSIONS` em `app.py`), tanto no `st.file_uploader` quanto na validação de extensão pós-upload.
- Atualiza a mensagem de erro/help text da aba Vídeo para citar mkv como formato aceito.
- Atualiza a spec `video-motion-analysis` (requirement "Upload e processamento de vídeo clínico") para listar mkv como formato suportado.
- Atualiza `README.md` (seção "As 4 abas") para citar mkv.
- Adiciona um caso de teste cobrindo upload de arquivo `.mkv` sendo aceito pela validação de extensão.

Nenhuma mudança em `video/pose.py`, `video/analysis.py` ou na função de decodificação (`_decode_video_frames` em `app.py`) — `cv2.VideoCapture` já decodifica mkv via FFmpeg sem alteração de código, apenas a lista de extensões permitidas bloqueava o upload antes de chegar lá.

## Capabilities

### New Capabilities
(nenhuma)

### Modified Capabilities
- `video-motion-analysis`: o requirement "Upload e processamento de vídeo clínico" passa a listar mkv como formato de vídeo suportado, além de mp4/avi/mov.

## Impact

- **Código alterado**: `app.py` (constante `VIDEO_ALLOWED_EXTENSIONS`, help text do uploader, mensagem de erro de formato inválido).
- **Testes**: novo caso de teste de validação de extensão para `.mkv`.
- **Documentação**: `README.md`; spec `video-motion-analysis` (delta desta change).
- **Sem impacto** em dependências, custos AWS, ou nas demais 3 abas.
