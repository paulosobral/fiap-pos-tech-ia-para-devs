## 1. Suporte a mkv na aba Vídeo

- [x] 1.1 Adicionar `"mkv"` à tupla `VIDEO_ALLOWED_EXTENSIONS` em `app.py`
- [x] 1.2 Atualizar help text do `st.file_uploader` e a mensagem de erro de formato inválido na aba Vídeo para citar mkv
- [x] 1.3 Adicionar teste cobrindo a validação de extensão aceitando `.mkv`
- [x] 1.4 Atualizar `README.md` (seção "As 4 abas") para citar mkv como formato aceito na aba Vídeo
- [x] 1.5 Verificar manualmente (ou via teste) que `_decode_video_frames`/`cv2.VideoCapture` decodifica um arquivo `.mkv` real sem alteração de código — confirmado: `ffmpeg` gerou um `.mkv` de 1s a partir de `data/demo_pose_walk.mp4`, e `_decode_video_frames` decodificou 10 frames corretamente (fps=10.0, shape (360, 480, 3)) sem qualquer alteração na função.
