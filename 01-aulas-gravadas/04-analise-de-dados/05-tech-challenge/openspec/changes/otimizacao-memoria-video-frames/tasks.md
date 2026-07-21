## 1. Redução de resolução e subamostragem no decode

- [x] 1.1 Adicionar constantes `VIDEO_MAX_DIMENSION` (640) e `VIDEO_TARGET_FPS` (10.0) em `app.py`
- [x] 1.2 Implementar `_downscale_frame(frame)` — limita o maior lado ao teto com `cv2.INTER_AREA`, preserva proporção, retorna inalterado se já pequeno
- [x] 1.3 Implementar `_frame_stride(source_fps)` — stride determinístico em `source_fps`, mínimo 1
- [x] 1.4 Reescrever `_decode_video_frames` para aplicar downscale + subamostragem por stride durante a leitura e retornar `(frames, effective_fps)` com `effective_fps = source_fps / stride`

## 2. Leitura sob demanda dos frames do relatório

- [x] 2.1 Implementar `_read_sampled_frames(video_bytes, extension, wanted_indices)` — passada sequencial de `cv2`, mesmo stride/downscale, retorna `dict {índice: frame}` apenas dos índices pedidos, com early-exit
- [x] 2.2 Remover `_decode_video_frames_cached`

## 3. Atualizar pontos de chamada na aba Vídeo

- [x] 3.1 Preview de zona: usar `_read_sampled_frames(..., [0])` e ler o frame 0 do dict
- [x] 3.2 Galeria de eventos: chamar `group_events_for_display` antes, coletar os `frame_index_pior` exibidos e ler somente esses via `_read_sampled_frames`
- [x] 3.3 Trocar a checagem `0 <= idx < len(report_frames)` por `idx in report_frames` (dict)

## 4. Verificação

- [x] 4.1 `app.py` faz parse sem erro de sintaxe (`ast.parse`)
- [x] 4.2 Confirmar que nenhum teste referencia `_decode_video_frames_cached` / helpers alterados
- [ ] 4.3 Teste manual: upload de vídeo longo/HD sem travar host, keypoints alinhados aos frames, timestamps de eventos coerentes
