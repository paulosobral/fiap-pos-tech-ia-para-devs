## Why

Ao enviar um vídeo longo em alta resolução (ex.: um clipe de ~3 min a 30 fps/1080p) a aba Vídeo decodificava **todos** os frames em resolução plena para uma lista Python e ainda os retinha uma segunda vez para desenhar o relatório. A ~6 MB por frame BGR 1080p, milhares de frames chegam a dezenas de GB residentes, esgotando a RAM: o app trava e, sob WSL2, congela a VM inteira (só reiniciar recupera). A análise não precisa de resolução plena nem de todos os frames — ângulos articulares e z-score são invariantes de escala, e o relatório visual só mostra o "pior" frame de cada evento.

## What Changes

- Decodificação de vídeo passa a **reduzir a resolução** de cada frame (lado maior limitado a um teto configurável, aspecto preservado) antes de retê-lo.
- Decodificação passa a **subamostrar** os frames para uma taxa-alvo aproximada (stride derivado do fps de origem), com `effective_fps = fps_origem / stride` para manter os timestamps por frame corretos.
- O relatório visual deixa de decodificar/reter o vídeo inteiro: passa a **ler sob demanda apenas os frames exibidos** (o `frame_index_pior` de cada evento mostrado, mais o frame 0 do preview de zona), em uma passada sequencial de `cv2`, retornando somente esses frames.
- Extração de pose e desenho do esqueleto usam **o mesmo teto de resolução e o mesmo stride**, garantindo que os keypoints (em pixels reduzidos) permaneçam alinhados aos frames sobre os quais são desenhados.

Sem mudança na lógica de detecção (limiares, agrupamento de eventos, alertas): apenas a escala/quantidade de frames processados muda.

## Capabilities

### New Capabilities
<!-- Nenhuma capability nova. -->

### Modified Capabilities
- `video-motion-analysis`: acrescenta requisitos de limite de recursos no processamento de vídeo — redução de resolução, subamostragem temporal com timestamps preservados, e leitura sob demanda apenas dos frames exibidos no relatório (em vez de reter o vídeo inteiro em memória).

## Impact

- Código: `app.py` (helpers de decodificação `_decode_video_frames`, novo `_read_sampled_frames` em substituição a `_decode_video_frames_cached`, e os dois pontos de chamada na aba Vídeo — preview de zona e galeria de eventos).
- Não afeta `video/pose.py`, `video/analysis.py` nem `video/draw.py` (ângulos e zona são invariantes de escala; keypoints e frames continuam na mesma escala em ambos os caminhos).
- Operacional (fora do software): documentar limite de RAM da VM via `~/.wslconfig` no host Windows como rede de segurança contra OOM.
