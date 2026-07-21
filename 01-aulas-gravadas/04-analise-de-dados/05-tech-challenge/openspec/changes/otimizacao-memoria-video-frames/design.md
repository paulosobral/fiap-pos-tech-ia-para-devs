## Context

A aba Vídeo processa um upload frame a frame com YOLOv8-pose. Antes desta mudança, dois caminhos independentes decodificavam o vídeo **inteiro** em resolução plena para listas Python:

1. `_extract_pose_frame_series` — decodifica todos os frames, roda inferência de pose e mantém a série (`@st.cache_data`).
2. `_decode_video_frames_cached` — decodifica **de novo** todos os frames e retém a lista completa, só para desenhar o relatório.

Cada frame BGR 1080p ocupa ~6 MB. Um clipe de ~3 min a 30 fps tem ~5400 frames → ~32 GB numa única lista, multiplicado pelas duas cópias em cache e pelos bytes crus do upload. A RAM esgota; sob WSL2 (que por padrão toma até ~50% da RAM do host e swap dinâmico) a VM inteira congela e só reinicia recupera.

Restrições que tornam a redução segura:
- Os ângulos articulares são calculados por produto escalar de vetores entre keypoints → **invariantes de escala**.
- O z-score é relativo à própria variação recente do vídeo → **invariante de escala**.
- A zona de risco é definida em coordenadas relativas `[0,1]` → **invariante de escala**.
- O relatório visual mostra apenas o "pior" frame (`frame_index_pior`) de cada evento exibido, não o vídeo todo.

## Goals / Non-Goals

**Goals:**
- Limitar o uso de memória do processamento de vídeo a uma fração pequena e previsível, independente da duração/resolução do upload.
- Manter a lógica de detecção (limiares, agrupamento de eventos, alertas, IDs) idêntica.
- Manter o alinhamento entre keypoints e frames desenhados.
- Manter timestamps de eventos corretos apesar da subamostragem.

**Non-Goals:**
- Alterar o modelo de pose, os limiares ou o algoritmo de detecção.
- Alterar `video/pose.py`, `video/analysis.py` ou `video/draw.py`.
- Streaming/paginação da inferência de pose em si (a série de pose continua cacheada por ser leve — só ângulos/keypoints, sem imagens cruas).
- Configurar o limite de RAM da VM (documentado como passo operacional, fora do software).

## Decisions

**D1 — Redução de resolução no decode (`_downscale_frame`), teto do maior lado = 640px, `cv2.INTER_AREA`.**
Por quê 640: preserva keypoints úteis para YOLOv8-pose mantendo ~9× menos memória por frame em 1080p. `INTER_AREA` é a interpolação recomendada para downscale (menos aliasing). Alternativa considerada: reduzir só quando a inferência degradasse — rejeitada, o teto fixo é previsível e simples. A proporção é preservada para não distorcer ângulos.

**D2 — Subamostragem temporal via stride derivado do fps (`_frame_stride`), alvo ~10 fps.**
Por quê stride determinístico em `fps_origem` apenas: garante que o caminho de decode e o leitor sob demanda derivem o **mesmo** stride do mesmo vídeo, logo concordam sobre qual índice subamostrado mapeia para qual frame de origem — condição necessária para keypoints e frames casarem. `effective_fps = fps_origem / stride` é retornado por `_decode_video_frames` e usado como `fps` na extração, preservando os timestamps. Alternativa: subamostrar por tempo absoluto (a cada X ms) — rejeitada por introduzir arredondamento divergente entre os dois caminhos.

**D3 — Leitura sob demanda (`_read_sampled_frames`) substitui `_decode_video_frames_cached`.**
Em vez de decodificar e reter o vídeo inteiro, uma única passada sequencial de `cv2` retém apenas os índices pedidos (`frame_index_pior` de cada evento exibido + frame 0 do preview), retornando um `dict {índice: frame}`. Sai cedo quando todos os pedidos foram coletados. Por quê `dict` e não lista: os índices pedidos são esparsos; a checagem de presença vira `idx in report_frames`. Alternativa: `cap.set(CAP_PROP_POS_FRAMES, idx)` por frame (seek aleatório) — rejeitada por ser não confiável em muitos codecs (seek para keyframe mais próximo); a passada sequencial é robusta e ainda barata (sem YOLO).

**D4 — Ordem de cálculo no relatório: agrupar eventos → coletar índices exibidos → ler só esses.**
`group_events_for_display` é chamado antes da leitura de frames para saber exatamente quais índices serão exibidos (top-N por seção), evitando ler frames de eventos que nunca aparecem.

## Risks / Trade-offs

- **Perda de fidelidade visual/temporal** (downscale + subsample) → aceitável: o relatório é indicativo de momentos irregulares, não medição clínica; os intervalos de tempo continuam corretos via `effective_fps`.
- **Detecção pode variar levemente** por menos frames e menos pixels → mitigado: ângulos/z-score são invariantes de escala; a menor densidade temporal apenas suaviza jitter por frame, coerente com o `DEFAULT_WINDOW` da análise de vídeo.
- **Leitor sob demanda relê o vídeo do disco** (uma passada por render) em vez de usar cache → trade-off deliberado: troca CPU/IO barato (decode sem YOLO) por memória, que era o gargalo real.
- **Bytes crus do upload permanecem na RAM do Streamlit** → não resolvido no software; rede de segurança é limitar a RAM da VM via `~/.wslconfig` no host (passo operacional).

## Migration Plan

Mudança contida em `app.py`, sem migração de dados. Rollback = reverter o commit. Passo operacional recomendado (host Windows): criar `%USERPROFILE%\.wslconfig` com `[wsl2] memory=<~50% RAM> swap=<2x memory>` e `wsl --shutdown`, para que um eventual OOM mate só o processo Python em vez de congelar a VM.

## Open Questions

- O teto de 640px e o alvo de 10 fps são constantes no módulo; se algum vídeo de demonstração exigir mais detalhe, avaliar torná-los ajustáveis na UI. Fora do escopo desta mudança.
