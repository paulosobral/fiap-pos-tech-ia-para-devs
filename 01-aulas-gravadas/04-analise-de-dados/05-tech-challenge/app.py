"""Streamlit entrypoint — Monitoramento Multimodal de Pacientes.

Single entrypoint per the Global Constraints (design.md D7): each
capability tab is thin UI glued to its own module's ``analyze(...)``
function, which internally pushes ``Alert``s to the shared feed
(``alerts/feed.py``).

Tabs implemented: Sinais Vitais (Task 3), Vídeo (Task 4), Áudio (Task 5),
Prescrições (Task 6).
"""
import math
import os
import tempfile
import uuid

import cv2
import pandas as pd
import streamlit as st

from alerts.feed import build_alerts_report, get_alerts, level_indicator, level_label
from audio.analysis import DEFAULT_THRESHOLD as AUDIO_DEFAULT_THRESHOLD
from audio.analysis import DEFAULT_WINDOW as AUDIO_DEFAULT_WINDOW
from audio.analysis import analyze as analyze_audio
from audio.aws_speech import (
    AUDIO_S3_BUCKET_ENV_VAR,
    DEFAULT_CRITICAL_TERMS,
    AudioProcessingError,
    build_clients,
    detect_entities,
    get_configured_bucket_name,
    raise_critical_term_alerts,
    transcribe_audio,
)
from audio.aws_speech import analyze_sentiment as analyze_audio_sentiment
from prescriptions.bedrock_review import (
    REQUIRED_COLUMNS as PRESCRIPTION_REQUIRED_COLUMNS,
)
from prescriptions.bedrock_review import (
    PrescriptionReviewError,
    PrescriptionValidationError,
    build_bedrock_client,
    load_prescriptions,
    review_patient_prescriptions,
)
from vital_signs.analysis import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    RECOGNIZED_VITAL_SIGN_COLUMNS,
    VitalSignsValidationError,
    analyze,
    build_vitals_summary,
    confidence_level,
    load_vital_signs_csv,
    vital_sign_description,
    vital_sign_label,
    zscore_threshold_is_reachable,
)
from video.analysis import DEFAULT_WINDOW as VIDEO_DEFAULT_WINDOW
from video.analysis import DEFAULT_ZONE as VIDEO_DEFAULT_ZONE
from video.analysis import FALLBACK_SENSITIVITY_THRESHOLD
from video.analysis import analyze as analyze_video
from video.analysis import (
    estimate_flagged_fraction,
    group_events_for_display,
    suggest_sensitivity_threshold,
)
from video.draw import draw_pose_on_frame, draw_zone_on_frame
from video.pose import extract_frame_series

st.set_page_config(page_title="Monitoramento Multimodal de Pacientes", layout="wide")
st.title("Monitoramento Multimodal de Pacientes")


def _render_alert_feed() -> None:
    """Render the unified cross-tab alert feed in the sidebar.

    Consumes ``alerts.feed.get_alerts()`` (clinical-alerting spec) to show
    every alert raised by any of the 4 tabs — Sinais Vitais, Vídeo, Áudio
    and Prescrições — in one place, newest-first, as a simulation of
    real-time notification to the medical team. This is in addition to
    each tab's own inline alert display; it does not replace it.
    """
    st.sidebar.header("Feed de Alertas (todas as abas)")
    alerts = get_alerts()

    if not alerts:
        st.sidebar.info("Nenhum alerta registrado na sessão atual.")
        return

    st.sidebar.caption(f"{len(alerts)} alerta(s) na sessão, do mais recente para o mais antigo.")

    # Session-wide export (clinical-alerting spec, change
    # export-relatorio-alertas): a compact per-tab/per-level summary plus a
    # button to download every session alert as one CSV. The CSV/summary
    # building lives in the pure helper build_alerts_report; here we only
    # render the summary and offer the download. The helper returns a str;
    # encode it UTF-8-SIG (BOM) so Excel shows PT accents correctly.
    csv_text, summary = build_alerts_report(alerts)
    por_origem = ", ".join(f"{n} {origem}" for origem, n in summary["por_origem"].items())
    por_nivel = ", ".join(
        f"{n} {level_label(nivel) or nivel}" for nivel, n in summary["por_nivel"].items()
    )
    st.sidebar.caption(f"**Total:** {summary['total']} — por aba: {por_origem}.")
    st.sidebar.caption(f"Por nível: {por_nivel}.")
    st.sidebar.download_button(
        "Baixar relatório (CSV)",
        data=csv_text.encode("utf-8-sig"),
        file_name="relatorio_alertas.csv",
        mime="text/csv",
    )

    for alert in alerts:
        # When the alert carries a structured level, show a per-level visual
        # indicator (clinical-alerting spec): an icon prefix, and a stronger
        # widget (st.error) for high-severity levels. Alerts without a level
        # fall back to a neutral icon + st.warning.
        icon, severity = level_indicator(alert.level)
        id_prefix = f"{alert.alert_id} " if alert.alert_id else ""
        body = (
            f"{icon} **[{alert.origin}]** {alert.timestamp:%Y-%m-%d %H:%M:%S}\n\n"
            f"{id_prefix}{alert.description}"
        )
        if severity == "high":
            st.sidebar.error(body)
        else:
            st.sidebar.warning(body)


@st.cache_resource(show_spinner=False)
def _load_pose_model():
    """Load the YOLOv8-pose model once per Streamlit process.

    ``yolov8n-pose.pt`` is downloaded automatically by ``ultralytics`` on
    first use if not already present locally (needs internet access).
    """
    from ultralytics import YOLO

    return YOLO("yolov8n-pose.pt")


# Decode budget for uploaded video. A long, high-resolution video decoded
# frame-for-frame at full resolution can exhaust RAM: each 1080p BGR frame
# is ~6 MB, so thousands of them (a few minutes at 30 fps) reach tens of GB
# held in one Python list — enough to freeze the machine (and, under WSL2,
# the whole VM). We bound this two ways, applied identically on both the
# pose-extraction path and the frame-drawing path so the keypoints (in
# downscaled pixels) line up exactly with the frames they are drawn on:
#   * downscale — cap the longest side to VIDEO_MAX_DIMENSION (aspect kept);
#   * subsample — keep roughly VIDEO_TARGET_FPS frames per second.
VIDEO_MAX_DIMENSION = 640
VIDEO_TARGET_FPS = 10.0


def _downscale_frame(frame):
    """Shrink a BGR frame so its longest side is <= ``VIDEO_MAX_DIMENSION``.

    Preserves aspect ratio; returns the frame unchanged when it is already
    small enough. Pose keypoints are extracted from (and later drawn on)
    these downscaled frames, so the coordinate scale is consistent
    end-to-end.
    """
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= VIDEO_MAX_DIMENSION:
        return frame
    scale = VIDEO_MAX_DIMENSION / longest
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _frame_stride(source_fps: float) -> int:
    """Keep roughly every Nth source frame to approximate ``VIDEO_TARGET_FPS``.

    Always >= 1. A 30 fps source with a 10 fps target yields stride 3.
    Deterministic in ``source_fps`` alone, so both the decode path and the
    on-demand frame reader derive the same stride from the same video and
    therefore agree on which subsampled index maps to which source frame.
    """
    if source_fps <= VIDEO_TARGET_FPS:
        return 1
    return max(1, int(round(source_fps / VIDEO_TARGET_FPS)))


def _decode_video_frames(video_bytes: bytes, extension: str):
    """Persist raw video bytes to a temp file and decode subsampled frames.

    ``cv2.VideoCapture`` needs a path (or a backend that supports
    in-memory buffers, which is not reliably available across
    platforms), so the raw bytes are written to a temporary file first.
    Takes plain ``bytes``/``extension`` rather than the Streamlit
    ``UploadedFile`` object so the cached function that wraps it can be
    called with a hashable, content-addressed argument.

    Frames are downscaled (``_downscale_frame``) and subsampled by
    ``_frame_stride`` as they are read, so only the reduced set is ever
    held in memory. Returns ``(frames, effective_fps)`` where
    ``effective_fps = source_fps / stride`` keeps per-frame timestamps
    correct despite the subsampling.
    """
    suffix = "." + extension
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        stride = _frame_stride(source_fps)
        frames = []
        source_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if source_index % stride == 0:
                frames.append(_downscale_frame(frame))
            source_index += 1
        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames, source_fps / stride


@st.cache_data(show_spinner=False)
def _extract_pose_frame_series(video_bytes, extension, _pose_model):
    """Decode ``video_bytes`` and run pose extraction, cached by video content.

    The expensive step here is ``extract_frame_series`` (YOLOv8-pose
    inference over every frame), not the cheap z-score/zone thresholding
    done afterwards by ``video.analysis.analyze``. Caching this step keyed
    on the uploaded video's own bytes (plus its extension) means that
    re-running it for the *same* video — e.g. clicking "Processar vídeo"
    again after only adjusting the sensitivity/zone sliders — reuses the
    previous inference result instead of re-running YOLO over every frame.

    ``_pose_model`` is prefixed with an underscore per Streamlit's caching
    convention, which excludes it from the cache key: a loaded model
    instance is not meaningfully stable/hashable across reruns, and isn't
    part of what should invalidate the cache anyway (the video content is).

    The progress bar is created and updated *inside* this function rather
    than passed in as an external callback: ``st.cache_data`` replays every
    Streamlit element call recorded during a cache miss when a later call
    is a cache hit, but only if those elements were created inside the
    cached function itself. A progress bar created by the caller and
    mutated from inside this function references a UI block from a
    *previous* script run on replay, which no longer exists — raising
    ``CacheReplayClosureError``. Keeping the whole progress bar self-
    contained here means the recorded replay is internally consistent on
    a cache hit (it just re-renders the bar briefly, then removes it).

    Returns:
        Tuple of ``(frame_series, frame_width, frame_height)``. ``frame_series``
        is ``[]`` and the dimensions are ``0`` when no frame could be
        decoded from the video.
    """
    frames, fps = _decode_video_frames(video_bytes, extension)
    if not frames:
        return [], 0, 0

    progress_bar = st.progress(0.0, text="Processando vídeo frame a frame...")

    def _update_progress(frame_index, total_frames):
        progress_bar.progress(
            (frame_index + 1) / total_frames,
            text=f"Processando vídeo frame a frame... ({frame_index + 1}/{total_frames})",
        )

    frame_series = extract_frame_series(
        _pose_model, frames, fps=fps, on_frame_processed=_update_progress
    )
    progress_bar.empty()

    frame_height, frame_width = frames[0].shape[:2]
    return frame_series, frame_width, frame_height


def _read_sampled_frames(video_bytes, extension, wanted_indices):
    """Read only ``wanted_indices`` of the subsampled frame stream.

    The event report only needs the original frame image at a handful of
    indices — each displayed event's ``frame_index_pior`` (and index 0 for
    the zone preview) — not the whole video. Decoding and holding every
    frame (the old behaviour) is what exhausted memory on long videos, so
    this instead does one sequential ``cv2`` pass (cheap — no YOLO) and
    keeps *only* the requested frames, downscaled and subsampled with the
    exact same ``stride``/``_downscale_frame`` as ``_decode_video_frames``
    so a subsampled index here refers to the same frame as in the pose
    series (and its keypoints line up).

    Args:
        video_bytes: Raw uploaded video bytes.
        extension: File extension (without dot), for the temp file suffix.
        wanted_indices: Iterable of subsampled-stream indices to return.

    Returns:
        Dict ``{subsampled_index: frame_bgr}`` holding only the requested
        (and successfully decoded) frames. Never a full-video list.
    """
    wanted = set(int(i) for i in wanted_indices)
    if not wanted:
        return {}

    suffix = "." + extension
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames_by_index: dict = {}
    try:
        cap = cv2.VideoCapture(tmp_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        stride = _frame_stride(source_fps)
        max_wanted = max(wanted)
        source_index = 0
        sampled_index = 0
        while sampled_index <= max_wanted:
            ret, frame = cap.read()
            if not ret:
                break
            if source_index % stride == 0:
                if sampled_index in wanted:
                    frames_by_index[sampled_index] = _downscale_frame(frame)
                    if len(frames_by_index) == len(wanted):
                        break
                sampled_index += 1
            source_index += 1
        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames_by_index


VIDEO_ALLOWED_EXTENSIONS = ("mp4", "avi", "mov", "mkv")
AUDIO_ALLOWED_EXTENSIONS = ("mp3", "wav")
PRESCRIPTIONS_ALLOWED_EXTENSIONS = ("csv", "xlsx", "xls")

(tab_vitals, tab_video, tab_audio, tab_prescriptions) = st.tabs(
    ["Sinais Vitais", "Vídeo", "Áudio", "Prescrições"]
)

with tab_vitals:
    st.header("Sinais Vitais")
    st.caption(
        "Upload de série temporal de sinais vitais (frequência cardíaca, "
        "oxigenação, pressão arterial etc.). Duas análises complementares são "
        "aplicadas: **Detecção em tempo real** (marca picos súbitos em relação "
        "às leituras recentes, leitura a leitura — internamente um *rolling "
        "z-score*) e **Análise do histórico completo** (marca leituras fora do "
        "padrão geral do paciente, avaliando a série toda de uma vez — "
        "internamente um *Isolation Forest*). Quando as duas concordam, a "
        "leitura tem alta confiança."
    )

    uploaded_file = st.file_uploader(
        "Selecione um CSV de sinais vitais",
        type=["csv"],
        key="vital_signs_uploader",
        help=f"Colunas reconhecidas (ao menos uma): {', '.join(RECOGNIZED_VITAL_SIGN_COLUMNS)}",
    )

    if uploaded_file is not None:
        try:
            vitals_df = load_vital_signs_csv(uploaded_file)
        except VitalSignsValidationError as exc:
            st.error(str(exc))
        else:
            signal_columns = [
                col for col in vitals_df.columns if col.strip().lower() in RECOGNIZED_VITAL_SIGN_COLUMNS
            ]

            friendly_signals = [vital_sign_label(col) for col in signal_columns]
            st.success(
                f"CSV carregado: {len(vitals_df)} leituras. "
                f"Sinais: {', '.join(friendly_signals)}."
            )
            with st.expander("O que cada sinal significa"):
                for col in signal_columns:
                    st.markdown(
                        f"- **{vital_sign_label(col)}**: {vital_sign_description(col)}"
                    )
                st.caption(
                    "As faixas são apenas referência geral para adulto, não "
                    "constituem diagnóstico."
                )

            st.subheader("Série temporal")
            chart_df = vitals_df.set_index("timestamp")[signal_columns]
            st.line_chart(chart_df)

            window = st.number_input(
                "Tamanho da janela de comparação",
                min_value=2,
                value=DEFAULT_WINDOW,
                step=1,
                help=(
                    "Quantas leituras recentes servem de referência para a "
                    "detecção em tempo real decidir se a leitura atual é um "
                    "pico. Exemplos: **13** (padrão) compara com as últimas 13 "
                    "leituras · valores **maiores** suavizam (é preciso um "
                    "desvio mais consistente para alertar) · valores "
                    "**menores** reagem mais rápido a picos curtos, mas geram "
                    "mais alertas por variação normal."
                ),
            )
            threshold = st.number_input(
                "Sensibilidade",
                min_value=0.1,
                value=DEFAULT_THRESHOLD,
                step=0.1,
                help=(
                    "O quanto uma leitura precisa se afastar do normal recente "
                    "para virar alerta (internamente, o limite de z-score). "
                    "Exemplos: **3.0** (padrão) · valores **menores** (ex.: "
                    "2.0) deixam a detecção mais sensível — marca até desvios "
                    "pequenos, gera mais alertas · valores **maiores** (ex.: "
                    "4.0) deixam menos sensível — só marca picos bem "
                    "acentuados."
                ),
            )

            if not zscore_threshold_is_reachable(int(window), float(threshold)):
                max_z = math.sqrt(int(window) - 1) if int(window) >= 2 else 0.0
                st.warning(
                    "Com esses valores, a detecção em tempo real nunca marcaria uma "
                    "anomalia: o desvio máximo detectável para uma janela de "
                    f"{int(window)} é sqrt(janela-1) ≈ {max_z:.2f}, que não supera a "
                    f"sensibilidade de {float(threshold):.2f}. Aumente a janela de "
                    "comparação ou reduza a sensibilidade para que a detecção em tempo "
                    "real volte a funcionar."
                )

            if st.button("Processar sinais vitais", key="vital_signs_process_button"):
                result = analyze(vitals_df, window=int(window), threshold=float(threshold))
                combined_report = result["combined_report"]

                st.subheader("Leituras que chamaram atenção")
                anomalies_only = combined_report[combined_report["agreement"] != "normal"]

                if anomalies_only.empty:
                    st.success(
                        "Nenhuma anomalia encontrada: todas as leituras ficaram "
                        "dentro do padrão esperado nas duas análises."
                    )
                else:
                    summary = build_vitals_summary(combined_report)

                    # Top summary: one clear sentence + short bullet list of the
                    # main critical readings (built by the tested helper).
                    criticas = summary["por_nivel"].get("alta_confianca", 0)
                    if criticas:
                        st.markdown(
                            f"**{summary['total_anomalias']} leitura(s) fora do padrão**, "
                            f"das quais **{criticas} de alta confiança** (as duas "
                            "análises concordaram)."
                        )
                    else:
                        st.markdown(
                            f"**{summary['total_anomalias']} leitura(s) fora do padrão** "
                            "(nenhuma de alta confiança — nenhuma leitura foi marcada "
                            "pelas duas análises ao mesmo tempo)."
                        )

                    for item in summary["itens"]:
                        nivel = confidence_level(item["nivel"])
                        valor_txt = f" ({item['valor']:g})" if item["valor"] is not None else ""
                        st.markdown(
                            f"- {nivel['icon']} **{item['sinal_label']}**{valor_txt} "
                            f"às {item['timestamp']} — {nivel['label']}"
                        )

                    # Friendly, translated table: one row per anomalous reading
                    # (max_itens=None ⇒ the same tested helper returns them all,
                    # with translated signal names and the responsible signal).
                    table_items = build_vitals_summary(combined_report, max_itens=None)["itens"]
                    display_rows = []
                    for item in table_items:
                        nivel = confidence_level(item["nivel"])
                        display_rows.append(
                            {
                                "ID": item["id"],
                                "Horário": item["timestamp"],
                                "Sinal vital": item["sinal_label"],
                                "Valor": item["valor"],
                                "Nível de confiança": f"{nivel['icon']} {nivel['label']}",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(display_rows), use_container_width=True, hide_index=True
                    )

                    # Compact legend so the user can decode the confidence column.
                    st.markdown("**Como ler o nível de confiança:**")
                    for agreement in ("alta_confianca", "zscore_only", "isolation_forest_only"):
                        nivel = confidence_level(agreement)
                        st.caption(
                            f"{nivel['icon']} **{nivel['label']}** — {nivel['short']}",
                            help=nivel["help"],
                        )

                if result["alerts"]:
                    st.subheader("Alertas gerados (feed compartilhado)")
                    for alert in result["alerts"]:
                        # Classify visually by confidence level (consistent
                        # with the table/legend): high confidence → st.error,
                        # otherwise st.warning; prefixed with the level icon
                        # and the linking id so the alert casa com a linha.
                        nivel = confidence_level(alert.level)
                        prefix = f"{nivel['icon']} {alert.alert_id} " if alert.alert_id else f"{nivel['icon']} "
                        render = st.error if alert.level == "alta_confianca" else st.warning
                        render(f"{prefix}[{alert.timestamp}] {alert.description}")
    else:
        st.info("Faça upload de um CSV para iniciar a análise.")

with tab_video:
    st.header("Vídeo")
    st.caption(
        "Upload de vídeo de fisioterapia/exercício ou cirurgia gravada. O "
        "vídeo é processado frame a frame com YOLOv8-pose, extraindo os "
        "ângulos de múltiplas articulações do corpo (cotovelos, joelhos, "
        "quadris/tronco e pescoço, ambos os lados) e uma velocidade de "
        "movimento global, além das detecções de pessoa no mesmo forward "
        "pass. Os desvios são detectados por rolling z-score (sensibilidade "
        "ajustável abaixo) e apresentados como eventos visuais: para cada "
        "momento irregular, a imagem do frame mais representativo com o "
        "esqueleto desenhado e a articulação afetada destacada. A detecção "
        "de zona crítica é opcional (desativada por padrão)."
    )

    video_file = st.file_uploader(
        "Selecione um vídeo (mp4, avi, mov ou mkv)",
        type=list(VIDEO_ALLOWED_EXTENSIONS),
        key="video_uploader",
        help="Formatos aceitos: " + ", ".join(VIDEO_ALLOWED_EXTENSIONS),
    )

    if video_file is not None:
        extension = video_file.name.rsplit(".", 1)[-1].lower() if "." in video_file.name else ""
        if extension not in VIDEO_ALLOWED_EXTENSIONS:
            st.error(
                "Formato de arquivo não suportado: "
                f"'.{extension or video_file.name}'. Formatos aceitos: "
                + ", ".join(VIDEO_ALLOWED_EXTENSIONS)
                + "."
            )
        else:
            pose_model = None
            frame_series: list = []
            frame_width = frame_height = 0

            with st.spinner(
                "Carregando modelo YOLOv8-pose e lendo o vídeo "
                "(pode baixar pesos na primeira execução)..."
            ):
                try:
                    pose_model = _load_pose_model()
                except Exception as exc:  # pragma: no cover - defensive, requires network/model failure
                    st.error(f"Não foi possível carregar o modelo YOLOv8-pose: {exc}")

                if pose_model is not None:
                    # Pose extraction (expensive YOLOv8-pose inference over
                    # every frame) runs right after upload — not only after
                    # clicking "Processar vídeo" — so the sensitivity slider
                    # below can already open pre-filled with a suggestion
                    # calibrated to this video's own motion. It's cached by
                    # the video's own bytes (see the function's docstring),
                    # so clicking "Processar vídeo" afterwards, or moving a
                    # slider, reuses this same result instead of re-running
                    # YOLO inference.
                    frame_series, frame_width, frame_height = _extract_pose_frame_series(
                        video_file.getvalue(),
                        extension,
                        pose_model,
                    )

            if pose_model is not None and not frame_series:
                st.error("Não foi possível ler nenhum frame do vídeo enviado. Verifique o arquivo.")

            suggested_threshold = (
                suggest_sensitivity_threshold(frame_series, window=VIDEO_DEFAULT_WINDOW)
                if frame_series
                else FALLBACK_SENSITIVITY_THRESHOLD
            )
            st.caption(
                f"Sensibilidade sugerida para este vídeo: **{suggested_threshold:.1f}** "
                "(calculada a partir da variação de movimento detectada nele). "
                "Você pode ajustar livremente abaixo."
            )
            sensitivity_threshold = st.slider(
                "Sensibilidade a movimentos anormais",
                min_value=0.5,
                max_value=5.0,
                value=suggested_threshold,
                step=0.1,
                help=(
                    "Controla o quão diferente um movimento precisa ser do padrão "
                    "recente do paciente para virar alerta. Exemplos: **1.0** "
                    "(bem sensível — sinaliza até pequenas variações, gera mais "
                    "alertas) · **2.0** (padrão, equilíbrio razoável) · **4.0** "
                    "(pouco sensível — só sinaliza desvios bem bruscos). O valor "
                    "inicial já vem sugerido para o vídeo carregado."
                ),
            )
            # Dynamic feedback: recomputed from this video on every slider
            # move (spec "Feedback do efeito da sensibilidade escolhida"), so
            # the user sees the concrete effect of the threshold without
            # interpreting a raw z-score value.
            if frame_series:
                flagged_fraction = estimate_flagged_fraction(
                    frame_series, float(sensitivity_threshold), VIDEO_DEFAULT_WINDOW
                )
                st.caption(
                    f"Neste nível, ~{flagged_fraction * 100:.0f}% do vídeo seria "
                    "marcado como irregular."
                )

            # Zone detection is opt-in (spec "Detecção opcional de zona
            # crítica"): the x/y sliders and the preview only appear when the
            # checkbox is on, and ``zone`` is only passed to ``analyze`` in
            # that case (``None`` otherwise ⇒ no zone events generated).
            analyze_zone = st.checkbox("Analisar zona de risco", value=False)
            critical_zone = None
            if analyze_zone:
                zone_x_range = st.slider(
                    "Área de risco no vídeo — de que ponto até que ponto, na horizontal",
                    min_value=0.0,
                    max_value=1.0,
                    value=(VIDEO_DEFAULT_ZONE[0], VIDEO_DEFAULT_ZONE[2]),
                    step=0.05,
                    help=(
                        "Marca uma faixa vertical do quadro do vídeo (de 0 = borda "
                        "esquerda até 1 = borda direita) como área de risco: se a "
                        "pessoa entrar nela, gera um alerta imediato. Exemplos: "
                        "**(0.0, 1.0)** cobre a largura toda do quadro · **(0.7, "
                        "1.0)** cobre só a faixa mais à direita (ex.: perto de uma "
                        "escada ou equipamento posicionado à direita na imagem)."
                    ),
                )
                zone_y_range = st.slider(
                    "Área de risco no vídeo — de que ponto até que ponto, na vertical",
                    min_value=0.0,
                    max_value=1.0,
                    value=(VIDEO_DEFAULT_ZONE[1], VIDEO_DEFAULT_ZONE[3]),
                    step=0.05,
                    help=(
                        "Marca uma faixa horizontal do quadro do vídeo (de 0 = topo "
                        "até 1 = base) como área de risco, combinada com a faixa "
                        "horizontal acima para formar um retângulo. Exemplos: "
                        "**(0.0, 1.0)** cobre a altura toda do quadro · **(0.0, "
                        "0.3)** cobre só a parte de cima da imagem (ex.: uma "
                        "prateleira alta ou porta)."
                    ),
                )
                critical_zone = (zone_x_range[0], zone_y_range[0], zone_x_range[1], zone_y_range[1])

                # Preview: draw the zone rectangle on the first frame so the
                # user sees where it lands before processing (spec "Prévia da
                # zona sobre o frame ao ativar"). ``draw_zone_on_frame`` maps
                # x_max/y_max=1.0 to width/height, i.e. one pixel past the last
                # index; clamp the bottom-right to width-1/height-1 here (Task
                # 3 review note) so a full-extent zone stays visible on-frame
                # rather than being clipped off. Handled at the call site to
                # keep video/draw.py (out of scope) untouched.
                preview_frames = _read_sampled_frames(video_file.getvalue(), extension, [0])
                if preview_frames:
                    first_frame = preview_frames[0]
                    ph, pw = first_frame.shape[:2]
                    preview_zone = (
                        critical_zone[0],
                        critical_zone[1],
                        min(critical_zone[2], (pw - 1) / pw),
                        min(critical_zone[3], (ph - 1) / ph),
                    )
                    st.image(
                        draw_zone_on_frame(first_frame, preview_zone),
                        channels="BGR",
                        caption="Prévia da zona de risco sobre o primeiro frame do vídeo.",
                    )

            if st.button("Processar vídeo", key="video_process_button"):
                if not frame_series:
                    st.error("Não foi possível ler nenhum frame do vídeo enviado. Verifique o arquivo.")
                else:
                    total_frames = len(frame_series)
                    frames_with_pose = sum(1 for f in frame_series if f["has_pose"])
                    st.success(
                        f"Vídeo processado: {total_frames} frames, "
                        f"{frames_with_pose} com pose detectada "
                        f"({total_frames - frames_with_pose} sem dados de pose)."
                    )

                    video_result = analyze_video(
                        frame_series,
                        threshold=float(sensitivity_threshold),
                        window=VIDEO_DEFAULT_WINDOW,
                        zone=critical_zone,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )

                    events = video_result["events"]
                    summary = video_result["summary"]

                    st.subheader("Relatório de desvios do vídeo")
                    if not events:
                        st.info("Nenhum desvio foi encontrado no vídeo processado.")
                    else:
                        # Summary: event count + most-affected joint. The
                        # affected-joint line is only shown when there are
                        # postural events (velocity/zone events aren't tied to
                        # a joint, so ``most_affected_label`` is then None).
                        if summary["most_affected_label"]:
                            st.markdown(
                                f"**{summary['total_events']} evento(s)** irregular(es) "
                                f"detectado(s). Articulação mais afetada: "
                                f"**{summary['most_affected_label']}**."
                            )
                        else:
                            st.markdown(
                                f"**{summary['total_events']} evento(s)** irregular(es) "
                                "detectado(s)."
                            )

                        # Group events by joint/type into collapsible sections
                        # with a top-N gallery each (change
                        # galeria-eventos-video-por-articulacao). The grouping/
                        # sorting/top-N logic lives in the tested analysis
                        # helper; here we only render. Sections come ordered
                        # most-affected first; each expander is closed by
                        # default so the page opens light even with hundreds of
                        # events.
                        secoes = group_events_for_display(events, top_n=10)

                        # Read only the worst frame of each *displayed* event
                        # (cheap — no YOLO), never the whole video: a long clip
                        # can hold tens of GB of raw frames and freeze the host.
                        # frame_index_pior indexes both frame_series and the
                        # subsampled decode, so the same index picks the frame
                        # and its keypoints.
                        wanted_indices = [
                            event["frame_index_pior"]
                            for secao in secoes
                            for event in secao["eventos"]
                        ]
                        report_frames = _read_sampled_frames(
                            video_file.getvalue(), extension, wanted_indices
                        )

                        columns_per_row = 3
                        for secao in secoes:
                            eventos = secao["eventos"]
                            with st.expander(
                                f"{secao['label']} — {secao['total']} evento(s)", expanded=False
                            ):
                                if secao["total"] > len(eventos):
                                    st.caption(f"Mostrando {len(eventos)} de {secao['total']}")

                                for row_start in range(0, len(eventos), columns_per_row):
                                    row_events = eventos[row_start : row_start + columns_per_row]
                                    cols = st.columns(columns_per_row)
                                    for col, event in zip(cols, row_events):
                                        idx = event["frame_index_pior"]
                                        # Prefix the same short unique id shown
                                        # in the alert text so the user can match
                                        # this frame to its feed alert (change
                                        # alerta-video-id-categoria, design.md D4).
                                        interval = (
                                            f"{event['event_id']} — "
                                            f"{event['t_inicio']:.1f}s a {event['t_fim']:.1f}s"
                                        )

                                        if idx in report_frames:
                                            frame = report_frames[idx]
                                            keypoints = frame_series[idx].get("keypoints_xy")

                                            if event["tipo"] == "postura":
                                                annotated = (
                                                    draw_pose_on_frame(
                                                        frame,
                                                        keypoints,
                                                        highlight_joint=event["articulacao"],
                                                    )
                                                    if keypoints is not None
                                                    else frame
                                                )
                                            elif event["tipo"] == "velocidade":
                                                # No single joint to highlight for
                                                # a global velocity event: draw the
                                                # whole skeleton.
                                                annotated = (
                                                    draw_pose_on_frame(frame, keypoints)
                                                    if keypoints is not None
                                                    else frame
                                                )
                                            else:  # zona_critica
                                                ph, pw = frame.shape[:2]
                                                zone_draw = (
                                                    critical_zone[0],
                                                    critical_zone[1],
                                                    min(critical_zone[2], (pw - 1) / pw),
                                                    min(critical_zone[3], (ph - 1) / ph),
                                                )
                                                annotated = draw_zone_on_frame(frame, zone_draw)

                                            col.image(annotated, channels="BGR", caption=interval)
                                        else:
                                            # Defensive: no decoded frame for this
                                            # index (shouldn't happen — same video).
                                            col.warning(interval)

                    if video_result["alerts"]:
                        st.subheader("Alertas gerados (feed compartilhado)")
                        for alert in video_result["alerts"]:
                            st.warning(f"[{alert.timestamp}] {alert.description}")
    else:
        st.info("Faça upload de um vídeo para iniciar a análise.")

with tab_audio:
    st.header("Áudio")
    st.caption(
        "Upload de áudio de consulta médica (mp3 ou wav). O áudio é "
        "transcrito via AWS Transcribe (texto + timestamps por palavra), "
        "o texto transcrito é analisado pelo AWS Comprehend (sentimento "
        "e entidades) e verificado contra uma lista configurável de "
        "termos críticos (ex.: \"dor\", \"não consigo respirar\"), cada "
        "ocorrência gerando um alerta no feed compartilhado. Séries de "
        "taxa de fala e duração de pausa são derivadas dos timestamps do "
        "Transcribe e passadas pelo mesmo detector de anomalia por "
        "rolling z-score usado nas demais abas, sinalizando segmentos "
        "compatíveis com fadiga ou disartria."
    )

    bucket_name = get_configured_bucket_name()
    if not bucket_name:
        st.warning(
            "Nenhum bucket S3 configurado para o AWS Transcribe. Defina a "
            f"variável de ambiente `{AUDIO_S3_BUCKET_ENV_VAR}` com o nome de "
            "um bucket S3 existente e acessível pelas credenciais AWS "
            "configuradas, e reinicie a aplicação."
        )

    audio_file = st.file_uploader(
        "Selecione um áudio (mp3 ou wav)",
        type=list(AUDIO_ALLOWED_EXTENSIONS),
        key="audio_uploader",
        help="Formatos aceitos: " + ", ".join(AUDIO_ALLOWED_EXTENSIONS),
    )

    if audio_file is not None:
        extension = audio_file.name.rsplit(".", 1)[-1].lower() if "." in audio_file.name else ""
        if extension not in AUDIO_ALLOWED_EXTENSIONS:
            st.error(
                "Formato de arquivo não suportado: "
                f"'.{extension or audio_file.name}'. Formatos aceitos: "
                + ", ".join(AUDIO_ALLOWED_EXTENSIONS)
                + "."
            )
        elif not bucket_name:
            st.error(
                "Não é possível processar o áudio: nenhum bucket S3 está "
                f"configurado (variável de ambiente `{AUDIO_S3_BUCKET_ENV_VAR}`). "
                "Configure a variável de ambiente e reinicie a aplicação antes "
                "de enviar um áudio."
            )
        else:
            critical_terms_input = st.text_input(
                "Termos críticos (separados por vírgula)",
                value=", ".join(DEFAULT_CRITICAL_TERMS),
                help="Lista configurável de termos que geram alerta imediato quando encontrados na transcrição.",
            )
            critical_terms = [t.strip() for t in critical_terms_input.split(",") if t.strip()]

            if st.button("Processar áudio", key="audio_process_button"):
                try:
                    clients = build_clients()
                    with st.spinner(
                        "Transcrevendo áudio via AWS Transcribe (pode levar até alguns minutos)..."
                    ):
                        transcription = transcribe_audio(
                            audio_bytes=audio_file.getvalue(),
                            file_extension=extension,
                            bucket_name=bucket_name,
                            s3_client=clients["s3"],
                            transcribe_client=clients["transcribe"],
                            job_name=f"tech-challenge-audio-{uuid.uuid4().hex}",
                        )
                except AudioProcessingError as exc:
                    st.error(f"Falha ao transcrever o áudio: {exc}")
                except Exception as exc:  # pragma: no cover - defensive, e.g. missing/invalid AWS credentials
                    st.error(f"Falha inesperada ao acessar os serviços AWS: {exc}")
                else:
                    text = transcription["text"]
                    words = transcription["words"]

                    st.subheader("Transcrição")
                    if not text:
                        st.info("Nenhuma fala foi identificada no áudio enviado.")
                    else:
                        st.write(text)
                        st.caption(f"{len(words)} palavra(s) com timestamp identificadas.")

                        try:
                            sentiment_result = analyze_audio_sentiment(text, client=clients["comprehend"])
                            entities = detect_entities(text, client=clients["comprehend"])
                        except AudioProcessingError as exc:
                            st.error(f"Falha ao analisar o texto via AWS Comprehend: {exc}")
                        else:
                            st.subheader("Sentimento (AWS Comprehend)")
                            st.write(
                                f"**{sentiment_result['sentiment']}** — "
                                f"{sentiment_result['sentiment_score']}"
                            )

                            st.subheader("Entidades (AWS Comprehend)")
                            if not entities:
                                st.info("Nenhuma entidade identificada no texto transcrito.")
                            else:
                                st.dataframe(pd.DataFrame(entities), use_container_width=True)

                        critical_alerts = raise_critical_term_alerts(text, terms=critical_terms)
                        st.subheader("Termos críticos")
                        if not critical_alerts:
                            st.info("Nenhum termo crítico foi identificado na transcrição.")
                        else:
                            for alert in critical_alerts:
                                st.warning(f"{alert.alert_id} [{alert.timestamp}] {alert.description}")

                        if words:
                            # Chain the alert_id sequence after the critical-term
                            # alerts above (change alertas-estruturados-audio-
                            # prescricoes, design.md D1) so #A01, #A02... follows
                            # the same top-to-bottom order the user sees them in.
                            audio_result = analyze_audio(
                                words,
                                window=AUDIO_DEFAULT_WINDOW,
                                threshold=AUDIO_DEFAULT_THRESHOLD,
                                start_index=len(critical_alerts) + 1,
                            )

                            st.subheader("Anomalias de fala (taxa de fala / duração de pausa)")
                            has_speech_anomaly = audio_result["speech_rate_anomalies"].any()
                            has_pause_anomaly = audio_result["pause_anomalies"].any()
                            if not has_speech_anomaly and not has_pause_anomaly:
                                st.info(
                                    "Nenhum indicador de fadiga ou disartria foi detectado "
                                    "(taxa de fala e duração de pausa dentro do esperado)."
                                )
                            else:
                                n_speech = int(audio_result["speech_rate_anomalies"].sum())
                                n_pause = int(audio_result["pause_anomalies"].sum())
                                st.write(
                                    f"{n_speech} segmento(s) com taxa de fala anômala, "
                                    f"{n_pause} pausa(s) anômala(s)."
                                )
                                for alert in audio_result["alerts"]:
                                    st.warning(f"{alert.alert_id} [{alert.timestamp}] {alert.description}")
    else:
        st.info("Faça upload de um áudio para iniciar a análise.")

with tab_prescriptions:
    st.header("Prescrições")
    st.caption(
        "Upload de histórico de prescrições (CSV ou Excel) por paciente ao "
        "longo do tempo. Para cada paciente, o histórico é enviado ao AWS "
        "Bedrock (Claude Sonnet), que analisa o texto/dados em busca de "
        "inconsistências: mudança abrupta de dose, possível interação "
        "medicamentosa ou alteração de dose sem justificativa clínica "
        "aparente. Cada inconsistência apontada gera um alerta no feed "
        "compartilhado."
    )

    prescriptions_file = st.file_uploader(
        "Selecione um arquivo de prescrições (CSV ou Excel)",
        type=list(PRESCRIPTIONS_ALLOWED_EXTENSIONS),
        key="prescriptions_uploader",
        help=f"Colunas obrigatórias: {', '.join(PRESCRIPTION_REQUIRED_COLUMNS)}.",
    )

    if prescriptions_file is not None:
        try:
            prescriptions_df = load_prescriptions(prescriptions_file, filename=prescriptions_file.name)
        except PrescriptionValidationError as exc:
            st.error(str(exc))
        else:
            st.success(f"Arquivo carregado: {len(prescriptions_df)} prescrição(ões).")

            patients = sorted(prescriptions_df["paciente"].unique())
            selected_patient = st.selectbox("Paciente", patients, key="prescriptions_patient_select")

            patient_history = prescriptions_df[prescriptions_df["paciente"] == selected_patient]

            st.subheader(f"Histórico de prescrições — {selected_patient}")
            st.dataframe(patient_history, use_container_width=True)

            if st.button("Revisar inconsistências via Bedrock", key="prescriptions_review_button"):
                try:
                    with st.spinner("Analisando histórico via AWS Bedrock (Claude Sonnet)..."):
                        client = build_bedrock_client()
                        review_result = review_patient_prescriptions(
                            prescriptions_df, selected_patient, client=client
                        )
                except PrescriptionReviewError as exc:
                    st.error(f"Falha ao revisar prescrições via AWS Bedrock: {exc}")
                except Exception as exc:  # pragma: no cover - defensive, e.g. missing/invalid AWS credentials
                    st.error(f"Falha inesperada ao acessar o AWS Bedrock: {exc}")
                else:
                    st.subheader("Inconsistências identificadas (AWS Bedrock)")
                    findings = review_result["findings"]
                    if not findings:
                        st.info("Nenhuma inconsistência encontrada para este paciente.")
                    else:
                        # zip is safe here: generate_alerts_for_findings creates
                        # exactly one alert per finding, in the same order.
                        for finding, alert in zip(findings, review_result["alerts"]):
                            st.warning(
                                f"{alert.alert_id} **{finding.get('tipo', 'inconsistencia')}** — "
                                f"{finding.get('explicacao', 'sem detalhes fornecidos.')}"
                            )
                        st.caption(
                            f"{len(review_result['alerts'])} alerta(s) gerado(s) no feed compartilhado."
                        )
    else:
        st.info("Faça upload de um CSV ou Excel de prescrições para iniciar a análise.")

# Unified cross-tab alert feed (clinical-alerting spec), rendered after
# all 4 tabs so it reflects any alert generated during this run, not just
# alerts accumulated in previous reruns. This is an additional view on
# top of each tab's own inline alert display above — not a replacement.
_render_alert_feed()
