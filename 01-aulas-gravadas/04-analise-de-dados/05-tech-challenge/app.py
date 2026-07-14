"""Streamlit entrypoint — Monitoramento Multimodal de Pacientes.

Single entrypoint per the Global Constraints (design.md D7): each
capability tab is thin UI glued to its own module's ``analyze(...)``
function, which internally pushes ``Alert``s to the shared feed
(``alerts/feed.py``).

Tabs implemented: Sinais Vitais (Task 3), Vídeo (Task 4), Áudio (Task 5),
Prescrições (Task 6).
"""
import os
import tempfile
import uuid

import cv2
import pandas as pd
import streamlit as st

from alerts.feed import get_alerts
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
    load_vital_signs_csv,
)
from video.analysis import DEFAULT_WINDOW as VIDEO_DEFAULT_WINDOW
from video.analysis import DEFAULT_ZONE as VIDEO_DEFAULT_ZONE
from video.analysis import FALLBACK_SENSITIVITY_THRESHOLD
from video.analysis import analyze as analyze_video
from video.analysis import suggest_sensitivity_threshold
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
    for alert in alerts:
        st.sidebar.warning(
            f"**[{alert.origin}]** {alert.timestamp:%Y-%m-%d %H:%M:%S}\n\n{alert.description}"
        )


@st.cache_resource(show_spinner=False)
def _load_pose_model():
    """Load the YOLOv8-pose model once per Streamlit process.

    ``yolov8n-pose.pt`` is downloaded automatically by ``ultralytics`` on
    first use if not already present locally (needs internet access).
    """
    from ultralytics import YOLO

    return YOLO("yolov8n-pose.pt")


def _decode_video_frames(video_bytes: bytes, extension: str):
    """Persist raw video bytes to a temp file and decode all frames.

    ``cv2.VideoCapture`` needs a path (or a backend that supports
    in-memory buffers, which is not reliably available across
    platforms), so the raw bytes are written to a temporary file first.
    Takes plain ``bytes``/``extension`` rather than the Streamlit
    ``UploadedFile`` object so this (and the cached function that wraps
    it below) can be called with a hashable, content-addressed argument.
    """
    suffix = "." + extension
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames, fps


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
        "oxigenação, pressão arterial etc.). Duas camadas de detecção de "
        "anomalia são aplicadas: rolling z-score (linha a linha, tempo "
        "real) e Isolation Forest (lote, validação complementar)."
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

            st.success(f"CSV carregado: {len(vitals_df)} leituras, sinais: {', '.join(signal_columns)}.")

            st.subheader("Série temporal")
            chart_df = vitals_df.set_index("timestamp")[signal_columns]
            st.line_chart(chart_df)

            window = st.number_input(
                "Janela do rolling z-score", min_value=2, value=DEFAULT_WINDOW, step=1
            )
            threshold = st.number_input(
                "Threshold do z-score (|z| >)", min_value=0.1, value=DEFAULT_THRESHOLD, step=0.1
            )

            if st.button("Processar sinais vitais", key="vital_signs_process_button"):
                result = analyze(vitals_df, window=int(window), threshold=float(threshold))
                combined_report = result["combined_report"]

                st.subheader("Relatório combinado de anomalias (z-score + Isolation Forest)")
                anomalies_only = combined_report[combined_report["agreement"] != "normal"]

                if anomalies_only.empty:
                    st.info("Nenhuma anomalia detectada por nenhuma das duas camadas.")
                else:
                    agreement_labels = {
                        "alta_confianca": "Alta confiança (ambas as camadas concordam)",
                        "zscore_only": "Somente rolling z-score",
                        "isolation_forest_only": "Somente Isolation Forest",
                    }
                    display_df = anomalies_only.copy()
                    display_df["agreement"] = display_df["agreement"].map(agreement_labels)
                    st.dataframe(display_df, use_container_width=True)

                    high_confidence = (combined_report["agreement"] == "alta_confianca").sum()
                    st.caption(
                        f"{len(anomalies_only)} leitura(s) anômala(s) no total, das quais "
                        f"{high_confidence} com alta confiança (ambas as camadas concordam)."
                    )

                if result["alerts"]:
                    st.subheader("Alertas gerados (feed compartilhado)")
                    for alert in result["alerts"]:
                        st.warning(f"[{alert.timestamp}] {alert.description}")
    else:
        st.info("Faça upload de um CSV para iniciar a análise.")

with tab_video:
    st.header("Vídeo")
    st.caption(
        "Upload de vídeo de fisioterapia/exercício ou cirurgia gravada. O "
        "vídeo é processado frame a frame com YOLOv8-pose, extraindo "
        "keypoints posturais (ângulo do cotovelo direito e velocidade do "
        "punho direito) e detecções de pessoa no mesmo forward pass. Duas "
        "detecções de desvio são independentes: anomalia postural "
        "(rolling z-score, sensibilidade ajustável pelo slider abaixo) e "
        "alerta de zona crítica (interseção de bounding box com uma área "
        "configurada do quadro)."
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

                    st.subheader("Relatório de desvios do vídeo")
                    deviation_report = video_result["deviation_report"]
                    if not deviation_report:
                        st.info("Nenhum desvio foi encontrado no vídeo processado.")
                    else:
                        report_df = pd.DataFrame(deviation_report)
                        report_df["kind"] = report_df["kind"].map(
                            {"postural": "Anomalia postural", "zona_critica": "Zona crítica"}
                        )
                        st.dataframe(report_df, use_container_width=True)

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
                                st.warning(f"[{alert.timestamp}] {alert.description}")

                        if words:
                            audio_result = analyze_audio(
                                words, window=AUDIO_DEFAULT_WINDOW, threshold=AUDIO_DEFAULT_THRESHOLD
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
                                    st.warning(f"[{alert.timestamp}] {alert.description}")
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
                        for finding in findings:
                            st.warning(
                                f"**{finding.get('tipo', 'inconsistencia')}** — "
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
