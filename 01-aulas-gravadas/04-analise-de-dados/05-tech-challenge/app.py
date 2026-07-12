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
from video.analysis import analyze as analyze_video
from video.pose import extract_frame_series

st.set_page_config(page_title="Monitoramento Multimodal de Pacientes", layout="wide")
st.title("Monitoramento Multimodal de Pacientes")


@st.cache_resource(show_spinner=False)
def _load_pose_model():
    """Load the YOLOv8-pose model once per Streamlit process.

    ``yolov8n-pose.pt`` is downloaded automatically by ``ultralytics`` on
    first use if not already present locally (needs internet access).
    """
    from ultralytics import YOLO

    return YOLO("yolov8n-pose.pt")


def _read_video_frames(uploaded_file):
    """Persist the uploaded video to a temp file and read all frames.

    ``cv2.VideoCapture`` needs a path (or a backend that supports
    in-memory buffers, which is not reliably available across
    platforms), so the Streamlit ``UploadedFile`` is written to a
    temporary file first.
    """
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
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


VIDEO_ALLOWED_EXTENSIONS = ("mp4", "avi", "mov")
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
        "Selecione um vídeo (mp4, avi ou mov)",
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
            sensitivity_threshold = st.slider(
                "Sensibilidade da detecção de anomalia postural (|z-score| >)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Valores menores tornam a detecção mais sensível (mais anomalias reportadas).",
            )

            zone_x_range = st.slider(
                "Zona crítica — faixa horizontal (x_min, x_max)",
                min_value=0.0,
                max_value=1.0,
                value=(VIDEO_DEFAULT_ZONE[0], VIDEO_DEFAULT_ZONE[2]),
                step=0.05,
                help=(
                    "Posição relativa da zona crítica no eixo horizontal do quadro "
                    "(0.0 = borda esquerda, 1.0 = borda direita). Ajuste conforme o "
                    "enquadramento do vídeo enviado."
                ),
            )
            zone_y_range = st.slider(
                "Zona crítica — faixa vertical (y_min, y_max)",
                min_value=0.0,
                max_value=1.0,
                value=(VIDEO_DEFAULT_ZONE[1], VIDEO_DEFAULT_ZONE[3]),
                step=0.05,
                help=(
                    "Posição relativa da zona crítica no eixo vertical do quadro "
                    "(0.0 = topo, 1.0 = base). Ajuste conforme o enquadramento do "
                    "vídeo enviado."
                ),
            )
            critical_zone = (zone_x_range[0], zone_y_range[0], zone_x_range[1], zone_y_range[1])

            with st.spinner("Carregando modelo YOLOv8-pose (pode baixar pesos na primeira execução)..."):
                try:
                    pose_model = _load_pose_model()
                except Exception as exc:  # pragma: no cover - defensive, requires network/model failure
                    pose_model = None
                    st.error(f"Não foi possível carregar o modelo YOLOv8-pose: {exc}")

            if pose_model is not None:
                frames, fps = _read_video_frames(video_file)

                if not frames:
                    st.error("Não foi possível ler nenhum frame do vídeo enviado. Verifique o arquivo.")
                else:
                    progress_bar = st.progress(0.0, text="Processando vídeo frame a frame...")

                    def _update_progress(frame_index, total_frames):
                        progress_bar.progress(
                            (frame_index + 1) / total_frames,
                            text=f"Processando vídeo frame a frame... ({frame_index + 1}/{total_frames})",
                        )

                    frame_series = extract_frame_series(
                        pose_model, frames, fps=fps, on_frame_processed=_update_progress
                    )
                    progress_bar.empty()

                    total_frames = len(frames)
                    frames_with_pose = sum(1 for f in frame_series if f["has_pose"])
                    st.success(
                        f"Vídeo processado: {total_frames} frames, "
                        f"{frames_with_pose} com pose detectada "
                        f"({total_frames - frames_with_pose} sem dados de pose)."
                    )

                    frame_height, frame_width = frames[0].shape[:2]
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
        elif bucket_name:
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
