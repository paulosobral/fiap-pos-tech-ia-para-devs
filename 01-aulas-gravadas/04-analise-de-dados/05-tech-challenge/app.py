"""Streamlit entrypoint — Monitoramento Multimodal de Pacientes.

Single entrypoint per the Global Constraints (design.md D7): each
capability tab is thin UI glued to its own module's ``analyze(...)``
function, which internally pushes ``Alert``s to the shared feed
(``alerts/feed.py``).

Tabs implemented so far: Sinais Vitais (Task 3), Vídeo (Task 4). The
remaining tabs (Áudio, Prescrições) are added by later tasks; no
placeholder tabs are created ahead of time.
"""
import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

from vital_signs.analysis import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    RECOGNIZED_VITAL_SIGN_COLUMNS,
    VitalSignsValidationError,
    analyze,
    load_vital_signs_csv,
)
from video.analysis import DEFAULT_WINDOW as VIDEO_DEFAULT_WINDOW
from video.analysis import analyze as analyze_video
from video.pose import extract_frame_series

st.set_page_config(page_title="Monitoramento Multimodal de Pacientes", layout="wide")
st.title("Monitoramento Multimodal de Pacientes")

VIDEO_ALLOWED_EXTENSIONS = ("mp4", "avi", "mov")


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


(tab_vitals, tab_video) = st.tabs(["Sinais Vitais", "Vídeo"])

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
                        zone=(0.7, 0.0, 1.0, 1.0),
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
