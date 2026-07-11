"""Streamlit entrypoint — Monitoramento Multimodal de Pacientes.

Single entrypoint per the Global Constraints (design.md D7): each
capability tab is thin UI glued to its own module's ``analyze(...)``
function, which internally pushes ``Alert``s to the shared feed
(``alerts/feed.py``).

This is the first tab implemented (Task 3 — Sinais Vitais). The remaining
tabs (Vídeo, Áudio, Prescrições) are added by later tasks; no placeholder
tabs are created ahead of time.
"""
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

st.set_page_config(page_title="Monitoramento Multimodal de Pacientes", layout="wide")
st.title("Monitoramento Multimodal de Pacientes")

(tab_vitals,) = st.tabs(["Sinais Vitais"])

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
