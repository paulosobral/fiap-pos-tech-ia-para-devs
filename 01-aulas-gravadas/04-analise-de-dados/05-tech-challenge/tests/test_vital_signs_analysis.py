"""Tests for vital_signs/analysis.py — CSV loading/validation, per-signal
rolling z-score alert generation and combined z-score / Isolation Forest
report.

Covers scenarios from the vital-signs-monitoring spec:
- Requirement: Upload de série temporal de sinais vitais
  - Scenario: Upload de CSV válido
  - Scenario: Upload de CSV com colunas inválidas
- Requirement: Detecção de anomalia em tempo real via rolling z-score
  - Scenario: Leitura de sinal vital fora do padrão
- Requirement: Validação batch via Isolation Forest
  - Scenario: Isolation Forest identifica anomalia não capturada pelo z-score
  - Scenario: Concordância entre as duas camadas de detecção
"""
import io
from datetime import datetime

import pandas as pd
import pytest
import streamlit as st

from vital_signs.analysis import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    RECOGNIZED_VITAL_SIGN_COLUMNS,
    VitalSignsValidationError,
    analyze,
    load_vital_signs_csv,
    zscore_threshold_is_reachable,
)


@pytest.fixture(autouse=True)
def clean_session_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


def _csv_bytes(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


# ── load_vital_signs_csv ─────────────────────────────────────────────


def test_load_vital_signs_csv_accepts_file_with_timestamp_and_recognized_signal():
    csv = _csv_bytes(
        "timestamp,heart_rate,spo2\n"
        "2024-01-01 00:00:00,80,98\n"
        "2024-01-01 01:00:00,82,97\n"
    )

    df = load_vital_signs_csv(csv)

    assert "timestamp" in df.columns
    assert "heart_rate" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert len(df) == 2


def test_load_vital_signs_csv_sorts_rows_by_timestamp():
    csv = _csv_bytes(
        "timestamp,heart_rate\n"
        "2024-01-01 02:00:00,90\n"
        "2024-01-01 00:00:00,80\n"
        "2024-01-01 01:00:00,85\n"
    )

    df = load_vital_signs_csv(csv)

    assert list(df["heart_rate"]) == [80, 85, 90]


def test_load_vital_signs_csv_rejects_file_without_recognized_vital_sign_column():
    csv = _csv_bytes("timestamp,foo,bar\n2024-01-01 00:00:00,1,2\n")

    with pytest.raises(VitalSignsValidationError) as exc_info:
        load_vital_signs_csv(csv)

    message = str(exc_info.value)
    # Error must be clear about which columns were expected.
    assert "heart_rate" in message or "sinal vital" in message.lower()


def test_load_vital_signs_csv_accepts_missing_timestamp_by_falling_back_to_row_order():
    # Spec's rejection scenario is specifically about missing vital-sign
    # columns; a CSV with a recognized signal but no timestamp column should
    # still degrade gracefully (sequential index) rather than crash.
    csv = _csv_bytes("heart_rate,spo2\n80,98\n82,97\n")

    df = load_vital_signs_csv(csv)

    assert "timestamp" in df.columns
    assert list(df["timestamp"]) == [0, 1]


def test_load_vital_signs_csv_rejects_unparseable_timestamp_value():
    # A recognized vital-sign column is present, but the timestamp values
    # are not parseable as dates. Must raise the module's validation error
    # (clean st.error in app.py), not crash with an uncaught
    # pandas.errors.DateParseError.
    csv = _csv_bytes("timestamp,heart_rate\nnot-a-date,80\nalso-bad,82\n")

    with pytest.raises(VitalSignsValidationError) as exc_info:
        load_vital_signs_csv(csv)

    message = str(exc_info.value)
    assert "timestamp" in message.lower()


def test_load_vital_signs_csv_rejects_non_numeric_value_in_signal_column():
    # A recognized vital-sign column has a non-numeric value in one row.
    # Must raise the module's validation error, not crash further down the
    # pipeline (e.g. rolling z-score / Isolation Forest) with an uncaught
    # pandas.errors.DataError.
    csv = _csv_bytes(
        "timestamp,heart_rate\n"
        "2024-01-01 00:00:00,80\n"
        "2024-01-01 01:00:00,bad\n"
        "2024-01-01 02:00:00,82\n"
    )

    with pytest.raises(VitalSignsValidationError) as exc_info:
        load_vital_signs_csv(csv)

    message = str(exc_info.value)
    assert "heart_rate" in message


def test_recognized_vital_sign_columns_include_common_signals():
    assert "heart_rate" in RECOGNIZED_VITAL_SIGN_COLUMNS
    assert "spo2" in RECOGNIZED_VITAL_SIGN_COLUMNS
    assert "systolic_bp" in RECOGNIZED_VITAL_SIGN_COLUMNS


# ── analyze() — z-score layer + alert generation ────────────────────


def test_analyze_flags_spike_and_generates_alert_referencing_signal_and_timestamp():
    timestamps = pd.date_range("2024-01-01", periods=12, freq="h")
    heart_rate = [80] * 8 + [200] + [80] * 3  # single clear spike
    df = pd.DataFrame({"timestamp": timestamps, "heart_rate": heart_rate})

    result = analyze(df, window=6, threshold=2.0)

    assert result["zscore_anomalies"]["heart_rate"].iloc[8] is True or bool(
        result["zscore_anomalies"]["heart_rate"].iloc[8]
    )
    assert len(result["alerts"]) >= 1
    alert = result["alerts"][0]
    assert alert.origin == "Sinais Vitais"
    assert "heart_rate" in alert.description
    assert "200" in alert.description


def test_analyze_flags_clear_spike_with_default_window_and_threshold():
    # Regression proof: with the shipped DEFAULT_WINDOW / DEFAULT_THRESHOLD,
    # a clearly anomalous spike must be flagged by the z-score layer. This
    # test FAILS under the old DEFAULT_WINDOW=6 (ceiling sqrt(5)≈2.24 < 3.0,
    # so |z| can never exceed the threshold) and PASSES under window=13
    # (ceiling sqrt(12)≈3.46 > 3.0).
    periods = 2 * DEFAULT_WINDOW
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h")
    heart_rate = [80] * periods
    spike_index = DEFAULT_WINDOW  # ensure a full trailing window exists
    heart_rate[spike_index] = 100000  # single, extreme, isolated spike
    df = pd.DataFrame({"timestamp": timestamps, "heart_rate": heart_rate})

    result = analyze(df, window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD)

    assert bool(result["zscore_anomalies"]["heart_rate"].iloc[spike_index])
    assert len(result["alerts"]) >= 1


def test_zscore_threshold_is_reachable_boundaries():
    # window=13, threshold=3.0 → sqrt(12)≈3.46 > 3.0 → reachable
    assert zscore_threshold_is_reachable(13, 3.0) is True
    # window=6, threshold=3.0 → sqrt(5)≈2.24 < 3.0 → NOT reachable (the bug)
    assert zscore_threshold_is_reachable(6, 3.0) is False
    # window=10, threshold=3.0 → sqrt(9)=3.0 is NOT > 3.0 → NOT reachable
    assert zscore_threshold_is_reachable(10, 3.0) is False
    # window<2 guard: ceiling undefined/zero → NOT reachable
    assert zscore_threshold_is_reachable(1, 0.1) is False
    assert zscore_threshold_is_reachable(0, 3.0) is False


def test_default_params_are_reachable():
    # The shipped defaults must be an effective (reachable) combination.
    assert zscore_threshold_is_reachable(DEFAULT_WINDOW, DEFAULT_THRESHOLD) is True


def test_analyze_alert_timestamp_is_wall_clock_generation_time_not_clinical_reading_time():
    # Alert.timestamp must use the default add_alert() wall-clock time
    # (consistent with the Vídeo/Áudio/Prescrições tabs) so the shared feed
    # sorts newest-first correctly instead of mixing in old clinical
    # timestamps from historical CSVs (e.g. MIMIC-III samples from 2024).
    # The clinical/CSV event time must still be present in the description.
    clinical_timestamps = pd.date_range("2024-01-01", periods=12, freq="h")
    heart_rate = [80] * 8 + [200] + [80] * 3
    df = pd.DataFrame({"timestamp": clinical_timestamps, "heart_rate": heart_rate})

    before = datetime.now()
    result = analyze(df, window=6, threshold=2.0)
    after = datetime.now()

    assert len(result["alerts"]) >= 1
    alert = result["alerts"][0]
    assert before <= alert.timestamp <= after
    assert "2024-01-01" in alert.description


def test_analyze_stable_series_generates_no_alerts():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "heart_rate": [80] * 10})

    result = analyze(df, window=4, threshold=3.0)

    assert result["alerts"] == []
    assert not result["zscore_anomalies"]["heart_rate"].any()


def test_analyze_raises_when_no_recognized_signal_column_present():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="h"), "foo": [1, 2, 3]})

    with pytest.raises(VitalSignsValidationError):
        analyze(df)


def test_analyze_pushes_alerts_to_shared_feed():
    from alerts.feed import get_alerts

    timestamps = pd.date_range("2024-01-01", periods=12, freq="h")
    heart_rate = [80] * 8 + [200] + [80] * 3
    df = pd.DataFrame({"timestamp": timestamps, "heart_rate": heart_rate})

    analyze(df, window=6, threshold=2.0)

    feed_origins = {a.origin for a in get_alerts()}
    assert "Sinais Vitais" in feed_origins


# ── analyze() — combined report (z-score + Isolation Forest) ────────


def test_analyze_combined_report_has_agreement_categories():
    timestamps = pd.date_range("2024-01-01", periods=30, freq="h")
    heart_rate = [80] * 30
    df = pd.DataFrame({"timestamp": timestamps, "heart_rate": heart_rate})

    result = analyze(df, window=6, threshold=2.0)

    report = result["combined_report"]
    assert "agreement" in report.columns
    assert set(report["agreement"].unique()) <= {
        "alta_confianca",
        "zscore_only",
        "isolation_forest_only",
        "normal",
    }


def test_analyze_combined_report_marks_agreement_when_both_layers_flag_same_row():
    timestamps = pd.date_range("2024-01-01", periods=14, freq="h")
    heart_rate = [80] * 10 + [300] + [80] * 3  # extreme outlier: both layers should catch it
    df = pd.DataFrame({"timestamp": timestamps, "heart_rate": heart_rate})

    result = analyze(df, window=6, threshold=2.0)

    report = result["combined_report"]
    spike_row = report.iloc[10]
    assert spike_row["zscore_anomaly"] in (True, 1)
    # Both layers should have flagged such an extreme, isolated spike.
    assert spike_row["agreement"] == "alta_confianca"
