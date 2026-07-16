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
    build_vitals_summary,
    confidence_level,
    load_vital_signs_csv,
    vital_sign_label,
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


# ── presentation helpers: vital_sign_label ──────────────────────────


def test_vital_sign_label_translates_known_signals():
    assert vital_sign_label("heart_rate") == "Frequência cardíaca"
    assert vital_sign_label("spo2") == "Saturação de O₂ (SpO₂)"
    assert vital_sign_label("resp_rate") == "Frequência respiratória"
    assert vital_sign_label("respiratory_rate") == "Frequência respiratória"
    assert vital_sign_label("systolic_bp") == "Pressão sistólica"
    assert vital_sign_label("diastolic_bp") == "Pressão diastólica"
    assert vital_sign_label("blood_pressure") == "Pressão arterial"
    assert vital_sign_label("temperature") == "Temperatura"


def test_vital_sign_label_is_case_insensitive_and_strips():
    assert vital_sign_label("Heart_Rate") == "Frequência cardíaca"
    assert vital_sign_label("  SPO2 ") == "Saturação de O₂ (SpO₂)"


def test_vital_sign_label_falls_back_to_raw_column_when_unknown():
    assert vital_sign_label("glucose") == "glucose"
    assert vital_sign_label("Custom Column") == "Custom Column"


# ── presentation helpers: confidence_level ──────────────────────────


def test_confidence_level_returns_full_dict_for_known_levels():
    for agreement, expected_icon in (
        ("alta_confianca", "🔴"),
        ("zscore_only", "🟠"),
        ("isolation_forest_only", "🟡"),
    ):
        level = confidence_level(agreement)
        assert set(level) >= {"label", "icon", "short", "help"}
        assert level["icon"] == expected_icon
        assert level["label"]
        assert level["short"]
        assert level["help"]


def test_confidence_level_labels_match_design():
    assert confidence_level("alta_confianca")["label"] == "Alta confiança"
    assert confidence_level("zscore_only")["label"] == "Só tempo real"
    assert confidence_level("isolation_forest_only")["label"] == "Só histórico"


def test_confidence_level_returns_sensible_default_for_normal_or_unknown():
    # Must not KeyError on 'normal' or an unexpected agreement string.
    for agreement in ("normal", "something_new", ""):
        level = confidence_level(agreement)
        assert set(level) >= {"label", "icon", "short", "help"}
        assert level["label"]


# ── presentation helpers: build_vitals_summary ──────────────────────


def _report_row(timestamp, zscore, iforest, agreement, **signals):
    row = {
        "timestamp": timestamp,
        "zscore_anomaly": zscore,
        "isolation_forest_anomaly": iforest,
        "agreement": agreement,
    }
    row.update(signals)
    return row


def _build_combined_report(rows):
    return pd.DataFrame(rows)


def test_build_vitals_summary_empty_when_no_anomalies():
    report = _build_combined_report(
        [
            _report_row("2024-01-01 00:00", False, False, "normal", heart_rate=80.0, spo2=98.0),
            _report_row("2024-01-01 01:00", False, False, "normal", heart_rate=81.0, spo2=97.0),
        ]
    )

    summary = build_vitals_summary(report)

    assert summary["total_anomalias"] == 0
    assert summary["por_nivel"] == {}
    assert summary["itens"] == []


def test_build_vitals_summary_counts_per_level():
    report = _build_combined_report(
        [
            _report_row("t0", True, True, "alta_confianca", heart_rate=200.0, spo2=98.0),
            _report_row("t1", True, False, "zscore_only", heart_rate=150.0, spo2=97.0),
            _report_row("t2", False, True, "isolation_forest_only", heart_rate=82.0, spo2=80.0),
            _report_row("t3", False, False, "normal", heart_rate=80.0, spo2=98.0),
        ]
    )

    summary = build_vitals_summary(report)

    assert summary["total_anomalias"] == 3
    assert summary["por_nivel"] == {
        "alta_confianca": 1,
        "zscore_only": 1,
        "isolation_forest_only": 1,
    }


def test_build_vitals_summary_prioritizes_alta_confianca_in_itens():
    report = _build_combined_report(
        [
            _report_row("t0", True, False, "zscore_only", heart_rate=150.0, spo2=97.0),
            _report_row("t1", True, True, "alta_confianca", heart_rate=200.0, spo2=98.0),
        ]
    )

    summary = build_vitals_summary(report)

    assert summary["itens"][0]["nivel"] == "alta_confianca"


def test_build_vitals_summary_uses_padrao_geral_when_only_isolation_forest_flags():
    report = _build_combined_report(
        [
            _report_row("t0", False, True, "isolation_forest_only", heart_rate=82.0, spo2=80.0),
        ]
    )

    summary = build_vitals_summary(report)

    assert len(summary["itens"]) == 1
    assert summary["itens"][0]["sinal_label"] == "padrão geral"
    assert summary["itens"][0]["nivel"] == "isolation_forest_only"
    # Isolation-Forest-only rows have no single responsible column, so no value.
    assert summary["itens"][0]["valor"] is None


def test_build_vitals_summary_item_has_translated_signal_and_value():
    report = _build_combined_report(
        [
            _report_row("t0", True, True, "alta_confianca", heart_rate=200.0, spo2=98.0),
        ]
    )

    summary = build_vitals_summary(report)
    item = summary["itens"][0]

    assert item["sinal_label"] in ("Frequência cardíaca", "Saturação de O₂ (SpO₂)")
    assert item["valor"] is not None
    assert item["timestamp"] == "t0"


def test_build_vitals_summary_picks_globally_most_extreme_signal_for_zscore_row():
    # Multi-row report so every column has a real (non-zero) distribution and
    # the "most extreme" proxy in _responsible_signal is actually exercised.
    # On the anomalous row heart_rate has a large spike (300 vs a ~80 baseline)
    # while spo2 barely moves, so heart_rate is unambiguously the most extreme
    # signal relative to its own series' mean/std. A regression that picked the
    # wrong column (e.g. first-by-order) would fail the exact-label assertion.
    report = _build_combined_report(
        [
            _report_row("t0", False, False, "normal", heart_rate=80.0, spo2=98.0),
            _report_row("t1", False, False, "normal", heart_rate=81.0, spo2=98.0),
            _report_row("t2", False, False, "normal", heart_rate=79.0, spo2=97.0),
            _report_row("t3", False, False, "normal", heart_rate=80.0, spo2=98.0),
            _report_row("t4", True, True, "alta_confianca", heart_rate=300.0, spo2=98.0),
        ]
    )

    summary = build_vitals_summary(report)
    item = summary["itens"][0]

    assert item["nivel"] == "alta_confianca"
    assert item["timestamp"] == "t4"
    # Exact: heart_rate wins the global z-score proxy by a clear margin.
    assert item["sinal_label"] == "Frequência cardíaca"
    assert item["valor"] == 300.0


def test_build_vitals_summary_caps_itens_list():
    rows = [
        _report_row(f"t{i}", True, True, "alta_confianca", heart_rate=200.0 + i)
        for i in range(20)
    ]
    report = _build_combined_report(rows)

    summary = build_vitals_summary(report)

    assert summary["total_anomalias"] == 20
    assert len(summary["itens"]) <= 8
