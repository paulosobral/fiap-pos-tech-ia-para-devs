"""Tests for anomaly/zscore.py — shared rolling z-score anomaly detector.

Covers the two scenarios described in the clinical-alerting spec:
- Requirement: Função genérica de detecção de anomalia por rolling z-score
  - Scenario: Série com ponto fora do threshold
  - Scenario: Série menor que a janela configurada
"""
import numpy as np
import pandas as pd
import pytest

from anomaly.zscore import detect_anomalies


def test_point_beyond_threshold_is_flagged_as_anomalous():
    # Stable baseline (value 1) with one clear spike (value 100). With an
    # inclusive rolling window of size 6, a lone spike among 5 baseline
    # points yields z = sqrt(window - 1) = sqrt(5) ~= 2.236 for the spike's
    # own row, which exceeds threshold=2 — while rows that merely have the
    # spike inside their window (but whose own value is still 1) stay well
    # under threshold (|z| ~= 0.447).
    series = pd.Series([1] * 7 + [100] + [1] * 4, dtype=float)

    result = detect_anomalies(series, window=6, threshold=2)

    assert isinstance(result, pd.Series)
    assert result.dtype == bool
    assert result.index.equals(series.index)
    assert bool(result[7]) is True  # the spike itself
    # Baseline points, including ones whose window contains the spike.
    assert bool(result[6]) is False
    assert bool(result[8]) is False
    assert bool(result[9]) is False


def test_constant_series_has_no_anomalies():
    # Zero variance series: std is 0 everywhere, must not raise (division by zero)
    # and must not flag any point as anomalous.
    series = pd.Series([5.0] * 10)

    result = detect_anomalies(series, window=3, threshold=2)

    assert not result.any()
    assert len(result) == len(series)


def test_series_shorter_than_window_returns_all_false_without_raising():
    series = pd.Series([1.0, 2.0, 3.0])

    result = detect_anomalies(series, window=10, threshold=2)

    assert len(result) == len(series)
    assert result.dtype == bool
    assert not result.any()


def test_empty_series_returns_empty_boolean_series_without_raising():
    series = pd.Series([], dtype=float)

    result = detect_anomalies(series, window=5, threshold=2)

    assert len(result) == 0
    assert result.dtype == bool


def test_result_has_no_nan_values():
    series = pd.Series(np.random.default_rng(42).normal(size=20))

    result = detect_anomalies(series, window=5, threshold=2)

    assert result.isna().sum() == 0
