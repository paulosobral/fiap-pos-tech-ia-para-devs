"""Tests for vital_signs/isolation_forest.py — batch anomaly validation layer.

Covers the vital-signs-monitoring spec requirement:
- Requirement: Validação batch via Isolation Forest
  - Scenario: Isolation Forest identifica anomalia não capturada pelo z-score
"""
import numpy as np
import pandas as pd

from vital_signs.isolation_forest import fit_and_predict


def test_fit_and_predict_returns_boolean_series_aligned_with_input_index():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "heart_rate": rng.normal(80, 2, size=50),
            "spo2": rng.normal(98, 1, size=50),
        }
    )

    result = fit_and_predict(df)

    assert isinstance(result, pd.Series)
    assert result.dtype == bool
    assert result.index.equals(df.index)
    assert len(result) == len(df)


def test_fit_and_predict_flags_obvious_outlier_in_otherwise_tight_cluster():
    rng = np.random.default_rng(7)
    heart_rate = rng.normal(80, 1, size=40).tolist()
    heart_rate[20] = 400.0  # extreme, isolated outlier
    df = pd.DataFrame({"heart_rate": heart_rate})

    result = fit_and_predict(df)

    assert bool(result.iloc[20]) is True
    # Most of the tight cluster should not be flagged.
    assert result.sum() < len(df) / 2


def test_fit_and_predict_handles_single_numeric_column():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"heart_rate": rng.normal(80, 2, size=30)})

    result = fit_and_predict(df)

    assert len(result) == 30
    assert result.dtype == bool


def test_fit_and_predict_ignores_non_numeric_columns():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=25, freq="h"),
            "heart_rate": rng.normal(80, 2, size=25),
        }
    )

    # Should not raise despite the non-numeric timestamp column.
    result = fit_and_predict(df)

    assert len(result) == 25
    assert result.dtype == bool


def test_fit_and_predict_on_small_dataframe_does_not_raise():
    df = pd.DataFrame({"heart_rate": [80.0, 81.0, 79.0]})

    result = fit_and_predict(df)

    assert len(result) == 3
    assert result.dtype == bool
