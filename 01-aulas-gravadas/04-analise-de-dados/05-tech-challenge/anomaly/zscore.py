"""Shared rolling z-score anomaly detector.

Used by the Vídeo, Áudio and Sinais Vitais capabilities to flag points in a
numeric time series that deviate from their recent local behaviour.

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/clinical-alerting/spec.md
"""
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Raw rolling z-score per point, ``NaN`` where undefined.

    The z-score of each point is computed against the mean and standard
    deviation of its own trailing window (``window`` points, inclusive of
    the point itself). ``NaN`` where there isn't enough history yet
    (fewer than ``window`` points seen), or where the window's standard
    deviation is exactly zero (a constant window — division by zero).

    Shared by ``detect_anomalies`` (thresholds this into a boolean flag)
    and by callers that need the raw magnitude instead, e.g.
    ``video.analysis.suggest_sensitivity_threshold``, which derives a
    sensitivity suggestion from the distribution of these values rather
    than from a fixed threshold.

    Args:
        series: Numeric time series to analyze.
        window: Size of the moving window used to compute the rolling
            mean/standard deviation.

    Returns:
        Series aligned with ``series.index``. Empty (matching
        ``series``'s length) when ``series`` has fewer points than
        ``window``.
    """
    if len(series) < window:
        return pd.Series(float("nan"), index=series.index)

    rolling = series.rolling(window=window, min_periods=window)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std(ddof=0)

    return (series - rolling_mean) / rolling_std


def detect_anomalies(series: pd.Series, window: int, threshold: float) -> "pd.Series[bool]":
    """Flag points in ``series`` whose rolling z-score exceeds ``threshold``.

    A point is anomalous when ``abs(z) > threshold``, where ``z`` is the
    point's rolling z-score (see ``rolling_zscore``).

    If ``series`` has fewer points than ``window`` (or is empty), every
    point is reported as not anomalous — no exception is raised.

    Args:
        series: Numeric time series to analyze.
        window: Size of the moving window used to compute the rolling
            mean/standard deviation.
        threshold: Z-score magnitude above which a point is considered
            anomalous.

    Returns:
        Boolean Series aligned with ``series.index``, ``True`` where the
        point is anomalous.
    """
    if len(series) < window:
        return pd.Series(False, index=series.index, dtype=bool)

    z_score = rolling_zscore(series, window)

    anomalies = z_score.abs() > threshold
    # Points without enough history (NaN z-score, e.g. std == 0 or the
    # first window-1 points) are never anomalous.
    anomalies = anomalies.fillna(False)

    return anomalies.astype(bool)
