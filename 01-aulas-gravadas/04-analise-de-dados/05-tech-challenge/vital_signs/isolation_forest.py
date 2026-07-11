"""Batch anomaly validation layer for vital signs using Isolation Forest.

Complements the row-by-row rolling z-score ("real time") detector in
``vital_signs/analysis.py`` with a batch model trained over the whole
uploaded series, per the vital-signs-monitoring spec's "Validação batch
via Isolation Forest" requirement.

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/vital-signs-monitoring/spec.md
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

# Fraction of points expected to be anomalous a priori. Fixed/documented
# per the Global Constraints (áudio e sinais vitais use a fixed threshold
# documented in code).
CONTAMINATION = 0.05
RANDOM_STATE = 42


def fit_and_predict(df: pd.DataFrame) -> "pd.Series[bool]":
    """Fit an Isolation Forest over the numeric vital-sign columns of ``df``
    and return a boolean Series flagging anomalous rows.

    Non-numeric columns (e.g. a ``timestamp`` column) are ignored. Rows
    with any missing numeric value are excluded from the model fit and
    reported as not anomalous (there isn't enough data to judge them).

    Args:
        df: DataFrame with one or more numeric vital-sign columns (and,
            optionally, a non-numeric timestamp column).

    Returns:
        Boolean Series aligned with ``df.index``, ``True`` where the row
        is flagged as anomalous by the model.
    """
    numeric_df = df.select_dtypes(include="number")

    result = pd.Series(False, index=df.index, dtype=bool)

    usable_rows = numeric_df.dropna()
    if numeric_df.shape[1] == 0 or len(usable_rows) < 2:
        # Not enough signal to fit a model — nothing to flag.
        return result

    model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    predictions = model.fit_predict(usable_rows)  # -1 = anomaly, 1 = normal

    is_anomaly = pd.Series(predictions == -1, index=usable_rows.index)
    result.loc[is_anomaly.index] = is_anomaly

    return result
