"""CSV loading/validation and anomaly analysis for the Sinais Vitais tab.

Combines two complementary detection layers, per the vital-signs-monitoring
spec:
- Rolling z-score (``anomaly.zscore.detect_anomalies``), applied row-by-row
  per vital-sign column, simulating a real-time alert.
- Isolation Forest (``vital_signs.isolation_forest.fit_and_predict``),
  fitted in batch over the whole uploaded series.

Every anomalous reading detected by the z-score layer generates an
``Alert`` in the shared feed (``alerts.feed``). The combined report exposes
agreement/disagreement between both layers so the medical team can tell
high-confidence anomalies (both layers agree) from single-layer signals.

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/vital-signs-monitoring/spec.md
"""
from typing import Any, Dict

import pandas as pd

from alerts.feed import add_alert
from anomaly.zscore import detect_anomalies
from vital_signs.isolation_forest import fit_and_predict

# Vital-sign columns recognized by the app. A CSV must contain at least one
# of these (case-insensitive) to be accepted.
RECOGNIZED_VITAL_SIGN_COLUMNS = [
    "heart_rate",
    "spo2",
    "resp_rate",
    "respiratory_rate",
    "systolic_bp",
    "diastolic_bp",
    "blood_pressure",
    "temperature",
]

ORIGIN = "Sinais Vitais"

# Fixed thresholds for the rolling z-score layer (Global Constraints:
# "áudio e sinais vitais use a fixed threshold documented in code").
DEFAULT_WINDOW = 6
DEFAULT_THRESHOLD = 3.0


class VitalSignsValidationError(ValueError):
    """Raised when an uploaded CSV has no recognized vital-sign column."""


def load_vital_signs_csv(file) -> pd.DataFrame:
    """Load and validate a vital-signs CSV.

    Args:
        file: Path, file-like object or buffer accepted by
            ``pandas.read_csv`` (e.g. a Streamlit ``UploadedFile``).

    Returns:
        DataFrame sorted by ``timestamp`` (parsed to ``datetime`` when the
        column is present; otherwise a sequential integer index is used as
        the timestamp column so downstream code can rely on its presence).

    Raises:
        VitalSignsValidationError: If the CSV has no column matching any
            entry in ``RECOGNIZED_VITAL_SIGN_COLUMNS``, or cannot be parsed
            as CSV at all.
    """
    try:
        df = pd.read_csv(file)
    except Exception as exc:  # pragma: no cover - defensive, exercised via error message
        raise VitalSignsValidationError(f"Não foi possível ler o arquivo como CSV: {exc}") from exc

    recognized = [col for col in df.columns if col.strip().lower() in RECOGNIZED_VITAL_SIGN_COLUMNS]
    if not recognized:
        raise VitalSignsValidationError(
            "Nenhuma coluna de sinal vital reconhecida foi encontrada no CSV. "
            f"Colunas esperadas (ao menos uma): {', '.join(RECOGNIZED_VITAL_SIGN_COLUMNS)}."
        )

    if "timestamp" in [c.strip().lower() for c in df.columns]:
        ts_col = next(c for c in df.columns if c.strip().lower() == "timestamp")
        df = df.rename(columns={ts_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df = df.reset_index(drop=True)
        df.insert(0, "timestamp", df.index)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _vital_sign_columns(df: pd.DataFrame) -> list:
    return [col for col in df.columns if col.strip().lower() in RECOGNIZED_VITAL_SIGN_COLUMNS]


def analyze(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Run both anomaly-detection layers over ``df`` and combine results.

    Args:
        df: Vital-signs DataFrame as produced by ``load_vital_signs_csv``
            (must contain a ``timestamp`` column and at least one
            recognized vital-sign column).
        window: Rolling window size for the z-score layer.
        threshold: Z-score magnitude above which a reading is anomalous.

    Returns:
        Dict with:
            - ``zscore_anomalies``: DataFrame of booleans, one column per
              vital sign, aligned with ``df.index``.
            - ``isolation_forest_anomalies``: boolean Series aligned with
              ``df.index``, from the batch Isolation Forest layer.
            - ``combined_report``: DataFrame indexed like ``df`` with
              ``timestamp``, one boolean column per signal's z-score flag
              collapsed into ``zscore_anomaly`` (True if any signal is
              anomalous in that row), ``isolation_forest_anomaly`` and an
              ``agreement`` column (one of ``"alta_confianca"``,
              ``"zscore_only"``, ``"isolation_forest_only"``, ``"normal"``).
            - ``alerts``: list of ``Alert`` objects generated for every
              z-score anomaly (also pushed to the shared feed).

    Raises:
        VitalSignsValidationError: If ``df`` has no recognized vital-sign
            column.
    """
    signal_columns = _vital_sign_columns(df)
    if not signal_columns:
        raise VitalSignsValidationError(
            "Nenhuma coluna de sinal vital reconhecida foi encontrada no CSV. "
            f"Colunas esperadas (ao menos uma): {', '.join(RECOGNIZED_VITAL_SIGN_COLUMNS)}."
        )

    zscore_anomalies = pd.DataFrame(index=df.index)
    alerts = []

    for column in signal_columns:
        flags = detect_anomalies(df[column], window=window, threshold=threshold)
        zscore_anomalies[column] = flags

        for row_index in flags[flags].index:
            timestamp = df.loc[row_index, "timestamp"]
            value = df.loc[row_index, column]
            alert = add_alert(
                origin=ORIGIN,
                description=(
                    f"Leitura anômala de {column} = {value} em {timestamp} "
                    f"(|z-score| > {threshold})."
                ),
                timestamp=timestamp if isinstance(timestamp, pd.Timestamp) else None,
            )
            alerts.append(alert)

    isolation_forest_anomalies = fit_and_predict(df[signal_columns])

    zscore_any = zscore_anomalies.any(axis=1)
    combined_report = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "zscore_anomaly": zscore_any,
            "isolation_forest_anomaly": isolation_forest_anomalies,
        }
    )
    for column in signal_columns:
        combined_report[column] = df[column]

    def _agreement(row) -> str:
        if row["zscore_anomaly"] and row["isolation_forest_anomaly"]:
            return "alta_confianca"
        if row["zscore_anomaly"]:
            return "zscore_only"
        if row["isolation_forest_anomaly"]:
            return "isolation_forest_only"
        return "normal"

    combined_report["agreement"] = combined_report.apply(_agreement, axis=1)

    return {
        "zscore_anomalies": zscore_anomalies,
        "isolation_forest_anomalies": isolation_forest_anomalies,
        "combined_report": combined_report,
        "alerts": alerts,
    }
