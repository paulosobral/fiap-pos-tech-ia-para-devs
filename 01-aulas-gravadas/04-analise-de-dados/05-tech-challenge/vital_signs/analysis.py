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
import math
import warnings
from typing import Any, Dict, Optional

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
#
# ``anomaly.zscore`` uses population std (ddof=0) over a window that
# includes the point itself, so the maximum achievable |z| for any point
# is ``sqrt(window - 1)`` (limit as a single outlier → ∞). The default
# window must therefore be large enough that this ceiling clears
# DEFAULT_THRESHOLD, otherwise the z-score layer can never fire. With
# threshold 3.0, window=13 gives a ceiling of sqrt(12) ≈ 3.46 (comfortable
# headroom); window=11 → sqrt(10) ≈ 3.16 would be too tight.
DEFAULT_WINDOW = 13
DEFAULT_THRESHOLD = 3.0


def zscore_threshold_is_reachable(window: int, threshold: float) -> bool:
    """Whether a z-score anomaly is mathematically detectable for these params.

    ``anomaly.zscore`` computes each point's z-score against the population
    mean/std (ddof=0) of a rolling window that includes the point itself.
    For a window of ``n`` points, the largest achievable |z| (a single
    extreme outlier, limit as it → ∞) is ``sqrt(n - 1)``. So a reading can
    only be flagged when ``threshold < sqrt(window - 1)``.

    Boundary: with ``window < 2`` the ceiling ``sqrt(window - 1)`` is either
    zero (window == 1) or undefined (window < 1), so no positive threshold is
    ever reachable — treated as not reachable (returns ``False``). Equality
    (``threshold == sqrt(window - 1)``) is also not reachable, since the
    detector requires ``|z| > threshold`` (strictly greater).

    Args:
        window: Rolling window size for the z-score layer.
        threshold: Z-score magnitude above which a reading is anomalous.

    Returns:
        ``True`` if some reading could exceed ``threshold``, else ``False``.
    """
    if window < 2:
        return False
    return threshold < math.sqrt(window - 1)


# ── Presentation vocabulary (friendly labels for the Sinais Vitais tab) ──
#
# Pure, testable presentation helpers, kept next to the layer that presents
# them (same pattern as ``video/analysis.py``'s ``JOINT_LABELS`` /
# ``joint_label`` / ``group_events_for_display``). ``app.py`` only renders
# what these produce — no data logic lives in the tab. Detection is not
# touched; this is presentation only (change ``repaginar-aba-sinais-vitais``).

# Friendly Portuguese labels for the internal vital-sign column keys. Keys
# are the canonical lowercased column names; ``vital_sign_label`` matches
# case-insensitively and falls back to the raw column for anything unknown.
VITAL_SIGN_LABELS: Dict[str, str] = {
    "heart_rate": "Frequência cardíaca",
    "spo2": "Saturação de O₂ (SpO₂)",
    "resp_rate": "Frequência respiratória",
    "respiratory_rate": "Frequência respiratória",
    "systolic_bp": "Pressão sistólica",
    "diastolic_bp": "Pressão diastólica",
    "blood_pressure": "Pressão arterial",
    "temperature": "Temperatura",
}

# Short, friendly descriptions of each vital sign for the upload
# confirmation: what it measures, its unit, and a general adult reference
# range. Same keys/case-insensitive matching style as ``VITAL_SIGN_LABELS``.
# Reference ranges are general adult references, NOT a diagnosis — the UI
# states this next to the descriptions.
VITAL_SIGN_DESCRIPTIONS: Dict[str, str] = {
    "heart_rate": "Batimentos por minuto (bpm). Referência adulto em repouso: ~60–100 bpm.",
    "spo2": "Saturação de oxigênio no sangue (%). Referência: ~95–100%.",
    "resp_rate": "Respirações por minuto (irpm). Referência adulto: ~12–20 irpm.",
    "respiratory_rate": "Respirações por minuto (irpm). Referência adulto: ~12–20 irpm.",
    "systolic_bp": "Pressão arterial sistólica (mmHg). Referência: ~90–120 mmHg.",
    "diastolic_bp": "Pressão arterial diastólica (mmHg). Referência: ~60–80 mmHg.",
    "blood_pressure": (
        "Pressão arterial (mmHg). Referência aproximada: sistólica 90–120 / "
        "diastólica 60–80."
    ),
    "temperature": "Temperatura corporal (°C). Referência: ~36,1–37,2 °C.",
}

# Fallback description for a recognized-but-undescribed / unknown column, so
# ``vital_sign_description`` always returns something the UI can render.
_DEFAULT_VITAL_SIGN_DESCRIPTION = "Sinal vital."

# Signal label used when a row was flagged only by the batch Isolation
# Forest layer, which scores the whole reading (all signals together)
# rather than a single column — there is no one "responsible" signal.
GENERAL_PATTERN_LABEL = "padrão geral"

# Friendly, self-explanatory decoding of each ``agreement`` level from the
# combined report. Each entry is ``{label, icon, short, help}``:
#   - ``label``: short name for the confidence column / legend.
#   - ``icon``: a colored dot conveying severity at a glance.
#   - ``short``: one-line phrase for the compact legend.
#   - ``help``: longer explanation for a tooltip / expander.
CONFIDENCE_LEVELS: Dict[str, Dict[str, str]] = {
    "alta_confianca": {
        "label": "Alta confiança",
        "icon": "🔴",
        "short": "As duas análises concordam — mais provável ser real",
        "help": (
            "Tanto a detecção em tempo real (pico súbito em relação às "
            "leituras recentes) quanto a análise do histórico completo "
            "(fora do padrão geral do paciente) marcaram esta leitura. "
            "Como as duas camadas concordam, é a que tem maior chance de "
            "ser uma anomalia real e merece atenção prioritária."
        ),
    },
    "zscore_only": {
        "label": "Só tempo real",
        "icon": "🟠",
        "short": "Pico isolado momentâneo",
        "help": (
            "Só a detecção em tempo real marcou esta leitura: houve um "
            "pico brusco em relação às leituras imediatamente anteriores, "
            "mas a análise do histórico completo não a considerou fora do "
            "padrão. Costuma indicar uma variação isolada e momentânea."
        ),
    },
    "isolation_forest_only": {
        "label": "Só histórico",
        "icon": "🟡",
        "short": "Fora do padrão geral, sem pico súbito",
        "help": (
            "Só a análise do histórico completo marcou esta leitura: ela "
            "está fora do padrão geral do paciente, mas sem um pico súbito "
            "que a detecção em tempo real captasse. Pode indicar um desvio "
            "sustentado e mais sutil ao longo do tempo."
        ),
    },
}

# Fallback level for ``normal`` or any unexpected agreement string, so
# ``confidence_level`` never raises a KeyError in the UI.
_DEFAULT_CONFIDENCE_LEVEL: Dict[str, str] = {
    "label": "Normal",
    "icon": "🟢",
    "short": "Dentro do padrão esperado",
    "help": "Nenhuma das duas camadas de análise marcou esta leitura como anômala.",
}

# Max number of readings listed in the summary's ``itens`` — the UI shows a
# short, scannable list, not the whole table.
_SUMMARY_MAX_ITEMS = 8

# Priority order used to sort the summary items (most important first).
_LEVEL_PRIORITY = {"alta_confianca": 0, "zscore_only": 1, "isolation_forest_only": 2}


def vital_sign_label(column: str) -> str:
    """Friendly Portuguese label for a vital-sign column name.

    Matches on the lowercased/stripped column name (CSVs may vary in case),
    e.g. ``"Heart_Rate"`` → ``"Frequência cardíaca"``. Falls back to the raw
    column string for any unmapped/unknown column.

    Args:
        column: Raw column name from the uploaded CSV / combined report.

    Returns:
        A human-readable label, or ``column`` unchanged when unknown.
    """
    return VITAL_SIGN_LABELS.get(column.strip().lower(), column)


def vital_sign_description(column: str) -> str:
    """Short friendly description of a vital-sign column (measure/unit/range).

    Matches on the lowercased/stripped column name (case-insensitive, like
    ``vital_sign_label``), e.g. ``"Heart_Rate"`` → the heart-rate description.
    Returns a generic fallback (``"Sinal vital."``) for any unmapped/unknown
    column so the UI always has something to render. Reference ranges are
    general adult references, not a diagnosis.

    Args:
        column: Raw column name from the uploaded CSV / combined report.

    Returns:
        A human-readable description, or a generic fallback when unknown.
    """
    return VITAL_SIGN_DESCRIPTIONS.get(column.strip().lower(), _DEFAULT_VITAL_SIGN_DESCRIPTION)


def confidence_level(agreement: str) -> Dict[str, str]:
    """Presentation dict for an ``agreement`` level from the combined report.

    Args:
        agreement: One of ``"alta_confianca"``, ``"zscore_only"``,
            ``"isolation_forest_only"``, ``"normal"``, or any other string.

    Returns:
        A ``{"label", "icon", "short", "help"}`` dict. For ``"normal"`` or
        any unexpected value, a sensible default is returned (never raises).
    """
    return CONFIDENCE_LEVELS.get(agreement, _DEFAULT_CONFIDENCE_LEVEL)


def _responsible_signal(row: dict, signal_columns: list, column_stats: Dict[str, tuple]):
    """Signal responsible for flagging one anomalous ``combined_report`` row.

    ``analyze`` stores a single row-level ``zscore_anomaly`` boolean (True
    when *any* signal's z-score fired that row), not a per-signal flag, so
    the combined report does not record *which* column tripped the z-score.
    We therefore derive the "responsible" signal from the data present:

    - If the row was flagged by the real-time (z-score) layer, the
      responsible signal is the one whose value is most extreme relative to
      the column's own distribution (largest ``|value - mean| / std`` across
      the whole series). This is the best available proxy for the column the
      rolling z-score reacted to, using only the data in ``combined_report``.
      Ties (or a zero-variance column) fall back to column order.
    - If the row was flagged *only* by the Isolation Forest layer (no
      z-score), the whole reading is out of pattern with no single
      responsible column, so the caller reports the sentinel
      ``GENERAL_PATTERN_LABEL`` ("padrão geral") instead of a specific signal.

    Documented choice: this proxy is GLOBAL (``|value - mean| / std`` over
    the whole series), while the rolling z-score is LOCAL (deviation vs the
    recent window). So the signal it points at may differ from the column
    that actually fired the z-score — not only when several signals spike
    together, but even for a single-signal row, if another signal is
    globally more extreme at that timestamp. It is presentation only and
    never changes detection (``analyze`` is not modified to add per-signal
    columns).

    Args:
        row: One combined-report row as a plain dict.
        signal_columns: Recognized vital-sign columns present in the report.
        column_stats: ``{column: (mean, std)}`` over the whole series.

    Returns:
        A ``(column, value)`` tuple for the z-score case, or ``None`` for the
        Isolation-Forest-only case (no single responsible column).
    """
    if not bool(row.get("zscore_anomaly")):
        return None

    best_col = None
    best_score = -1.0
    best_value = None
    for col in signal_columns:
        value = row.get(col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        mean, std = column_stats.get(col, (0.0, 0.0))
        score = abs(value - mean) / std if std > 0 else 0.0
        if score > best_score:
            best_score = score
            best_col = col
            best_value = value
    if best_col is None:
        return None
    return (best_col, best_value)


def build_vitals_summary(
    combined_report: pd.DataFrame, max_itens: Optional[int] = _SUMMARY_MAX_ITEMS
) -> Dict[str, Any]:
    """Build a friendly, deterministic summary of the anomalous readings.

    Pure presentation helper over the ``combined_report`` returned by
    ``analyze``. Does not touch detection; only reshapes the report into a
    short, human-readable structure for the Sinais Vitais tab.

    Args:
        combined_report: DataFrame with ``timestamp``, ``zscore_anomaly``,
            ``isolation_forest_anomaly``, one column per recognized vital
            sign, and an ``agreement`` column.
        max_itens: Cap on the ``itens`` list length (short list for the UI
            top summary). ``None`` returns every anomalous reading — used to
            drive the full translated table.

    Returns:
        ``{
            "total_anomalias": int,           # rows whose agreement != normal
            "por_nivel": {agreement: count},  # count per non-normal level
            "itens": [                        # alta_confianca first, capped
                {"nivel": str, "sinal_label": str, "valor": float|None,
                 "timestamp": Any}, ...
            ],
        }``
        The "responsible signal" per item is derived by
        ``_responsible_signal`` (documented there): the most-extreme signal
        for z-score rows, or ``"padrão geral"`` when only Isolation Forest
        flagged the row. ``itens`` is capped at ``max_itens``.
    """
    empty = {"total_anomalias": 0, "por_nivel": {}, "itens": []}
    if combined_report is None or combined_report.empty or "agreement" not in combined_report:
        return empty

    anomalies = combined_report[combined_report["agreement"] != "normal"]
    if anomalies.empty:
        return empty

    signal_columns = [
        col
        for col in combined_report.columns
        if col not in ("timestamp", "zscore_anomaly", "isolation_forest_anomaly", "agreement")
        and col.strip().lower() in RECOGNIZED_VITAL_SIGN_COLUMNS
    ]

    # Per-column mean/std over the whole series, so "most extreme" is judged
    # against each signal's own distribution (deterministic for the report).
    column_stats: Dict[str, tuple] = {}
    for col in signal_columns:
        series = pd.to_numeric(combined_report[col], errors="coerce")
        column_stats[col] = (float(series.mean()), float(series.std(ddof=0)))

    por_nivel: Dict[str, int] = {}
    for level in anomalies["agreement"]:
        por_nivel[level] = por_nivel.get(level, 0) + 1

    # Prioritize by confidence level (alta_confianca first), keeping the
    # original row order within each level for determinism.
    ordered = sorted(
        anomalies.to_dict("records"),
        key=lambda r: _LEVEL_PRIORITY.get(r["agreement"], 99),
    )

    selected = ordered if max_itens is None else ordered[:max_itens]
    itens = []
    for row_dict in selected:
        responsible = _responsible_signal(row_dict, signal_columns, column_stats)
        if responsible is not None:
            col, value = responsible
            sinal_label = vital_sign_label(col)
            valor = float(value) if value is not None and not pd.isna(value) else None
        else:
            sinal_label = GENERAL_PATTERN_LABEL
            valor = None
        itens.append(
            {
                "nivel": row_dict["agreement"],
                "sinal_label": sinal_label,
                "valor": valor,
                "timestamp": row_dict["timestamp"],
            }
        )

    return {
        "total_anomalias": int(len(anomalies)),
        "por_nivel": por_nivel,
        "itens": itens,
    }


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
            entry in ``RECOGNIZED_VITAL_SIGN_COLUMNS``, cannot be parsed as
            CSV at all, has an unparseable ``timestamp`` value, or has a
            non-numeric value in a recognized vital-sign column.
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

    for column in recognized:
        numeric_column = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = numeric_column.isna() & df[column].notna()
        if invalid_mask.any():
            bad_row = df.index[invalid_mask][0]
            bad_value = df.loc[bad_row, column]
            raise VitalSignsValidationError(
                f"Valor não numérico encontrado na coluna '{column}' "
                f"(linha {bad_row + 2} do CSV: '{bad_value}')."
            )
        df[column] = numeric_column

    if "timestamp" in [c.strip().lower() for c in df.columns]:
        ts_col = next(c for c in df.columns if c.strip().lower() == "timestamp")
        df = df.rename(columns={ts_col: "timestamp"})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        except (ValueError, TypeError) as exc:
            raise VitalSignsValidationError(
                f"Não foi possível interpretar a coluna 'timestamp' como data/hora: {exc}"
            ) from exc
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
