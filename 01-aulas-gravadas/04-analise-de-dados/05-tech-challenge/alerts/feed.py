"""Shared alert feed consumed by the Vídeo, Áudio, Sinais Vitais and
Prescrições tabs, simulating real-time notification to the medical team.

Alerts live in ``st.session_state`` so they persist across Streamlit
reruns within the same browser session and are visible to every tab.

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/clinical-alerting/spec.md
"""
import csv
import io
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import streamlit as st

_SESSION_KEY = "alerts"


@dataclass
class Alert:
    """A single alert raised by any capability tab.

    Attributes:
        origin: Tab/capability that generated the alert (e.g. "Vídeo",
            "Áudio", "Sinais Vitais", "Prescrições").
        description: Human-readable description of the anomaly/issue.
        timestamp: When the alert was generated.
        alert_id: Optional short identifier that links this alert to the
            row/image/finding it refers to in the originating tab (e.g.
            ``"#S01"`` in Sinais Vitais, ``"#V03"`` in Vídeo). ``None``
            for tabs/alerts that have no natural id.
        category: Optional structured category for the alert (e.g. the
            vital-sign name, the video body-region, "Termo crítico"),
            kept as a first-class field so a later export can read clean
            columns instead of parsing the description text.
        level: Optional structured severity/confidence level (e.g. the
            vital-signs ``agreement`` value, or the video event ``tipo``),
            used by the feed to show a visual indicator when present.
    """

    origin: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    alert_id: Optional[str] = None
    category: Optional[str] = None
    level: Optional[str] = None


def add_alert(
    origin: str,
    description: str,
    timestamp: Optional[datetime] = None,
    alert_id: Optional[str] = None,
    category: Optional[str] = None,
    level: Optional[str] = None,
) -> Alert:
    """Create an ``Alert`` and append it to the shared feed in ``st.session_state``.

    Args:
        origin: Tab/capability that generated the alert.
        description: Human-readable description of the anomaly/issue.
        timestamp: Optional explicit timestamp; defaults to ``datetime.now()``.
        alert_id: Optional short identifier linking the alert to its
            source row/image/finding (e.g. ``"#S01"``/``"#V03"``).
        category: Optional structured category (e.g. vital-sign name).
        level: Optional structured severity/confidence level.

    Returns:
        The ``Alert`` instance that was added to the feed.
    """
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = []

    alert = Alert(
        origin=origin,
        description=description,
        timestamp=timestamp or datetime.now(),
        alert_id=alert_id,
        category=category,
        level=level,
    )
    st.session_state[_SESSION_KEY].append(alert)
    return alert


def get_alerts() -> List[Alert]:
    """Return all alerts in the shared feed, ordered newest-first.

    Returns:
        List of ``Alert`` objects. Empty list when no alert has been
        generated in the current session.
    """
    alerts = st.session_state.get(_SESSION_KEY, [])
    return sorted(alerts, key=lambda alert: alert.timestamp, reverse=True)


# Visual severity indicator per structured ``level``, used by the unified
# sidebar feed (clinical-alerting spec) to show a per-level indicator when
# an alert carries a level. Different tabs use different level vocabularies
# (Sinais Vitais: agreement between the two layers; Vídeo/Prescrições: the
# event/finding kind), so this maps the known level strings across all tabs
# to (icon, "high"|"normal") — "high" picks a stronger widget (st.error) in
# the feed. Unknown/absent levels fall back to a neutral indicator so tabs
# that set no level still render fine.
_LEVEL_INDICATORS = {
    # Sinais Vitais — confidence between the real-time and history layers.
    "alta_confianca": ("🔴", "high"),
    "zscore_only": ("🟠", "normal"),
    "isolation_forest_only": ("🟡", "normal"),
    # Vídeo — event kinds.
    "postura": ("🟠", "normal"),
    "velocidade": ("🟠", "normal"),
    "zona_critica": ("🔴", "high"),
}
_DEFAULT_LEVEL_INDICATOR = ("🔔", "normal")


def level_indicator(level: Optional[str]) -> tuple:
    """Return ``(icon, severity)`` for an alert ``level`` for the feed.

    ``severity`` is ``"high"`` or ``"normal"``; the feed uses it to choose a
    stronger widget for high-severity alerts. Unknown or ``None`` levels
    return a neutral default (never raises), so alerts without a structured
    level still render.
    """
    if not level:
        return _DEFAULT_LEVEL_INDICATOR
    return _LEVEL_INDICATORS.get(level, _DEFAULT_LEVEL_INDICATOR)


# Fixed column order for the exported session report (change
# export-relatorio-alertas). Kept as a module constant so the tests and the
# UI share one source of truth.
_REPORT_COLUMNS = [
    "id",
    "origem",
    "timestamp",
    "categoria",
    "nivel",
    "nivel_label",
    "descricao",
]

# Short PT labels for the Vídeo event levels (their level vocabulary is the
# event ``tipo``). The Sinais Vitais levels reuse CONFIDENCE_LEVELS[...]["label"].
# Any level not covered here nor in CONFIDENCE_LEVELS exports an empty label.
_VIDEO_LEVEL_LABELS = {
    "postura": "Postura",
    "velocidade": "Velocidade",
    "zona_critica": "Zona crítica",
}

# Key used in the summary's per-level breakdown for alerts that carry no level.
_NO_LEVEL_KEY = "(sem nível)"


def _level_label(level: Optional[str]) -> str:
    """Friendly label for a raw ``level`` value, or ``""`` when unknown.

    Reuses the Sinais Vitais confidence vocabulary (CONFIDENCE_LEVELS) and the
    small Vídeo event-label map. Never raises.
    """
    if not level:
        return ""
    # Imported lazily: vital_signs.analysis imports add_alert from this
    # module, so a top-level import here would be a circular import.
    from vital_signs.analysis import CONFIDENCE_LEVELS

    if level in CONFIDENCE_LEVELS:
        return CONFIDENCE_LEVELS[level]["label"]
    return _VIDEO_LEVEL_LABELS.get(level, "")


def build_alerts_report(alerts: List[Alert]) -> Tuple[str, dict]:
    """Build a CSV report + summary from a list of ``Alert`` (pure function).

    Takes the same ``Alert`` objects ``get_alerts()`` returns (does NOT read
    ``session_state``) so it is directly unit-testable.

    Returns ``(csv_text, summary)`` where:
      - ``csv_text`` is a CSV string built with the stdlib ``csv`` module: a
        header row plus one row per alert. Columns are exactly
        ``id, origem, timestamp, categoria, nivel, nivel_label, descricao``.
        Missing ``alert_id``/``category``/``level`` become empty cells;
        ``timestamp`` is ISO-formatted ``%Y-%m-%d %H:%M:%S``; ``nivel`` is the
        raw ``alert.level`` value; ``nivel_label`` is a friendly label when the
        level is known, else empty. The function returns ``str``; the UI is
        responsible for encoding it as UTF-8-SIG so Excel shows accents.
      - ``summary`` is
        ``{"total": int, "por_origem": {origin: count}, "por_nivel": {level_or_"(sem nível)": count}}``,
        deterministic (insertion-ordered by first appearance).
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_REPORT_COLUMNS)

    por_origem: "OrderedDict[str, int]" = OrderedDict()
    por_nivel: "OrderedDict[str, int]" = OrderedDict()

    for alert in alerts:
        level = alert.level or ""
        writer.writerow(
            [
                alert.alert_id or "",
                alert.origin,
                alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                alert.category or "",
                level,
                _level_label(alert.level),
                alert.description,
            ]
        )

        por_origem[alert.origin] = por_origem.get(alert.origin, 0) + 1
        level_key = alert.level if alert.level else _NO_LEVEL_KEY
        por_nivel[level_key] = por_nivel.get(level_key, 0) + 1

    summary = {
        "total": len(alerts),
        "por_origem": dict(por_origem),
        "por_nivel": dict(por_nivel),
    }
    return buffer.getvalue(), summary
