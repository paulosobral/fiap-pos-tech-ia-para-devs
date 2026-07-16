"""Shared alert feed consumed by the Vídeo, Áudio, Sinais Vitais and
Prescrições tabs, simulating real-time notification to the medical team.

Alerts live in ``st.session_state`` so they persist across Streamlit
reruns within the same browser session and are visible to every tab.

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/clinical-alerting/spec.md
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

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
