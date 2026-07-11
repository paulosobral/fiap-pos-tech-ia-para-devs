"""Tests for alerts/feed.py — shared alert feed over st.session_state.

Covers scenarios from the clinical-alerting spec:
- Requirement: Feed unificado de alertas para a equipe médica
  - Scenario: Alerta gerado em qualquer aba aparece no feed
  - Scenario: Feed exibido em ordem cronológica
  - Scenario: Nenhum alerta gerado na sessão

Streamlit's `st.session_state` works as a plain dict-like object even
outside a running `streamlit run` script (verified empirically: only
emits bare-mode warnings, does not raise), so tests exercise the real
`streamlit.session_state` object directly instead of a mock/wrapper.
"""
from datetime import datetime, timedelta

import pytest
import streamlit as st

from alerts.feed import Alert, add_alert, get_alerts


@pytest.fixture(autouse=True)
def clean_session_state():
    """Ensure each test starts and ends with an empty session_state."""
    st.session_state.clear()
    yield
    st.session_state.clear()


def test_get_alerts_returns_empty_list_when_no_alert_was_generated():
    assert get_alerts() == []


def test_add_alert_appends_alert_with_origin_timestamp_and_description():
    add_alert(origin="Vídeo", description="Movimento anômalo detectado")

    alerts = get_alerts()

    assert len(alerts) == 1
    alert = alerts[0]
    assert isinstance(alert, Alert)
    assert alert.origin == "Vídeo"
    assert alert.description == "Movimento anômalo detectado"
    assert isinstance(alert.timestamp, datetime)


def test_add_alert_accepts_explicit_timestamp():
    ts = datetime(2026, 1, 1, 12, 0, 0)

    add_alert(origin="Áudio", description="Termo crítico: dor", timestamp=ts)

    alerts = get_alerts()
    assert alerts[0].timestamp == ts


def test_alerts_from_any_tab_are_added_to_the_shared_feed():
    add_alert(origin="Vídeo", description="Alerta de vídeo")
    add_alert(origin="Áudio", description="Alerta de áudio")
    add_alert(origin="Sinais Vitais", description="Alerta de sinais vitais")
    add_alert(origin="Prescrições", description="Alerta de prescrição")

    origins = {alert.origin for alert in get_alerts()}

    assert origins == {"Vídeo", "Áudio", "Sinais Vitais", "Prescrições"}


def test_get_alerts_returns_newest_first():
    now = datetime(2026, 1, 1, 12, 0, 0)
    add_alert(origin="Vídeo", description="mais antigo", timestamp=now - timedelta(minutes=10))
    add_alert(origin="Áudio", description="meio", timestamp=now - timedelta(minutes=5))
    add_alert(origin="Sinais Vitais", description="mais recente", timestamp=now)

    alerts = get_alerts()

    assert [a.description for a in alerts] == ["mais recente", "meio", "mais antigo"]


def test_get_alerts_orders_newest_first_even_when_added_out_of_order():
    now = datetime(2026, 1, 1, 12, 0, 0)
    add_alert(origin="Vídeo", description="meio", timestamp=now - timedelta(minutes=5))
    add_alert(origin="Áudio", description="mais antigo", timestamp=now - timedelta(minutes=10))
    add_alert(origin="Sinais Vitais", description="mais recente", timestamp=now)

    alerts = get_alerts()

    assert [a.description for a in alerts] == ["mais recente", "meio", "mais antigo"]


def test_get_alerts_does_not_mutate_underlying_session_state_order():
    # Guards against get_alerts() accidentally sorting session_state in place
    # in a way that breaks add_alert's own ordering assumptions.
    add_alert(origin="Vídeo", description="first")
    add_alert(origin="Áudio", description="second")

    get_alerts()
    get_alerts()

    assert len(st.session_state["alerts"]) == 2
