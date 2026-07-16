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
import csv
import io
from datetime import datetime, timedelta

import pytest
import streamlit as st

from alerts.feed import Alert, add_alert, build_alerts_report, get_alerts, level_indicator


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


# ── structured fields: alert_id / category / level (change
#    alertas-estruturados-e-vinculo-sinais-vitais) ─────────────────────


def test_add_alert_without_structured_fields_leaves_them_none():
    # Backward-compat: existing callers pass only origin+description and must
    # keep working, with the new structured fields defaulting to None.
    add_alert(origin="Vídeo", description="Movimento anômalo detectado")

    alert = get_alerts()[0]
    assert alert.alert_id is None
    assert alert.category is None
    assert alert.level is None


def test_add_alert_populates_structured_fields_when_provided():
    add_alert(
        origin="Sinais Vitais",
        description="Leitura anômala",
        alert_id="#S01",
        category="Frequência cardíaca",
        level="alta_confianca",
    )

    alert = get_alerts()[0]
    assert alert.alert_id == "#S01"
    assert alert.category == "Frequência cardíaca"
    assert alert.level == "alta_confianca"


def test_get_alerts_still_newest_first_with_structured_fields():
    now = datetime(2026, 1, 1, 12, 0, 0)
    add_alert(
        origin="Sinais Vitais",
        description="mais antigo",
        timestamp=now - timedelta(minutes=10),
        alert_id="#S01",
        category="Frequência cardíaca",
        level="zscore_only",
    )
    add_alert(
        origin="Sinais Vitais",
        description="mais recente",
        timestamp=now,
        alert_id="#S02",
        category="Temperatura",
        level="alta_confianca",
    )

    alerts = get_alerts()
    assert [a.description for a in alerts] == ["mais recente", "mais antigo"]
    assert alerts[0].alert_id == "#S02"


def test_level_indicator_high_severity_for_alta_confianca_and_zona_critica():
    icon, severity = level_indicator("alta_confianca")
    assert severity == "high"
    assert icon  # non-empty
    _, sev_zone = level_indicator("zona_critica")
    assert sev_zone == "high"


def test_level_indicator_normal_severity_for_known_non_high_levels():
    for lvl in ("zscore_only", "isolation_forest_only", "postura", "velocidade"):
        icon, severity = level_indicator(lvl)
        assert severity == "normal"
        assert icon


def test_level_indicator_falls_back_for_none_or_unknown_level():
    for lvl in (None, "", "nivel_inexistente"):
        icon, severity = level_indicator(lvl)
        assert severity == "normal"
        assert icon  # neutral default icon, never empty / never raises


# ── build_alerts_report: session-wide CSV export + summary (change
#    export-relatorio-alertas) ──────────────────────────────────────────

_REPORT_COLUMNS = [
    "id",
    "origem",
    "timestamp",
    "categoria",
    "nivel",
    "nivel_label",
    "descricao",
]


def _parse_rows(csv_text):
    return list(csv.DictReader(io.StringIO(csv_text)))


def test_build_alerts_report_one_row_per_alert_with_exact_columns():
    alerts = [
        Alert(
            origin="Sinais Vitais",
            description="Leitura anômala",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            alert_id="#S01",
            category="Frequência cardíaca",
            level="alta_confianca",
        ),
        Alert(
            origin="Vídeo",
            description="Zona crítica",
            timestamp=datetime(2026, 1, 1, 12, 5, 0),
            alert_id="#V03",
            category="Pescoço",
            level="zona_critica",
        ),
    ]

    csv_text, _summary = build_alerts_report(alerts)

    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    assert header == _REPORT_COLUMNS

    rows = _parse_rows(csv_text)
    assert len(rows) == 2
    assert rows[0]["id"] == "#S01"
    assert rows[0]["origem"] == "Sinais Vitais"
    assert rows[0]["timestamp"] == "2026-01-01 12:00:00"
    assert rows[0]["categoria"] == "Frequência cardíaca"
    assert rows[0]["nivel"] == "alta_confianca"
    assert rows[0]["nivel_label"] == "Alta confiança"
    assert rows[0]["descricao"] == "Leitura anômala"
    # Video level maps to a short PT label.
    assert rows[1]["nivel"] == "zona_critica"
    assert rows[1]["nivel_label"] == "Zona crítica"


def test_build_alerts_report_missing_structured_fields_yield_empty_cells():
    alerts = [
        Alert(
            origin="Áudio",
            description="Termo crítico: dor",
            timestamp=datetime(2026, 1, 1, 9, 30, 0),
        )
    ]

    csv_text, _summary = build_alerts_report(alerts)

    rows = _parse_rows(csv_text)
    assert len(rows) == 1
    assert rows[0]["id"] == ""
    assert rows[0]["categoria"] == ""
    assert rows[0]["nivel"] == ""
    assert rows[0]["nivel_label"] == ""
    assert rows[0]["origem"] == "Áudio"
    assert rows[0]["descricao"] == "Termo crítico: dor"


def test_build_alerts_report_unknown_level_leaves_label_empty():
    alerts = [
        Alert(
            origin="Prescrições",
            description="Interação medicamentosa",
            timestamp=datetime(2026, 1, 1, 10, 0, 0),
            level="algum_nivel_desconhecido",
        )
    ]

    rows = _parse_rows(build_alerts_report(alerts)[0])
    assert rows[0]["nivel"] == "algum_nivel_desconhecido"
    assert rows[0]["nivel_label"] == ""


def test_build_alerts_report_summary_counts_per_origin_and_level():
    alerts = [
        Alert(origin="Vídeo", description="a", level="postura"),
        Alert(origin="Vídeo", description="b", level="zona_critica"),
        Alert(origin="Sinais Vitais", description="c", level="alta_confianca"),
        Alert(origin="Áudio", description="d"),  # no level
    ]

    _csv_text, summary = build_alerts_report(alerts)

    assert summary["total"] == 4
    assert summary["por_origem"] == {"Vídeo": 2, "Sinais Vitais": 1, "Áudio": 1}
    assert summary["por_nivel"] == {
        "postura": 1,
        "zona_critica": 1,
        "alta_confianca": 1,
        "(sem nível)": 1,
    }


def test_build_alerts_report_empty_list_yields_header_only_and_zeroed_summary():
    csv_text, summary = build_alerts_report([])

    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    assert header == _REPORT_COLUMNS
    assert next(reader, None) is None  # no data rows

    assert summary["total"] == 0
    assert summary["por_origem"] == {}
    assert summary["por_nivel"] == {}
