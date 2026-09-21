from datetime import datetime, timezone

from service import kpi_calc

NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)


def iso(day: int, hour: int = 10) -> str:
    return f"2026-09-{day:02d}T{hour:02d}:00:00+00:00"


def profile(created_at=None, status="new", intent=None, area=None, route=None):
    item = {"lead_id": "l1", "telegram_user_id": 42, "created_at": created_at}
    if status:
        item["status"] = status
    if intent:
        item["intent"] = intent
    if area:
        item["area"] = area
    if route:
        item["route"] = route
    return item


class TestParseIso8601:
    def test_valid_zulu_and_offset(self):
        parsed = kpi_calc.parse_iso8601("2026-09-20T12:00:00Z")
        assert parsed == NOW
        assert kpi_calc.parse_iso8601("2026-09-20T09:00:00-03:00") == NOW

    def test_invalid_returns_none(self):
        assert kpi_calc.parse_iso8601("not-a-date") is None
        assert kpi_calc.parse_iso8601(None) is None
        assert kpi_calc.parse_iso8601("") is None
        assert kpi_calc.parse_iso8601(42) is None

    def test_naive_assumed_utc(self):
        parsed = kpi_calc.parse_iso8601("2026-09-20T12:00:00")
        assert parsed is not None and parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0


class TestLeadCounts:
    def test_counts_today_and_week_skipping_invalid(self):
        profiles = {
            "a": profile(created_at=iso(20, 9)),
            "b": profile(created_at=iso(17, 9)),
            "c": profile(created_at=iso(10, 9)),
            "d": profile(created_at="inválido"),
            "e": profile(created_at=None),
        }
        assert kpi_calc.count_new_leads(profiles, kpi_calc.start_of_today(NOW), NOW) == 1
        assert kpi_calc.count_new_leads(profiles, kpi_calc.days_ago(NOW, 7), NOW) == 2

    def test_empty_profiles_are_zero(self):
        assert kpi_calc.count_new_leads({}, kpi_calc.start_of_today(NOW), NOW) == 0

    def test_future_created_at_not_counted(self):
        profiles = {"a": profile(created_at="2026-09-21T00:00:00+00:00")}
        assert kpi_calc.count_new_leads(profiles, kpi_calc.days_ago(NOW, 7), NOW) == 0


class TestQualificationAndFunnel:
    def _profiles(self):
        return {"a": profile(), "b": profile(), "c": profile(), "d": profile()}

    def test_qualification_sources(self):
        latest = {
            "a": {"current_state": "scheduling", "context": {}},
            "b": {"current_state": "qualification", "context": {"lead_qualified": True}},
            "c": {"current_state": "greeting", "context": {}},
            "d": {"current_state": "greeting", "context": {}},
        }
        assert kpi_calc.qualification_rate(self._profiles(), latest) == 0.5

    def test_qualified_status_on_profile(self):
        latest = {"a": {"current_state": "greeting", "context": {}}}
        profiles = {"a": profile(status="qualified")}
        assert kpi_calc.qualification_rate(profiles, latest) == 1.0

    def test_zero_when_no_leads(self):
        assert kpi_calc.qualification_rate({}, {}) == 0.0

    def test_state_funnel_includes_all_states_and_unknown(self):
        latest = {
            "a": {"current_state": "greeting"},
            "b": {"current_state": "scheduling"},
            "c": {"current_state": "estado_exótico"},
        }
        funnel = kpi_calc.state_funnel(latest)
        assert funnel["greeting"] == 1
        assert funnel["scheduling"] == 1
        assert funnel["outros"] == 1
        assert set(funnel) == set(kpi_calc.PIPELINE_STATES) | {"outros"}

    def test_scheduled_visits_counts_scheduling_and_handoff(self):
        latest = {
            "a": {"current_state": "scheduling"},
            "b": {"current_state": "handoff"},
            "c": {"current_state": "recommendation"},
        }
        assert kpi_calc.scheduled_visits_count(latest) == 2

    def test_latest_conversation_picks_most_recent_per_lead(self):
        conversations = [
            {"session_id": "s1", "lead_id": "l1", "created_at": iso(10), "current_state": "greeting"},
            {"session_id": "s2", "lead_id": "l1", "created_at": iso(18), "current_state": "scheduling"},
            {"session_id": "s3", "lead_id": "l2", "created_at": iso(19), "current_state": "intent"},
            {"session_id": "s4", "lead_id": None, "created_at": iso(19)},
        ]
        latest = kpi_calc.latest_conversation_by_lead(conversations)
        assert latest["l1"]["session_id"] == "s2"
        assert latest["l2"]["session_id"] == "s3"
        assert len(latest) == 2


class TestIntentAndRoute:
    def test_intent_volume_skips_missing(self):
        profiles = {
            "a": profile(intent="compra"),
            "b": profile(intent="compra"),
            "c": profile(intent="locação"),
            "d": profile(status="new"),
        }
        assert kpi_calc.intent_volume(profiles) == {"compra": 2, "locação": 1}

    def test_route_rules_follow_u1_roleta(self):
        profiles = {
            "small": profile(area="400 m²"),
            "boundary": profile(area="500 m²"),
            "big": profile(area="800 m²"),
            "override": profile(area="100 m²", route="diretor"),
            "thousands": profile(area="1.200 m²"),
        }
        distribution = kpi_calc.route_distribution(profiles)
        assert distribution == {"consultores": 2, "diretor": 3}

    def test_missing_area_rotates_to_consultores(self):
        assert kpi_calc.route_for_profile(profile(area=None)) == "consultores"
        assert kpi_calc.parse_area_m2("sem número") is None

    def test_parse_area_m2_thousands_separator(self):
        assert kpi_calc.parse_area_m2("1.200 m²") == 1200.0


class TestAlertsWindow:
    def test_alerts_last_24h_window_and_sorting(self):
        alerts = [
            {"anomaly_id": "old", "detected_at": "2026-09-18T00:00:00Z"},
            {"anomaly_id": "mid", "detected_at": "2026-09-19T20:00:00Z"},
            {"anomaly_id": "fresh", "detected_at": "2026-09-20T11:00:00Z"},
            {"anomaly_id": "broken", "detected_at": "ontem"},
            {"anomaly_id": "future", "detected_at": "2026-09-20T13:00:00Z"},
        ]
        window = kpi_calc.alerts_last_24h(alerts, NOW)
        assert [alert["anomaly_id"] for alert in window] == ["fresh", "mid"]
