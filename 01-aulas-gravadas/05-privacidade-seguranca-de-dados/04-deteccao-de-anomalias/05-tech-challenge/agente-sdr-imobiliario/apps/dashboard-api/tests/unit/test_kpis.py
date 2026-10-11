import pytest

import service.kpis as kpis_module
from infra.alert_store import AlertStoreError
from infra.conversation_store import ConversationStoreError
from service.kpis import KpiService

NOW_ISO = "2026-09-20T12:00:00+00:00"
NOW_PARTS = {"day": 20, "hour": 10}


def iso(day: int, hour: int = 10) -> str:
    return f"2026-09-{day:02d}T{hour:02d}:00:00+00:00"


def fake_now():
    from datetime import datetime, timezone

    return datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)


class FakeConversations:
    def __init__(self, conversations=None, profiles=None, error=None):
        self.conversations = conversations or []
        self.profiles = profiles or {}
        self.error = error

    def list_conversations(self):
        if self.error:
            raise self.error
        return self.conversations

    def get_lead_profiles(self):
        if self.error:
            raise self.error
        return self.profiles


class FakeAlerts:
    def __init__(self, alerts=None, error=None):
        self.alerts = alerts or []
        self.error = error

    def list_alerts(self):
        if self.error:
            raise self.error
        return self.alerts


class FakeMeter:
    def __init__(self, data=None, error=None):
        self.data = data if data is not None else {"response_time_p90": 0.8, "cost_monthly": 15.0}
        self.error = error

    def read(self):
        if self.error:
            raise self.error
        return self.data


def make_service(conversations=None, alerts=None, meter=None):
    return KpiService(
        conversations=conversations or FakeConversations(),
        alerts=alerts or FakeAlerts(),
        meter=meter or FakeMeter(),
        now_fn=fake_now,
    )


def profile(lead_id="l1", created_at=NOW_ISO, **extra):
    base = {"lead_id": lead_id, "created_at": created_at, "status": "new"}
    base.update(extra)
    return base


class TestSnapshot:
    def test_happy_snapshot_contract_fields(self):
        conversations = [
            {"session_id": "s1", "lead_id": "l1", "created_at": iso(20), "current_state": "scheduling", "context": {}},
            {"session_id": "s2", "lead_id": "l2", "created_at": iso(15), "current_state": "recommendation", "context": {}},
        ]
        profiles = {
            "l1": profile(created_at=iso(20), intent="compra", area="800 m²"),
            "l2": profile(created_at=iso(15), intent="locação", area="300 m²"),
        }
        alerts = [
            {"anomaly_id": "a1", "lead_id": "l1", "detected_at": iso(20, 11), "confidence": 0.9,
             "type": "volume", "status": "open", "action_taken": "alert_issued", "features": {"x": 1}},
        ]
        snapshot = make_service(
            FakeConversations(conversations, profiles), FakeAlerts(alerts)
        ).snapshot()
        assert snapshot["leads_today"] == 1
        assert snapshot["leads_week"] == 2
        assert snapshot["response_time_p90"] == 0.8
        assert snapshot["qualification_rate"] == 1.0
        assert snapshot["scheduled_visits"] == 1
        assert snapshot["anomalies_count"] == 1
        assert snapshot["cost_monthly"] == 15.0
        assert snapshot["generated_at"] == NOW_ISO
        assert snapshot["intents"] == {"compra": 1, "locação": 1}
        assert snapshot["route_distribution"] == {"diretor": 1, "consultores": 1}
        assert snapshot["funnel"]["scheduling"] == 1
        assert snapshot["alerts"][0]["anomaly_id"] == "a1"
        assert "features" not in snapshot["alerts"][0]

    def test_empty_sources_yield_zeros_and_empty_state(self):
        snapshot = make_service().snapshot()
        assert snapshot["leads_today"] == 0
        assert snapshot["leads_week"] == 0
        assert snapshot["qualification_rate"] == 0.0
        assert snapshot["scheduled_visits"] == 0
        assert snapshot["anomalies_count"] == 0
        assert snapshot["alerts"] == []
        assert snapshot["intents"] == {}
        assert snapshot["route_distribution"] == {}
        assert set(snapshot["funnel"].values()) == {0}

    def test_store_failure_propagates_and_logs(self, caplog):
        service = make_service(FakeConversations(error=ConversationStoreError("down")))
        with pytest.raises(ConversationStoreError):
            service.snapshot()
        assert "conversation_read_failed" in caplog.text

    def test_alerts_failure_propagates_and_logs(self, caplog):
        service = make_service(alerts=FakeAlerts(error=AlertStoreError("down")))
        with pytest.raises(AlertStoreError):
            service.snapshot()
        assert "alerts_read_failed" in caplog.text

    def test_meter_none_values_become_zeros(self):
        snapshot = make_service(meter=FakeMeter(data={"response_time_p90": None, "cost_monthly": None})).snapshot()
        assert snapshot["response_time_p90"] == 0.0
        assert snapshot["cost_monthly"] == 0.0

    def test_meter_failure_and_invalid_payload_degrade(self):
        broken = make_service(meter=FakeMeter(error=RuntimeError("cw down"))).snapshot()
        assert broken["response_time_p90"] == 0.0
        invalid = make_service(meter=FakeMeter(data="nope")).snapshot()
        assert invalid["cost_monthly"] == 0.0

    def test_anomalies_count_uses_24h_window(self):
        alerts = [
            {"anomaly_id": "fresh", "detected_at": iso(20, 11)},
            {"anomaly_id": "old", "detected_at": iso(15, 11)},
        ]
        snapshot = make_service(alerts=FakeAlerts(alerts)).snapshot()
        assert snapshot["anomalies_count"] == 1
        assert [alert["anomaly_id"] for alert in snapshot["alerts"]] == ["fresh"]

    def test_alerts_payload_capped(self, monkeypatch):
        monkeypatch.setattr(kpis_module, "ALERTS_PAYLOAD_LIMIT", 2)
        alerts = [
            {"anomaly_id": f"a{i}", "detected_at": iso(20, 11 - i // 60), "lead_id": f"l{i}"}
            for i in range(5)
        ]
        snapshot = make_service(alerts=FakeAlerts(alerts)).snapshot()
        assert len(snapshot["alerts"]) == 2
        assert snapshot["anomalies_count"] == 5
