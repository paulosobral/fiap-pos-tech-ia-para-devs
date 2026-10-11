import pytest

from service.scheduler_gate import SchedulingGate


class FakeAlerts:
    def __init__(self, restricted_item=None):
        self.restrict_calls = []
        self.restricted_item = restricted_item

    def restrict_scheduling(self, anomaly_id):
        self.restrict_calls.append(anomaly_id)

    def is_scheduling_restricted(self, lead_id):
        return self.restricted_item is not None and self.restricted_item.get("lead_id") == lead_id

    def find_open_restriction(self, lead_id):
        if self.is_scheduling_restricted(lead_id):
            return self.restricted_item
        return None


class TestSchedulingGate:
    def test_restrict_delegates_to_alert_store(self):
        alerts = FakeAlerts()
        gate = SchedulingGate(alerts)
        gate.restrict("l-1", "a-1")
        assert alerts.restrict_calls == ["a-1"]

    def test_restrict_requires_lead_and_anomaly_ids(self):
        gate = SchedulingGate(FakeAlerts())
        with pytest.raises(ValueError):
            gate.restrict("", "a-1")
        with pytest.raises(ValueError):
            gate.restrict("l-1", "")

    def test_is_restricted_true_for_flagged_lead(self):
        alerts = FakeAlerts(restricted_item={"lead_id": "l-1", "restriction_reason": "schedule_restricted"})
        gate = SchedulingGate(alerts)
        assert gate.is_restricted("l-1") is True
        assert gate.is_restricted("other") is False

    def test_is_restricted_false_for_blank_lead(self):
        assert SchedulingGate(FakeAlerts()).is_restricted("") is False
        assert SchedulingGate(FakeAlerts()).restriction_reason("") is None

    def test_restriction_reason_falls_back_to_action_taken(self):
        alerts = FakeAlerts(restricted_item={"lead_id": "l-1", "action_taken": "schedule_restricted"})
        gate = SchedulingGate(alerts)
        assert gate.restriction_reason("l-1") == "schedule_restricted"
        assert gate.restriction_reason("unknown") is None

    def test_restrict_logs_json_event(self, caplog):
        import json
        import logging

        caplog.set_level(logging.INFO)
        gate = SchedulingGate(FakeAlerts())
        gate.restrict("l-1", "a-1")
        record = [r for r in caplog.records if r.name == "infra.logging_utils"][-1]
        event = json.loads(record.getMessage())
        assert event["event"] == "scheduling_restricted"
        assert event["lead_id"] == "l-1"
        assert event["anomaly_id"] == "a-1"
