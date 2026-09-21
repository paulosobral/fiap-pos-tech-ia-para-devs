from datetime import datetime, timezone

import pytest

from infra.alert_store import AlertStoreError
from infra.conversation_store import ConversationStoreError
from service.anomaly_detector import AnomalyDetector
from service.feature_extractor import (
    FEATURE_HOURS,
    FEATURE_LENGTH,
    FEATURE_SENTIMENT,
    FEATURE_VOLUME,
)
from service.scorer import HeuristicScorer

FIXED_NOW = datetime(2026, 9, 20, 3, 30, tzinfo=timezone.utc)


def normal_conversation(lead_id="lead-1", session_id="sess-1"):
    return {
        "session_id": session_id,
        "lead_id": lead_id,
        "messages": [{"role": "lead", "text": "bom dia", "ts": "2026-09-19T10:00:00+00:00"}],
    }


def suspicious_conversation(lead_id="lead-x", session_id="sess-x"):
    text = "isso é um golpe e um absurdo" * 50
    return {
        "session_id": session_id,
        "lead_id": lead_id,
        "messages": [
            {"role": "lead", "text": text, "ts": "2026-09-20T03:00:00+00:00"} for _ in range(30)
        ],
    }


class FakeReader:
    def __init__(self, conversations=None, error=None):
        self.conversations = conversations or []
        self.error = error

    def list_conversations(self):
        if self.error:
            raise self.error
        return list(self.conversations)


class FakeAlerts:
    def __init__(self, error=None):
        self.saved = []
        self.restricted = []
        self.error = error

    def save_anomaly(self, anomaly):
        if self.error:
            raise self.error
        self.saved.append(anomaly)
        return anomaly

    def restrict_scheduling(self, anomaly_id):
        self.restricted.append(anomaly_id)


class FakeGate:
    def __init__(self, alerts=None):
        self.alerts = alerts
        self.calls = []

    def restrict(self, lead_id, anomaly_id):
        self.calls.append((lead_id, anomaly_id))
        if self.alerts is not None:
            self.alerts.restrict_scheduling(anomaly_id)


def build_detector(reader, alerts, gate=None):
    return AnomalyDetector(
        conversations=reader,
        alerts=alerts,
        scorer=HeuristicScorer(),
        gate=gate,
        now_fn=lambda: FIXED_NOW,
    )


class TestAnomalyDetector:
    def test_run_flags_anomaly_saves_alert_and_restricts_lead(self):
        alerts = FakeAlerts()
        gate = FakeGate(alerts)
        reader = FakeReader([normal_conversation(), suspicious_conversation()])
        summary = build_detector(reader, alerts, gate).run()
        assert summary == {"conversations": 2, "scored": 2, "anomalies": 1, "restricted": 1, "errors": 0}
        assert len(alerts.saved) == 1
        item = alerts.saved[0]
        assert item["anomaly_id"] == "sess-x#2026-09-20"
        assert item["lead_id"] == "lead-x"
        assert item["status"] == "open"
        assert item["action_taken"] == "alert_issued"
        assert item["detected_at"] == FIXED_NOW.isoformat()
        assert item["features"] == {
            FEATURE_VOLUME: 30,
            FEATURE_LENGTH: 1400.0,
            FEATURE_SENTIMENT: 1.0,
            FEATURE_HOURS: 1.0,
        }
        assert gate.calls == [("lead-x", "sess-x#2026-09-20")]
        assert alerts.restricted == ["sess-x#2026-09-20"]

    def test_empty_conversation_set_is_handled_gracefully(self):
        summary = build_detector(FakeReader([]), FakeAlerts()).run()
        assert summary == {"conversations": 0, "scored": 0, "anomalies": 0, "restricted": 0, "errors": 0}

    def test_all_normal_conversations_produce_no_alerts(self):
        alerts = FakeAlerts()
        reader = FakeReader([normal_conversation(f"lead-{i}", f"sess-{i}") for i in range(3)])
        summary = build_detector(reader, alerts).run()
        assert summary["anomalies"] == 0
        assert alerts.saved == []

    def test_conversation_without_lead_id_is_counted_not_scored(self):
        broken = {"session_id": "sess-9", "messages": []}
        summary = build_detector(FakeReader([broken]), FakeAlerts()).run()
        assert summary == {"conversations": 1, "scored": 1, "anomalies": 0, "restricted": 0, "errors": 0}

    def test_feature_extraction_failure_counts_error_and_continues(self):
        class ExplodingExtractor:
            def extract(self, conversation):
                if conversation["session_id"] == "bad":
                    raise RuntimeError("boom")
                return {FEATURE_VOLUME: 0, FEATURE_LENGTH: 0.0, FEATURE_SENTIMENT: 0.0, FEATURE_HOURS: 0.0}

        alerts = FakeAlerts()
        reader = FakeReader([{"session_id": "bad", "lead_id": "lead-b", "messages": []}, normal_conversation()])
        detector = AnomalyDetector(
            conversations=reader, alerts=alerts, scorer=HeuristicScorer(), extractor=ExplodingExtractor(), now_fn=lambda: FIXED_NOW
        )
        summary = detector.run()
        assert summary["errors"] == 1
        assert summary["scored"] == 1

    def test_reader_failure_propagates_for_lambda_retry(self):
        reader = FakeReader(error=ConversationStoreError("conversation store unavailable"))
        with pytest.raises(ConversationStoreError):
            build_detector(reader, FakeAlerts()).run()

    def test_alert_store_failure_propagates(self):
        reader = FakeReader([suspicious_conversation()])
        alerts = FakeAlerts(error=AlertStoreError("alert store unavailable"))
        with pytest.raises(AlertStoreError):
            build_detector(reader, alerts).run()

    def test_same_day_rerun_produces_deterministic_anomaly_id(self):
        alerts_first, alerts_second = FakeAlerts(), FakeAlerts()
        build_detector(FakeReader([suspicious_conversation()]), alerts_first).run()
        build_detector(FakeReader([suspicious_conversation()]), alerts_second).run()
        assert alerts_first.saved[0]["anomaly_id"] == alerts_second.saved[0]["anomaly_id"]
