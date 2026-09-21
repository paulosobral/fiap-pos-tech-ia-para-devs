import json
import sys
import types
from datetime import datetime, timezone

import pytest

import handler as anomaly_handler
from infra.conversation_store import unmarshal_item


def conversation_raw(session_id, lead_id, text, ts, count=1):
    return {
        "PK": {"S": f"LEAD#{lead_id}"},
        "SK": {"S": f"CONV#{session_id}"},
        "session_id": {"S": session_id},
        "lead_id": {"S": lead_id},
        "messages": {"S": json.dumps([{"role": "lead", "text": text, "ts": ts} for _ in range(count)])},
        "context": {"S": "{}"},
        "current_state": {"S": "greeting"},
        "pii_masked": {"BOOL": False},
        "ttl": {"N": "7776000"},
    }


SUSPICIOUS_TEXT = "isso é um golpe e um absurdo " * 45

FIXED_NOW = datetime(2026, 9, 20, 3, 30, tzinfo=timezone.utc)


class FakeDynamo:
    """Cliente DynamoDB in-memory: scan com filtro CONV#, put/update/query de alertas."""

    def __init__(self, conversation_items=()):
        self._items: dict[str, dict] = {}
        for raw in conversation_items:
            self._items[raw["PK"]["S"] + "|" + raw["SK"]["S"]] = raw

    def scan(self, **kwargs):
        prefix = kwargs["ExpressionAttributeValues"][":prefix"]["S"]
        items = [
            raw
            for raw in self._items.values()
            if raw.get("SK", {}).get("S", "").startswith(prefix)
        ]
        return {"Items": items}

    def put_item(self, TableName=None, Item=None, **_kwargs):
        self._items[Item["PK"]["S"] + "|"] = Item
        return {}

    def update_item(self, TableName=None, Key=None, UpdateExpression=None, ExpressionAttributeValues=None, **_kwargs):
        stored = self._items.get(Key["PK"]["S"] + "|")
        if stored is None:
            stored = {"PK": Key["PK"]}
            self._items[Key["PK"]["S"] + "|"] = stored
        assignments = UpdateExpression.split("SET", 1)[1].split(",")
        for assignment, placeholder in zip(assignments, ExpressionAttributeValues.values()):
            attribute = assignment.strip().split("=", 1)[0].strip()
            stored[attribute] = placeholder
        return {}

    def query(self, TableName=None, IndexName=None, KeyConditionExpression=None, ExpressionAttributeValues=None, **_kwargs):
        lead_id = next(iter(ExpressionAttributeValues.values()))["S"]
        items = [
            raw
            for raw in self._items.values()
            if raw.get("lead_id", {}).get("S") == lead_id
        ]
        return {"Items": items}


def install_fake_boto3(monkeypatch, fake_dynamo):
    module = types.ModuleType("boto3")

    def client(service):
        assert service == "dynamodb"
        return fake_dynamo

    module.client = client
    monkeypatch.setitem(sys.modules, "boto3", module)


@pytest.fixture()
def env_tables(monkeypatch):
    monkeypatch.setenv("SESSIONS_TABLE", "sdr-sessions-test")
    monkeypatch.setenv("ALERTS_TABLE", "sdr-alerts-test")
    monkeypatch.delenv("ANOMALY_SCORER", raising=False)
    monkeypatch.delenv("ANOMALY_THRESHOLD", raising=False)


class TestAnomalyPipeline:
    def test_daily_job_detects_restricts_and_persists_alert(self, env_tables):
        dynamo = FakeDynamo(
            [
                conversation_raw("s-ok", "lead-ok", "bom dia, quero ver imóveis", "2026-09-19T10:00:00+00:00"),
                conversation_raw("s-bad", "lead-bad", SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30),
            ]
        )
        summary = run_handler(dynamo)
        assert summary == {"conversations": 2, "scored": 2, "anomalies": 1, "restricted": 1, "errors": 0}
        alert = unmarshal_item(dynamo._items["s-bad#2026-09-20|"])
        assert alert["lead_id"] == "lead-bad"
        assert alert["type"] == "negative_sentiment"
        assert alert["confidence"] >= 0.7
        assert alert["scheduling_restricted"] is True
        assert alert["action_taken"] == "schedule_restricted"
        assert alert["status"] == "open"
        assert alert["features"] and json.loads(alert["features"])["message_volume"] == 30

    def test_restriction_is_queryable_by_lead_after_job(self, env_tables):
        dynamo = FakeDynamo(
            [conversation_raw("s-bad", "lead-bad", SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30)]
        )
        run_handler(dynamo)
        from infra.alert_store import AlertStore

        alerts = AlertStore(dynamo, "sdr-alerts-test")
        assert alerts.is_scheduling_restricted("lead-bad") is True
        assert alerts.is_scheduling_restricted("lead-ok") is False

    def test_empty_job_completes_without_alerts(self, env_tables):
        dynamo = FakeDynamo([])
        summary = run_handler(dynamo)
        assert summary["conversations"] == 0
        assert summary["anomalies"] == 0

    def test_job_is_idempotent_same_day(self, env_tables):
        dynamo = FakeDynamo(
            [conversation_raw("s-bad", "lead-bad", SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30)]
        )
        run_handler(dynamo)
        run_handler(dynamo)
        alert_items = [key for key in dynamo._items if key.startswith("s-bad#")]
        assert alert_items == ["s-bad#2026-09-20|"]

    def test_structured_logs_never_leak_conversation_text(self, env_tables, caplog):
        import logging

        caplog.set_level(logging.INFO)
        dynamo = FakeDynamo(
            [conversation_raw("s-bad", "lead-bad", "PII ana@empresa.com 11988887777 " + SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30)]
        )
        run_handler(dynamo)
        assert "ana@empresa.com" not in caplog.text
        assert "11988887777" not in caplog.text
        assert "golpe" not in caplog.text

    def test_handler_runs_with_eventbridge_event_shape(self, env_tables):
        dynamo = FakeDynamo([])
        event = {
            "id": "scheduled-event-id",
            "source": "aws.events",
            "detail-type": "Scheduled Event",
            "detail": {},
        }
        with pytest.MonkeyPatch.context() as patcher:
            install_fake_boto3(patcher, dynamo)
            summary = anomaly_handler.handler(event)
        assert summary["conversations"] == 0

    def test_handler_with_sklearn_scorer_missing_dependency(self, env_tables, monkeypatch):
        for name in ("sklearn", "sklearn.ensemble", "sklearn.decomposition", "sklearn.preprocessing"):
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setenv("ANOMALY_SCORER", "sklearn")
        dynamo = FakeDynamo(
            [conversation_raw(f"s-{i}", f"lead-{i}", SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30) for i in range(6)]
        )
        install_fake_boto3(monkeypatch, dynamo)
        from service.scorer import ScorerDependencyError

        with pytest.raises(ScorerDependencyError, match="scikit-learn indisponível"):
            anomaly_handler.handler({"source": "aws.events"})

    def test_handler_rejects_unknown_scorer_name(self, env_tables, monkeypatch):
        monkeypatch.setenv("ANOMALY_SCORER", "naive-bayes")
        dynamo = FakeDynamo([])
        install_fake_boto3(monkeypatch, dynamo)
        with pytest.raises(ValueError, match="unknown ANOMALY_SCORER"):
            anomaly_handler.handler({"source": "aws.events"})


def run_handler(dynamo, event=None):
    with pytest.MonkeyPatch.context() as patcher:
        install_fake_boto3(patcher, dynamo)
        patcher.setattr(anomaly_handler, "utc_now", lambda: FIXED_NOW)
        return anomaly_handler.handler(event or {"source": "aws.events", "detail-type": "Scheduled Event"})
