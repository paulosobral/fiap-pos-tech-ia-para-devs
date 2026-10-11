import importlib.util
import json
import logging
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

import handler as anomaly_handler
from infra.conversation_store import unmarshal_item

U1_RESTRICTION_PATH = (
    Path(__file__).resolve().parents[4] / "apps" / "conversation-router" / "service" / "restriction.py"
)


def conversation_raw(session_id, lead_id, text, at, count=1):
    return {
        "PK": {"S": f"LEAD#{lead_id}"},
        "SK": {"S": f"CONV#{session_id}"},
        "session_id": {"S": session_id},
        "lead_id": {"S": lead_id},
        "messages": {"S": json.dumps([{"role": "lead", "text": text, "at": at} for _ in range(count)])},
        "context": {"S": "{}"},
        "current_state": {"S": "greeting"},
        "pii_masked": {"BOOL": False},
        "ttl": {"N": "7776000"},
    }


SUSPICIOUS_TEXT = "isso é um golpe e um absurdo " * 45
BUSINESS_HOUR_TEXT = "bom dia, quero ver imóveis"

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

    def update_item(self, TableName=None, Key=None, UpdateExpression=None, ExpressionAttributeValues=None, ExpressionAttributeNames=None, **_kwargs):
        stored = self._items.get(Key["PK"]["S"] + "|")
        if stored is None:
            stored = {"PK": Key["PK"]}
            self._items[Key["PK"]["S"] + "|"] = stored
        assignments = UpdateExpression.split("SET", 1)[1].split(",")
        for assignment, placeholder in zip(assignments, ExpressionAttributeValues.values()):
            attribute = assignment.strip().split("=", 1)[0].strip()
            if attribute.startswith("#"):
                attribute = ExpressionAttributeNames[attribute]
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


def run_handler(dynamo, event=None):
    with pytest.MonkeyPatch.context() as patcher:
        install_fake_boto3(patcher, dynamo)
        patcher.setattr(anomaly_handler, "utc_now", lambda: FIXED_NOW)
        return anomaly_handler.handler(event or {"source": "aws.events", "detail-type": "Scheduled Event"})


def load_u1_restriction_check():
    assert U1_RESTRICTION_PATH.exists(), f"checker da U1 não encontrado: {U1_RESTRICTION_PATH}"
    spec = importlib.util.spec_from_file_location("u1_restriction", str(U1_RESTRICTION_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DynamoRestrictionCheck


class TestAnomalyPipeline:
    def test_daily_job_detects_restricts_and_persists_alert(self, env_tables):
        dynamo = FakeDynamo(
            [
                conversation_raw("s-ok", "lead-ok", BUSINESS_HOUR_TEXT, "2026-09-19T13:00:00+00:00"),
                conversation_raw("s-bad", "lead-bad", SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30),
            ]
        )
        summary = run_handler(dynamo)
        assert summary == {
            "conversations": 2,
            "scored": 2,
            "anomalies": 1,
            "restricted": 1,
            "resolved": 0,
            "errors": 0,
        }
        alert = unmarshal_item(dynamo._items["s-bad#2026-09-20|"])
        assert alert["lead_id"] == "lead-bad"
        assert alert["type"] == "negative_sentiment"
        assert alert["confidence"] >= 0.7
        assert alert["scheduling_restricted"] is True
        assert alert["action_taken"] == "schedule_restricted"
        assert alert["status"] == "open"
        assert alert["features"] and json.loads(alert["features"])["message_volume"] == 30
        assert json.loads(alert["features"])["atypical_hour_ratio"] == 1.0

    def test_restriction_is_queryable_by_lead_after_job(self, env_tables):
        dynamo = FakeDynamo(
            [conversation_raw("s-bad", "lead-bad", SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30)]
        )
        run_handler(dynamo)
        from infra.alert_store import AlertStore

        alerts = AlertStore(dynamo, "sdr-alerts-test")
        assert alerts.is_scheduling_restricted("lead-bad") is True
        assert alerts.is_scheduling_restricted("lead-ok") is False

    def test_normal_lead_with_open_restriction_is_auto_resolved_by_rerun(self, env_tables):
        dynamo = FakeDynamo(
            [conversation_raw("s-bad", "lead-bad", SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30)]
        )
        run_handler(dynamo)
        dynamo._items.pop("LEAD#lead-bad|CONV#s-bad", None)
        normalized = conversation_raw("s-bad2", "lead-bad", "bom dia, obrigado!", "2026-09-21T13:00:00+00:00")
        dynamo._items[normalized["PK"]["S"] + "|" + normalized["SK"]["S"]] = normalized
        summary = run_handler(dynamo)
        assert summary["resolved"] == 1
        alert = unmarshal_item(dynamo._items["s-bad#2026-09-20|"])
        assert alert["status"] == "resolved"
        assert alert["resolved_reason"] == "not_anomalous_in_run"
        from infra.alert_store import AlertStore

        assert AlertStore(dynamo, "sdr-alerts-test").is_scheduling_restricted("lead-bad") is False

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
        caplog.set_level(logging.INFO)
        dynamo = FakeDynamo(
            [conversation_raw("s-bad", "lead-bad", "PII ana@empresa.com 11988887777 " + SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30)]
        )
        run_handler(dynamo)
        assert "ana@empresa.com" not in caplog.text
        assert "11988887777" not in caplog.text
        assert "golpe" not in caplog.text

    def test_all_component_logs_are_json_events(self, env_tables, caplog):
        caplog.set_level(logging.INFO)
        dynamo = FakeDynamo(
            [
                conversation_raw("s-bad", "lead-bad", SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30),
            ]
        )
        run_handler(dynamo)
        records = [r for r in caplog.records if r.name == "infra.logging_utils"]
        events = [json.loads(record.getMessage()) for record in records]
        assert len(events) >= 4
        assert all("event" in event for event in events)
        assert {event["event"] for event in events} >= {
            "job_started",
            "anomaly_alerted",
            "scheduling_restricted",
            "job_completed",
        }

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

    def test_handler_runs_anyway_with_unexpected_event_payload(self, env_tables, caplog):
        caplog.set_level(logging.INFO)
        dynamo = FakeDynamo([])
        with pytest.MonkeyPatch.context() as patcher:
            install_fake_boto3(patcher, dynamo)
            summary = anomaly_handler.handler("not-a-dict")
        assert summary["conversations"] == 0
        record = [r for r in caplog.records if r.name == "infra.logging_utils"][0]
        assert json.loads(record.getMessage())["event"] == "unexpected_eventbridge_payload"

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


class TestU1RestrictionContract:
    """Contrato U5→U1 (FR9.4): o checker REAL da U1 consome os alertas persistidos."""

    def test_persisted_alert_satisfies_real_u1_checker(self, env_tables):
        dynamo = FakeDynamo(
            [conversation_raw("s-bad", "lead-bad", SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30)]
        )
        run_handler(dynamo)
        raw = dynamo._items["s-bad#2026-09-20|"]
        assert raw["lead_id"] == {"S": "lead-bad"}
        assert raw["scheduling_restricted"] == {"BOOL": True}
        assert raw["status"] == {"S": "open"}
        checker = load_u1_restriction_check()
        assert checker(dynamo, "sdr-alerts-test")("lead-bad") is True

    def test_clean_lead_is_not_restricted_for_u1_checker(self, env_tables):
        dynamo = FakeDynamo(
            [conversation_raw("s-ok", "lead-ok", BUSINESS_HOUR_TEXT, "2026-09-19T13:00:00+00:00")]
        )
        run_handler(dynamo)
        checker = load_u1_restriction_check()
        assert checker(dynamo, "sdr-alerts-test")("lead-ok") is False

    def test_auto_resolved_restriction_releases_lead_for_u1_checker(self, env_tables):
        dynamo = FakeDynamo(
            [conversation_raw("s-bad", "lead-bad", SUSPICIOUS_TEXT, "2026-09-20T03:00:00+00:00", count=30)]
        )
        run_handler(dynamo)
        checker = load_u1_restriction_check()
        assert checker(dynamo, "sdr-alerts-test")("lead-bad") is True
        dynamo._items.pop("LEAD#lead-bad|CONV#s-bad", None)
        normalized = conversation_raw("s-bad2", "lead-bad", "bom dia, obrigado!", "2026-09-21T13:00:00+00:00")
        dynamo._items[normalized["PK"]["S"] + "|" + normalized["SK"]["S"]] = normalized
        run_handler(dynamo)
        assert checker(dynamo, "sdr-alerts-test")("lead-bad") is False
