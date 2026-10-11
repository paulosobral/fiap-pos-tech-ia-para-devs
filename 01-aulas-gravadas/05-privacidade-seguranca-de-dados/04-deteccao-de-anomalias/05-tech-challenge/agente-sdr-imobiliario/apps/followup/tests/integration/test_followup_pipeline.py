import json
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import handler as followup_handler
from infra.followup_state import FOLLOWUP_SK, FollowupStateStore
from infra.silence_window import utc_now


def iso(dt):
    return dt.isoformat()


def conversation_raw(lead_id, session_id, created_at, messages=None, intent=None):
    context = {"channel": "telegram"}
    if intent:
        context["intent"] = intent
    return {
        "PK": {"S": f"LEAD#{lead_id}"},
        "SK": {"S": f"CONV#{session_id}"},
        "session_id": {"S": session_id},
        "lead_id": {"S": lead_id},
        "messages": {"S": json.dumps(messages or [{"role": "lead", "text": "quero locar", "at": created_at}])},
        "context": {"S": json.dumps(context)},
        "current_state": {"S": "qualification"},
        "pii_masked": {"BOOL": False},
        "created_at": {"S": created_at},
        "ttl": {"N": "7776000"},
    }


def profile_raw(lead_id, telegram_user_id=42, intent="locação"):
    return {
        "PK": {"S": f"LEAD#{lead_id}"},
        "SK": {"S": "PROFILE"},
        "lead_id": {"S": lead_id},
        "telegram_user_id": {"N": str(telegram_user_id)},
        "intent": {"S": intent},
        "status": {"S": "new"},
    }


class FakeDynamo:
    """Cliente DynamoDB in-memory: scan com filtro SK, query de estado, put de estado."""

    def __init__(self, items=()):
        self._items: dict[tuple[str, str], dict] = {}
        for raw in items:
            self._items[(raw["PK"]["S"], raw["SK"]["S"])] = raw

    def scan(self, **kwargs):
        prefix = kwargs["ExpressionAttributeValues"][":prefix"]["S"]
        items = [raw for (_, sk), raw in self._items.items() if sk.startswith(prefix)]
        return {"Items": items}

    def query(self, TableName=None, KeyConditionExpression=None, ExpressionAttributeValues=None, **_kwargs):
        pk = ExpressionAttributeValues[":pk"]["S"]
        sk = ExpressionAttributeValues[":sk"]["S"]
        raw = self._items.get((pk, sk))
        return {"Items": [raw] if raw else []}

    def put_item(self, TableName=None, Item=None, **_kwargs):
        self._items[(Item["PK"]["S"], Item["SK"]["S"])] = Item
        return {}


class FakeHttp:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return SimpleNamespace(ok=self.ok, status_code=200 if self.ok else 500)


def install_fakes(monkeypatch, dynamo, http):
    boto3_module = types.ModuleType("boto3")
    boto3_module.client = lambda service: dynamo
    requests_module = types.ModuleType("requests")
    requests_module.Session = lambda: http
    monkeypatch.setitem(sys.modules, "boto3", boto3_module)
    monkeypatch.setitem(sys.modules, "requests", requests_module)


@pytest.fixture()
def env_tables(monkeypatch):
    monkeypatch.setenv("SESSIONS_TABLE", "sdr-sessions-test")
    monkeypatch.setenv("FOLLOWUP_TABLE", "sdr-followup-state-test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TIMEZONE", "UTC")
    monkeypatch.setenv("SILENCE_WINDOW_START", "0")
    monkeypatch.setenv("SILENCE_WINDOW_END", "0")
    for name in ("FOLLOWUP_CADENCE_DAYS",):
        monkeypatch.delenv(name, raising=False)


class TestFollowupPipeline:
    def test_day_two_lead_receives_followup_and_state_is_persisted(self, env_tables, monkeypatch):
        now = utc_now()
        dynamo = FakeDynamo(
            [
                conversation_raw("lead-1", "sess-1", iso(now - timedelta(days=2))),
                profile_raw("lead-1"),
            ]
        )
        http = FakeHttp()
        install_fakes(monkeypatch, dynamo, http)
        summary = followup_handler.handler({"source": "aws.events", "detail-type": "Scheduled Event"})
        assert summary["sent"] == 1
        assert http.calls[0]["url"].endswith("/bottest-token/sendMessage")
        state = FollowupStateStore(dynamo, "sdr-followup-state-test").get_state("lead-1")
        assert state["last_step"] == 2
        assert state["next_step"] == 5

    def test_silence_window_closed_defers_entire_tick(self, env_tables, monkeypatch):
        hour = utc_now().hour
        monkeypatch.setenv("SILENCE_WINDOW_START", str((hour + 1) % 24))
        monkeypatch.setenv("SILENCE_WINDOW_END", str((hour + 2) % 24))
        dynamo = FakeDynamo(
            [
                conversation_raw("lead-1", "sess-1", iso(utc_now() - timedelta(days=2))),
                profile_raw("lead-1"),
            ]
        )
        http = FakeHttp()
        install_fakes(monkeypatch, dynamo, http)
        summary = followup_handler.handler({"source": "aws.events"})
        assert summary["deferred"] == 1
        assert summary["sent"] == 0
        assert http.calls == []

    def test_custom_cadence_day_completes_cadence(self, env_tables, monkeypatch):
        monkeypatch.setenv("FOLLOWUP_CADENCE_DAYS", "1")
        dynamo = FakeDynamo(
            [
                conversation_raw("lead-1", "sess-1", iso(utc_now() - timedelta(days=1))),
                profile_raw("lead-1"),
            ]
        )
        install_fakes(monkeypatch, dynamo, FakeHttp())
        summary = followup_handler.handler({"source": "aws.events"})
        assert summary["sent"] == 1
        state = FollowupStateStore(dynamo, "sdr-followup-state-test").get_state("lead-1")
        assert state["last_step"] == 1
        assert state["done"] is True

    def test_lead_reply_after_followup_suppresses_next_step(self, env_tables, monkeypatch):
        now = utc_now()
        followup_at = now - timedelta(days=1, hours=2)
        replied = conversation_raw(
            "lead-1",
            "sess-1",
            iso(now - timedelta(days=5)),
            messages=[
                {"role": "agent", "text": "follow-up", "at": iso(followup_at)},
                {"role": "lead", "text": "ainda tenho interesse", "at": iso(now - timedelta(hours=1))},
            ],
        )
        dynamo = FakeDynamo([replied, profile_raw("lead-1")])
        store = FollowupStateStore(dynamo, "sdr-followup-state-test")
        store.record_followup("lead-1", 2, iso(followup_at), 5)
        http = FakeHttp()
        install_fakes(monkeypatch, dynamo, http)
        summary = followup_handler.handler({"source": "aws.events"})
        assert summary["skipped_replied"] == 1
        assert http.calls == []

    def test_rerun_same_day_never_resends_followup(self, env_tables, monkeypatch):
        dynamo = FakeDynamo(
            [
                conversation_raw("lead-1", "sess-1", iso(utc_now() - timedelta(days=2))),
                profile_raw("lead-1"),
            ]
        )
        http = FakeHttp()
        install_fakes(monkeypatch, dynamo, http)
        first = followup_handler.handler({"source": "aws.events"})
        second = followup_handler.handler({"source": "aws.events"})
        assert first["sent"] == 1
        assert second["sent"] == 0
        assert len(http.calls) == 1

    def test_structured_logs_never_leak_conversation_text(self, env_tables, monkeypatch, caplog):
        import logging

        caplog.set_level(logging.INFO)
        now = utc_now()
        dynamo = FakeDynamo(
            [
                conversation_raw(
                    "lead-1",
                    "sess-1",
                    iso(now - timedelta(days=2)),
                    messages=[
                        {
                            "role": "lead",
                            "text": "sou ana@empresa.com, tel 11988887777, laje na Faria Lima",
                            "at": iso(now - timedelta(days=2)),
                        }
                    ],
                ),
                profile_raw("lead-1"),
            ]
        )
        install_fakes(monkeypatch, dynamo, FakeHttp())
        followup_handler.handler({"source": "aws.events"})
        assert "ana@empresa.com" not in caplog.text
        assert "11988887777" not in caplog.text
        assert "Faria Lima" not in caplog.text

    def test_handler_missing_telegram_token_raises(self, env_tables, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        install_fakes(monkeypatch, FakeDynamo([]), FakeHttp())
        with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
            followup_handler.handler({"source": "aws.events"})

    def test_handler_tolerates_non_dict_event_payload(self, env_tables, monkeypatch):
        install_fakes(monkeypatch, FakeDynamo([]), FakeHttp())
        summary = followup_handler.handler(None)
        assert summary["leads"] == 0
