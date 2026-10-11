import json
import logging

import pytest

from infra.followup_state import FollowupStateError, FollowupStateStore


class FakeStateClient:
    def __init__(self, error=None):
        self.items = {}
        self.error = error

    def query(self, TableName=None, KeyConditionExpression=None, ExpressionAttributeValues=None, **_kwargs):
        if self.error:
            raise self.error
        pk = ExpressionAttributeValues[":pk"]["S"]
        sk = ExpressionAttributeValues[":sk"]["S"]
        raw = self.items.get((pk, sk))
        return {"Items": [raw] if raw else []}

    def put_item(self, TableName=None, Item=None, **_kwargs):
        if self.error:
            raise self.error
        self.items[(Item["PK"]["S"], Item["SK"]["S"])] = Item


class TestFollowupStateStore:
    def test_get_state_returns_none_without_history(self):
        assert FollowupStateStore(FakeStateClient()).get_state("lead-1") is None

    def test_record_and_read_roundtrip(self):
        client = FakeStateClient()
        store = FollowupStateStore(client)
        store.record_followup("lead-1", 2, "2026-09-20T12:00:00+00:00", 5)
        state = store.get_state("lead-1")
        assert state["lead_id"] == "lead-1"
        assert state["cadence_step"] == 2
        assert state["last_step"] == 2
        assert state["last_followup_at"] == "2026-09-20T12:00:00+00:00"
        assert state["next_step"] == 5
        assert state["done"] is False

    def test_final_step_marks_done_and_omits_next(self):
        store = FollowupStateStore(FakeStateClient())
        store.record_followup("lead-1", 9, "2026-09-20T12:00:00+00:00", None)
        state = store.get_state("lead-1")
        assert state["done"] is True
        assert "next_step" not in state

    def test_re_record_overwrites_previous_state(self):
        client = FakeStateClient()
        store = FollowupStateStore(client)
        store.record_followup("lead-1", 2, "2026-09-18T12:00:00+00:00", 5)
        store.record_followup("lead-1", 5, "2026-09-21T12:00:00+00:00", 9)
        state = store.get_state("lead-1")
        assert state["last_step"] == 5
        assert state["last_followup_at"] == "2026-09-21T12:00:00+00:00"

    def test_write_failure_raises_followup_state_error(self):
        with pytest.raises(FollowupStateError, match="write failed"):
            FollowupStateStore(FakeStateClient(error=RuntimeError("dynamo down"))).record_followup(
                "lead-1", 2, "2026-09-20T12:00:00+00:00", 5
            )

    def test_read_failure_raises_followup_state_error(self):
        with pytest.raises(FollowupStateError, match="unavailable"):
            FollowupStateStore(FakeStateClient(error=RuntimeError("dynamo down"))).get_state("lead-1")

    def test_failure_logs_are_structured_json(self, caplog):
        caplog.set_level(logging.INFO)
        with pytest.raises(FollowupStateError):
            FollowupStateStore(FakeStateClient(error=RuntimeError("dynamo down"))).get_state("lead-1")
        with pytest.raises(FollowupStateError):
            FollowupStateStore(FakeStateClient(error=RuntimeError("dynamo down"))).record_followup(
                "lead-1", 2, "2026-09-20T12:00:00+00:00", 5
            )
        events = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
        names = {e.get("event") for e in events}
        assert "followup_state_unavailable" in names
        assert "followup_state_write_failed" in names
