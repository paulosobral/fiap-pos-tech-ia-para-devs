from datetime import datetime, timezone

from service.duplicate_guard import DuplicateGuard

FIXED_NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


class FakeState:
    def __init__(self, states=None):
        self.states = states or {}

    def get_state(self, lead_id):
        return self.states.get(lead_id)


def state(last_step, last_followup_at=None):
    return {"last_step": last_step, "last_followup_at": last_followup_at}


class TestDuplicateGuard:
    def test_no_history_allows_followup(self):
        assert DuplicateGuard(FakeState()).evaluate("lead-1", 2) is None

    def test_same_step_already_sent_suppresses(self):
        guard = DuplicateGuard(FakeState({"lead-1": state(2, "2026-09-20T11:00:00+00:00")}))
        assert guard.evaluate("lead-1", 2) == "already_followed_up"

    def test_earlier_step_than_sent_suppresses(self):
        guard = DuplicateGuard(FakeState({"lead-1": state(5, "2026-09-21T11:00:00+00:00")}))
        assert guard.evaluate("lead-1", 2) == "already_followed_up"

    def test_lead_replied_after_last_followup_suppresses(self):
        guard = DuplicateGuard(
            FakeState({"lead-1": state(2, "2026-09-18T12:00:00+00:00")})
        )
        reason = guard.evaluate("lead-1", 5, "2026-09-19T10:00:00+00:00")
        assert reason == "lead_replied"

    def test_lead_silent_since_last_followup_allows_next_step(self):
        guard = DuplicateGuard(
            FakeState({"lead-1": state(2, "2026-09-18T12:00:00+00:00")})
        )
        assert guard.evaluate("lead-1", 5, "2026-09-17T10:00:00+00:00") is None

    def test_reply_check_requires_both_timestamps(self):
        guard = DuplicateGuard(FakeState({"lead-1": state(2)}))
        assert guard.evaluate("lead-1", 5, None) is None
        guard = DuplicateGuard(FakeState({}))
        assert guard.evaluate("lead-1", 5, "2026-09-19T10:00:00+00:00") is None

    def test_unparseable_reply_timestamp_is_ignored(self):
        guard = DuplicateGuard(
            FakeState({"lead-1": state(2, "2026-09-18T12:00:00+00:00")})
        )
        assert guard.evaluate("lead-1", 5, "sem-data") is None
