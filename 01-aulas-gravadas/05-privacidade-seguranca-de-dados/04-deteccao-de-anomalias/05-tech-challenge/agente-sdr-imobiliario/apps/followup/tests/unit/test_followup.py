from datetime import datetime, timezone

import pytest

from infra.silence_window import SilenceWindow
from service.cadence import CadenceCalculator
from service.followup import FollowupService
from service.telegram_gateway import GatewayError
from service.message_builder import FollowupMessageBuilder

FIXED_NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)

DAY_2 = "2026-09-18T10:00:00+00:00"
DAY_5 = "2026-09-15T10:00:00+00:00"
DAY_9 = "2026-09-11T10:00:00+00:00"
DAY_10 = "2026-09-10T10:00:00+00:00"


def conversation(lead_id="lead-1", session_id="sess-1", created_at=DAY_2, messages=None, context=None, current_state="qualification"):
    return {
        "session_id": session_id,
        "lead_id": lead_id,
        "created_at": created_at,
        "messages": messages if messages is not None else [{"role": "lead", "text": "quero locar", "ts": created_at}],
        "context": context or {"channel": "telegram"},
        "current_state": current_state,
    }


def profile(lead_id="lead-1", telegram_user_id=42, intent="locação"):
    return {"lead_id": lead_id, "telegram_user_id": telegram_user_id, "intent": intent}


class FakeConversations:
    def __init__(self, conversations=(), profiles=None, error=None):
        self._conversations = list(conversations)
        self._profiles = profiles or {}
        self._error = error

    def list_conversations(self):
        if self._error:
            raise self._error
        return list(self._conversations)

    def get_lead_profiles(self):
        if self._error:
            raise self._error
        return dict(self._profiles)


class FakeState:
    def __init__(self, states=None, record_error=False):
        self.states = dict(states or {})
        self.recorded = []
        self.record_error = record_error

    def get_state(self, lead_id):
        return self.states.get(lead_id)

    def record_followup(self, lead_id, step, sent_at, next_step):
        if self.record_error:
            raise RuntimeError("dynamo down")
        self.recorded.append({"lead_id": lead_id, "step": step, "sent_at": sent_at, "next_step": next_step})
        self.states[lead_id] = {"last_step": step, "last_followup_at": sent_at}


class FakeGuard:
    def __init__(self, states=None):
        self.states = states or {}

    def evaluate(self, lead_id, step, last_lead_message_at=None):
        state = self.states.get(lead_id) or {}
        if state.get("last_step", 0) >= step:
            return "already_followed_up"
        if state.get("replied") and state.get("last_followup_at") and last_lead_message_at:
            if last_lead_message_at > state["last_followup_at"]:
                return "lead_replied"
        return None


class FakeTelegram:
    def __init__(self, error=False):
        self.sent = []
        self.error = error

    def send_message(self, chat_id, text):
        if self.error:
            raise GatewayError("sendMessage rejected with status 500")
        self.sent.append({"chat_id": chat_id, "text": text})


def build_service(conversations, state=None, guard=None, telegram=None, silence=None):
    return FollowupService(
        conversations=conversations,
        state_store=state or FakeState(),
        guard=guard or FakeGuard(),
        cadence=CadenceCalculator(now_fn=lambda: FIXED_NOW),
        silence_window=silence or SilenceWindow(now_fn=lambda: FIXED_NOW),
        builder=FollowupMessageBuilder(),
        telegram=telegram or FakeTelegram(),
        now_fn=lambda: FIXED_NOW,
    )


class TestFollowupService:
    def test_due_day_two_lead_gets_followup_and_state_recorded(self):
        state = FakeState()
        telegram = FakeTelegram()
        conversations = FakeConversations([conversation()], {"lead-1": profile()})
        summary = build_service(conversations, state, telegram=telegram).run()
        assert summary["sent"] == 1
        assert summary["due"] == 1
        assert telegram.sent[0]["chat_id"] == 42
        assert "Retomando" in telegram.sent[0]["text"]
        assert state.recorded[0]["step"] == 2
        assert state.recorded[0]["next_step"] == 5

    def test_silence_window_closed_defers_without_sending(self):
        state = FakeState()
        telegram = FakeTelegram()
        conversations = FakeConversations([conversation()], {"lead-1": profile()})
        silence = SilenceWindow(start_hour=13, end_hour=18, now_fn=lambda: FIXED_NOW)
        summary = build_service(conversations, state, telegram=telegram, silence=silence).run()
        assert summary["deferred"] == 1
        assert summary["sent"] == 0
        assert telegram.sent == []
        assert state.recorded == []

    def test_step_already_sent_is_suppressed_not_resent(self):
        guard = FakeGuard({"lead-1": {"last_step": 2}})
        telegram = FakeTelegram()
        conversations = FakeConversations([conversation()], {"lead-1": profile()})
        summary = build_service(conversations, guard=guard, telegram=telegram).run()
        assert summary["skipped_already_followed"] == 1
        assert summary["sent"] == 0
        assert telegram.sent == []

    def test_lead_replied_after_followup_is_not_double_messaged(self):
        replied = conversation(
            created_at=DAY_5,
            messages=[
                {"role": "agent", "text": "follow-up", "ts": DAY_2},
                {"role": "lead", "text": "ainda tenho interesse", "ts": "2026-09-19T14:00:00+00:00"},
            ],
        )
        guard = FakeGuard({"lead-1": {"last_step": 2, "last_followup_at": DAY_2, "replied": True}})
        telegram = FakeTelegram()
        conversations = FakeConversations([replied], {"lead-1": profile()})
        summary = build_service(conversations, guard=guard, telegram=telegram).run()
        assert summary["skipped_replied"] == 1
        assert telegram.sent == []

    def test_lead_without_context_is_dropped_explicitly(self):
        telegram = FakeTelegram()
        empty = conversation(messages=[], context={})
        conversations = FakeConversations([empty], {"lead-1": {"lead_id": "lead-1", "telegram_user_id": 42}})
        summary = build_service(conversations, telegram=telegram).run()
        assert summary["dropped"] == 1
        assert telegram.sent == []

    def test_lead_with_invalid_created_at_is_dropped(self):
        conversations = FakeConversations([conversation(created_at="sem-data")], {"lead-1": profile()})
        summary = build_service(conversations).run()
        assert summary["dropped"] == 1

    def test_expired_cadence_without_completion_is_dropped(self):
        conversations = FakeConversations([conversation(created_at=DAY_10)], {"lead-1": profile()})
        summary = build_service(conversations).run()
        assert summary["dropped"] == 1
        assert summary["due"] == 0

    def test_completed_cadence_is_neutral_not_due(self):
        state = FakeState({"lead-1": {"last_step": 9}})
        conversations = FakeConversations([conversation(created_at=DAY_10)], {"lead-1": profile()})
        summary = build_service(conversations, state).run()
        assert summary["not_due"] == 1
        assert summary["dropped"] == 0

    def test_lead_without_telegram_channel_is_dropped(self):
        conversations = FakeConversations([conversation()], {"lead-1": {"lead_id": "lead-1"}})
        summary = build_service(conversations).run()
        assert summary["dropped"] == 1

    def test_send_failure_counts_error_and_skips_state_write(self):
        state = FakeState()
        telegram = FakeTelegram(error=True)
        conversations = FakeConversations([conversation()], {"lead-1": profile()})
        summary = build_service(conversations, state, telegram=telegram).run()
        assert summary["errors"] == 1
        assert summary["sent"] == 0
        assert state.recorded == []

    def test_state_write_failure_counts_error(self):
        state = FakeState(record_error=True)
        conversations = FakeConversations([conversation()], {"lead-1": profile()})
        summary = build_service(conversations, state).run()
        assert summary["errors"] == 1
        assert summary["sent"] == 0

    def test_only_latest_conversation_per_lead_is_processed(self):
        conversations = FakeConversations(
            [conversation(session_id="old", created_at=DAY_9), conversation(session_id="new")],
            {"lead-1": profile()},
        )
        summary = build_service(conversations).run()
        assert summary["leads"] == 1

    def test_conversation_read_failure_propagates_for_lambda_retry(self):
        conversations = FakeConversations(error=RuntimeError("dynamo down"))
        with pytest.raises(RuntimeError):
            build_service(conversations).run()

    def test_day_nine_send_marks_cadence_done(self):
        state = FakeState()
        conversations = FakeConversations([conversation(created_at=DAY_9)], {"lead-1": profile()})
        summary = build_service(conversations, state).run()
        assert summary["sent"] == 1
        assert state.recorded[0]["next_step"] is None
