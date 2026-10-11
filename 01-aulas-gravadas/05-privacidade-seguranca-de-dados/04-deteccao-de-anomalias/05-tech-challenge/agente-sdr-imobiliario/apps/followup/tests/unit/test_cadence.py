from datetime import datetime, timezone

import pytest

from service.cadence import (
    DEFAULT_CADENCE_DAYS,
    CadenceCalculator,
    parse_cadence_days,
    parse_iso8601,
)

FIXED_NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def days_ago(days: int) -> str:
    return datetime(2026, 9, 20 - days, 10, 0, tzinfo=timezone.utc).isoformat()


class TestCadenceCalculator:
    def test_step_is_due_on_day_two(self):
        cadence = CadenceCalculator(now_fn=lambda: FIXED_NOW)
        assert cadence.due_step(days_ago(2)) == 2

    def test_step_is_due_on_day_five_and_day_nine(self):
        cadence = CadenceCalculator(now_fn=lambda: FIXED_NOW)
        assert cadence.due_step(days_ago(5)) == 5
        assert cadence.due_step(days_ago(9)) == 9

    def test_no_step_before_first_cadence_day(self):
        cadence = CadenceCalculator(now_fn=lambda: FIXED_NOW)
        assert cadence.due_step(days_ago(1)) is None
        assert cadence.due_step(days_ago(0)) is None

    def test_missed_tick_self_heals_with_latest_due_step(self):
        cadence = CadenceCalculator(now_fn=lambda: FIXED_NOW)
        assert cadence.due_step(days_ago(3)) == 2
        assert cadence.due_step(days_ago(7)) == 5

    def test_future_created_at_returns_none(self):
        cadence = CadenceCalculator(now_fn=lambda: FIXED_NOW)
        assert cadence.due_step(datetime(2026, 9, 25, tzinfo=timezone.utc).isoformat()) is None

    def test_is_expired_after_last_day_only(self):
        cadence = CadenceCalculator(now_fn=lambda: FIXED_NOW)
        assert cadence.is_expired(days_ago(10)) is True
        assert cadence.is_expired(days_ago(9)) is False

    def test_invalid_created_at_is_expired_and_undue(self):
        cadence = CadenceCalculator(now_fn=lambda: FIXED_NOW)
        assert cadence.due_step("not-a-date") is None
        assert cadence.is_expired(None) is True
        assert cadence.is_expired("") is True

    def test_next_step_walks_cadence_until_exhausted(self):
        cadence = CadenceCalculator(now_fn=lambda: FIXED_NOW)
        assert cadence.next_step(2) == 5
        assert cadence.next_step(5) == 9
        assert cadence.next_step(9) is None

    def test_custom_days_and_invalid_days(self):
        cadence = CadenceCalculator(days=(1, 3), now_fn=lambda: FIXED_NOW)
        assert cadence.due_step(days_ago(1)) == 1
        assert cadence.last_day == 3
        with pytest.raises(ValueError):
            CadenceCalculator(days=(0, 2))
        with pytest.raises(ValueError):
            CadenceCalculator(days=())


class TestCadenceParsers:
    def test_parse_cadence_days_defaults_to_2_5_9(self):
        assert parse_cadence_days(None) == DEFAULT_CADENCE_DAYS
        assert parse_cadence_days("  ") == DEFAULT_CADENCE_DAYS

    def test_parse_cadence_days_sorts_and_deduplicates(self):
        assert parse_cadence_days("5, 2, 2") == (2, 5)
        assert parse_cadence_days("9,2,5") == (2, 5, 9)

    def test_parse_cadence_days_rejects_invalid(self):
        with pytest.raises(ValueError, match="invalid FOLLOWUP_CADENCE_DAYS"):
            parse_cadence_days("dois,5")
        with pytest.raises(ValueError, match="positive day numbers"):
            parse_cadence_days("0,5")

    def test_parse_iso8601_handles_z_and_naive(self):
        parsed = parse_iso8601("2026-09-18T10:00:00Z")
        assert parsed is not None and parsed.tzinfo is timezone.utc
        naive = parse_iso8601("2026-09-18T10:00:00")
        assert naive is not None and naive.tzinfo is timezone.utc

    def test_parse_iso8601_rejects_invalid_input(self):
        assert parse_iso8601("not-a-date") is None
        assert parse_iso8601("") is None
        assert parse_iso8601(None) is None
        assert parse_iso8601(12345) is None
