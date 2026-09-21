from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from infra.silence_window import DEFAULT_TIMEZONE, SilenceWindow

SP = ZoneInfo("America/Sao_Paulo")


def at(hour: int) -> datetime:
    return datetime(2026, 9, 20, hour, 0, tzinfo=timezone.utc)


def brt(hour: int) -> datetime:
    return datetime(2026, 9, 20, hour, 0, tzinfo=SP)


class TestSilenceWindow:
    def test_default_timezone_is_sao_paulo(self):
        assert DEFAULT_TIMEZONE == "America/Sao_Paulo"
        assert SilenceWindow().timezone_name == "America/Sao_Paulo"

    def test_default_window_is_8_to_18_in_sao_paulo(self):
        window = SilenceWindow()
        assert window.is_open(at(12)) is True
        assert window.is_open(at(10)) is False
        assert window.is_open(at(22)) is False

    def test_boundaries_are_evaluated_in_local_timezone(self):
        window = SilenceWindow()
        assert window.is_open(brt(8)) is True
        assert window.is_open(brt(7)) is False
        assert window.is_open(brt(17)) is True
        assert window.is_open(brt(18)) is False

    def test_timezone_override_restores_hour_semantics(self):
        window = SilenceWindow(timezone_name="UTC")
        assert window.is_open(at(8)) is True
        assert window.is_open(at(7)) is False
        assert window.is_open(at(18)) is False

    def test_naive_now_is_interpreted_in_window_timezone(self):
        window = SilenceWindow()
        assert window.is_open(datetime(2026, 9, 20, 9, 0)) is True
        assert window.is_open(datetime(2026, 9, 20, 7, 0)) is False

    def test_overnight_window_wraps_midnight(self):
        window = SilenceWindow(start_hour=22, end_hour=6)
        assert window.is_open(brt(23)) is True
        assert window.is_open(brt(5)) is True
        assert window.is_open(brt(12)) is False

    def test_equal_hours_mean_always_open(self):
        window = SilenceWindow(start_hour=0, end_hour=0)
        assert window.is_open(brt(0)) is True
        assert window.is_open(brt(12)) is True
        assert window.is_open(brt(23)) is True

    def test_invalid_hours_are_rejected(self):
        with pytest.raises(ValueError):
            SilenceWindow(start_hour=24)
        with pytest.raises(ValueError):
            SilenceWindow(end_hour=-1)

    def test_unknown_timezone_is_rejected(self):
        with pytest.raises(ValueError, match="unknown timezone"):
            SilenceWindow(timezone_name="Marte/Central")

    def test_from_env_defaults_and_custom_values(self):
        default = SilenceWindow.from_env(environ={})
        assert default.start_hour == 8 and default.end_hour == 18
        assert default.timezone_name == "America/Sao_Paulo"
        custom = SilenceWindow.from_env(
            environ={"SILENCE_WINDOW_START": "9", "SILENCE_WINDOW_END": "17", "TIMEZONE": "UTC"}
        )
        assert custom.is_open(at(9)) is True
        assert custom.is_open(at(17)) is False

    def test_from_env_rejects_non_numeric_values(self):
        with pytest.raises(ValueError, match="must be integers"):
            SilenceWindow.from_env(environ={"SILENCE_WINDOW_START": "nove"})
