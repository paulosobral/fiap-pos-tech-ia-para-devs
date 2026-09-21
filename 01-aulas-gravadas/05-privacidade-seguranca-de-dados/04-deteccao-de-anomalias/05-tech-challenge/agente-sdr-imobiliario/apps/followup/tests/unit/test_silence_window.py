from datetime import datetime, timezone

import pytest

from infra.silence_window import SilenceWindow


def at(hour: int) -> datetime:
    return datetime(2026, 9, 20, hour, 0, tzinfo=timezone.utc)


class TestSilenceWindow:
    def test_open_inside_default_window(self):
        assert SilenceWindow().is_open(at(10)) is True

    def test_closed_outside_default_window(self):
        assert SilenceWindow().is_open(at(3)) is False
        assert SilenceWindow().is_open(at(20)) is False

    def test_start_boundary_is_inclusive(self):
        assert SilenceWindow().is_open(at(8)) is True

    def test_end_boundary_is_exclusive(self):
        assert SilenceWindow().is_open(at(18)) is False

    def test_overnight_window_wraps_midnight(self):
        window = SilenceWindow(start_hour=22, end_hour=6)
        assert window.is_open(at(23)) is True
        assert window.is_open(at(5)) is True
        assert window.is_open(at(12)) is False

    def test_equal_hours_mean_always_open(self):
        window = SilenceWindow(start_hour=0, end_hour=0)
        assert window.is_open(at(0)) is True
        assert window.is_open(at(12)) is True
        assert window.is_open(at(23)) is True

    def test_invalid_hours_are_rejected(self):
        with pytest.raises(ValueError):
            SilenceWindow(start_hour=24)
        with pytest.raises(ValueError):
            SilenceWindow(end_hour=-1)

    def test_from_env_defaults_and_custom_values(self):
        default = SilenceWindow.from_env(environ={})
        assert default.start_hour == 8 and default.end_hour == 18
        custom = SilenceWindow.from_env(
            environ={"SILENCE_WINDOW_START": "9", "SILENCE_WINDOW_END": "17"}
        )
        assert custom.is_open(at(9)) is True
        assert custom.is_open(at(17)) is False

    def test_from_env_rejects_non_numeric_values(self):
        with pytest.raises(ValueError, match="must be integers"):
            SilenceWindow.from_env(environ={"SILENCE_WINDOW_START": "nove"})
