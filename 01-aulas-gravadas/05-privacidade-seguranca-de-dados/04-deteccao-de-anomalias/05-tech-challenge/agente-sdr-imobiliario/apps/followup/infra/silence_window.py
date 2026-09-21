from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TIMEZONE = "America/Sao_Paulo"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SilenceWindow:
    """FR8.3 — janela de silêncio configurável em horas LOCAIS (fim exclusivo).

    O fuso é configurável via `timezone_name` e o padrão é
    `America/Sao_Paulo` (público do produto): com os defaults 8–18, a janela
    abre 08h e fecha 18h no horário de São Paulo (fim exclusivo). Fora da
    janela nenhum follow-up é enviado: o tick marca os leads como `deferred`
    e eles permanecem pendentes para o próximo tick dentro da janela.
    Suporta janela que cruza a meia-noite (start > end) e janela 24h
    (start == end). O clock é injetável para testar os boundaries.
    """

    def __init__(
        self,
        start_hour: int = 8,
        end_hour: int = 18,
        now_fn: Callable[[], datetime] = utc_now,
        timezone_name: str = DEFAULT_TIMEZONE,
    ) -> None:
        for hour in (start_hour, end_hour):
            if not 0 <= hour <= 23:
                raise ValueError(f"silence window hours must be 0-23, got {hour}")
        try:
            self._tz = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone: {timezone_name}") from exc
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.timezone_name = timezone_name
        self.now_fn = now_fn

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, now_fn: Callable[[], datetime] = utc_now) -> "SilenceWindow":
        env = environ if environ is not None else os.environ
        try:
            start = int(env.get("SILENCE_WINDOW_START", "8"))
            end = int(env.get("SILENCE_WINDOW_END", "18"))
        except ValueError as exc:
            raise ValueError("SILENCE_WINDOW_START/SILENCE_WINDOW_END must be integers 0-23") from exc
        return cls(
            start_hour=start,
            end_hour=end,
            now_fn=now_fn,
            timezone_name=env.get("TIMEZONE", DEFAULT_TIMEZONE),
        )

    def is_open(self, now: datetime | None = None) -> bool:
        moment = now or self.now_fn()
        if moment.tzinfo is None:
            local = moment.replace(tzinfo=self._tz)
        else:
            local = moment.astimezone(self._tz)
        hour = local.hour
        if self.start_hour == self.end_hour:
            return True
        if self.start_hour < self.end_hour:
            return self.start_hour <= hour < self.end_hour
        return hour >= self.start_hour or hour < self.end_hour
