from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

DEFAULT_CADENCE_DAYS = (2, 5, 9)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_cadence_days(raw: str | None) -> tuple[int, ...]:
    """Dias da cadência (FR8.1) — configurável via `FOLLOWUP_CADENCE_DAYS`.

    Valores separados por vírgula, deduplicados e ordenados; vazio usa o
    padrão 2/5/9. Entrada inválida (não numérica, dia < 1) levanta `ValueError`.
    """
    if not raw or not raw.strip():
        return DEFAULT_CADENCE_DAYS
    try:
        days = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError(f"invalid FOLLOWUP_CADENCE_DAYS {raw!r}: day numbers expected") from exc
    if not days or any(day < 1 for day in days):
        raise ValueError(f"invalid FOLLOWUP_CADENCE_DAYS {raw!r}: positive day numbers required")
    return days


def parse_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class CadenceCalculator:
    """Cadência dia 2/5/9 (FR8.1) — clock injetável.

    Na POC os wait states do Step Functions são simulados: o tick diário do
    EventBridge resolve, por lead, o passo vencido como o maior dia da
    cadência já alcançado (`delta >= dia`). Um tick perdido se autocorrige no
    tick seguinte; a supressão de passos repetidos é responsabilidade da
    DuplicateGuard. Após o último dia a cadência expira (lead droppable).
    """

    def __init__(
        self,
        days: Iterable[int] = DEFAULT_CADENCE_DAYS,
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        normalized = tuple(sorted({int(day) for day in days}))
        if not normalized or any(day < 1 for day in normalized):
            raise ValueError("cadence days must be positive day numbers")
        self.days = normalized
        self.now_fn = now_fn

    @property
    def last_day(self) -> int:
        return self.days[-1]

    def due_step(self, created_at: Any) -> int | None:
        """Passo vencido: maior dia da cadência já alcançado; None se ainda não."""
        created = parse_iso8601(created_at)
        if created is None:
            return None
        delta = (self.now_fn() - created).days
        eligible = [day for day in self.days if day <= delta]
        return eligible[-1] if eligible else None

    def is_expired(self, created_at: Any) -> bool:
        """Delta além do último dia da cadência (ou created_at inválido)."""
        created = parse_iso8601(created_at)
        if created is None:
            return True
        return (self.now_fn() - created).days > self.last_day

    def next_step(self, step: int) -> int | None:
        later = [day for day in self.days if day > step]
        return later[0] if later else None
