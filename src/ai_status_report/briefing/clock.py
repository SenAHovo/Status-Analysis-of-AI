"""Injectable clock for briefing and later run orchestration.

Production reads the system clock in the configured zone; tests inject a
fixed moment so calendar and leap-day boundaries are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol
from zoneinfo import ZoneInfo


class Clock(Protocol):
    """Reads the current local time in the user's zone."""

    zone_name: str

    def now(self) -> datetime: ...
    def today(self) -> date: ...


@dataclass(frozen=True)
class SystemClock:
    zone_name: str

    def now(self) -> datetime:
        return datetime.now(ZoneInfo(self.zone_name))

    def today(self) -> date:
        return self.now().date()


@dataclass(frozen=True)
class FixedClock:
    """Deterministic clock for tests and recovery scenarios."""

    zone_name: str
    moment: datetime

    def __post_init__(self):
        if self.moment.tzinfo is None:
            raise ValueError("FixedClock moment must carry a timezone")

    def now(self) -> datetime:
        return self.moment

    def today(self) -> date:
        return self.moment.date()
