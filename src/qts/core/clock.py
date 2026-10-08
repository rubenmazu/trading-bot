"""Ceasuri injectabile. Nucleul nu apelează niciodată direct `datetime.now`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


class NaiveDatetimeError(ValueError):
    """Ridicată pentru `datetime` fără fus orar sau cu alt fus decât UTC."""


def ensure_utc(ts: datetime) -> datetime:
    """Acceptă numai `datetime` cu fus orar și îl convertește în UTC."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise NaiveDatetimeError(f"datetime fără fus orar: {ts!r}")
    return ts.astimezone(UTC)


class Clock(Protocol):
    def now(self) -> datetime: ...


class WallClock:
    """Ceas real, în UTC (Shadow / Demo)."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class SimClock:
    """Ceas simulat, avansat numai de evenimente (Backtest). Nu poate merge înapoi."""

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, ts: datetime) -> None:
        ts = ensure_utc(ts)
        if ts < self._now:
            raise ValueError(f"SimClock nu poate merge înapoi: {ts} < {self._now}")
        self._now = ts
