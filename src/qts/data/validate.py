"""Validarea barelor OHLCV înainte de a ajunge la Strategy (Req 5.4).

O bară este acceptată numai dacă:
- toate câmpurile obligatorii sunt prezente și au tipuri valide;
- prețurile sunt > 0, `low ≤ min(open, close) ≤ max(open, close) ≤ high` și `volume ≥ 0`;
- `ts_close - ts_open == interval_min` minute;
- timpul este strict crescător per instrument (bara nu începe înaintea închiderii celei anterioare).

Barele invalide sunt excluse, iar motivul este returnat cu un cod stabil și jurnalizat.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from decimal import DecimalException
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from qts.core.models import Bar

logger = logging.getLogger(__name__)

REQUIRED_FIELDS: Final[tuple[str, ...]] = tuple(Bar.model_fields)


class BarRejectReason(StrEnum):
    MISSING_FIELD = "BAR_MISSING_FIELD"
    INVALID_FIELD = "BAR_INVALID_FIELD"
    NON_POSITIVE_PRICE = "BAR_NON_POSITIVE_PRICE"
    OHLC_INCONSISTENT = "BAR_OHLC_INCONSISTENT"
    NEGATIVE_VOLUME = "BAR_NEGATIVE_VOLUME"
    INTERVAL_MISMATCH = "BAR_INTERVAL_MISMATCH"
    TIME_NOT_INCREASING = "BAR_TIME_NOT_INCREASING"


class BarVerdict(BaseModel):
    """Rezultatul validării. `bar` este setat numai pentru barele acceptate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    bar: Bar | None = None
    reason: BarRejectReason | None = None
    detail: str = ""

    @classmethod
    def accept(cls, bar: Bar) -> BarVerdict:
        return cls(ok=True, bar=bar)

    @classmethod
    def reject(cls, reason: BarRejectReason, detail: str) -> BarVerdict:
        return cls(ok=False, reason=reason, detail=detail)


def _coerce(raw: Bar | Mapping[str, Any]) -> Bar | BarVerdict:
    if isinstance(raw, Bar):
        return raw
    missing = [f for f in REQUIRED_FIELDS if raw.get(f) is None]
    if missing:
        return BarVerdict.reject(BarRejectReason.MISSING_FIELD, f"câmpuri absente: {missing}")
    try:
        return Bar.model_validate(dict(raw))
    except (ValidationError, ValueError, TypeError, DecimalException) as exc:
        return BarVerdict.reject(BarRejectReason.INVALID_FIELD, str(exc).splitlines()[0])


def validate_bar(raw: Bar | Mapping[str, Any], prev_ts_close: datetime | None = None) -> BarVerdict:
    """Validează o bară izolată; `prev_ts_close` este închiderea ultimei bare acceptate."""
    coerced = _coerce(raw)
    if isinstance(coerced, BarVerdict):
        return coerced
    bar = coerced

    if min(bar.open, bar.high, bar.low, bar.close) <= 0:
        return BarVerdict.reject(BarRejectReason.NON_POSITIVE_PRICE, "prețurile trebuie > 0")
    if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
        return BarVerdict.reject(
            BarRejectReason.OHLC_INCONSISTENT,
            f"o={bar.open} h={bar.high} l={bar.low} c={bar.close}",
        )
    if bar.volume < 0:
        return BarVerdict.reject(BarRejectReason.NEGATIVE_VOLUME, f"volume={bar.volume}")
    if bar.ts_close - bar.ts_open != timedelta(minutes=bar.interval_min):
        return BarVerdict.reject(
            BarRejectReason.INTERVAL_MISMATCH,
            f"ts_close - ts_open = {bar.ts_close - bar.ts_open}, interval_min = {bar.interval_min}",
        )
    if prev_ts_close is not None and bar.ts_open < prev_ts_close:
        return BarVerdict.reject(
            BarRejectReason.TIME_NOT_INCREASING,
            f"ts_open={bar.ts_open.isoformat()} < ts_close anterior={prev_ts_close.isoformat()}",
        )
    return BarVerdict.accept(bar)


def _instrument_of(raw: Bar | Mapping[str, Any]) -> str | None:
    if isinstance(raw, Bar):
        return raw.instrument
    value = raw.get("instrument")
    return value if isinstance(value, str) else None


class BarValidator:
    """Validator cu stare: urmărește ultima bară acceptată per instrument."""

    def __init__(self) -> None:
        self._last_close: dict[str, datetime] = {}

    def last_close(self, instrument: str) -> datetime | None:
        return self._last_close.get(instrument)

    def validate(self, raw: Bar | Mapping[str, Any]) -> BarVerdict:
        instrument = _instrument_of(raw)
        prev = self._last_close.get(instrument) if instrument is not None else None
        verdict = validate_bar(raw, prev)
        if verdict.bar is not None:
            self._last_close[verdict.bar.instrument] = verdict.bar.ts_close
        else:
            logger.warning(
                "bară exclusă instrument=%s reason=%s detail=%s",
                instrument,
                verdict.reason,
                verdict.detail,
            )
        return verdict


def filter_bars(
    bars: Iterable[Bar | Mapping[str, Any]],
) -> tuple[list[Bar], list[BarVerdict]]:
    """Separă barele valide de cele respinse, în ordinea de intrare."""
    validator = BarValidator()
    accepted: list[Bar] = []
    rejected: list[BarVerdict] = []
    for raw in bars:
        verdict = validator.validate(raw)
        if verdict.bar is not None:
            accepted.append(verdict.bar)
        else:
            rejected.append(verdict)
    return accepted, rejected
