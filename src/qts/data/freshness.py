"""Prospețimea datelor per instrument (Req 5.5).

`FreshnessTracker` blochează `Order_Intent` pentru un instrument când
`now - ts_receipt_last > freshness_threshold[instrument]`. Comportamentul este fail-closed:
- un instrument pentru care nu s-a recepționat niciun eveniment este considerat expirat;
- un `ts_receipt` aflat în viitorul ceasului injectat indică o anomalie de ceas și blochează.

Timpul vine exclusiv din `Clock` injectat; modulul nu citește ceasul de perete.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from enum import StrEnum

from qts.config.schema import DataConfig
from qts.core.clock import Clock, ensure_utc
from qts.core.models import Frozen, MarketEvent


class FreshnessReason(StrEnum):
    NEVER_RECEIVED = "DATA_NEVER_RECEIVED"
    STALE = "DATA_STALE"
    RECEIPT_IN_FUTURE = "DATA_RECEIPT_IN_FUTURE"


class FreshnessVerdict(Frozen):
    instrument: str
    ok: bool
    threshold: timedelta
    last_receipt: datetime | None = None
    age: timedelta | None = None
    reason: FreshnessReason | None = None

    @property
    def blocks_order_intent(self) -> bool:
        return not self.ok


def _positive(threshold: timedelta, label: str) -> timedelta:
    if threshold <= timedelta(0):
        raise ValueError(f"pragul de prospețime pentru {label} trebuie să fie > 0: {threshold}")
    return threshold


class FreshnessTracker:
    """Urmărește ultimul `ts_receipt` per instrument și decide blocarea `Order_Intent`."""

    def __init__(
        self,
        clock: Clock,
        default_threshold: timedelta,
        overrides: Mapping[str, timedelta] | None = None,
    ) -> None:
        self._clock = clock
        self._default = _positive(default_threshold, "implicit")
        self._overrides = {k: _positive(v, k) for k, v in (overrides or {}).items()}
        self._last_receipt: dict[str, datetime] = {}

    @classmethod
    def from_config(cls, config: DataConfig, clock: Clock) -> FreshnessTracker:
        return cls(
            clock,
            timedelta(seconds=config.default_freshness_seconds),
            {k: timedelta(seconds=v) for k, v in config.freshness_seconds.items()},
        )

    def threshold(self, instrument: str) -> timedelta:
        return self._overrides.get(instrument, self._default)

    def last_receipt(self, instrument: str) -> datetime | None:
        return self._last_receipt.get(instrument)

    def record_receipt(self, instrument: str, ts_receipt: datetime) -> None:
        """Înregistrează o recepție; un timp mai vechi nu reduce prospețimea deja observată."""
        ts = ensure_utc(ts_receipt)
        current = self._last_receipt.get(instrument)
        if current is None or ts > current:
            self._last_receipt[instrument] = ts

    def record(self, event: MarketEvent) -> None:
        self.record_receipt(event.instrument, event.ts_receipt)

    def check(self, instrument: str) -> FreshnessVerdict:
        threshold = self.threshold(instrument)
        last = self._last_receipt.get(instrument)
        if last is None:
            return FreshnessVerdict(
                instrument=instrument,
                ok=False,
                threshold=threshold,
                reason=FreshnessReason.NEVER_RECEIVED,
            )
        age = ensure_utc(self._clock.now()) - last
        reason: FreshnessReason | None = None
        if age < timedelta(0):
            reason = FreshnessReason.RECEIPT_IN_FUTURE
        elif age > threshold:
            reason = FreshnessReason.STALE
        return FreshnessVerdict(
            instrument=instrument,
            ok=reason is None,
            threshold=threshold,
            last_receipt=last,
            age=age,
            reason=reason,
        )

    def is_fresh(self, instrument: str) -> bool:
        return self.check(instrument).ok

    def blocked_instruments(self, instruments: Iterable[str]) -> list[str]:
        """Instrumentele, în ordinea primită, pentru care `Order_Intent` este blocat."""
        return [i for i in instruments if not self.check(i).ok]
