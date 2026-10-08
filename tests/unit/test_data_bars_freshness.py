"""Teste unitare pentru `qts.data.bars` (Req 4.5) și `qts.data.freshness` (Req 5.5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from qts.config.schema import DataConfig
from qts.core.clock import SimClock
from qts.core.models import Bar, MarketEvent, Trade
from qts.data.bars import (
    BarAggregationError,
    BarIntervalError,
    IncompleteReason,
    aggregate_bars,
    aggregate_trades,
    bucket_start,
    check_interval,
)
from qts.data.freshness import FreshnessReason, FreshnessTracker

T0 = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)


def src_bar(
    i: int,
    interval: int = 5,
    *,
    o: str = "10",
    h: str = "11",
    lo: str = "9",
    c: str = "10.5",
    v: str = "100",
    instrument: str = "XYZ",
) -> Bar:
    ts_open = T0 + timedelta(minutes=interval * i)
    return Bar(
        instrument=instrument,
        ts_open=ts_open,
        ts_close=ts_open + timedelta(minutes=interval),
        interval_min=interval,
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=v,
    )


def trade(minutes: float, price: str, size: str = "1", instrument: str = "XYZ") -> Trade:
    return Trade(instrument=instrument, ts=T0 + timedelta(minutes=minutes), price=price, size=size)


# --------------------------------------------------------------------------- interval


@pytest.mark.parametrize("interval", [5, 15, 30, 60])
def test_check_interval_accepts_bounds(interval: int) -> None:
    assert check_interval(interval) == interval


@pytest.mark.parametrize("interval", [0, 1, 4, 61, 120, -5])
def test_check_interval_refuses_outside_range(interval: int) -> None:
    with pytest.raises(BarIntervalError, match=r"\[5, 60\]"):
        check_interval(interval)


@pytest.mark.parametrize("interval", [True, 15.0, "15"])
def test_check_interval_refuses_non_int(interval: object) -> None:
    with pytest.raises(BarIntervalError):
        check_interval(interval)  # type: ignore[arg-type]


def test_aggregate_refuses_out_of_range_target() -> None:
    with pytest.raises(BarIntervalError):
        aggregate_bars([src_bar(0)], 90)
    with pytest.raises(BarIntervalError):
        aggregate_trades([trade(0, "10")], 1)


def test_bucket_start_aligned_to_utc_boundaries() -> None:
    assert bucket_start(T0 + timedelta(minutes=14, seconds=59), 15) == T0
    assert bucket_start(T0 + timedelta(minutes=15), 15) == T0 + timedelta(minutes=15)
    assert bucket_start(T0 + timedelta(minutes=59), 60) == T0


# --------------------------------------------------------------------------- bare


def test_aggregate_bars_ohlcv_decimal() -> None:
    bars = [
        src_bar(0, o="10", h="10.4", lo="9.9", c="10.2", v="1.5"),
        src_bar(1, o="10.2", h="10.9", lo="10.1", c="10.6", v="2"),
        src_bar(2, o="10.6", h="10.7", lo="9.7", c="10.0", v="0.25"),
    ]
    result = aggregate_bars(bars, 15)
    assert result.incomplete == ()
    (bar,) = result.bars
    assert bar.ts_open == T0 and bar.ts_close == T0 + timedelta(minutes=15)
    assert bar.interval_min == 15
    assert (bar.open, bar.high, bar.low, bar.close) == (
        Decimal("10"),
        Decimal("10.9"),
        Decimal("9.7"),
        Decimal("10.0"),
    )
    assert bar.volume == Decimal("3.75")
    assert isinstance(bar.volume, Decimal)


def test_aggregate_bars_flags_missing_source_bars() -> None:
    bars = [src_bar(i) for i in (0, 1, 2, 3, 5)]  # lipsește bara 4 din a doua găleată
    result = aggregate_bars(bars, 15)
    assert [b.ts_open for b in result.bars] == [T0]
    (partial,) = result.incomplete
    assert partial.reason is IncompleteReason.MISSING_SOURCE_BARS
    assert (partial.observed, partial.expected) == (2, 3)
    assert partial.provisional.ts_open == T0 + timedelta(minutes=15)


def test_aggregate_bars_requires_multiple_of_source() -> None:
    with pytest.raises(BarIntervalError, match="multiplu"):
        aggregate_bars([src_bar(0, 15)], 20)


def test_aggregate_bars_rejects_disorder_and_mixed_instruments() -> None:
    with pytest.raises(BarAggregationError):
        aggregate_bars([src_bar(1), src_bar(0)], 15)
    with pytest.raises(BarAggregationError):
        aggregate_bars([src_bar(0), src_bar(1, instrument="ABC")], 15)
    with pytest.raises(BarAggregationError):
        aggregate_bars([src_bar(0, 5), src_bar(1, 15)], 30)


def test_aggregate_bars_rejects_bar_straddling_boundary() -> None:
    shifted = Bar(
        instrument="XYZ",
        ts_open=T0 + timedelta(minutes=25),
        ts_close=T0 + timedelta(minutes=35),
        interval_min=10,
        open="1",
        high="1",
        low="1",
        close="1",
        volume="0",
    )
    with pytest.raises(BarAggregationError, match="granița"):
        aggregate_bars([shifted], 30)


def test_aggregate_empty() -> None:
    assert aggregate_bars([], 15).bars == ()
    assert aggregate_trades([], 15).bars == ()


# --------------------------------------------------------------------------- tranzacții


def test_aggregate_trades_buckets_and_open_last_bucket() -> None:
    trades = [
        trade(0, "10", "1"),
        trade(3, "12", "2"),
        trade(7, "9", "1"),
        trade(14.9, "11", "0.5"),
        trade(15, "11.5", "1"),  # granița aparține găleții următoare
    ]
    result = aggregate_trades(trades, 15)
    (bar,) = result.bars
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (
        Decimal("10"),
        Decimal("12"),
        Decimal("9"),
        Decimal("11"),
        Decimal("4.5"),
    )
    (partial,) = result.incomplete
    assert partial.reason is IncompleteReason.NOT_CLOSED
    assert partial.provisional.ts_open == T0 + timedelta(minutes=15)


def test_aggregate_trades_as_of_closes_last_bucket() -> None:
    trades = [trade(1, "10"), trade(2, "10.5")]
    still_open = aggregate_trades(trades, 5, as_of=T0 + timedelta(minutes=4, seconds=59))
    assert still_open.bars == () and len(still_open.incomplete) == 1
    closed = aggregate_trades(trades, 5, as_of=T0 + timedelta(minutes=5))
    assert len(closed.bars) == 1 and closed.incomplete == ()


def test_aggregate_trades_rejects_disorder() -> None:
    with pytest.raises(BarAggregationError):
        aggregate_trades([trade(2, "10"), trade(1, "10")], 5)


# --------------------------------------------------------------------------- prospețime


def event(ts_receipt: datetime, instrument: str = "XYZ") -> MarketEvent:
    return MarketEvent(
        source_id="s",
        instrument=instrument,
        ts_source=ts_receipt,
        ts_receipt=ts_receipt,
        kind="trade",
        payload=Trade(instrument=instrument, ts=ts_receipt, price="10", size="1"),
    )


def test_never_seen_instrument_is_blocked() -> None:
    tracker = FreshnessTracker(SimClock(T0), timedelta(seconds=60))
    verdict = tracker.check("XYZ")
    assert verdict.blocks_order_intent
    assert verdict.reason is FreshnessReason.NEVER_RECEIVED


def test_freshness_threshold_boundary_is_inclusive() -> None:
    clock = SimClock(T0)
    tracker = FreshnessTracker(clock, timedelta(seconds=60))
    tracker.record(event(T0))
    clock.advance_to(T0 + timedelta(seconds=60))
    assert tracker.is_fresh("XYZ")
    clock.advance_to(T0 + timedelta(seconds=60, microseconds=1))
    verdict = tracker.check("XYZ")
    assert not verdict.ok and verdict.reason is FreshnessReason.STALE
    assert verdict.age == timedelta(seconds=60, microseconds=1)


def test_per_instrument_overrides_block_only_affected_instrument() -> None:
    clock = SimClock(T0)
    tracker = FreshnessTracker(clock, timedelta(seconds=300), {"FAST": timedelta(seconds=10)})
    tracker.record(event(T0, "FAST"))
    tracker.record(event(T0, "SLOW"))
    clock.advance_to(T0 + timedelta(seconds=11))
    assert tracker.blocked_instruments(["FAST", "SLOW", "NEW"]) == ["FAST", "NEW"]
    tracker.record(event(T0 + timedelta(seconds=11), "FAST"))
    assert tracker.blocked_instruments(["FAST", "SLOW"]) == []


def test_older_receipt_does_not_regress() -> None:
    clock = SimClock(T0 + timedelta(seconds=30))
    tracker = FreshnessTracker(clock, timedelta(seconds=60))
    tracker.record_receipt("XYZ", T0 + timedelta(seconds=30))
    tracker.record_receipt("XYZ", T0)
    assert tracker.last_receipt("XYZ") == T0 + timedelta(seconds=30)


def test_receipt_in_future_fails_closed() -> None:
    tracker = FreshnessTracker(SimClock(T0), timedelta(seconds=60))
    tracker.record_receipt("XYZ", T0 + timedelta(seconds=1))
    assert tracker.check("XYZ").reason is FreshnessReason.RECEIPT_IN_FUTURE


def test_from_config_uses_default_and_overrides() -> None:
    cfg = DataConfig(
        source_id="s",
        bar_interval_min=15,
        default_freshness_seconds=1800,
        freshness_seconds={"XYZ": 120},
    )
    tracker = FreshnessTracker.from_config(cfg, SimClock(T0))
    assert tracker.threshold("XYZ") == timedelta(seconds=120)
    assert tracker.threshold("OTHER") == timedelta(seconds=1800)


def test_non_positive_threshold_refused() -> None:
    with pytest.raises(ValueError):
        FreshnessTracker(SimClock(T0), timedelta(0))
    with pytest.raises(ValueError):
        FreshnessTracker(SimClock(T0), timedelta(seconds=1), {"X": timedelta(seconds=-1)})
