"""Agregarea barelor OHLCV în intervale de 5–60 de minute inclusiv (Req 4.5).

Reguli:
- intervalul țintă trebuie să fie un întreg în [5, 60]; altfel `BarIntervalError`;
- la agregarea din bare, intervalul țintă trebuie să fie multiplu al intervalului sursă;
- gălețile sunt aliniate la granițele UTC ale intervalului, măsurate de la epoca Unix
  (`[k·interval, (k+1)·interval)`), deci o tranzacție la exact granița aparține găleții următoare;
- OHLCV se calculează numai în `Decimal`;
- gălețile incomplete (bare sursă lipsă sau găleată încă deschisă) nu sunt emise ca bare
  complete, ci raportate separat ca `PartialBucket`, cu motivul și bara provizorie.

Intrarea trebuie să fie deja validată (`qts.data.validate`) și deduplicată
(`qts.data.normalize`): ordinea nestrict crescătoare sau instrumentele amestecate ridică
`BarAggregationError` în loc să fie corectate tacit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Final

from qts.core.clock import ensure_utc
from qts.core.models import Bar, Frozen, Trade
from qts.core.money import ZERO

MIN_INTERVAL_MIN: Final = 5
MAX_INTERVAL_MIN: Final = 60

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


class BarIntervalError(ValueError):
    """Interval țintă în afara [5, 60] minute sau incompatibil cu intervalul sursă."""


class BarAggregationError(ValueError):
    """Intrare care nu poate fi agregată determinist (ordine, instrument, aliniere)."""


class IncompleteReason(StrEnum):
    MISSING_SOURCE_BARS = "BUCKET_MISSING_SOURCE_BARS"
    NOT_CLOSED = "BUCKET_NOT_CLOSED"


class PartialBucket(Frozen):
    """Găleată care nu poate fi emisă drept bară completă."""

    reason: IncompleteReason
    observed: int  # bare sursă sau tranzacții observate
    expected: int | None  # bare sursă așteptate; None pentru tranzacții
    provisional: Bar


class AggregationResult(Frozen):
    bars: tuple[Bar, ...] = ()
    incomplete: tuple[PartialBucket, ...] = ()


def check_interval(interval_min: int) -> int:
    """Refuză orice interval care nu este un întreg între 5 și 60 de minute inclusiv."""
    if isinstance(interval_min, bool) or not isinstance(interval_min, int):
        raise BarIntervalError(
            f"intervalul trebuie să fie un număr întreg de minute: {interval_min!r}"
        )
    if not MIN_INTERVAL_MIN <= interval_min <= MAX_INTERVAL_MIN:
        raise BarIntervalError(
            f"intervalul {interval_min} min este în afara intervalului permis "
            f"[{MIN_INTERVAL_MIN}, {MAX_INTERVAL_MIN}] minute (Req 4.5)"
        )
    return interval_min


def bucket_start(ts: datetime, interval_min: int) -> datetime:
    """Începutul găleții UTC care conține `ts` (granița inferioară inclusă)."""
    check_interval(interval_min)
    ts = ensure_utc(ts)
    step = timedelta(minutes=interval_min)
    return _EPOCH + ((ts - _EPOCH) // step) * step


def _make_bar(
    instrument: str,
    ts_open: datetime,
    interval_min: int,
    opens: Decimal,
    highs: Decimal,
    lows: Decimal,
    closes: Decimal,
    volume: Decimal,
) -> Bar:
    return Bar(
        instrument=instrument,
        ts_open=ts_open,
        ts_close=ts_open + timedelta(minutes=interval_min),
        interval_min=interval_min,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        volume=volume,
    )


def _single_instrument(instruments: Iterable[str]) -> str:
    seen = set(instruments)
    if len(seen) != 1:
        raise BarAggregationError(f"agregarea cere un singur instrument, primit: {sorted(seen)}")
    return seen.pop()


def _merge(bars: Sequence[Bar], ts_open: datetime, interval_min: int) -> Bar:
    return _make_bar(
        bars[0].instrument,
        ts_open,
        interval_min,
        bars[0].open,
        max(b.high for b in bars),
        min(b.low for b in bars),
        bars[-1].close,
        sum((b.volume for b in bars), ZERO),
    )


def aggregate_bars(bars: Iterable[Bar], target_interval_min: int) -> AggregationResult:
    """Agregă bare mai fine în bare de `target_interval_min` minute.

    O găleată este completă numai dacă conține toate cele `target / source` bare sursă.
    """
    check_interval(target_interval_min)
    items = list(bars)
    if not items:
        return AggregationResult()
    _single_instrument(b.instrument for b in items)

    source_intervals = {b.interval_min for b in items}
    if len(source_intervals) != 1:
        raise BarAggregationError(f"intervale sursă amestecate: {sorted(source_intervals)}")
    source = source_intervals.pop()
    if target_interval_min % source != 0:
        raise BarIntervalError(
            f"intervalul țintă {target_interval_min} min nu este multiplu al intervalului "
            f"sursă {source} min"
        )
    expected = target_interval_min // source
    span = timedelta(minutes=target_interval_min)

    groups: list[tuple[datetime, list[Bar]]] = []
    prev: Bar | None = None
    for bar in items:
        if prev is not None and bar.ts_open < prev.ts_close:
            raise BarAggregationError(
                f"bare neordonate sau suprapuse: {bar.ts_open.isoformat()} < "
                f"{prev.ts_close.isoformat()}"
            )
        start = bucket_start(bar.ts_open, target_interval_min)
        if bar.ts_close > start + span:
            raise BarAggregationError(
                f"bara {bar.ts_open.isoformat()} traversează granița găleții "
                f"{(start + span).isoformat()}"
            )
        if groups and groups[-1][0] == start:
            groups[-1][1].append(bar)
        else:
            groups.append((start, [bar]))
        prev = bar

    complete: list[Bar] = []
    partial: list[PartialBucket] = []
    for start, members in groups:
        merged = _merge(members, start, target_interval_min)
        if len(members) == expected:
            complete.append(merged)
        else:
            partial.append(
                PartialBucket(
                    reason=IncompleteReason.MISSING_SOURCE_BARS,
                    observed=len(members),
                    expected=expected,
                    provisional=merged,
                )
            )
    return AggregationResult(bars=tuple(complete), incomplete=tuple(partial))


def aggregate_trades(
    trades: Iterable[Trade], target_interval_min: int, *, as_of: datetime | None = None
) -> AggregationResult:
    """Agregă tranzacții în bare de `target_interval_min` minute.

    O găleată este închisă când există o tranzacție ulterioară la sau după granița ei
    superioară, ori când `as_of` (timpul ceasului injectat) a atins granița. Gălețile fără
    nicio tranzacție nu produc bare (golurile sunt tratate de consumatori).
    """
    check_interval(target_interval_min)
    items = list(trades)
    if not items:
        return AggregationResult()
    _single_instrument(t.instrument for t in items)
    cutoff = ensure_utc(as_of) if as_of is not None else None
    span = timedelta(minutes=target_interval_min)

    groups: list[tuple[datetime, list[Trade]]] = []
    prev_ts: datetime | None = None
    for trade in items:
        if trade.price <= 0 or trade.size < 0:
            raise BarAggregationError(
                f"tranzacție invalidă la {trade.ts.isoformat()}: price={trade.price} "
                f"size={trade.size}"
            )
        if prev_ts is not None and trade.ts < prev_ts:
            raise BarAggregationError(
                f"tranzacții neordonate: {trade.ts.isoformat()} < {prev_ts.isoformat()}"
            )
        start = bucket_start(trade.ts, target_interval_min)
        if groups and groups[-1][0] == start:
            groups[-1][1].append(trade)
        else:
            groups.append((start, [trade]))
        prev_ts = trade.ts

    complete: list[Bar] = []
    partial: list[PartialBucket] = []
    for idx, (start, members) in enumerate(groups):
        prices = [t.price for t in members]
        bar = _make_bar(
            members[0].instrument,
            start,
            target_interval_min,
            prices[0],
            max(prices),
            min(prices),
            prices[-1],
            sum((t.size for t in members), ZERO),
        )
        closed = idx < len(groups) - 1 or (cutoff is not None and cutoff >= start + span)
        if closed:
            complete.append(bar)
        else:
            partial.append(
                PartialBucket(
                    reason=IncompleteReason.NOT_CLOSED,
                    observed=len(members),
                    expected=None,
                    provisional=bar,
                )
            )
    return AggregationResult(bars=tuple(complete), incomplete=tuple(partial))
