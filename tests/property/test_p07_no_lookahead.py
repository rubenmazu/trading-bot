"""P7: absența look-ahead-ului (Req 7.2).

Pentru orice set de date, modificarea barelor de după momentul *t* (înlocuire, adăugare,
eliminare, inclusiv bare viitoare invalide) nu schimbă niciun semnal și nicio stare emise la
momente ≤ *t*. Strategia rulează bară cu bară cu o `HistoryView` care conține seria COMPLETĂ
(inclusiv viitorul), avansată la `ts_close` al fiecărei bare. Complementar, o strategie care
încearcă să citească viitorul prin vedere primește `LookAheadError`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qts.core.models import Bar, Signal, canonical_json
from qts.strategy.base import REASON_NO_RULE_MATCHED, StrategyState, none_signal
from qts.strategy.history_view import HistoryView, LookAheadError
from qts.strategy.mean_reversion import MeanReversionParams, MeanReversionStrategy
from tests.fixtures.synthetic import DEFAULT_START, SCENARIOS, SyntheticSpec, generate_bars

INSTRUMENT = "XYZ"
STEP = timedelta(minutes=15)
SNAPSHOT_ID = "snap-p7"


# ----------------------------------------------------------------------------- generatoare


def _price(lo: str = "1", hi: str = "200") -> st.SearchStrategy[Decimal]:
    return st.decimals(
        min_value=Decimal(lo),
        max_value=Decimal(hi),
        places=2,
        allow_nan=False,
        allow_infinity=False,
    )


@st.composite
def _arbitrary_series(draw: st.DrawFn) -> list[Bar]:
    """Serie validă de bare de 15 minute, cu goluri arbitrare între bare."""
    closes = draw(st.lists(_price(), min_size=1, max_size=60))
    wicks = draw(
        st.lists(
            st.tuples(_price("0", "3"), _price("0", "3")),
            min_size=len(closes),
            max_size=len(closes),
        )
    )
    gaps = draw(st.lists(st.integers(0, 3), min_size=len(closes), max_size=len(closes)))
    bars: list[Bar] = []
    ts_open = DEFAULT_START
    prev_close = closes[0]
    for close, (up, down), gap in zip(closes, wicks, gaps, strict=True):
        ts_open += gap * STEP
        o = prev_close
        low = max(min(o, close) - down, Decimal("0.01"))
        bars.append(
            Bar(
                instrument=INSTRUMENT,
                ts_open=ts_open,
                ts_close=ts_open + STEP,
                interval_min=15,
                open=o,
                high=max(o, close) + up,
                low=min(low, o, close),
                close=close,
                volume=Decimal(draw(st.integers(0, 10_000))),
            )
        )
        prev_close = close
        ts_open += STEP
    return bars


@st.composite
def _synthetic_series(draw: st.DrawFn) -> list[Bar]:
    spec = SyntheticSpec(
        scenario=draw(st.sampled_from(SCENARIOS)),
        seed=draw(st.integers(0, 2**32 - 1)),
        n_bars=draw(st.integers(1, 80)),
        instrument=INSTRUMENT,
        volatility=draw(st.sampled_from((0.001, 0.005, 0.02))),
    )
    return generate_bars(spec)


_series = st.one_of(_arbitrary_series(), _synthetic_series())


@st.composite
def _params(draw: st.DrawFn) -> MeanReversionParams:
    entry_z = draw(_price("0.30", "3.00"))
    exit_z = draw(_price(str(-entry_z + Decimal("0.01")), "1.00"))
    return MeanReversionParams.from_mapping(
        {
            "lookback": draw(st.integers(2, 25)),
            "entry_z": str(entry_z),
            "exit_z": str(exit_z),
            "stop_k": str(draw(_price("0.10", "4.00"))),
        }
    )


@st.composite
def _future_bar_values(draw: st.DrawFn) -> dict[str, object]:
    """Valori arbitrare pentru o bară viitoare, adesea invalide (OHLC incoerent, ≤ 0)."""
    any_price = _price("-50", "500")
    return {
        "interval_min": draw(st.sampled_from((15, 15, 15, 5, 30, 60))),
        "open": draw(any_price),
        "high": draw(any_price),
        "low": draw(any_price),
        "close": draw(any_price),
        "volume": Decimal(draw(st.integers(-10, 10_000))),
    }


@st.composite
def _mutated_tail(draw: st.DrawFn, original_tail: list[Bar], t_close: datetime) -> list[Bar]:
    """Coada nouă (strict după *t*): ștergere, perturbare, înlocuire completă sau adăugare."""
    mode = draw(st.sampled_from(("drop", "perturb", "replace", "append")))
    if mode == "drop":
        keep = draw(st.integers(0, max(0, len(original_tail) - 1)))
        return original_tail[:keep]
    if mode == "perturb" and original_tail:
        out: list[Bar] = []
        for bar in original_tail:
            if draw(st.booleans()):
                out.append(bar.model_copy(update=draw(_future_bar_values())))
            else:
                out.append(bar)
        return out
    base = original_tail if mode == "append" else []
    last_close = base[-1].ts_close if base else t_close
    n_new = draw(st.integers(1 if mode == "append" else 0, 20))
    out = list(base)
    for _ in range(n_new):
        values = draw(_future_bar_values())
        # Timpi strict crescători (cerința de structură a `HistoryView`); durata poate fi aberantă.
        ts_close = last_close + draw(st.integers(1, 4)) * STEP
        width = draw(st.sampled_from((STEP, STEP, timedelta(minutes=7), timedelta(hours=1))))
        out.append(
            Bar(instrument=INSTRUMENT, ts_open=ts_close - width, ts_close=ts_close, **values)
        )
        last_close = ts_close
    return out


# ----------------------------------------------------------------------------- rulare


def _run(strategy: MeanReversionStrategy, bars: list[Bar]) -> list[tuple[str, str, str]]:
    """Rulează bară cu bară cu vederea peste seria completă; întoarce (ts, semnal, stare)."""
    full = HistoryView({INSTRUMENT: bars}, bars[0].ts_close)
    state: StrategyState = strategy.initial_state()
    out: list[tuple[str, str, str]] = []
    for bar in bars:
        signal, state = strategy.on_bar(bar, full.advance(bar.ts_close), state)
        out.append((bar.ts_close.isoformat(), canonical_json(signal), canonical_json(state)))
    return out


@given(data=st.data(), bars=_series, params=_params())
def test_property_7_future_mutation_does_not_change_past_signals(
    data: st.DataObject, bars: list[Bar], params: MeanReversionParams
) -> None:
    """**Validates: Requirements 7.2**"""
    cut = data.draw(st.integers(0, len(bars) - 1), label="cut")
    t = bars[cut].ts_close
    tail = data.draw(_mutated_tail(bars[cut + 1 :], t), label="tail")
    mutated = bars[: cut + 1] + tail
    assert all(b.ts_close > t for b in tail)

    strategy = MeanReversionStrategy(params, config_snapshot_id=SNAPSHOT_ID)
    original_run = _run(strategy, bars)
    mutated_run = _run(strategy, mutated)

    past_original = [row for row in original_run if datetime.fromisoformat(row[0]) <= t]
    past_mutated = [row for row in mutated_run if datetime.fromisoformat(row[0]) <= t]
    assert len(past_original) == cut + 1
    assert past_original == past_mutated


# ----------------------------------------------------------------------------- strategie trișoare


class _PeekingStrategy:
    """Strategie deliberat incorectă: încearcă să citească bara/timpul de după cursor."""

    strategy_id: str = "peeking"
    version: str = "0.0.0"

    def __init__(self, probe: str) -> None:
        self.probe = probe

    def initial_state(self) -> StrategyState:
        return StrategyState()

    def on_bar(
        self, bar: Bar, view: HistoryView, state: StrategyState
    ) -> tuple[Signal, StrategyState]:
        n = view.count(bar.instrument)
        if self.probe == "bar":
            view.bar(bar.instrument, n)
        elif self.probe == "getitem":
            view[bar.instrument, n]
        elif self.probe == "slice":
            view.slice(bar.instrument, 0, n + 1)
        else:
            view.until(bar.instrument, view.cursor + timedelta(microseconds=1))
        signal = none_signal(
            self, bar, reason_code=REASON_NO_RULE_MATCHED, config_snapshot_id=SNAPSHOT_ID
        )
        return signal, state


@given(
    data=st.data(),
    bars=_series,
    probe=st.sampled_from(("bar", "getitem", "slice", "until")),
)
def test_property_7_future_access_raises_lookahead_error(
    data: st.DataObject, bars: list[Bar], probe: str
) -> None:
    """**Validates: Requirements 7.2**"""
    cut = data.draw(st.integers(0, len(bars) - 1), label="cut")
    bar = bars[cut]
    view = HistoryView({INSTRUMENT: bars}, bars[0].ts_close).advance(bar.ts_close)
    strategy = _PeekingStrategy(probe)
    with pytest.raises(LookAheadError):
        strategy.on_bar(bar, view, strategy.initial_state())
    # Accesul onest rămâne limitat la trecut: ultima bară vizibilă este bara curentă.
    assert view.current(INSTRUMENT) == bar
    assert view.last(INSTRUMENT, len(bars) + 5) == tuple(bars[: cut + 1])
