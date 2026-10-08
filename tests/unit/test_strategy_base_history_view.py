"""Teste pentru `strategy.base` și `strategy.history_view` (Req 6.1, 6.2, 6.6, 7.2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from qts.core.clock import NaiveDatetimeError
from qts.core.models import Bar, Signal, canonical_json
from qts.strategy.base import (
    REASON_INVALID_INPUT,
    Strategy,
    StrategyState,
    bar_data_id,
    build_signal,
    make_signal_id,
    none_signal,
)
from qts.strategy.history_view import HistoryView, ImmutableViewError, LookAheadError

T0 = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)


def make_bar(instrument: str, i: int, close: str = "100") -> Bar:
    ts_open = T0 + timedelta(minutes=15 * i)
    return Bar(
        instrument=instrument,
        ts_open=ts_open,
        ts_close=ts_open + timedelta(minutes=15),
        interval_min=15,
        open=close,
        high=close,
        low=close,
        close=close,
        volume="10",
    )


def series(instrument: str, n: int) -> list[Bar]:
    return [make_bar(instrument, i, close=str(100 + i)) for i in range(n)]


@pytest.fixture
def full() -> dict[str, list[Bar]]:
    return {"AAA": series("AAA", 10), "BBB": series("BBB", 5)}


# --------------------------------------------------------------------------- HistoryView


def test_view_exposes_only_bars_up_to_cursor(full: dict[str, list[Bar]]) -> None:
    cursor = full["AAA"][3].ts_close
    view = HistoryView(full, cursor)
    assert view.count("AAA") == 4
    assert view.current("AAA") == full["AAA"][3]
    assert view.bar("AAA", -1) == full["AAA"][3]
    assert view["AAA", 0] == full["AAA"][0]
    assert view.last("AAA", 2) == tuple(full["AAA"][2:4])
    assert view.last("AAA", 100) == tuple(full["AAA"][:4])
    assert view.closes("AAA", 3) == (Decimal("101"), Decimal("102"), Decimal("103"))
    assert view.slice("AAA", 1, 4) == tuple(full["AAA"][1:4])
    assert all(b.ts_close <= cursor for b in view.last("BBB", 100))


def test_future_access_raises_lookahead(full: dict[str, list[Bar]]) -> None:
    view = HistoryView(full, full["AAA"][3].ts_close)
    with pytest.raises(LookAheadError):
        view.bar("AAA", 4)
    with pytest.raises(LookAheadError):
        view["AAA", 9]
    with pytest.raises(LookAheadError):
        view.slice("AAA", 0, 5)
    with pytest.raises(LookAheadError):
        view.until("AAA", view.cursor + timedelta(seconds=1))


def test_until_and_bounds(full: dict[str, list[Bar]]) -> None:
    view = HistoryView(full, full["AAA"][5].ts_close)
    assert view.until("AAA", full["AAA"][2].ts_close) == tuple(full["AAA"][:3])
    with pytest.raises(IndexError):
        view.bar("AAA", -7)
    with pytest.raises(KeyError):
        view.count("ZZZ")
    with pytest.raises(NaiveDatetimeError):
        view.until("AAA", datetime(2024, 1, 2, 9, 0))  # noqa: DTZ001 - naive, intenționat


def test_cursor_before_history_is_empty(full: dict[str, list[Bar]]) -> None:
    view = HistoryView(full, T0)
    assert view.count("AAA") == 0
    assert view.current("AAA") is None
    assert view.last("AAA", 3) == ()
    with pytest.raises(LookAheadError):
        view.bar("AAA", 0)


def test_view_is_immutable(full: dict[str, list[Bar]]) -> None:
    view = HistoryView(full, full["AAA"][3].ts_close)
    with pytest.raises(ImmutableViewError):
        view._cursor = full["AAA"][9].ts_close
    with pytest.raises(ImmutableViewError):
        view.extra = 1
    with pytest.raises(ImmutableViewError):
        del view._series
    assert isinstance(view.last("AAA", 2), tuple)
    with pytest.raises(TypeError):
        view._visible["AAA"] = 10  # type: ignore[index]
    assert view.count("AAA") == 4


def test_source_mutation_does_not_leak(full: dict[str, list[Bar]]) -> None:
    view = HistoryView(full, full["AAA"][3].ts_close)
    full["AAA"].clear()
    assert view.count("AAA") == 4


def test_advance_returns_new_view(full: dict[str, list[Bar]]) -> None:
    v1 = HistoryView(full, full["AAA"][1].ts_close)
    v2 = v1.advance(full["AAA"][6].ts_close)
    assert v1.count("AAA") == 2
    assert v2.count("AAA") == 7
    assert v2.count("BBB") == 5


def test_cursor_normalised_to_utc(full: dict[str, list[Bar]]) -> None:
    tz = timezone(timedelta(hours=2))
    view = HistoryView(full, full["AAA"][3].ts_close.astimezone(tz))
    assert view.cursor.tzinfo == UTC
    assert view.count("AAA") == 4
    with pytest.raises(NaiveDatetimeError):
        HistoryView(full, datetime(2024, 1, 2, 10, 0))  # noqa: DTZ001 - naive, intenționat


def test_from_bars_groups_and_sorts(full: dict[str, list[Bar]]) -> None:
    mixed = list(reversed(full["AAA"])) + full["BBB"]
    view = HistoryView.from_bars(mixed, full["AAA"][9].ts_close)
    assert view.instruments() == ("AAA", "BBB")
    assert view.last("AAA", 10) == tuple(full["AAA"])


def test_invalid_series_rejected(full: dict[str, list[Bar]]) -> None:
    with pytest.raises(ValueError, match="strict"):
        HistoryView({"AAA": list(reversed(full["AAA"]))}, T0)
    with pytest.raises(ValueError, match="aparține"):
        HistoryView({"AAA": full["BBB"]}, T0)


# --------------------------------------------------------------------------- Strategy base


class CounterState(StrategyState):
    bars_seen: int = 0


class EchoStrategy:
    """Strategie minimală de test: contorizează barele și nu acționează."""

    strategy_id = "echo"
    version = "1.0.0"

    def initial_state(self) -> StrategyState:
        return CounterState()

    def on_bar(
        self, bar: Bar, view: HistoryView, state: StrategyState
    ) -> tuple[Signal, StrategyState]:
        assert isinstance(state, CounterState)
        signal = none_signal(
            self,
            bar,
            reason_code="NO_RULE_MATCHED",
            config_snapshot_id="cfg",
            inputs={"close": bar.close},
            rules_evaluated=["noop"],
        )
        return signal, CounterState(bars_seen=state.bars_seen + 1)


def test_strategy_protocol_and_determinism(full: dict[str, list[Bar]]) -> None:
    strat = EchoStrategy()
    assert isinstance(strat, Strategy)

    def run() -> list[str]:
        state = strat.initial_state()
        out: list[str] = []
        view = HistoryView(full, T0)
        for bar in full["AAA"]:
            view = view.advance(bar.ts_close)
            sig, state = strat.on_bar(bar, view, state)
            out.append(canonical_json(sig))
        out.append(canonical_json(state))
        return out

    first, second = run(), run()
    assert first == second
    assert first[-1] == '{"bars_seen":10}'


def test_state_is_frozen() -> None:
    state = CounterState(bars_seen=1)
    with pytest.raises(ValidationError):
        state.bars_seen = 2  # type: ignore[misc]


def test_none_signal_fields() -> None:
    bar = make_bar("AAA", 0)
    sig = none_signal(
        EchoStrategy(), bar, reason_code=REASON_INVALID_INPUT, config_snapshot_id="cfg-1"
    )
    assert sig.action == "NONE"
    assert sig.stop_price is None
    assert sig.reason_code == REASON_INVALID_INPUT
    assert sig.strategy_id == "echo" and sig.strategy_version == "1.0.0"
    assert sig.ts == bar.ts_close
    assert sig.data_ids == [bar_data_id(bar)]
    assert sig.signal_id == make_signal_id("echo", "1.0.0", bar)


def test_build_entry_signal_requires_stop() -> None:
    bar = make_bar("AAA", 0)
    sig = build_signal(
        EchoStrategy(),
        bar,
        action="ENTER_LONG",
        reason_code="ENTRY",
        config_snapshot_id="cfg",
        stop_price=Decimal("95"),
    )
    assert sig.stop_price == Decimal("95")
    with pytest.raises(ValidationError):
        build_signal(
            EchoStrategy(), bar, action="ENTER_LONG", reason_code="ENTRY", config_snapshot_id="c"
        )


def test_append_extends_series_and_moves_cursor_without_mutation() -> None:
    a = series("AAA", 3)
    base = HistoryView({"AAA": a[:2]}, a[1].ts_close)
    grown = base.append(a[2]).append(make_bar("BBB", 2))
    assert base.count("AAA") == 2  # vederea inițială rămâne neschimbată
    assert grown.cursor == a[2].ts_close
    assert grown.count("AAA") == 3 and grown.current("AAA") == a[2]
    assert grown.current("BBB") == make_bar("BBB", 2)
    with pytest.raises(ValueError):
        grown.append(a[1])  # nu strict după ultima bară a instrumentului
