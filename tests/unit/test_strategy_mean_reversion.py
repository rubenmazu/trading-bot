"""Teste pentru strategia de referință mean-reversion (Req 6.1, 6.3, 6.4)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from qts.config.schema import StrategyConfig
from qts.core.models import Bar, Signal, canonical_json
from qts.strategy.base import (
    REASON_INSUFFICIENT_HISTORY,
    REASON_NO_RULE_MATCHED,
    Strategy,
    StrategyState,
    bar_data_id,
)
from qts.strategy.history_view import HistoryView
from qts.strategy.mean_reversion import (
    REASON_ENTRY_ZSCORE,
    REASON_EXIT_MEAN,
    REASON_EXIT_STOP,
    REASON_HISTORY_MISMATCH,
    REASON_INVALID_BAR,
    REASON_INVALID_INTERVAL,
    REASON_ZERO_VOLATILITY,
    MeanReversionParams,
    MeanReversionState,
    MeanReversionStrategy,
)

T0 = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)
SQRT6 = Decimal(6).sqrt()


def bar(i: int, close: str, low: str | None = None, *, interval: int = 15) -> Bar:
    ts_open = T0 + timedelta(minutes=15 * i)
    c = Decimal(close)
    lo = Decimal(low) if low is not None else c
    return Bar(
        instrument="AAA",
        ts_open=ts_open,
        ts_close=ts_open + timedelta(minutes=interval),
        interval_min=interval,
        open=c,
        high=c,
        low=lo,
        close=c,
        volume="10",
    )


def strategy() -> MeanReversionStrategy:
    params = MeanReversionParams.from_mapping(
        {"lookback": 4, "entry_z": "1.5", "exit_z": "0", "stop_k": "1"}
    )
    return MeanReversionStrategy(params, config_snapshot_id="cfg-1")


def run(
    strat: MeanReversionStrategy, bars: list[Bar], state: StrategyState | None = None
) -> list[tuple[Signal, StrategyState]]:
    """Rulează strategia bară cu bară, cu vederea avansată la fiecare `ts_close`."""
    state = state if state is not None else strat.initial_state()
    view = HistoryView({"AAA": bars}, T0)
    out: list[tuple[Signal, StrategyState]] = []
    for b in bars:
        view = view.advance(b.ts_close)
        sig, state = strat.on_bar(b, view, state)
        out.append((sig, state))
    return out


# Ferestrele: [101, 99, 101, 95] → mean 99, std √6, z = -4/√6 ≈ -1.633 ≤ -1.5.
ENTRY_BARS = [bar(0, "101"), bar(1, "99"), bar(2, "101"), bar(3, "95")]
EXPECTED_STOP = Decimal("92.5505")  # floor(95 - √6) la 0.0001
FLAT_STATE = MeanReversionState()


def test_satisfies_protocol() -> None:
    assert isinstance(strategy(), Strategy)


def test_entry_on_low_zscore() -> None:
    results = run(strategy(), ENTRY_BARS)
    for sig, _ in results[:3]:
        assert sig.action == "NONE"
        assert sig.reason_code == REASON_INSUFFICIENT_HISTORY
    sig, state = results[3]
    assert sig.action == "ENTER_LONG"
    assert sig.reason_code == REASON_ENTRY_ZSCORE
    assert sig.stop_price == EXPECTED_STOP
    assert sig.inputs["close"] == Decimal("95")
    assert sig.inputs["mean"] == Decimal("99")
    assert sig.inputs["std"] == SQRT6
    assert sig.inputs["z"] < Decimal("-1.5")
    assert sig.inputs["stop"] == EXPECTED_STOP
    assert sig.inputs["param.lookback"] == Decimal(4)
    assert "z<=-entry_z:True" in sig.rules_evaluated
    assert sig.config_snapshot_id == "cfg-1"
    assert sig.data_ids == [bar_data_id(b) for b in ENTRY_BARS]
    assert state == MeanReversionState(
        in_position=True, entry_price=Decimal("95"), stop_price=EXPECTED_STOP
    )


def test_exit_at_mean() -> None:
    # [99, 101, 95, 101] → mean 99, z = 2/√6 ≥ 0.
    sig, state = run(strategy(), [*ENTRY_BARS, bar(4, "101")])[-1]
    assert sig.action == "EXIT"
    assert sig.reason_code == REASON_EXIT_MEAN
    assert "stop_hit(low<=stop):False" in sig.rules_evaluated
    assert "z>=exit_z:True" in sig.rules_evaluated
    assert state == MeanReversionState()


def test_exit_at_stop_uses_low() -> None:
    sig, state = run(strategy(), [*ENTRY_BARS, bar(4, "94", low="92.5505")])[-1]
    assert sig.action == "EXIT"
    assert sig.reason_code == REASON_EXIT_STOP
    assert sig.inputs["low"] == Decimal("92.5505")
    assert "stop_hit(low<=stop):True" in sig.rules_evaluated
    assert state == MeanReversionState()


def test_hold_when_no_exit_rule() -> None:
    # [99, 101, 95, 96] → mean 97.75, z < 0, low > stop.
    sig, state = run(strategy(), [*ENTRY_BARS, bar(4, "96", low="95")])[-1]
    assert sig.action == "NONE"
    assert sig.reason_code == REASON_NO_RULE_MATCHED
    assert state.in_position  # type: ignore[attr-defined]


def test_no_reentry_while_in_position() -> None:
    # A doua scădere puternică nu produce un nou ENTER_LONG cât timp poziția e deschisă.
    sig, _ = run(strategy(), [*ENTRY_BARS, bar(4, "93", low="93")])[-1]
    assert sig.action == "NONE"
    assert not any(r.startswith("z<=-entry_z") for r in sig.rules_evaluated)


def test_zero_volatility_is_none() -> None:
    bars = [bar(i, "100") for i in range(4)]
    sig, state = run(strategy(), bars)[-1]
    assert sig.action == "NONE"
    assert sig.reason_code == REASON_ZERO_VOLATILITY
    assert sig.inputs["std"] == Decimal(0)
    assert state == MeanReversionState()


def test_wrong_interval_is_none() -> None:
    strat = strategy()
    b = bar(0, "100", interval=30)
    sig, state = strat.on_bar(b, HistoryView({"AAA": [b]}, b.ts_close), strat.initial_state())
    assert sig.action == "NONE"
    assert sig.reason_code == REASON_INVALID_INTERVAL
    assert sig.rules_evaluated == ["bar_valid:False"]
    assert state == strat.initial_state()


def test_invalid_bar_is_none() -> None:
    strat = strategy()
    b = bar(0, "100", low="101")  # low > close
    sig, _ = strat.on_bar(b, HistoryView({"AAA": [b]}, b.ts_close), strat.initial_state())
    assert sig.reason_code == REASON_INVALID_BAR


def test_view_not_matching_bar_is_none() -> None:
    strat = strategy()
    sig, _ = strat.on_bar(
        ENTRY_BARS[3], HistoryView({"AAA": ENTRY_BARS}, ENTRY_BARS[2].ts_close), FLAT_STATE
    )
    assert sig.reason_code == REASON_HISTORY_MISMATCH
    sig, _ = strat.on_bar(ENTRY_BARS[0], HistoryView({}, T0), FLAT_STATE)
    assert sig.reason_code == REASON_HISTORY_MISMATCH


def test_deterministic_sequence() -> None:
    bars = [*ENTRY_BARS, bar(4, "96", low="95"), bar(5, "101"), bar(6, "97"), bar(7, "90")]

    def serialise() -> list[str]:
        return [canonical_json(s) + canonical_json(st) for s, st in run(strategy(), bars)]

    assert serialise() == serialise()


def test_params_validation_and_from_config() -> None:
    with pytest.raises(ValidationError):
        MeanReversionParams.from_mapping({"lookback": 1})
    with pytest.raises(ValidationError):
        MeanReversionParams.from_mapping({"entry_z": "0"})
    with pytest.raises(ValidationError):
        MeanReversionParams.from_mapping({"entry_z": "1", "exit_z": "-1"})
    with pytest.raises(ValidationError):
        MeanReversionParams.from_mapping({"unknown": 1})
    cfg = StrategyConfig(strategy_id="mean_reversion_v1", params={"lookback": 30})
    strat = MeanReversionStrategy.from_config(cfg, config_snapshot_id="snap")
    assert strat.params.lookback == 30
    with pytest.raises(ValueError, match="strategy_id"):
        MeanReversionStrategy.from_config(
            StrategyConfig(strategy_id="other"), config_snapshot_id="snap"
        )
    with pytest.raises(ValueError, match="config_snapshot_id"):
        MeanReversionStrategy(MeanReversionParams(), config_snapshot_id="")


def test_state_validation() -> None:
    with pytest.raises(ValidationError):
        MeanReversionState(in_position=True)
    with pytest.raises(ValidationError):
        MeanReversionState(in_position=False, entry_price="1", stop_price="0.5")
