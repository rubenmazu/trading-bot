from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qts.core.clock import NaiveDatetimeError, SimClock, WallClock
from qts.core.money import convert, floor_to_step, notional, quantize_money, round_to_tick

T0 = datetime(2025, 1, 2, 9, 0, tzinfo=UTC)

_qty = st.decimals(min_value=Decimal(0), max_value=Decimal(10**6), places=6)
_step = st.sampled_from([Decimal("1"), Decimal("0.1"), Decimal("0.01"), Decimal("0.001")])


@given(qty=_qty, step=_step)
def test_floor_to_step_is_multiple_and_never_increases(qty: Decimal, step: Decimal) -> None:
    out = floor_to_step(qty, step)
    assert out <= qty
    assert qty - out < step
    assert out % step == 0


@given(price=_qty, tick=_step)
def test_round_to_tick_is_nearest_multiple(price: Decimal, tick: Decimal) -> None:
    out = round_to_tick(price, tick)
    assert out % tick == 0
    assert abs(out - price) <= tick / 2


def test_convert_and_notional() -> None:
    assert notional(Decimal("10.5"), Decimal("3")) == Decimal("31.5")
    assert convert(Decimal("100"), Decimal("0.92")) == Decimal("92.00")
    with pytest.raises(ValueError, match="> 0"):
        convert(Decimal(1), Decimal(0))
    assert quantize_money(Decimal("1.000000005")) == Decimal("1.00000000")  # half-even


def test_wall_clock_is_utc_aware() -> None:
    now = WallClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_sim_clock_normalizes_to_utc_and_rejects_naive() -> None:
    plus2 = datetime(2025, 1, 2, 11, 0, tzinfo=timezone(timedelta(hours=2)))
    clock = SimClock(plus2)
    assert clock.now() == T0
    assert clock.now().tzinfo is UTC
    with pytest.raises(NaiveDatetimeError):
        SimClock(datetime(2025, 1, 1))  # noqa: DTZ001
    with pytest.raises(NaiveDatetimeError):
        clock.advance_to(datetime(2025, 1, 3))  # noqa: DTZ001


def test_sim_clocks_are_independent() -> None:
    a, b = SimClock(T0), SimClock(T0)
    a.advance_to(T0 + timedelta(hours=1))
    assert b.now() == T0
