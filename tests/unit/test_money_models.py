import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from qts.core.clock import NaiveDatetimeError, SimClock, ensure_utc
from qts.core.models import (
    Bar,
    CostBreakdown,
    ExecKind,
    ExecutionEvent,
    Instrument,
    MarketEvent,
    Order,
    Signal,
    canonical_bytes,
    canonical_hash,
    canonical_json,
)
from qts.core.money import FloatNotAllowedError, ceil_to_step, dec, floor_to_step, round_to_tick

T0 = datetime(2025, 1, 2, 9, 0, tzinfo=UTC)


def test_dec_rejects_float_and_nonfinite() -> None:
    with pytest.raises(FloatNotAllowedError):
        dec(0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nefinit"):
        dec("NaN")
    assert dec("0.10") == Decimal("0.10")


@pytest.mark.parametrize(
    ("qty", "step", "expected"),
    [
        ("1.239", "0.01", "1.23"),
        ("1.0", "0.25", "1.00"),
        ("0.24", "0.25", "0"),
        ("-1.01", "1", "-2"),
    ],
)
def test_floor_to_step_never_rounds_up(qty: str, step: str, expected: str) -> None:
    assert floor_to_step(Decimal(qty), Decimal(step)) == Decimal(expected)


def test_ceil_and_tick_rounding() -> None:
    assert ceil_to_step(Decimal("0.101"), Decimal("0.01")) == Decimal("0.11")
    assert round_to_tick(Decimal("10.0049"), Decimal("0.01")) == Decimal("10.00")
    assert round_to_tick(Decimal("10.005"), Decimal("0.01")) == Decimal("10.00")  # half-even
    with pytest.raises(ValueError, match="> 0"):
        floor_to_step(Decimal(1), Decimal(0))


def test_utc_enforced() -> None:
    with pytest.raises(NaiveDatetimeError):
        ensure_utc(datetime(2025, 1, 1))  # noqa: DTZ001
    plus2 = datetime(2025, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    assert ensure_utc(plus2).hour == 10


def test_sim_clock_monotonic() -> None:
    clock = SimClock(T0)
    clock.advance_to(T0 + timedelta(minutes=5))
    with pytest.raises(ValueError, match="înapoi"):
        clock.advance_to(T0)


def _bar(**overrides: object) -> Bar:
    data: dict[str, object] = {
        "instrument": "XYZ",
        "ts_open": T0,
        "ts_close": T0 + timedelta(minutes=15),
        "interval_min": 15,
        "open": "10",
        "high": "11",
        "low": "9",
        "close": "10.5",
        "volume": "100",
    }
    data.update(overrides)
    return Bar.model_validate(data)


def test_models_are_frozen_and_strict() -> None:
    bar = _bar()
    with pytest.raises(ValidationError):
        bar.open = Decimal(1)  # type: ignore[misc]
    with pytest.raises(ValidationError):
        _bar(unknown_field=1)
    with pytest.raises(ValidationError):
        _bar(open=10.0)
    with pytest.raises(ValidationError):
        _bar(ts_open=datetime(2025, 1, 1))  # noqa: DTZ001


def test_market_event_kind_must_match_payload() -> None:
    bar = _bar()
    MarketEvent(
        source_id="s", instrument="XYZ", ts_source=T0, ts_receipt=T0, kind="bar", payload=bar
    )
    with pytest.raises(ValidationError):
        MarketEvent(
            source_id="s", instrument="XYZ", ts_source=T0, ts_receipt=T0, kind="quote", payload=bar
        )


def test_signal_entry_requires_stop() -> None:
    with pytest.raises(ValidationError):
        Signal(
            signal_id="1",
            strategy_id="s",
            strategy_version="1",
            instrument="XYZ",
            ts=T0,
            action="ENTER_LONG",
            reason_code="x",
            inputs={},
            rules_evaluated=[],
            config_snapshot_id="c",
            data_ids=[],
        )


def test_order_quantity_invariant() -> None:
    order = Order(client_order_id="c", intent_id="i", instrument="XYZ", side="BUY", qty=Decimal(2))
    assert order.remaining_qty == Decimal(2)
    with pytest.raises(ValidationError):
        order.model_copy(update={"filled_qty": Decimal(3)}).model_validate(
            order.model_copy(update={"filled_qty": Decimal(3)}).model_dump()
        )


def test_fill_requires_qty_and_price() -> None:
    with pytest.raises(ValidationError):
        ExecutionEvent(
            broker_exec_id="e", client_order_id="c", kind=ExecKind.FILL, ts_broker=T0, ts_receipt=T0
        )


def test_cost_breakdown_total_and_add() -> None:
    a = CostBreakdown(spread=Decimal("0.01"), commission=Decimal("0.02"))
    b = CostBreakdown(slippage=Decimal("0.03"), taxes=Decimal("0.04"))
    assert (a + b).total == Decimal("0.10")


def test_canonical_json_is_stable() -> None:
    inst = Instrument(
        symbol="XYZ",
        venue="X",
        asset_class="etf",
        currency="EUR",
        tick_size=Decimal("0.01"),
        qty_step=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        calendar_id="XETR",
    )
    assert canonical_json(inst) == canonical_json(Instrument.model_validate(inst.model_dump()))
    assert '"tick_size":"0.01"' in canonical_json(inst)


@pytest.mark.parametrize("interval", [4, 61, 0])
def test_bar_interval_out_of_range_rejected(interval: int) -> None:
    with pytest.raises(ValidationError):
        _bar(interval_min=interval)


@pytest.mark.parametrize("interval", [5, 60])
def test_bar_interval_bounds_accepted(interval: int) -> None:
    assert _bar(interval_min=interval).interval_min == interval


def test_canonical_form_is_deterministic_and_utc() -> None:
    plus2 = timezone(timedelta(hours=2))
    a = _bar(ts_open=T0.astimezone(plus2), ts_close=(T0 + timedelta(minutes=15)).astimezone(plus2))
    b = _bar()
    # același moment în fusuri diferite => aceeași formă canonică (normalizată la UTC)
    assert canonical_bytes(a) == canonical_bytes(b)
    assert canonical_hash(a) == canonical_hash(b)
    text = canonical_json(b)
    assert '"ts_open":"2025-01-02T09:00:00Z"' in text
    assert '"close":"10.5"' in text
    keys = list(json.loads(text).keys())
    assert keys == sorted(keys)
    assert canonical_hash(_bar(close="10.6")) != canonical_hash(b)


def test_canonical_roundtrip_of_nested_models() -> None:
    order = Order(
        client_order_id="c",
        intent_id="i",
        instrument="XYZ",
        side="BUY",
        qty=Decimal("2.5"),
        costs=CostBreakdown(commission=Decimal("1.00")),
    )
    restored = Order.model_validate_json(canonical_json(order))
    assert restored == order
    assert canonical_bytes(restored) == canonical_bytes(order)
