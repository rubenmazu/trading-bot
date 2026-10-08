"""Teste unitare suplimentare: rotunjire monetară și validarea modelelor (Req 13.9)."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from qts.core.models import (
    CostBreakdown,
    ExecKind,
    ExecutionEvent,
    Instrument,
    Order,
    OrderIntent,
    Signal,
)
from qts.core.money import FloatNotAllowedError, ceil_to_step, dec, floor_to_step, round_to_tick

T0 = datetime(2025, 1, 2, 9, 0, tzinfo=UTC)

_amount = st.decimals(min_value=Decimal(0), max_value=Decimal(10**6), places=6)
_step = st.sampled_from([Decimal("1"), Decimal("0.25"), Decimal("0.01"), Decimal("0.001")])


# --------------------------------------------------------------------------- rotunjire


@given(value=_amount, step=_step)
def test_ceil_to_step_never_decreases_costs(value: Decimal, step: Decimal) -> None:
    out = ceil_to_step(value, step)
    assert out >= value
    assert out - value < step
    assert out % step == 0


@pytest.mark.parametrize(
    ("value", "step", "expected"),
    [("0.10", "0.01", "0.10"), ("0", "0.01", "0"), ("-1.01", "1", "-1"), ("1.01", "0.25", "1.25")],
)
def test_ceil_to_step_examples(value: str, step: str, expected: str) -> None:
    assert ceil_to_step(Decimal(value), Decimal(step)) == Decimal(expected)


@given(
    qty=_amount,
    step=_step,
    price=st.decimals(min_value=Decimal("0.01"), max_value=Decimal(1000), places=2),
)
def test_floor_qty_never_increases_notional_risk(
    qty: Decimal, step: Decimal, price: Decimal
) -> None:
    # riscul/expunerea după rotunjire nu poate depăși valoarea dinainte de rotunjire
    assert floor_to_step(qty, step) * price <= qty * price


@pytest.mark.parametrize("fn", [floor_to_step, ceil_to_step, round_to_tick])
@pytest.mark.parametrize("step", ["0", "-0.01"])
def test_rounding_rejects_non_positive_step(fn: object, step: str) -> None:
    with pytest.raises(ValueError, match="> 0"):
        fn(Decimal(1), Decimal(step))  # type: ignore[operator]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "sNaN"])
def test_dec_rejects_nonfinite(value: str) -> None:
    with pytest.raises(ValueError, match="nefinit"):
        dec(value)
    with pytest.raises(ValueError, match="nefinit"):
        dec(Decimal(value))


def test_dec_rejects_bool_and_float() -> None:
    with pytest.raises(TypeError):
        dec(True)
    with pytest.raises(FloatNotAllowedError):
        dec(float("nan"))  # type: ignore[arg-type]
    assert dec(3) == Decimal(3)


# --------------------------------------------------------------------------- modele


def _instrument(**overrides: object) -> Instrument:
    data: dict[str, object] = {
        "symbol": "XYZ",
        "venue": "X",
        "asset_class": "etf",
        "currency": "EUR",
        "tick_size": "0.01",
        "qty_step": "0.001",
        "min_qty": "0.001",
        "calendar_id": "XETR",
    }
    data.update(overrides)
    return Instrument.model_validate(data)


@pytest.mark.parametrize("bad", [0.01, "NaN", "Infinity", True, None, [1]])
def test_decimal_fields_reject_float_nonfinite_and_wrong_types(bad: object) -> None:
    with pytest.raises(ValidationError):
        _instrument(tick_size=bad)


def test_instrument_frozen_extra_forbid_and_positive_steps() -> None:
    inst = _instrument()
    with pytest.raises(ValidationError):
        inst.symbol = "ABC"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        _instrument(leverage=2)
    with pytest.raises(ValidationError):
        _instrument(asset_class="cfd")
    for field in ("tick_size", "qty_step"):
        with pytest.raises(ValidationError):
            _instrument(**{field: "0"})
    with pytest.raises(ValidationError):
        _instrument(min_qty="-1")


def _order(**overrides: object) -> Order:
    data: dict[str, object] = {
        "client_order_id": "c",
        "intent_id": "i",
        "instrument": "XYZ",
        "side": "BUY",
        "qty": "2",
    }
    data.update(overrides)
    return Order.model_validate(data)


@pytest.mark.parametrize(
    ("qty", "filled"), [("0", "0"), ("-1", "0"), ("2", "-0.1"), ("2", "2.001")]
)
def test_order_quantity_invariant_rejections(qty: str, filled: str) -> None:
    with pytest.raises(ValidationError):
        _order(qty=qty, filled_qty=filled)


def test_order_quantity_invariant_bounds_and_remaining() -> None:
    assert _order(filled_qty="0").remaining_qty == Decimal(2)
    assert _order(filled_qty="2").remaining_qty == Decimal(0)
    assert _order(filled_qty="0.5").remaining_qty == Decimal("1.5")
    with pytest.raises(ValidationError):
        _order(side="SHORT")


def test_cost_breakdown_rejects_float() -> None:
    with pytest.raises(ValidationError):
        CostBreakdown(commission=0.5)


def _exec(kind: ExecKind, **overrides: object) -> ExecutionEvent:
    data: dict[str, object] = {
        "broker_exec_id": "e",
        "client_order_id": "c",
        "kind": kind,
        "ts_broker": T0,
        "ts_receipt": T0,
    }
    data.update(overrides)
    return ExecutionEvent.model_validate(data)


@pytest.mark.parametrize("kind", [ExecKind.FILL, ExecKind.PARTIAL_FILL])
@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"qty": "1"},
        {"price": "10"},
        {"qty": "0", "price": "10"},
        {"qty": "1", "price": "0"},
        {"qty": "-1", "price": "10"},
        {"qty": "1", "price": "-10"},
    ],
)
def test_fill_events_require_positive_qty_and_price(kind: ExecKind, fields: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        _exec(kind, **fields)


def test_valid_fill_and_non_fill_events() -> None:
    fill = _exec(ExecKind.FILL, qty="1.5", price="10.01", commission="0.5")
    assert fill.qty == Decimal("1.5")
    for kind in (ExecKind.ACK, ExecKind.REJECT, ExecKind.CANCELLED, ExecKind.EXPIRED):
        assert _exec(kind).qty is None
    with pytest.raises(ValidationError):
        _exec(ExecKind.ACK, ts_broker=datetime(2025, 1, 1))  # noqa: DTZ001


def _signal(**overrides: object) -> Signal:
    data: dict[str, object] = {
        "signal_id": "1",
        "strategy_id": "s",
        "strategy_version": "1",
        "instrument": "XYZ",
        "ts": T0,
        "action": "ENTER_LONG",
        "stop_price": "9.5",
        "reason_code": "x",
        "inputs": {"atr": "0.5"},
        "rules_evaluated": ["r1"],
        "config_snapshot_id": "c",
        "data_ids": ["d1"],
    }
    data.update(overrides)
    return Signal.model_validate(data)


def test_signal_stop_rules() -> None:
    assert _signal().stop_price == Decimal("9.5")
    with pytest.raises(ValidationError):
        _signal(stop_price=None)
    assert _signal(action="EXIT", stop_price=None).stop_price is None
    assert _signal(action="NONE", stop_price=None).action == "NONE"
    with pytest.raises(ValidationError):
        _signal(action="ENTER_SHORT")
    with pytest.raises(ValidationError):
        _signal(inputs={"atr": 0.5})


def _intent(**overrides: object) -> OrderIntent:
    data: dict[str, object] = {
        "intent_id": "i",
        "signal_id": "1",
        "instrument": "XYZ",
        "side": "BUY",
        "ref_price": "10",
    }
    data.update(overrides)
    return OrderIntent.model_validate(data)


def test_order_intent_limit_requires_limit_price_and_positive_ref() -> None:
    assert _intent().order_type == "MARKET"
    with pytest.raises(ValidationError):
        _intent(order_type="LIMIT")
    assert _intent(order_type="LIMIT", limit_price="10.05").limit_price == Decimal("10.05")
    for ref in ("0", "-1"):
        with pytest.raises(ValidationError):
            _intent(ref_price=ref)
    with pytest.raises(ValidationError):
        _intent(order_type="STOP")
