from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from qts.broker.adapter import (
    BrokerAdapter,
    BrokerCapabilities,
    OrderRequest,
    ReasonCode,
    check_request,
    order_request_from,
)
from qts.broker.sim import SimBarContext, SimBroker, SimBrokerConfig
from qts.core.clock import SimClock
from qts.core.models import (
    Bar,
    ExecKind,
    ExecutionEvent,
    Instrument,
    Order,
    OrderIntent,
    OrderState,
    Quote,
)
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import (
    CompleteCostModel,
    CostModelConfig,
    CostModelIncomplete,
    FxConfig,
    LatencyConfig,
    SlippageConfig,
    SpreadSchedule,
    TaxConfig,
)

D = Decimal
T0 = datetime(2025, 1, 2, 9, 0, tzinfo=UTC)
STEP = timedelta(minutes=15)
CTX = SimBarContext(sigma_bar=D("0.01"), adv=D("10000"))


def _inst(**kw: Any) -> Instrument:
    data: dict[str, Any] = {
        "symbol": "XYZ",
        "venue": "XETR",
        "asset_class": "etf",
        "currency": "EUR",
        "tick_size": D("0.01"),
        "qty_step": D("1"),
        "min_qty": D("1"),
        "calendar_id": "XETR",
    }
    data.update(kw)
    return Instrument.model_validate(data)


def _cost_model(k: str = "0", min_ticks: str = "1", spread: str = "0.002") -> CompleteCostModel:
    table = CommissionTable(
        broker="sim",
        version="v1",
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        currency="EUR",
        percent=D("0.001"),
        minimum=D("1"),
    )
    return CompleteCostModel(
        CostModelConfig(
            version="costs-v1",
            commissions=CommissionSchedule(tables=(table,)),
            spreads={"XYZ": SpreadSchedule(default=D(spread))},
            slippage=SlippageConfig(k=D(k), min_ticks=D(min_ticks)),
            latency=LatencyConfig(latency_ms=500),
            fx=FxConfig(conversion_spread=D("0.002")),
            taxes=TaxConfig(approved=True, reference="operator"),
        )
    )


def _bar(
    i: int, o: str = "10", h: str = "10.5", lo: str = "9.5", c: str = "10", v: str = "1000"
) -> Bar:
    return Bar(
        instrument="XYZ",
        ts_open=T0 + i * STEP,
        ts_close=T0 + (i + 1) * STEP,
        interval_min=15,
        open=D(o),
        high=D(h),
        low=D(lo),
        close=D(c),
        volume=D(v),
    )


def _broker(
    clock: SimClock | None = None, cost: CompleteCostModel | None = None, **cfg: Any
) -> tuple[SimBroker, SimClock]:
    clock = clock or SimClock(T0 + STEP)  # închiderea barei 0
    broker = SimBroker(
        account_id="acc",
        instruments=[_inst()],
        cost_model=cost or _cost_model(),
        clock=clock,
        config=SimBrokerConfig(**cfg),
    )
    return broker, clock


def _req(coid: str = "c1", side: str = "BUY", qty: str = "10", **kw: Any) -> OrderRequest:
    return OrderRequest.model_validate(
        {"client_order_id": coid, "instrument": "XYZ", "side": side, "qty": qty, **kw}
    )


def _drain(broker: SimBroker) -> list[ExecutionEvent]:
    return list(broker.events())


def test_protocol_conformance() -> None:
    broker, _ = _broker()
    assert isinstance(broker, BrokerAdapter)
    assert broker.environment == "sim"


def test_market_executes_at_open_of_next_bar_with_spread_and_slippage() -> None:
    broker, _ = _broker()
    ack = broker.submit(_req())
    assert ack.accepted and ack.broker_order_id == "SIM-00000001"
    [ev_ack] = _drain(broker)
    assert ev_ack.kind is ExecKind.ACK and ev_ack.seq == 1
    # bara 0 (deja închisă la trimitere) nu execută ordinul
    broker.on_bar(_bar(0), CTX)
    assert _drain(broker) == []
    broker.on_bar(_bar(1, o="10"), CTX)
    [fill] = _drain(broker)
    # 10 + 10×0.002/2 + 1 tick = 10.02
    assert fill.kind is ExecKind.FILL and fill.seq == 2
    assert fill.qty == D(10) and fill.price == D("10.02")
    assert fill.ts_broker == _bar(1).ts_open
    assert fill.commission == D(1)  # max(100.2 × 0.001, minim 1)
    assert fill.broker_exec_id == "acc:c1:2"


def test_sell_price_is_adverse_and_quote_spread_used() -> None:
    broker, clock = _broker()
    broker.submit(_req())
    broker.on_bar(_bar(1), CTX)
    clock.advance_to(T0 + 2 * STEP)
    broker.submit(_req("c2", side="SELL"))
    q = Quote(instrument="XYZ", ts=T0 + 2 * STEP, bid=D("9.98"), ask=D("10.02"))
    broker.on_bar(_bar(2), SimBarContext(quote=q, sigma_bar=D("0.01"), adv=D("10000")))
    events = _drain(broker)
    sell = events[-1]
    assert sell.kind is ExecKind.FILL and sell.price == D("9.97")  # 10 - 0.02 - 0.01
    snap = broker.snapshot()
    assert snap.positions == {}
    assert snap.cash == D("-100.20") - 1 + D("99.70") - 1


def test_partial_fills_across_bars_by_participation() -> None:
    broker, _ = _broker(max_participation=D("0.1"))
    broker.submit(_req(qty="25"))
    broker.on_bar(_bar(1, v="100"), CTX)
    broker.on_bar(_bar(2, v="100"), CTX)
    broker.on_bar(_bar(3, v="100"), CTX)
    fills = [e for e in _drain(broker) if e.kind in (ExecKind.PARTIAL_FILL, ExecKind.FILL)]
    assert [(e.kind, e.qty, e.seq) for e in fills] == [
        (ExecKind.PARTIAL_FILL, D(10), 2),
        (ExecKind.PARTIAL_FILL, D(10), 3),
        (ExecKind.FILL, D(5), 4),
    ]
    # comisionul minim se aplică o singură dată per ordin
    assert sum(e.commission or 0 for e in fills) == D(1)
    status = broker.snapshot().orders[0]
    assert status.state is OrderState.FILLED and status.filled_qty == D(25)
    assert status.avg_fill_price == D("10.02")


def test_partial_fills_by_max_qty_per_bar() -> None:
    broker, _ = _broker(max_fill_qty_per_bar=D("4"))
    broker.submit(_req(qty="10"))
    for i in range(1, 4):
        broker.on_bar(_bar(i), CTX)
    qtys = [e.qty for e in _drain(broker) if e.kind is not ExecKind.ACK]
    assert qtys == [D(4), D(4), D(2)]


def test_limit_order_fills_only_when_crossed() -> None:
    broker, _ = _broker()
    broker.submit(_req(order_type="LIMIT", limit_price="9.80"))
    broker.on_bar(_bar(1, lo="9.80"), CTX)  # atingere fără traversare
    broker.on_bar(_bar(2, lo="9.70"), CTX)
    events = _drain(broker)
    fill = events[-1]
    assert [e.kind for e in events] == [ExecKind.ACK, ExecKind.FILL]
    assert fill.price == D("9.80") and fill.ts_broker == _bar(2).ts_open


def test_limit_order_marketable_at_open_gets_open_price() -> None:
    broker, _ = _broker()
    broker.submit(_req(order_type="LIMIT", limit_price="10.50"))
    broker.on_bar(_bar(1), CTX)
    assert _drain(broker)[-1].price == D("10.02")


def test_cancel_before_fill_and_repeat() -> None:
    broker, _ = _broker()
    broker.submit(_req())
    ack = broker.cancel("c1")
    assert ack.accepted
    assert broker.cancel("c1") == ack  # idempotent, fără eveniment nou
    broker.on_bar(_bar(1), CTX)
    events = _drain(broker)
    assert [(e.kind, e.seq, e.qty) for e in events] == [
        (ExecKind.ACK, 1, None),
        (ExecKind.CANCELLED, 2, D(10)),
    ]
    assert broker.snapshot().orders[0].state is OrderState.CANCELLED


def test_cancel_after_partial_and_after_full_fill() -> None:
    broker, _ = _broker(max_fill_qty_per_bar=D("4"))
    broker.submit(_req())
    broker.on_bar(_bar(1), CTX)
    assert broker.cancel("c1").accepted
    assert _drain(broker)[-1].qty == D(6)

    broker2, _ = _broker()
    broker2.submit(_req())
    broker2.on_bar(_bar(1), CTX)
    ack = broker2.cancel("c1")
    assert not ack.accepted and ack.reason_code is ReasonCode.ORDER_NOT_OPEN
    assert _drain(broker2)[-1].kind is ExecKind.CANCEL_REJECTED
    unknown = broker2.cancel("nope")
    assert unknown.reason_code is ReasonCode.UNKNOWN_ORDER


def test_idempotent_submit() -> None:
    broker, _ = _broker()
    first = broker.submit(_req())
    assert broker.submit(_req()) == first
    conflict = broker.submit(_req(qty="11"))
    assert conflict.reason_code is ReasonCode.CLIENT_ORDER_ID_CONFLICT
    assert len(_drain(broker)) == 1
    assert len(broker.snapshot().orders) == 1


@pytest.mark.parametrize(
    ("req_kw", "inst_kw", "caps_kw", "code"),
    [
        ({"instrument": "ABC"}, {}, {}, ReasonCode.CAPABILITY_UNSUPPORTED_INSTRUMENT),
        ({}, {"asset_class": "crypto"}, {}, ReasonCode.CAPABILITY_UNSUPPORTED_ASSET_CLASS),
        ({}, {"is_derivative": True}, {}, ReasonCode.CAPABILITY_UNSUPPORTED_DERIVATIVE),
        ({}, {"requires_leverage": True}, {}, ReasonCode.CAPABILITY_UNSUPPORTED_LEVERAGE),
        (
            {"order_type": "LIMIT", "limit_price": "10"},
            {},
            {"order_types": frozenset({"MARKET"})},
            ReasonCode.CAPABILITY_UNSUPPORTED_ORDER_TYPE,
        ),
        ({"time_in_force": "IOC"}, {}, {}, ReasonCode.CAPABILITY_UNSUPPORTED_TIME_IN_FORCE),
        (
            {"qty": "0.5"},
            {"qty_step": D("0.1"), "min_qty": D("0.1")},
            {},
            ReasonCode.CAPABILITY_UNSUPPORTED_FRACTIONAL,
        ),
        ({"qty": "2.5"}, {"fractional": True}, {"fractional": True}, ReasonCode.INVALID_QTY),
        (
            {"order_type": "LIMIT", "limit_price": "10.005"},
            {},
            {},
            ReasonCode.INVALID_PRICE,
        ),
        ({"side": "SELL"}, {}, {}, ReasonCode.CAPABILITY_UNSUPPORTED_SHORT),
    ],
)
def test_check_request_reason_codes(
    req_kw: dict[str, Any], inst_kw: dict[str, Any], caps_kw: dict[str, Any], code: ReasonCode
) -> None:
    caps_data: dict[str, Any] = {
        "order_types": frozenset({"MARKET", "LIMIT"}),
        "time_in_force": frozenset({"GTC"}),
        "asset_classes": frozenset({"etf"}),
        **caps_kw,
    }
    inst = _inst(**inst_kw)
    req = _req(**req_kw)
    found = check_request(
        BrokerCapabilities(**caps_data), req, inst if req.instrument == "XYZ" else None
    )
    assert found is not None and found.code is code


def test_submit_rejection_emits_reject_event_with_reason() -> None:
    broker, _ = _broker()
    ack = broker.submit(_req(side="SELL"))
    assert not ack.accepted and ack.reason_code is ReasonCode.CAPABILITY_UNSUPPORTED_SHORT
    [ev] = _drain(broker)
    assert ev.kind is ExecKind.REJECT and ev.seq == 1
    assert ev.reason == ReasonCode.CAPABILITY_UNSUPPORTED_SHORT.value
    assert broker.snapshot().orders[0].state is OrderState.REJECTED_BROKER


def test_sell_reserves_position() -> None:
    broker, clock = _broker()
    broker.submit(_req(qty="10"))
    broker.on_bar(_bar(1), CTX)
    clock.advance_to(T0 + 2 * STEP)
    assert broker.submit(_req("s1", side="SELL", qty="6")).accepted
    second = broker.submit(_req("s2", side="SELL", qty="6"))
    assert second.reason_code is ReasonCode.CAPABILITY_UNSUPPORTED_SHORT


def test_missing_cost_inputs_raise() -> None:
    broker, _ = _broker()
    broker.submit(_req())
    with pytest.raises(CostModelIncomplete):
        broker.on_bar(_bar(1))  # fără sigma_bar / ADV


def test_slippage_uses_square_root_impact() -> None:
    broker, _ = _broker(cost=_cost_model(k="1", min_ticks="0", spread="0"))
    broker.submit(_req(qty="100"))
    broker.on_bar(_bar(1), CTX)
    # 0.01 × sqrt(100/10000) × 10 = 0.001 → rotunjit în sus la 10.01
    assert _drain(broker)[-1].price == D("10.01")


def test_snapshot_and_determinism() -> None:
    def run() -> tuple[list[ExecutionEvent], Any]:
        broker, _ = _broker(max_participation=D("0.05"))
        broker.submit(_req(qty="30"))
        broker.submit(_req("c2", qty="5"))
        for i in range(1, 5):
            broker.on_bar(_bar(i, v="200"), CTX)
        return _drain(broker), broker.snapshot()

    ev1, snap1 = run()
    ev2, snap2 = run()
    assert ev1 == ev2 and snap1 == snap2
    assert snap1.complete and snap1.positions == {"XYZ": D(35)}
    assert set(snap1.execution_ids) == {e.broker_exec_id for e in ev1}
    assert len(snap1.execution_ids) == len(set(snap1.execution_ids))


def test_order_request_validation_and_translation() -> None:
    with pytest.raises(ValidationError):
        _req(order_type="LIMIT")
    with pytest.raises(ValidationError):
        _req(limit_price="10")
    intent = OrderIntent(
        intent_id="i1",
        signal_id="s1",
        instrument="XYZ",
        side="BUY",
        ref_price=D("10"),
        order_type="LIMIT",
        limit_price=D("9.9"),
    )
    order = Order(client_order_id="c1", intent_id="i1", instrument="XYZ", side="BUY", qty=D(3))
    req = order_request_from(order, intent)
    assert (req.client_order_id, req.qty, req.order_type, req.limit_price) == (
        "c1",
        D(3),
        "LIMIT",
        D("9.9"),
    )
