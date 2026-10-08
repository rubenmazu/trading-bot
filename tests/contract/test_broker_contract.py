"""Suita contract comună pentru adaptoarele broker (Req 3.1–3.4).

Fiecare test rulează pe toate fabricile din `ADAPTER_FACTORIES` (sim, fake; ulterior demo).
Pentru a adăuga un adaptor, vedeți docstring-ul din `tests/contract/broker_contract.py`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from qts.broker.adapter import (
    BrokerAdapter,
    BrokerCapabilities,
    OrderRequest,
    ReasonCode,
    order_request_from,
)
from qts.core.models import ExecKind, OrderIntent, OrderState
from qts.oms.manager import ExecStatus, ListSink, OrderManager, RecordType
from qts.risk.engine import RiskDecision

from .broker_contract import (
    ADAPTER_FACTORIES,
    CONTRACT_QTY,
    CONTRACT_SYMBOL,
    Harness,
    check_event_stream,
    check_snapshot,
)

D = Decimal
_ALL_TIF = ("GTC", "DAY", "IOC")


@pytest.fixture(params=sorted(ADAPTER_FACTORIES))
def factory_name(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    return name


@pytest.fixture
def h(factory_name: str) -> Harness:
    return ADAPTER_FACTORIES[factory_name](None)


def _req(
    coid: str = "c1", side: str = "BUY", qty: Decimal = CONTRACT_QTY, **kw: Any
) -> OrderRequest:
    return OrderRequest.model_validate(
        {"client_order_id": coid, "instrument": CONTRACT_SYMBOL, "side": side, "qty": qty, **kw}
    )


class _Tracker:
    """Cererile trimise (cantitate și sens pe ordin), pentru verificările de flux."""

    def __init__(self, h: Harness) -> None:
        self.h = h
        self.qty: dict[str, Decimal] = {}
        self.side: dict[str, str] = {}

    def submit(self, req: OrderRequest) -> bool:
        ack = self.h.adapter.submit(req)
        self.qty.setdefault(req.client_order_id, req.qty)
        self.side.setdefault(req.client_order_id, req.side)
        return ack.accepted

    def verify(self) -> None:
        self.h.drain()
        summaries = check_event_stream(self.h.log, self.qty)
        check_snapshot(self.h.adapter.snapshot(), self.h.adapter, summaries, self.side)


# --------------------------------------------------------------------------- identitate


def test_conforms_to_protocol(h: Harness) -> None:
    assert isinstance(h.adapter, BrokerAdapter)
    assert h.adapter.environment in ("sim", "demo", "live")
    assert h.adapter.account_id
    caps = h.adapter.capabilities()
    assert isinstance(caps, BrokerCapabilities)
    assert caps.order_types and caps.time_in_force
    snap = h.adapter.snapshot()
    assert snap.complete and snap.orders == () and snap.positions == {}
    assert h.drain() == []


# --------------------------------------------------------------------------- idempotență


def test_submit_is_idempotent_on_client_order_id(h: Harness) -> None:
    first = h.adapter.submit(_req())
    assert first.accepted and first.reason_code is None
    assert first.broker_order_id
    assert h.adapter.submit(_req()) == first
    events = h.drain()
    assert [e.kind for e in events] == [ExecKind.ACK]
    assert events[0].broker_order_id == first.broker_order_id
    assert len(h.adapter.snapshot().orders) == 1


def test_same_client_order_id_with_different_content_conflicts(h: Harness) -> None:
    assert h.adapter.submit(_req()).accepted
    h.drain()
    conflict = h.adapter.submit(_req(qty=CONTRACT_QTY + 1))
    assert not conflict.accepted
    assert conflict.reason_code is ReasonCode.CLIENT_ORDER_ID_CONFLICT
    assert h.drain() == []
    [status] = h.adapter.snapshot().orders
    assert status.qty == CONTRACT_QTY and status.state is OrderState.ACKNOWLEDGED


# --------------------------------------------------------------------------- capabilități


def _restricted(h: Harness, factory_name: str, **update: Any) -> Harness:
    """Adaptor proaspăt cu capabilitățile curente modificate (`update`)."""
    caps = BrokerCapabilities.model_validate({**h.adapter.capabilities().model_dump(), **update})
    return ADAPTER_FACTORIES[factory_name](caps)


def _assert_rejected(h: Harness, req: OrderRequest, code: ReasonCode) -> None:
    tracker = _Tracker(h)
    ack = h.adapter.submit(req)
    tracker.qty[req.client_order_id] = req.qty
    tracker.side[req.client_order_id] = req.side
    assert not ack.accepted and ack.reason_code is code, ack
    [event] = h.drain()
    assert event.kind is ExecKind.REJECT and event.seq == 1
    assert event.client_order_id == req.client_order_id and event.reason == code.value
    [status] = h.adapter.snapshot().orders
    assert status.state is OrderState.REJECTED_BROKER and status.filled_qty == 0
    tracker.verify()


def test_unsupported_instrument_rejected(h: Harness) -> None:
    req = _req().model_copy(update={"instrument": "NOT-OFFERED"})
    _assert_rejected(h, req, ReasonCode.CAPABILITY_UNSUPPORTED_INSTRUMENT)


def test_unsupported_order_type_rejected(h: Harness, factory_name: str) -> None:
    restricted = _restricted(h, factory_name, order_types=frozenset({"MARKET"}))
    req = _req(order_type="LIMIT", limit_price="9.50")
    _assert_rejected(restricted, req, ReasonCode.CAPABILITY_UNSUPPORTED_ORDER_TYPE)


def test_unsupported_time_in_force_rejected(h: Harness) -> None:
    supported = h.adapter.capabilities().time_in_force
    missing = [t for t in _ALL_TIF if t not in supported]
    if not missing:
        pytest.skip(f"{h.name}: toate valorile TIF sunt oferite")
    req = _req(time_in_force=missing[0])
    _assert_rejected(h, req, ReasonCode.CAPABILITY_UNSUPPORTED_TIME_IN_FORCE)


def test_fractional_qty_on_non_fractional_instrument_rejected(h: Harness) -> None:
    _assert_rejected(h, _req(qty=D("0.5")), ReasonCode.CAPABILITY_UNSUPPORTED_FRACTIONAL)


def test_short_sale_rejected_when_not_offered(h: Harness, factory_name: str) -> None:
    restricted = _restricted(h, factory_name, short=False)
    _assert_rejected(restricted, _req(side="SELL"), ReasonCode.CAPABILITY_UNSUPPORTED_SHORT)


# --------------------------------------------------------------------------- flux de evenimente


def test_lifecycle_ack_partial_fill_emits_normalized_events(h: Harness) -> None:
    tracker = _Tracker(h)
    assert tracker.submit(_req())
    h.step_until_terminal("c1")
    h.drain()
    kinds = [e.kind for e in h.log]
    assert kinds[0] is ExecKind.ACK and kinds[-1] is ExecKind.FILL
    assert ExecKind.PARTIAL_FILL in kinds
    assert sum((e.qty or D(0)) for e in h.log if e.kind is not ExecKind.ACK) == CONTRACT_QTY
    tracker.verify()


def test_positions_follow_fills_and_short_guard(h: Harness) -> None:
    tracker = _Tracker(h)
    assert tracker.submit(_req("b1"))
    h.step_until_terminal("b1")
    assert tracker.submit(_req("s1", side="SELL", qty=D(4)))
    # 6 unități rămase disponibile: o vânzare de 7 ar deschide o poziție scurtă.
    assert not tracker.submit(_req("s2", side="SELL", qty=D(7)))
    h.step_until_terminal("s1")
    assert h.adapter.snapshot().positions == {CONTRACT_SYMBOL: D(6)}
    tracker.verify()


# --------------------------------------------------------------------------- anulare


def test_cancel_open_order_reports_remaining_qty(h: Harness) -> None:
    tracker = _Tracker(h)
    assert tracker.submit(_req())
    h.step()  # o execuție parțială
    ack = h.adapter.cancel("c1")
    assert ack.accepted and ack.reason_code is None
    assert h.adapter.cancel("c1") == ack  # repetarea nu produce eveniment nou
    h.drain()
    cancelled = [e for e in h.log if e.kind is ExecKind.CANCELLED]
    assert len(cancelled) == 1
    [status] = h.adapter.snapshot().orders
    assert status.state is OrderState.CANCELLED
    assert cancelled[0].qty == status.remaining_qty > 0
    h.step()  # ordinul anulat nu se mai execută
    tracker.verify()


def test_cancel_terminal_order_rejected(h: Harness) -> None:
    tracker = _Tracker(h)
    assert tracker.submit(_req())
    h.step_until_terminal("c1")
    h.drain()
    ack = h.adapter.cancel("c1")
    assert not ack.accepted and ack.reason_code is ReasonCode.ORDER_NOT_OPEN
    [event] = h.drain()
    assert event.kind is ExecKind.CANCEL_REJECTED
    assert event.reason == ReasonCode.ORDER_NOT_OPEN.value
    assert h.adapter.snapshot().orders[0].state is OrderState.FILLED
    tracker.verify()


def test_cancel_unknown_order(h: Harness) -> None:
    ack = h.adapter.cancel("nope")
    assert not ack.accepted and ack.reason_code is ReasonCode.UNKNOWN_ORDER
    assert h.drain() == []
    assert h.adapter.snapshot().orders == ()


# --------------------------------------------------------------------------- integrare OMS


def _oms_submit(m: OrderManager, h: Harness, n: int) -> str:
    """Ordin aprobat → OMS (write-ahead) → `order_request_from` → adaptor."""
    intent = OrderIntent(
        intent_id=f"i{n}",
        signal_id=f"s{n}",
        instrument=CONTRACT_SYMBOL,
        side="BUY",
        ref_price=D(10),
    )
    decision = RiskDecision(intent_id=intent.intent_id, approved=True, qty=CONTRACT_QTY)
    ts = h.clock.now()
    coid = m.create_order(
        intent, decision, run_id="run", strategy_id="mr", signal_seq=n, ts=ts
    ).order.client_order_id
    assert m.submit(coid, ts).send
    ack = h.adapter.submit(order_request_from(m.order(coid), intent))
    assert ack.accepted, ack
    return coid


def _feed(m: OrderManager, h: Harness) -> list[ExecStatus]:
    statuses: list[ExecStatus] = []
    for event in h.drain():
        statuses.extend(o.status for o in m.on_execution(event))
    return statuses


def test_events_consumed_by_order_manager(h: Harness) -> None:
    """Același cod OMS, indiferent de adaptor (Req 3.2, 3.3)."""
    sink = ListSink()
    m = OrderManager(sink)
    filled = _oms_submit(m, h, 1)
    cancelled = _oms_submit(m, h, 2)
    statuses = _feed(m, h)
    assert m.order(filled).state is OrderState.ACKNOWLEDGED

    h.step()
    statuses += _feed(m, h)
    assert m.order(filled).state is OrderState.PARTIALLY_FILLED
    assert m.request_cancel(cancelled, h.clock.now()).send
    assert h.adapter.cancel(cancelled).accepted
    h.step_until_terminal(filled)
    statuses += _feed(m, h)

    assert statuses and set(statuses) == {ExecStatus.APPLIED}
    assert not [r for r in sink.records if r.type is RecordType.INCIDENT]
    assert not any(m.is_frozen(c) for c in (filled, cancelled))
    snap = {o.client_order_id: o for o in h.adapter.snapshot().orders}
    for coid, state in ((filled, OrderState.FILLED), (cancelled, OrderState.CANCELLED)):
        order = m.order(coid)
        assert order.state is state is snap[coid].state
        assert order.filled_qty == snap[coid].filled_qty
        assert order.broker_order_id == snap[coid].broker_order_id
