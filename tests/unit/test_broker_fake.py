"""Teste unitare pentru `broker/fake.py` (Req 9.1, 10.4, 11.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from qts.broker.adapter import BrokerAdapter, OrderRequest, ReasonCode, order_request_from
from qts.broker.fake import (
    BrokerDisconnectedError,
    BrokerTimeoutError,
    CancelFault,
    FakeBroker,
    FaultPlan,
    SnapshotFault,
    SnapshotUnavailableError,
    SubmitFault,
)
from qts.core.clock import SimClock
from qts.core.models import ExecKind, ExecutionEvent, Instrument, OrderIntent, OrderState
from qts.oms.manager import ExecOutcome, ExecStatus, ListSink, OrderManager, RecordType
from qts.risk.engine import RiskDecision

D = Decimal
S = OrderState
T0 = datetime(2026, 1, 5, 9, tzinfo=UTC)
ACCT = "FAKE-DEMO-1"


def _inst(symbol: str = "XYZ") -> Instrument:
    data: dict[str, Any] = {
        "symbol": symbol,
        "venue": "XETR",
        "asset_class": "etf",
        "currency": "EUR",
        "tick_size": D("0.01"),
        "qty_step": D("1"),
        "min_qty": D("1"),
        "calendar_id": "XETR",
    }
    return Instrument.model_validate(data)


def _broker(**kw: Any) -> FakeBroker:
    kw.setdefault("initial_cash", D(10000))
    return FakeBroker(instruments=[_inst("XYZ"), _inst("ABC")], clock=SimClock(T0), **kw)


def _req(coid: str = "c1", qty: str = "10", side: str = "BUY", inst: str = "XYZ") -> OrderRequest:
    return OrderRequest.model_validate(
        {"client_order_id": coid, "instrument": inst, "side": side, "qty": D(qty)}
    )


def _drain(b: FakeBroker) -> list[ExecutionEvent]:
    return list(b.events())


def _kinds(events: list[ExecutionEvent]) -> list[tuple[ExecKind, int | None]]:
    return [(e.kind, e.seq) for e in events]


# --------------------------------------------------------------------------- protocol


def test_satisfies_protocol_and_environment_rules() -> None:
    b = _broker()
    assert isinstance(b, BrokerAdapter)
    assert b.environment == "demo" and b.account_id == ACCT and b.endpoint == "fake://local"
    assert _broker(environment="sim").environment == "sim"
    with pytest.raises(ValueError):
        _broker(environment="live")
    with pytest.raises(ValueError):
        _broker(endpoint="https://api.broker.com")


def test_submit_is_idempotent_and_detects_conflict() -> None:
    b = _broker()
    first = b.submit(_req())
    assert first.accepted and first.broker_order_id == "FAKE-00000001"
    assert b.submit(_req()) == first
    conflict = b.submit(_req(qty="11"))
    assert not conflict.accepted
    assert conflict.reason_code is ReasonCode.CLIENT_ORDER_ID_CONFLICT
    events = _drain(b)
    assert _kinds(events) == [(ExecKind.ACK, 1)]
    assert events[0].broker_exec_id == f"{ACCT}:c1:1"
    assert len(b.truth().orders) == 1


# --------------------------------------------------------------------------- respingeri


def test_capability_rejections_use_check_request() -> None:
    b = _broker()
    unknown = b.submit(_req("c1", inst="NOPE"))
    assert unknown.reason_code is ReasonCode.CAPABILITY_UNSUPPORTED_INSTRUMENT
    short = b.submit(_req("c2", side="SELL"))
    assert short.reason_code is ReasonCode.CAPABILITY_UNSUPPORTED_SHORT
    events = _drain(b)
    assert [(e.kind, e.reason) for e in events] == [
        (ExecKind.REJECT, "CAPABILITY_UNSUPPORTED_INSTRUMENT"),
        (ExecKind.REJECT, "CAPABILITY_UNSUPPORTED_SHORT"),
    ]
    assert {o.state for o in b.truth().orders} == {S.REJECTED_BROKER}


def test_injected_sync_and_async_rejections() -> None:
    plan = FaultPlan().on_submit("c1", SubmitFault.REJECT, ReasonCode.INVALID_PRICE)
    plan.on_submit("c2", SubmitFault.DEFER_ACK).on_submit("c3", SubmitFault.DEFER_ACK)
    b = _broker(faults=plan)
    ack1 = b.submit(_req("c1"))
    assert not ack1.accepted and ack1.reason_code is ReasonCode.INVALID_PRICE
    assert b.submit(_req("c2")).accepted and b.submit(_req("c3")).accepted
    assert _kinds(_drain(b)) == [(ExecKind.REJECT, 1)]  # c2/c3 fără ACK încă
    b.reject("c2", ReasonCode.INVALID_QTY)
    b.ack("c3")
    events = _drain(b)
    assert [(e.client_order_id, e.kind, e.seq) for e in events] == [
        ("c2", ExecKind.REJECT, 1),
        ("c3", ExecKind.ACK, 1),
    ]
    with pytest.raises(ValueError):
        b.ack("c3")
    states = {o.client_order_id: o.state for o in b.truth().orders}
    assert states == {"c1": S.REJECTED_BROKER, "c2": S.REJECTED_BROKER, "c3": S.ACKNOWLEDGED}


# --------------------------------------------------------------------------- timeout


def test_timeout_before_accept_leaves_no_order() -> None:
    b = _broker(faults=FaultPlan().on_submit("c1", SubmitFault.TIMEOUT_BEFORE_ACCEPT))
    with pytest.raises(BrokerTimeoutError):
        b.submit(_req())
    assert b.truth().orders == () and _drain(b) == []
    assert b.submit(_req()).accepted  # directiva s-a consumat
    assert _kinds(_drain(b)) == [(ExecKind.ACK, 1)]


def test_timeout_after_accept_keeps_order_and_retry_is_idempotent() -> None:
    b = _broker(faults=FaultPlan().on_submit("c1", SubmitFault.TIMEOUT_AFTER_ACCEPT))
    with pytest.raises(TimeoutError):
        b.submit(_req())
    (status,) = b.truth().orders
    assert status.state is S.ACKNOWLEDGED
    retry = b.submit(_req())
    assert retry.accepted and retry.broker_order_id == status.broker_order_id
    assert _kinds(_drain(b)) == [(ExecKind.ACK, 1)]


# --------------------------------------------------------------------------- canal de livrare


def test_duplicate_and_redeliver_reuse_exec_id() -> None:
    b = _broker(faults=FaultPlan().duplicate("c1", 1, copies=2))
    b.submit(_req())
    events = _drain(b)
    assert len(events) == 3 and len({e.broker_exec_id for e in events}) == 1
    b.redeliver("c1", 1)
    (again,) = _drain(b)
    assert again.model_dump(exclude={"ts_receipt"}) == events[0].model_dump(exclude={"ts_receipt"})


def test_reorder_explicit_and_seeded_is_deterministic() -> None:
    def scripted(seed: int) -> FakeBroker:
        b = _broker(seed=seed)
        b.submit(_req())
        b.fill("c1", D(3), D(100))
        b.fill("c1", D(3), D(101))
        b.fill("c1", D(4), D(102))
        return b

    b = scripted(0)
    assert b.reorder([3, 1, 2, 0]) == [3, 1, 2, 0]
    assert [e.seq for e in _drain(b)] == [4, 2, 3, 1]
    with pytest.raises(ValueError):
        scripted(0).reorder([0, 0, 1, 2])
    p1, p2 = scripted(7).reorder(), scripted(7).reorder()
    assert p1 == p2 and sorted(p1) == [0, 1, 2, 3]


def test_dropped_message_creates_gap_and_resend_fills_it() -> None:
    b = _broker(faults=FaultPlan().drop("c1", 2))
    b.submit(_req())
    b.fill("c1", D(4), D(100))
    b.fill("c1", D(6), D(100))
    assert [e.seq for e in _drain(b)] == [1, 3]
    assert f"{ACCT}:c1:2" in b.truth().execution_ids  # emis la broker, pierdut în tranzit
    (resent,) = b.resend("c1", [2])
    assert resent.kind is ExecKind.PARTIAL_FILL and [e.seq for e in _drain(b)] == [2]
    with pytest.raises(KeyError):
        b.resend("c1", [9])


def test_disconnect_buffers_broker_side_events_until_reconnect() -> None:
    b = _broker()
    b.submit(_req())
    b.disconnect()
    for call in (b.events, b.snapshot, lambda: b.submit(_req("c2")), lambda: b.cancel("c1")):
        with pytest.raises(ConnectionError):
            call()
    b.fill("c1", D(10), D(100))  # brokerul execută și în timpul deconectării
    b.reconnect()
    assert _kinds(_drain(b)) == [(ExecKind.ACK, 1), (ExecKind.FILL, 2)]
    assert b.snapshot().positions == {"XYZ": D(10)}


def test_disconnect_losing_in_flight_messages_is_recoverable() -> None:
    b = _broker()
    b.submit(_req())
    b.disconnect(lose_in_flight=True)
    with pytest.raises(BrokerDisconnectedError):
        b.events()
    b.reconnect()
    assert _drain(b) == []
    b.resend("c1", range(1, b.next_seq("c1")))
    assert _kinds(_drain(b)) == [(ExecKind.ACK, 1)]


# --------------------------------------------------------------------------- snapshot


def test_incomplete_and_failing_snapshot() -> None:
    plan = FaultPlan().fail_snapshot(SnapshotFault.INCOMPLETE).fail_snapshot(SnapshotFault.RAISE)
    b = _broker(faults=plan)
    b.submit(_req("c1"))
    b.submit(_req("c2", inst="ABC"))
    partial = b.snapshot()
    assert not partial.complete
    assert [o.client_order_id for o in partial.orders] == ["c1"]
    assert partial.execution_ids == (f"{ACCT}:c1:1",)
    with pytest.raises(SnapshotUnavailableError):
        b.snapshot()
    full = b.snapshot()
    assert full.complete and len(full.orders) == 2 and full == b.truth()


# --------------------------------------------------------------------------- anulare


def test_fill_during_deferred_cancel_then_cancelled() -> None:
    b = _broker(faults=FaultPlan().on_cancel("c1", CancelFault.DEFER))
    b.submit(_req())
    ack = b.cancel("c1")
    assert ack.accepted and b.is_cancel_pending("c1")
    b.fill("c1", D(4), D(100))
    assert b.cancel("c1") == ack  # repetarea întoarce aceeași confirmare
    done = b.confirm_cancel("c1")
    assert done.kind is ExecKind.CANCELLED and done.qty == D(6)
    assert _kinds(_drain(b)) == [
        (ExecKind.ACK, 1),
        (ExecKind.PARTIAL_FILL, 2),
        (ExecKind.CANCELLED, 3),
    ]
    (status,) = b.truth().orders
    assert status.state is S.CANCELLED and status.filled_qty == D(4)
    with pytest.raises(ValueError):
        b.confirm_cancel("c1")


def test_full_fill_during_deferred_cancel_rejects_cancel() -> None:
    b = _broker(faults=FaultPlan().on_cancel("c1", CancelFault.DEFER))
    b.submit(_req())
    b.cancel("c1")
    b.fill("c1", D(10), D(100))
    late = b.confirm_cancel("c1")
    assert late.kind is ExecKind.CANCEL_REJECTED and late.reason == "ORDER_NOT_OPEN"
    assert not b.is_cancel_pending("c1")


def test_cancel_faults_and_unknown_order() -> None:
    plan = FaultPlan().on_cancel("c1", CancelFault.REJECT, ReasonCode.UNKNOWN_ORDER)
    plan.on_cancel("c2", CancelFault.TIMEOUT_AFTER_ACCEPT)
    b = _broker(faults=plan)
    b.submit(_req("c1"))
    b.submit(_req("c2"))
    rejected = b.cancel("c1")
    assert not rejected.accepted and rejected.reason_code is ReasonCode.UNKNOWN_ORDER
    assert b.cancel("c1").accepted  # directiva consumată; anulare imediată
    with pytest.raises(BrokerTimeoutError):
        b.cancel("c2")
    assert b.cancel("c2").accepted  # anularea fusese aplicată la broker
    assert b.cancel("c9").reason_code is ReasonCode.UNKNOWN_ORDER
    events = [(e.client_order_id, e.kind) for e in _drain(b)]
    assert events == [
        ("c1", ExecKind.ACK),
        ("c2", ExecKind.ACK),
        ("c1", ExecKind.CANCEL_REJECTED),
        ("c1", ExecKind.CANCELLED),
        ("c2", ExecKind.CANCELLED),
    ]


# --------------------------------------------------------------------------- stare reală


def test_ground_truth_positions_cash_and_fills() -> None:
    b = _broker()
    b.submit(_req("c1", qty="10"))
    b.fill("c1", D(10), D("100.50"), D("1.25"))
    b.submit(_req("c2", qty="4", side="SELL"))
    b.fill("c2", D(4), D(110), D(1))
    truth = b.truth()
    assert truth.positions == {"XYZ": D(6)}
    assert truth.cash == D(10000) - D("1005.00") - D("1.25") + D(440) - D(1)
    assert [f.qty for f in b.fills] == [D(10), D(4)]
    with pytest.raises(ValueError):
        b.fill("c1", D(1), D(100))  # ordin executat complet
    b.submit(_req("c3", qty="2"))
    expired = b.expire("c3")
    assert expired.kind is ExecKind.EXPIRED and expired.qty == D(2)


# --------------------------------------------------------------------------- cap la cap cu OMS


def _oms_order(m: OrderManager, n: int = 1, qty: str = "10") -> tuple[str, OrderRequest]:
    intent = OrderIntent(
        intent_id=f"i-{n}", signal_id=f"s-{n}", instrument="XYZ", side="BUY", ref_price=D(100)
    )
    decision = RiskDecision(intent_id=intent.intent_id, approved=True, qty=D(qty))
    res = m.create_order(intent, decision, run_id="r", strategy_id="mr", signal_seq=n, ts=T0)
    coid = res.order.client_order_id
    assert m.submit(coid, T0).send
    return coid, order_request_from(m.order(coid), intent)


def _pump(m: OrderManager, b: FakeBroker) -> list[ExecOutcome]:
    out: list[ExecOutcome] = []
    for event in b.events():
        out.extend(m.on_execution(event))
    return out


def test_e2e_duplicates_are_retransmissions() -> None:
    m, b = OrderManager(ListSink()), _broker()
    coid, req = _oms_order(m)
    b.faults.duplicate(coid, 2)
    b.submit(req)
    b.fill(coid, D(10), D(100), D(1))
    statuses = [o.status for o in _pump(m, b)]
    assert statuses == [ExecStatus.APPLIED, ExecStatus.APPLIED, ExecStatus.RETRANSMISSION]
    order = m.order(coid)
    assert order.state is S.FILLED and order.filled_qty == D(10)


def test_e2e_reorder_buffers_and_drains() -> None:
    m, b = OrderManager(ListSink()), _broker()
    coid, req = _oms_order(m)
    b.submit(req)
    b.fill(coid, D(4), D(100))
    b.fill(coid, D(6), D(101))
    b.reorder([2, 1, 0])
    out = _pump(m, b)
    assert [(o.status, o.event.seq) for o in out] == [
        (ExecStatus.BUFFERED, 3),
        (ExecStatus.BUFFERED, 2),
        (ExecStatus.APPLIED, 1),
        (ExecStatus.APPLIED, 2),
        (ExecStatus.APPLIED, 3),
    ]
    assert out[0].request is not None and out[0].request.missing_seqs == (1, 2)
    assert m.order(coid).state is S.FILLED and m.missing_seqs(coid) == ()


def test_e2e_dropped_message_is_requested_and_resent() -> None:
    m, b = OrderManager(ListSink()), _broker()
    coid, req = _oms_order(m)
    b.faults.drop(coid, 2)
    b.submit(req)
    b.fill(coid, D(4), D(100))
    b.fill(coid, D(6), D(100))
    out = _pump(m, b)
    request = out[-1].request
    assert out[-1].status is ExecStatus.BUFFERED and request is not None
    assert request.missing_seqs == (2,)
    b.resend(request.client_order_id, request.missing_seqs)
    assert [o.status for o in _pump(m, b)] == [ExecStatus.APPLIED, ExecStatus.APPLIED]
    assert m.order(coid).filled_qty == D(10) and m.order(coid).state is S.FILLED


def test_e2e_timeout_after_accept_reconciled_from_snapshot() -> None:
    sink = ListSink()
    m, b = OrderManager(sink), _broker()
    coid, req = _oms_order(m)
    b.faults.on_submit(coid, SubmitFault.TIMEOUT_AFTER_ACCEPT)
    with pytest.raises(BrokerTimeoutError):
        b.submit(req)
    assert m.submit_timeout(coid, T0).order.state is S.UNKNOWN
    assert [o.status for o in _pump(m, b)] == [ExecStatus.HELD]  # ACK reținut cât e înghețat
    snap = b.snapshot()
    assert snap.complete
    (status,) = [o for o in snap.orders if o.client_order_id == coid]
    reflected = [x for x in snap.execution_ids if x.startswith(f"{ACCT}:{coid}:")]
    replay = m.reconcile(
        coid,
        ts=T0,
        state=status.state,
        filled_qty=status.filled_qty,
        avg_fill_price=status.avg_fill_price,
        broker_order_id=status.broker_order_id,
        reflected_exec_ids=reflected,
        next_seq=len(reflected) + 1,
    )
    assert [o.status for o in replay] == [ExecStatus.RETRANSMISSION]
    assert m.order(coid).state is S.ACKNOWLEDGED and not m.is_frozen(coid)
    b.fill(coid, D(10), D(100))
    assert [o.status for o in _pump(m, b)] == [ExecStatus.APPLIED]
    assert m.order(coid).state is S.FILLED
    assert RecordType.RECONCILED in [r.type for r in sink.records]


def test_e2e_fill_during_cancel() -> None:
    m, b = OrderManager(ListSink()), _broker()
    coid, req = _oms_order(m)
    b.faults.on_cancel(coid, CancelFault.DEFER)
    b.submit(req)
    _pump(m, b)
    assert m.request_cancel(coid, T0).send
    assert b.cancel(coid).accepted
    b.fill(coid, D(4), D(100))
    b.confirm_cancel(coid)
    assert all(o.status is ExecStatus.APPLIED for o in _pump(m, b))
    order = m.order(coid)
    assert order.state is S.CANCELLED and order.filled_qty == D(4)
