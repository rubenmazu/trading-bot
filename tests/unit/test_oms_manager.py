"""Teste unitare pentru `oms/manager.py` (Req 9.3–9.6, 10.1–10.5)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from qts.core.models import CostBreakdown, ExecKind, ExecutionEvent, Order, OrderIntent, OrderState
from qts.oms.idempotency import idempotency_key
from qts.oms.manager import (
    ExecStatus,
    IncidentCode,
    InstrumentBlockedError,
    JournalOmsSink,
    ListSink,
    OmsRecord,
    OrderManager,
    ReconciliationMismatchError,
    RecordType,
)
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.risk.engine import RiskDecision
from qts.risk.limits import RejectReason

T0 = datetime(2026, 1, 5, 9, tzinfo=UTC)
S = OrderState
D = Decimal


def _intent(instrument: str = "XYZ", n: int = 1) -> OrderIntent:
    return OrderIntent(
        intent_id=f"i-{instrument}-{n}",
        signal_id=f"s-{n}",
        instrument=instrument,
        side="BUY",
        ref_price=D(100),
    )


def _approve(intent: OrderIntent, qty: str = "10") -> RiskDecision:
    return RiskDecision(intent_id=intent.intent_id, approved=True, qty=D(qty))


def _create(m: OrderManager, instrument: str = "XYZ", n: int = 1, qty: str = "10") -> str:
    intent = _intent(instrument, n)
    res = m.create_order(
        intent, _approve(intent, qty), run_id="run1", strategy_id="mr", signal_seq=n, ts=T0
    )
    return res.order.client_order_id


def _ev(
    coid: str,
    exec_id: str,
    kind: ExecKind,
    qty: str | None = None,
    price: str | None = None,
    *,
    commission: str | None = None,
    seq: int | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        broker_exec_id=exec_id,
        client_order_id=coid,
        kind=kind,
        qty=D(qty) if qty else None,
        price=D(price) if price else None,
        commission=D(commission) if commission else None,
        broker_order_id="B-" + coid[:4] if kind is ExecKind.ACK else None,
        ts_broker=T0,
        ts_receipt=T0,
        seq=seq,
    )


def _live(m: OrderManager, instrument: str = "XYZ", n: int = 1, qty: str = "10") -> str:
    """Ordin ACKNOWLEDGED, fără secvențiere."""
    coid = _create(m, instrument, n, qty)
    assert m.submit(coid, T0).send
    m.on_execution(_ev(coid, f"ack-{coid}", ExecKind.ACK))
    assert m.order(coid).state is S.ACKNOWLEDGED
    return coid


def _types(sink: ListSink) -> list[RecordType]:
    return [r.type for r in sink.records]


def _invariant(order: Order) -> None:
    assert order.filled_qty + order.remaining_qty == order.qty


@pytest.fixture
def sink() -> ListSink:
    return ListSink()


@pytest.fixture
def m(sink: ListSink) -> OrderManager:
    return OrderManager(sink)


# --------------------------------------------------------------------------- creare / trimitere


def test_create_assigns_deterministic_key_and_is_idempotent(
    m: OrderManager, sink: ListSink
) -> None:
    coid = _create(m)
    assert coid == idempotency_key("run1", "mr", "XYZ", 1)
    order = m.order(coid)
    assert order.state is S.APPROVED and order.qty == D(10)
    again = _create(m)  # aceeași cheie, de exemplu după repornire
    assert again == coid and len(m.orders) == 1
    assert _types(sink) == [RecordType.ORDER_CREATED, RecordType.ORDER_DUPLICATE]


def test_rejected_or_mismatched_decision_is_refused(m: OrderManager) -> None:
    intent = _intent()
    rejected = RiskDecision(
        intent_id=intent.intent_id, approved=False, reason=RejectReason.KILL_SWITCH_ACTIVE
    )
    with pytest.raises(ValueError):
        m.create_order(intent, rejected, run_id="r", strategy_id="s", signal_seq=1, ts=T0)
    other = RiskDecision(intent_id="other", approved=True, qty=D(1))
    with pytest.raises(ValueError):
        m.create_order(intent, other, run_id="r", strategy_id="s", signal_seq=1, ts=T0)


def test_submit_is_write_ahead_and_sent_once(m: OrderManager, sink: ListSink) -> None:
    coid = _create(m)
    first = m.submit(coid, T0)
    assert first.send and first.order.state is S.SUBMITTED
    last = sink.records[-1]
    assert last.seq == first.record_seq and last.payload["to_state"] == "SUBMITTED"
    second = m.submit(coid, T0)
    assert not second.send and sink.records[-1].type is RecordType.COMMAND_DUPLICATE


class _FailingSink:
    def __init__(self) -> None:
        self.fail = False

    def record(self, record: OmsRecord) -> None:
        if self.fail:
            raise OSError("disc plin")


def test_sink_failure_leaves_state_unchanged() -> None:
    sink = _FailingSink()
    m = OrderManager(sink)
    coid = _create(m)
    sink.fail = True
    with pytest.raises(OSError):
        m.submit(coid, T0)
    assert m.order(coid).state is S.APPROVED


# --------------------------------------------------------------------------- execuții


def test_partial_fills_update_vwap_costs_and_quantities(m: OrderManager) -> None:
    coid = _live(m)
    out1 = m.on_execution(_ev(coid, "f1", ExecKind.PARTIAL_FILL, "4", "100", commission="1"))
    o = m.order(coid)
    assert out1[0].status is ExecStatus.APPLIED and o.state is S.PARTIALLY_FILLED
    assert (o.filled_qty, o.remaining_qty, o.avg_fill_price) == (D(4), D(6), D(100))
    _invariant(o)
    fill = out1[0].fill
    assert fill is not None and fill.fill_id == "f1" and fill.costs.commission == D(1)

    # parțială care epuizează restul: normalizată la FILL
    out2 = m.on_execution(_ev(coid, "f2", ExecKind.PARTIAL_FILL, "6", "110", commission="1"))
    o = m.order(coid)
    assert out2[0].status is ExecStatus.APPLIED and o.state is S.FILLED
    assert o.filled_qty == D(10) and o.remaining_qty == 0
    assert o.avg_fill_price == D(106)  # (4×100 + 6×110) / 10
    assert o.costs.commission == D(2)
    _invariant(o)


def test_injected_cost_function_is_used(sink: ListSink) -> None:
    def costs(order: Order, event: ExecutionEvent) -> CostBreakdown:
        return CostBreakdown(commission=D("0.5"), spread=D("0.1"))

    m = OrderManager(sink, cost_fn=costs)
    coid = _live(m)
    out = m.on_execution(_ev(coid, "f1", ExecKind.FILL, "10", "100"))
    assert out[0].fill is not None and out[0].fill.costs.total == D("0.6")
    assert m.order(coid).costs.total == D("0.6")


def test_duplicate_execution_is_ignored_and_correlated(m: OrderManager, sink: ListSink) -> None:
    coid = _live(m)
    event = _ev(coid, "f1", ExecKind.PARTIAL_FILL, "4", "100", commission="1")
    first = m.on_execution(event)[0]
    before = m.order(coid)
    dup = m.on_execution(
        event.model_copy(update={"ts_receipt": datetime(2026, 1, 5, 10, tzinfo=UTC)})
    )
    assert len(dup) == 1 and dup[0].status is ExecStatus.RETRANSMISSION and dup[0].fill is None
    assert m.order(coid) == before
    rec = sink.records[-1]
    assert rec.type is RecordType.EXEC_RETRANSMISSION
    assert rec.payload["original_record_seq"] == first.record_seq
    assert rec.payload["content_matches"] is True


def test_reused_exec_id_with_other_content_is_incident(m: OrderManager) -> None:
    coid = _live(m)
    m.on_execution(_ev(coid, "f1", ExecKind.PARTIAL_FILL, "4", "100"))
    before = m.order(coid)
    out = m.on_execution(_ev(coid, "f1", ExecKind.PARTIAL_FILL, "5", "100"))
    assert out[-1].incident is IncidentCode.EXEC_ID_CONFLICT
    assert m.order(coid) == before and m.is_frozen(coid)


@pytest.mark.parametrize(
    ("kind", "qty", "detail"),
    [
        (ExecKind.PARTIAL_FILL, "11", "remaining"),  # supra-execuție
        (ExecKind.FILL, "11", "remaining"),
        (ExecKind.FILL, "4", "!="),  # FILL ambiguu, mai mic decât restul
    ],
)
def test_overfill_and_ambiguous_fill_are_incidents(
    m: OrderManager, kind: ExecKind, qty: str, detail: str
) -> None:
    coid = _live(m)
    before = m.order(coid)
    out = m.on_execution(_ev(coid, "bad", kind, qty, "100"))
    assert out[0].status is ExecStatus.INCIDENT
    assert out[0].incident is IncidentCode.TRANSITION_INVALID and detail in out[0].detail
    assert m.order(coid) == before and out[0].fill is None
    assert m.is_frozen(coid) and m.is_instrument_blocked("XYZ")


def test_invalid_broker_transition_keeps_state_and_freezes(m: OrderManager, sink: ListSink) -> None:
    coid = _create(m)
    m.submit(coid, T0)
    before = m.order(coid)
    out = m.on_execution(_ev(coid, "x1", ExecKind.CANCELLED))  # SUBMITTED + CANCELLED
    assert out[0].incident is IncidentCode.TRANSITION_INVALID
    assert m.order(coid) == before
    rec = sink.records[-1]
    assert rec.type is RecordType.INCIDENT and rec.payload["fsm_reason"] == "TRANSITION_NOT_ALLOWED"
    # modificările sunt înghețate: comenzile sunt refuzate, execuțiile reținute
    assert not m.request_cancel(coid, T0).send
    held = m.on_execution(_ev(coid, "a1", ExecKind.ACK))
    assert held[0].status is ExecStatus.HELD and m.order(coid) == before


def test_unknown_order_execution_is_incident(m: OrderManager) -> None:
    out = m.on_execution(_ev("nope", "e1", ExecKind.ACK))
    assert out[0].incident is IncidentCode.UNKNOWN_ORDER and out[0].order is None


def test_invalid_cost_is_incident_without_effect(m: OrderManager) -> None:
    coid = _live(m)
    before = m.order(coid)
    out = m.on_execution(_ev(coid, "f1", ExecKind.FILL, "10", "100", commission="-1"))
    assert out[0].incident is IncidentCode.COST_INVALID and m.order(coid) == before


# --------------------------------------------------------------------------- anulări


def test_internal_invalid_command_is_rejected_without_freezing(
    m: OrderManager, sink: ListSink
) -> None:
    coid = _create(m)
    m.submit(coid, T0)
    res = m.request_cancel(coid, T0)  # SUBMITTED nu acceptă CANCEL_REQUEST
    assert not res.send and res.order.state is S.SUBMITTED
    assert sink.records[-1].type is RecordType.TRANSITION_REJECTED
    assert not m.is_frozen(coid)


def test_cancel_partial_fill_then_cancelled(m: OrderManager, sink: ListSink) -> None:
    coid = _live(m)
    assert m.request_cancel(coid, T0).send
    assert not m.request_cancel(coid, T0).send  # o singură cerere în curs
    out = m.on_execution(_ev(coid, "f1", ExecKind.PARTIAL_FILL, "3", "100"))
    assert out[0].status is ExecStatus.APPLIED and out[0].fill is not None
    assert m.order(coid).state is S.CANCEL_PENDING and m.cancel_outstanding(coid)
    restored = sink.records[-1]
    assert restored.type is RecordType.CANCEL_PENDING_RESTORED
    assert restored.payload["broker_request_sent"] is False
    assert sink.records[-2].payload["to_state"] == "PARTIALLY_FILLED"

    m.on_execution(_ev(coid, "c1", ExecKind.CANCELLED))
    o = m.order(coid)
    assert o.state is S.CANCELLED and o.filled_qty == D(3) and o.remaining_qty == D(7)
    assert sink.records[-1].payload["order"]["remaining_qty"] == "7"
    assert not m.cancel_outstanding(coid)


def test_cancel_partial_fill_then_cancel_rejected(m: OrderManager) -> None:
    coid = _live(m)
    m.request_cancel(coid, T0)
    m.on_execution(_ev(coid, "f1", ExecKind.PARTIAL_FILL, "3", "100"))
    m.on_execution(_ev(coid, "cr", ExecKind.CANCEL_REJECTED))
    o = m.order(coid)
    assert o.state is S.PARTIALLY_FILLED and o.remaining_qty == D(7)
    assert not m.cancel_outstanding(coid) and not m.is_frozen(coid)


def test_cancel_rejected_without_fills_returns_to_acknowledged(m: OrderManager) -> None:
    coid = _live(m)
    m.request_cancel(coid, T0)
    m.on_execution(_ev(coid, "cr", ExecKind.CANCEL_REJECTED))
    assert m.order(coid).state is S.ACKNOWLEDGED
    assert m.request_cancel(coid, T0).send  # o nouă anulare este posibilă


def test_fill_during_cancel_then_late_cancel_reject_is_ignored(m: OrderManager) -> None:
    coid = _live(m)
    m.request_cancel(coid, T0)
    m.on_execution(_ev(coid, "f1", ExecKind.FILL, "10", "100"))
    assert m.order(coid).state is S.FILLED
    out = m.on_execution(_ev(coid, "cr", ExecKind.CANCEL_REJECTED))
    assert out[0].status is ExecStatus.IGNORED
    assert not m.is_frozen(coid) and not m.is_instrument_blocked("XYZ")


# --------------------------------------------------------------------------- UNKNOWN


def test_unknown_freezes_order_and_instrument_until_reconciliation(
    m: OrderManager, sink: ListSink
) -> None:
    coid = _create(m)
    m.submit(coid, T0)
    m.submit_timeout(coid, T0)
    assert m.order(coid).state is S.UNKNOWN and m.is_frozen(coid)
    assert m.blocked_instruments() == frozenset({"XYZ"})
    with pytest.raises(InstrumentBlockedError):
        _create(m, n=2)
    assert sink.records[-1].type is RecordType.ORDER_BLOCKED
    other = _create(m, instrument="ABC", n=3)  # instrumentele independente continuă
    assert m.order(other).state is S.APPROVED
    assert not m.request_cancel(coid, T0).send

    # mesajele sosite în UNKNOWN sunt reținute
    held = [
        m.on_execution(_ev(coid, "ack", ExecKind.ACK))[0],
        m.on_execution(_ev(coid, "f1", ExecKind.PARTIAL_FILL, "4", "100"))[0],
        m.on_execution(_ev(coid, "f2", ExecKind.PARTIAL_FILL, "2", "102"))[0],
    ]
    assert all(h.status is ExecStatus.HELD for h in held)
    assert m.order(coid).state is S.UNKNOWN

    with pytest.raises(ReconciliationMismatchError):
        m.reconcile(coid, ts=T0, state=S.FILLED, filled_qty=D(4), avg_fill_price=D(100))
    assert m.is_frozen(coid)

    # snapshot-ul brokerului include deja ACK și f1; f2 este nou
    out = m.reconcile(
        coid,
        ts=T0,
        state=S.PARTIALLY_FILLED,
        filled_qty=D(4),
        avg_fill_price=D(100),
        reflected_exec_ids=["ack", "f1"],
    )
    assert [o.status for o in out] == [
        ExecStatus.RETRANSMISSION,
        ExecStatus.RETRANSMISSION,
        ExecStatus.APPLIED,
    ]
    o = m.order(coid)
    assert o.state is S.PARTIALLY_FILLED and o.filled_qty == D(6)
    assert o.avg_fill_price == D(604) / D(6)
    _invariant(o)
    assert not m.is_instrument_blocked("XYZ")
    _create(m, n=4)  # ordinele noi sunt din nou permise


# --------------------------------------------------------------------------- secvență


def test_sequence_gap_suspends_only_dependent_order(m: OrderManager, sink: ListSink) -> None:
    a = _create(m, "XYZ", 1)
    b = _create(m, "ABC", 2)
    m.submit(a, T0)
    m.submit(b, T0)
    assert m.on_execution(_ev(a, "a1", ExecKind.ACK, seq=1))[0].status is ExecStatus.APPLIED

    out3 = m.on_execution(_ev(a, "a3", ExecKind.PARTIAL_FILL, "2", "100", seq=3))
    assert out3[0].status is ExecStatus.BUFFERED
    req = out3[0].request
    assert req is not None and req.client_order_id == a and req.missing_seqs == (2,)
    assert sink.records[-1].type is RecordType.MISSING_MESSAGES_REQUEST
    assert m.missing_seqs(a) == (2,)

    out4 = m.on_execution(_ev(a, "a4", ExecKind.PARTIAL_FILL, "3", "100", seq=4))
    assert out4[0].status is ExecStatus.BUFFERED and out4[0].request is None  # deja cerut
    assert m.order(a).filled_qty == 0

    # retransmisia unui mesaj din buffer este corelată, nu dublată
    dup = m.on_execution(_ev(a, "a3", ExecKind.PARTIAL_FILL, "2", "100", seq=3))
    assert dup[0].status is ExecStatus.RETRANSMISSION
    assert (
        dup[0].record_seq and sink.records[-1].payload["original_record_seq"] == out3[0].record_seq
    )

    # ordinul independent continuă
    assert m.on_execution(_ev(b, "b1", ExecKind.ACK, seq=1))[0].status is ExecStatus.APPLIED
    assert m.order(b).state is S.ACKNOWLEDGED

    # completarea golului eliberează bufferul în ordine
    out2 = m.on_execution(_ev(a, "a2", ExecKind.PARTIAL_FILL, "1", "100", seq=2))
    assert [o.event.broker_exec_id for o in out2] == ["a2", "a3", "a4"]
    assert all(o.status is ExecStatus.APPLIED for o in out2)
    assert m.order(a).filled_qty == D(6) and m.missing_seqs(a) == ()
    _invariant(m.order(a))


def test_consumed_sequence_with_new_exec_id_is_incident(m: OrderManager) -> None:
    coid = _create(m)
    m.submit(coid, T0)
    m.on_execution(_ev(coid, "a1", ExecKind.ACK, seq=1))
    out = m.on_execution(_ev(coid, "zz", ExecKind.PARTIAL_FILL, "1", "100", seq=1))
    assert out[0].incident is IncidentCode.SEQ_CONFLICT and m.is_frozen(coid)


# --------------------------------------------------------------------------- jurnal


def test_journal_sink_appends_correlated_records() -> None:
    journal = Journal(open_db(":memory:"))
    m = OrderManager(JournalOmsSink(journal))
    coid = _live(m)
    m.on_execution(_ev(coid, "f1", ExecKind.FILL, "10", "100"))
    records = journal.by_correlation(coid)
    assert [r.type for r in records] == [
        "oms.ORDER_CREATED",
        "oms.TRANSITION",
        "oms.EXEC_APPLIED",
        "oms.EXEC_APPLIED",
    ]
    assert [r.payload["oms_seq"] for r in records] == [1, 2, 3, 4]
    assert records[-1].payload["order"]["state"] == "FILLED"
