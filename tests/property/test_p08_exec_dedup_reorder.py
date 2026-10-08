"""P8: duplicatele și reordonările execuțiilor nu schimbă starea finală (Req 10.2, 10.3).

Se generează un istoric canonic valid de execuții pentru unul sau mai multe ordine (ACK, apoi
execuții parțiale cu suma ≤ qty, opțional execuția finală, comisioane; alternativ REJECT sau
EXPIRED fără execuții), fiecare mesaj numerotat pe ordin cu `seq = 1..n`. Livrarea către
managerul B este o permutare arbitrară (în cadrul ordinului și între ordine) în care sunt
intercalate oricâte duplicate (același `broker_exec_id` și conținut identic, eventual alt
`ts_receipt`). Managerul A primește istoricul canonic, în ordine, fără duplicate. Fiecare
`ExecOutcome.fill` este aplicat unui `Portfolio` propriu. Se verifică egalitatea stărilor finale
ale ordinelor și portofoliilor, absența incidentelor și faptul că fiecare duplicat produce numai
`RETRANSMISSION` (niciodată o a doua execuție).

Varianta nesecvențiată (`seq=None`): fără numărul de secvență, managerul aplică mesajele direct,
deci ordinea FSM a mesajelor unui ordin contează (o execuție înaintea ACK ar fi o tranziție
invalidă — comportament corect, nu o încălcare a P8). Acolo se testează numai duplicatele
(livrate oricând după original) și intercalarea între ordine diferite, cu ordinea păstrată pe
fiecare ordin.

Instrumentele sunt distincte pe ordin și cantitățile/prețurile au puține zecimale, astfel încât
aritmetica portofoliului este exactă și comutativă între ordine.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

from hypothesis import event, given
from hypothesis import strategies as st

from qts.core.models import CostBreakdown, ExecKind, ExecutionEvent, OrderIntent, OrderState
from qts.oms.manager import ExecOutcome, ExecStatus, ListSink, OrderManager
from qts.portfolio.portfolio import Portfolio, PortfolioState
from qts.risk.engine import RiskDecision

T0 = datetime(2026, 1, 5, 9, tzinfo=UTC)
D = Decimal
UNIT = D("0.01")
CASH = D(10_000_000)
STATS: Counter[str] = Counter()

price_st = st.decimals(min_value=D("1.00"), max_value=D("500.00"), places=2)
commission_st = st.one_of(
    st.none(), st.decimals(min_value=D("0.00"), max_value=D("5.00"), places=2)
)


@dataclass(frozen=True)
class OrderPlan:
    instrument: str
    qty: Decimal
    events: tuple[ExecutionEvent, ...]  # istoricul canonic, cu seq = 1..n (sau None)


def _coid(manager: OrderManager, plan: OrderPlan, n: int) -> str:
    intent = OrderIntent(
        intent_id=f"i-{n}",
        signal_id=f"s-{n}",
        instrument=plan.instrument,
        side="BUY",
        ref_price=D(100),
    )
    decision = RiskDecision(intent_id=intent.intent_id, approved=True, qty=plan.qty)
    res = manager.create_order(
        intent, decision, run_id="run", strategy_id="p8", signal_seq=n, ts=T0
    )
    coid = res.order.client_order_id
    assert manager.submit(coid, T0).send
    return coid


def _coid_for(n: int, instrument: str) -> str:
    # client_order_id este deterministic: îl calculăm cu un manager de probă.
    probe = OrderManager(ListSink())
    plan = OrderPlan(instrument, D(1), ())
    return _coid(probe, plan, n)


@st.composite
def _order_plan(draw: st.DrawFn, n: int, sequenced: bool) -> OrderPlan:
    instrument = f"I{n}"
    coid = _coid_for(n, instrument)
    units = draw(st.integers(2, 10_000))  # qty în pași de 0.01
    qty = D(units) * UNIT
    ending = draw(st.sampled_from(["fill", "open", "expire", "reject"]))
    clock = iter(range(1, 10_000))

    def ev(
        kind: ExecKind,
        *,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        commission: Decimal | None = None,
        broker_order_id: str | None = None,
        reason: str | None = None,
    ) -> ExecutionEvent:
        idx = next(clock)
        ts = T0 + timedelta(seconds=idx, milliseconds=n)
        return ExecutionEvent(
            broker_exec_id=f"x-{n}-{idx}",
            client_order_id=coid,
            kind=kind,
            qty=qty,
            price=price,
            commission=commission,
            broker_order_id=broker_order_id,
            reason=reason,
            ts_broker=ts,
            ts_receipt=ts,
            seq=idx if sequenced else None,
        )

    events: list[ExecutionEvent] = []
    if ending == "reject":
        events.append(ev(ExecKind.REJECT, reason="insufficient"))
        return OrderPlan(instrument, qty, tuple(events))
    events.append(ev(ExecKind.ACK, broker_order_id=f"B-{n}"))
    if ending == "expire":
        events.append(ev(ExecKind.EXPIRED))
        return OrderPlan(instrument, qty, tuple(events))
    cuts = sorted(
        draw(st.lists(st.integers(1, units - 1), unique=True, max_size=min(4, units - 1)))
    )
    bounds = [0, *cuts, units]
    segments = [b - a for a, b in pairwise(bounds)]
    if ending == "open":
        segments = segments[:-1]  # ultimul segment rămâne neexecutat
    for i, seg in enumerate(segments):
        last = ending == "fill" and i == len(segments) - 1
        # Ultima execuție poate fi `FILL` sau un `PARTIAL_FILL` care epuizează restul (normalizat).
        kind = ExecKind.PARTIAL_FILL
        if last:
            kind = draw(st.sampled_from([ExecKind.FILL, ExecKind.PARTIAL_FILL]))
        events.append(
            ev(kind, qty=D(seg) * UNIT, price=draw(price_st), commission=draw(commission_st))
        )
    return OrderPlan(instrument, qty, tuple(events))


@st.composite
def _plans(draw: st.DrawFn, sequenced: bool) -> list[OrderPlan]:
    k = draw(st.integers(1, 3))
    return [draw(_order_plan(n, sequenced)) for n in range(1, k + 1)]


def _dup(e: ExecutionEvent, offset_ms: int) -> ExecutionEvent:
    """Retransmisie: conținut identic, eventual alt moment de recepție locală."""
    return e.model_copy(update={"ts_receipt": e.ts_receipt + timedelta(milliseconds=offset_ms)})


@st.composite
def _sequenced_case(draw: st.DrawFn) -> tuple[list[OrderPlan], list[ExecutionEvent]]:
    plans = draw(_plans(True))
    pool: list[ExecutionEvent] = []
    for plan in plans:
        for e in plan.events:
            pool.append(e)
            for _ in range(draw(st.integers(0, 2))):
                pool.append(_dup(e, draw(st.integers(0, 5_000))))
    delivery = draw(st.permutations(pool))
    return plans, list(delivery)


@st.composite
def _unsequenced_case(draw: st.DrawFn) -> tuple[list[OrderPlan], list[ExecutionEvent]]:
    plans = draw(_plans(False))
    # Intercalare între ordine, cu ordinea canonică păstrată pe fiecare ordin.
    queues = [list(p.events) for p in plans]
    merged: list[ExecutionEvent] = []
    while any(queues):
        live = [i for i, q in enumerate(queues) if q]
        merged.append(queues[draw(st.sampled_from(live))].pop(0))
    # Duplicate inserate oriunde după prima apariție a originalului.
    delivery = list(merged)
    for e in merged:
        for _ in range(draw(st.integers(0, 2))):
            first = delivery.index(e)
            pos = draw(st.integers(first + 1, len(delivery)))
            delivery.insert(pos, _dup(e, draw(st.integers(1, 5_000))))
    return plans, delivery


def _setup(plans: list[OrderPlan]) -> tuple[OrderManager, Portfolio, list[str]]:
    manager = OrderManager(ListSink())
    coids = [_coid(manager, plan, n) for n, plan in enumerate(plans, start=1)]
    return manager, Portfolio.with_cash(CASH), coids


def _route(outcomes: list[ExecOutcome], portfolio: Portfolio) -> None:
    for o in outcomes:
        assert o.status is not ExecStatus.INCIDENT, (o.incident, o.detail)
        if o.fill is not None:
            assert o.status is ExecStatus.APPLIED
            portfolio.apply_fill(o.fill)


def _run_canonical(plans: list[OrderPlan]) -> tuple[OrderManager, Portfolio, list[str]]:
    manager, portfolio, coids = _setup(plans)
    for plan in plans:
        for e in plan.events:
            outcomes = manager.on_execution(e)
            assert [o.status for o in outcomes] == [ExecStatus.APPLIED]
            _route(outcomes, portfolio)
    return manager, portfolio, coids


def _run_noisy(
    plans: list[OrderPlan], delivery: list[ExecutionEvent]
) -> tuple[OrderManager, Portfolio, list[str]]:
    manager, portfolio, coids = _setup(plans)
    seen: set[str] = set()
    pending: set[str] = set()  # trimise, încă neaplicate (în buffer)
    applied: Counter[str] = Counter()
    for e in delivery:
        outcomes = manager.on_execution(e)
        _route(outcomes, portfolio)
        if e.broker_exec_id in seen:
            # Duplicatul produce exclusiv o retransmisie, fără efect.
            assert [o.status for o in outcomes] == [ExecStatus.RETRANSMISSION]
            assert outcomes[0].fill is None
            kind = "dup_of_buffered" if e.broker_exec_id in pending else "dup_of_applied"
            STATS[kind] += 1
            event(kind)
            continue
        seen.add(e.broker_exec_id)
        statuses = [o.status for o in outcomes]
        if statuses == [ExecStatus.BUFFERED]:
            pending.add(e.broker_exec_id)
            STATS["buffered"] += 1
            event("buffered")
        else:
            assert ExecStatus.BUFFERED not in statuses
            if len(outcomes) > 1:
                STATS["drained"] += 1
                event("buffer drained")
        for o in outcomes:
            if o.status is ExecStatus.APPLIED:
                applied[o.event.broker_exec_id] += 1
                pending.discard(o.event.broker_exec_id)
    assert not pending
    assert all(v == 1 for v in applied.values())
    assert set(applied) == {e.broker_exec_id for p in plans for e in p.events}
    for coid in coids:
        assert manager.missing_seqs(coid) == ()
        assert not manager.is_frozen(coid)
    return manager, portfolio, coids


def _assert_same(
    a: tuple[OrderManager, Portfolio, list[str]], b: tuple[OrderManager, Portfolio, list[str]]
) -> None:
    ma, pa, coids = a
    mb, pb, coids_b = b
    assert coids == coids_b
    for coid in coids:
        oa, ob = ma.order(coid), mb.order(coid)
        assert oa.state is ob.state
        assert oa.filled_qty == ob.filled_qty
        assert oa.avg_fill_price == ob.avg_fill_price
        assert oa.costs == ob.costs
        assert oa.broker_order_id == ob.broker_order_id
        assert oa == ob
        event(f"final {oa.state}")
    sa: PortfolioState = pa.snapshot()
    sb: PortfolioState = pb.snapshot()
    assert sa.positions == sb.positions
    assert sa.cash_eur == sb.cash_eur
    assert sa.realized == sb.realized
    assert sa.realized_by_instrument == sb.realized_by_instrument
    assert sa.incidents == () and sb.incidents == ()
    assert sa == sb
    # Sanity: costurile cumulate ale ordinelor = costurile realizate ale portofoliului.
    total = sum((ma.order(c).costs for c in coids), CostBreakdown())
    assert total == sa.realized.costs


@given(_sequenced_case())
def test_property_8_duplicates_and_reorderings_do_not_change_final_state(
    case: tuple[list[OrderPlan], list[ExecutionEvent]],
) -> None:
    """**Validates: Requirements 10.2, 10.3**"""
    plans, delivery = case
    canonical = _run_canonical(plans)
    noisy = _run_noisy(plans, delivery)
    _assert_same(canonical, noisy)
    for coid in canonical[2]:
        if canonical[0].order(coid).state is OrderState.FILLED:
            STATS["filled"] += 1


@given(_unsequenced_case())
def test_property_8_unsequenced_duplicates_do_not_change_final_state(
    case: tuple[list[OrderPlan], list[ExecutionEvent]],
) -> None:
    """Fără `seq`, numai duplicatele (și intercalarea între ordine) sunt neutre.

    **Validates: Requirements 10.2, 10.3**
    """
    plans, delivery = case
    canonical = _run_canonical(plans)
    noisy = _run_noisy(plans, delivery)
    _assert_same(canonical, noisy)


def test_property_8_non_vacuity() -> None:
    """Se exercită buffere, goliri, duplicate ale mesajelor din buffer și ale celor aplicate."""
    STATS.clear()
    test_property_8_duplicates_and_reorderings_do_not_change_final_state()
    for key in ("buffered", "drained", "dup_of_buffered", "dup_of_applied", "filled"):
        assert STATS[key] > 0, f"scenariu neexercitat: {key} ({dict(STATS)})"
    STATS.clear()
    test_property_8_unsequenced_duplicates_do_not_change_final_state()
    assert STATS["dup_of_applied"] > 0
    assert STATS["buffered"] == 0  # fără seq nu există buffer
