"""P9 și P10: conservarea cantității și mașina de stări închisă (Req 9.2, 9.3, 9.4).

P9: pentru orice ordin, în orice stare, `0 ≤ filled_qty ≤ qty` și
`filled_qty + remaining_qty = qty`.
P10: orice tranziție absentă din tabel (sau incompatibilă cu cantitățile) lasă starea ordinului
neschimbată; orice tranziție acceptată ajunge într-o țintă din `TRANSITIONS[(stare, declanșator)]`.

Driverul de test este local (nu depinde de `oms/manager.py`): aplică secvențe arbitrare de
declanșatori unui `Order` folosind exclusiv `fsm.apply`. `filled_qty` se actualizează numai la o
tranziție acceptată: incremental pentru `PARTIAL_FILL`/`FILL`, absolut (generat consistent cu
starea țintă) pentru reconcilierile `RECONCILED_*` din `UNKNOWN`. Secvențele includ declanșatori
invalizi, supraexecuții, cantități zero/negative, declanșatori pe stări terminale, fluxuri de
anulare și reconcilieri.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from hypothesis import event, given
from hypothesis import strategies as st

from qts.core.models import Order, OrderState
from qts.oms.fsm import (
    TERMINAL_STATES,
    TRANSITIONS,
    TransitionCode,
    TransitionResult,
    Trigger,
    allowed,
    apply,
    is_terminal,
    targets,
)

Q = Decimal("0.0001")  # toate cantitățile au 4 zecimale → aritmetică exactă
ZERO = Decimal(0)

ALL_TRIGGERS = sorted(Trigger, key=str)
ALL_STATES = sorted(OrderState, key=str)
MISSING_PAIRS = [(s, t) for s in ALL_STATES for t in ALL_TRIGGERS if (s, t) not in TRANSITIONS]
FILL_TRIGGERS = frozenset({Trigger.PARTIAL_FILL, Trigger.FILL})
RECON_TRIGGERS = frozenset(t for t in Trigger if t.startswith("RECONCILED_"))

qty_st = st.decimals(min_value=Q, max_value=Decimal("10000"), places=4)
any_qty_st = st.decimals(
    min_value=Decimal("-100"), max_value=Decimal("20000"), places=4, allow_nan=False
)
fraction_st = st.tuples(st.integers(0, 10), st.integers(1, 10)).map(
    lambda kn: Decimal(kn[0]) / Decimal(kn[1])
)

# Specificația cantității executate: relativă la cantitatea rămasă sau brută (inclusiv ≤ 0,
# supraexecuție) sau absentă.
exec_spec_st = st.one_of(
    st.just(("remaining", None)),
    st.tuples(st.just("fraction"), fraction_st),
    st.tuples(st.just("over"), st.decimals(min_value=Q, max_value=Decimal("100"), places=4)),
    st.tuples(st.just("raw"), any_qty_st),
    st.just(("none", None)),
)

STATS: Counter[str] = Counter()


def _q(x: Decimal) -> Decimal:
    return x.quantize(Q, rounding=ROUND_DOWN)


def _resolve_exec(spec: tuple[str, Decimal | None], remaining: Decimal) -> Decimal | None:
    kind, value = spec
    if kind == "remaining":
        return remaining
    if kind == "fraction":
        assert value is not None
        return _q(remaining * value)
    if kind == "over":
        assert value is not None
        return remaining + value
    if kind == "raw":
        return value
    return None


def _make_order(qty: Decimal) -> Order:
    return Order(client_order_id="c-1", intent_id="i-1", instrument="TEST", side="BUY", qty=qty)


def _assert_p9(order: Order) -> None:
    assert ZERO <= order.filled_qty <= order.qty
    assert order.filled_qty + order.remaining_qty == order.qty
    assert order.remaining_qty >= ZERO
    # Coerența stării cu cantitatea executată.
    if order.state is OrderState.FILLED:
        assert order.remaining_qty == ZERO
    if order.state is OrderState.PARTIALLY_FILLED:
        assert ZERO < order.filled_qty < order.qty
    if order.state in (
        OrderState.CREATED,
        OrderState.APPROVED,
        OrderState.SUBMITTED,
        OrderState.UNKNOWN,
        OrderState.REJECTED_RISK,
    ):
        assert order.filled_qty == ZERO


def _recon_filled(trigger: Trigger, order: Order, frac: Decimal) -> Decimal:
    """Cantitatea executată absolută din snapshot-ul brokerului, consistentă cu ținta."""
    if trigger is Trigger.RECONCILED_FILLED:
        return order.qty
    if trigger is Trigger.RECONCILED_PARTIALLY_FILLED:
        partial = _q(order.qty * frac)
        if partial <= ZERO or partial >= order.qty:
            # Fallback: cea mai mică execuție parțială posibilă (sau 0 dacă qty == Q).
            partial = Q if order.qty > Q else ZERO
        return partial
    if trigger is Trigger.RECONCILED_CANCELLED:
        return min(_q(order.qty * frac), order.qty - Q) if order.qty > Q else ZERO
    return order.filled_qty  # ACKNOWLEDGED / REJECTED: nimic executat


@dataclass
class Step:
    trigger: Trigger
    exec_spec: tuple[str, Decimal | None]
    recon_frac: Decimal


def _drive(order: Order, step: Step) -> tuple[Order, TransitionResult]:
    state = order.state
    exec_qty = _resolve_exec(step.exec_spec, order.remaining_qty)
    res = apply(
        state,
        step.trigger,
        order_qty=order.qty,
        filled_qty=order.filled_qty,
        exec_qty=exec_qty,
    )
    assert res.from_state is state
    assert res.trigger is step.trigger
    if not res.ok:
        # P10: respingerea lasă starea (și cantitățile) neschimbate.
        assert res.new_state is state
        assert res.reason is not None
        return order, res

    # Închidere: tranziția acceptată există în tabel și ajunge într-o țintă declarată.
    assert (state, step.trigger) in TRANSITIONS
    assert res.new_state in TRANSITIONS[(state, step.trigger)]
    assert res.new_state in targets(state)

    filled = order.filled_qty
    if step.trigger in FILL_TRIGGERS:
        assert exec_qty is not None and exec_qty > ZERO
        filled = filled + exec_qty
    elif step.trigger in RECON_TRIGGERS:
        filled = _recon_filled(step.trigger, order, step.recon_frac)
    if (
        step.trigger is Trigger.RECONCILED_PARTIALLY_FILLED and filled == ZERO
    ):  # qty minimă: nu există execuție parțială validă; nu aplicăm reconcilierea
        return order, res

    # Reconstrucție validată (validatorul modelului impune invariantul P9).
    new_order = Order(**{**order.model_dump(), "state": res.new_state, "filled_qty": filled})
    return new_order, res


@st.composite
def _scenario(draw: st.DrawFn) -> tuple[Decimal, list[Step]]:
    qty = draw(qty_st)
    n = draw(st.integers(1, 30))
    steps: list[Step] = []
    # Declanșatorii se trag din toți, cu o preferință pentru cei permiși în starea simulată
    # (pentru a parcurge ciclul de viață complet); starea „simulată” e doar o euristică —
    # driverul real decide.
    sim = _make_order(qty)
    for _ in range(n):
        ok_triggers = sorted(allowed(sim.state), key=str)
        # Declanșatori care nu duc direct într-o stare terminală (ciclul de viață avansează).
        live = [
            t for t in ok_triggers if not TERMINAL_STATES.issuperset(TRANSITIONS[(sim.state, t)])
        ]
        fills = [t for t in ok_triggers if t in FILL_TRIGGERS]
        roll = draw(st.integers(0, 9))
        if roll >= 6 and fills:
            trig = draw(st.sampled_from(fills))
        elif roll >= 3 and live:
            trig = draw(st.sampled_from(live))
        elif roll >= 1 and ok_triggers:
            trig = draw(st.sampled_from(ok_triggers))
        else:
            trig = draw(st.sampled_from(ALL_TRIGGERS))
        step = Step(trig, draw(exec_spec_st), draw(fraction_st))
        sim, _ = _drive(sim, step)
        steps.append(step)
    return qty, steps


@given(_scenario())
def test_property_9_10_order_driver_conserves_qty_and_rejects_unchanged(
    scenario: tuple[Decimal, list[Step]],
) -> None:
    """**Validates: Requirements 9.2, 9.3, 9.4**"""
    qty, steps = scenario
    order = _make_order(qty)
    _assert_p9(order)
    for step in steps:
        before = order
        order, res = _drive(order, step)
        _assert_p9(order)
        if res.ok:
            STATS["accepted"] += 1
            if step.trigger in FILL_TRIGGERS:
                STATS["fill_accepted"] += 1
                event("fill accepted")
            if step.trigger is Trigger.CANCEL_CONFIRMED:
                STATS["cancel_confirmed"] += 1
            if step.trigger is Trigger.CANCEL_REJECTED:
                STATS["cancel_rejected"] += 1
            if step.trigger in RECON_TRIGGERS:
                STATS["reconciled"] += 1
                event("reconciled")
        else:
            STATS["rejected"] += 1
            STATS[f"reject:{res.reason}"] += 1
            event(f"rejected {res.reason}")
            # P10: driverul nu și-a modificat starea sau cantitățile.
            assert order == before
    if is_terminal(order.state):
        STATS["ended_terminal"] += 1
        event(f"ended in {order.state}")


def test_property_9_10_non_vacuity() -> None:
    """Testul de driver exercită atât execuții acceptate, cât și respingeri de toate felurile."""
    STATS.clear()
    test_property_9_10_order_driver_conserves_qty_and_rejects_unchanged()
    for key in (
        "fill_accepted",
        "cancel_confirmed",
        "cancel_rejected",
        "reconciled",
        "ended_terminal",
        f"reject:{TransitionCode.OVERFILL}",
        f"reject:{TransitionCode.EXEC_QTY_INVALID}",
        f"reject:{TransitionCode.TERMINAL_STATE}",
        f"reject:{TransitionCode.TRANSITION_NOT_ALLOWED}",
        f"reject:{TransitionCode.FILL_QTY_MISMATCH}",
        f"reject:{TransitionCode.QTY_CONTEXT_MISSING}",
    ):
        assert STATS[key] > 0, f"scenariu neexercitat: {key} ({dict(STATS)})"


@given(
    st.sampled_from(ALL_STATES),
    st.sampled_from(ALL_TRIGGERS),
    qty_st,
    st.decimals(min_value=ZERO, max_value=Decimal("10000"), places=4),
    exec_spec_st,
)
def test_property_9_fill_never_exceeds_order_qty(
    state: OrderState,
    trigger: Trigger,
    qty: Decimal,
    filled_raw: Decimal,
    exec_spec: tuple[str, Decimal | None],
) -> None:
    """Orice execuție acceptată păstrează `0 < filled + exec ≤ qty` (P9, într-un singur pas).

    **Validates: Requirements 9.2, 9.3**
    """
    filled = min(filled_raw, qty)
    exec_qty = _resolve_exec(exec_spec, qty - filled)
    res = apply(state, trigger, order_qty=qty, filled_qty=filled, exec_qty=exec_qty)
    if res.ok and trigger in FILL_TRIGGERS:
        assert exec_qty is not None
        new_filled = filled + exec_qty
        assert filled < new_filled <= qty
        assert (new_filled == qty) == (res.new_state is OrderState.FILLED)


@given(
    st.sampled_from(MISSING_PAIRS),
    st.one_of(st.none(), any_qty_st),
    st.one_of(st.none(), any_qty_st),
    st.one_of(st.none(), any_qty_st),
)
def test_property_10_transitions_outside_table_are_rejected(
    pair: tuple[OrderState, Trigger],
    order_qty: Decimal | None,
    filled_qty: Decimal | None,
    exec_qty: Decimal | None,
) -> None:
    """Orice (stare, declanșator) absent din tabel e respins, cu starea neschimbată.

    **Validates: Requirements 9.2, 9.4**
    """
    state, trigger = pair
    res = apply(state, trigger, order_qty=order_qty, filled_qty=filled_qty, exec_qty=exec_qty)
    assert not res.ok
    assert res.from_state is state
    assert res.new_state is state
    expected = (
        TransitionCode.TERMINAL_STATE
        if state in TERMINAL_STATES
        else TransitionCode.TRANSITION_NOT_ALLOWED
    )
    assert res.reason is expected


@given(
    st.sampled_from(ALL_STATES),
    st.sampled_from(ALL_TRIGGERS),
    st.one_of(st.none(), any_qty_st),
    st.one_of(st.none(), any_qty_st),
    st.one_of(st.none(), any_qty_st),
)
def test_property_10_closure_any_context(
    state: OrderState,
    trigger: Trigger,
    order_qty: Decimal | None,
    filled_qty: Decimal | None,
    exec_qty: Decimal | None,
) -> None:
    """Pentru context arbitrar: acceptat ⇒ țintă din tabel; respins ⇒ stare neschimbată.

    **Validates: Requirements 9.2, 9.4**
    """
    res = apply(state, trigger, order_qty=order_qty, filled_qty=filled_qty, exec_qty=exec_qty)
    assert res.from_state is state
    if res.ok:
        assert res.reason is None
        assert not is_terminal(state)
        assert res.new_state in TRANSITIONS[(state, trigger)]
    else:
        assert res.new_state is state
        assert res.reason is not None


def test_property_10_exhaustive_missing_pairs() -> None:
    """Enumerare completă a perechilor absente, pentru câteva contexte de cantitate fixe."""
    contexts = [
        (None, None, None),
        (Decimal("10"), ZERO, Decimal("10")),
        (Decimal("10"), Decimal("4"), Decimal("1")),
        (Decimal("10"), Decimal("10"), Decimal("0")),
        (Decimal("-1"), Decimal("-5"), Decimal("-2")),
    ]
    assert MISSING_PAIRS  # tabelul nu este complet
    for state, trigger in MISSING_PAIRS:
        for oq, fq, eq in contexts:
            res = apply(state, trigger, order_qty=oq, filled_qty=fq, exec_qty=eq)
            assert not res.ok and res.new_state is state, (state, trigger, oq, fq, eq)
    # Stările terminale nu au niciun declanșator permis.
    for s in TERMINAL_STATES:
        assert allowed(s) == frozenset()
