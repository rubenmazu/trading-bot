"""Teste unitare pentru mașina de stări a ordinului (Req 9.1, 9.2, 9.4)."""

from __future__ import annotations

import itertools
from decimal import Decimal

import pytest

from qts.core.models import ExecKind, OrderState
from qts.oms.fsm import (
    TERMINAL_STATES,
    TRANSITIONS,
    TransitionCode,
    Trigger,
    allowed,
    apply,
    is_terminal,
    targets,
    trigger_for_exec,
)

S = OrderState
T = Trigger
D = Decimal

# Tabelul așteptat, scris independent de implementare, după diagrama din design.
EXPECTED: dict[tuple[OrderState, Trigger], OrderState] = {
    (S.CREATED, T.RISK_APPROVE): S.APPROVED,
    (S.CREATED, T.RISK_REJECT): S.REJECTED_RISK,
    (S.APPROVED, T.SUBMIT): S.SUBMITTED,
    (S.SUBMITTED, T.ACK): S.ACKNOWLEDGED,
    (S.SUBMITTED, T.BROKER_REJECT): S.REJECTED_BROKER,
    (S.SUBMITTED, T.SUBMIT_TIMEOUT): S.UNKNOWN,
    (S.ACKNOWLEDGED, T.PARTIAL_FILL): S.PARTIALLY_FILLED,
    (S.ACKNOWLEDGED, T.FILL): S.FILLED,
    (S.ACKNOWLEDGED, T.CANCEL_REQUEST): S.CANCEL_PENDING,
    (S.ACKNOWLEDGED, T.EXPIRE): S.EXPIRED,
    (S.PARTIALLY_FILLED, T.PARTIAL_FILL): S.PARTIALLY_FILLED,
    (S.PARTIALLY_FILLED, T.FILL): S.FILLED,
    (S.PARTIALLY_FILLED, T.CANCEL_REQUEST): S.CANCEL_PENDING,
    (S.CANCEL_PENDING, T.CANCEL_CONFIRMED): S.CANCELLED,
    (S.CANCEL_PENDING, T.PARTIAL_FILL): S.PARTIALLY_FILLED,
    (S.CANCEL_PENDING, T.FILL): S.FILLED,
    (S.UNKNOWN, T.RECONCILED_ACKNOWLEDGED): S.ACKNOWLEDGED,
    (S.UNKNOWN, T.RECONCILED_PARTIALLY_FILLED): S.PARTIALLY_FILLED,
    (S.UNKNOWN, T.RECONCILED_FILLED): S.FILLED,
    (S.UNKNOWN, T.RECONCILED_REJECTED): S.REJECTED_BROKER,
    (S.UNKNOWN, T.RECONCILED_CANCELLED): S.CANCELLED,
}
CANCEL_REJECTED_KEY = (S.CANCEL_PENDING, T.CANCEL_REJECTED)
ALL_PAIRS = list(itertools.product(OrderState, Trigger))


def _qty_for(state: OrderState, trigger: Trigger) -> dict[str, Decimal]:
    """Context de cantitate valid: ordin de 10, 4 deja executate (0 dacă nu s-a executat)."""
    filled = D(4) if state in (S.PARTIALLY_FILLED, S.CANCEL_PENDING) else D(0)
    remaining = D(10) - filled
    exec_qty = remaining if trigger is T.FILL else D(1)
    return {"order_qty": D(10), "filled_qty": filled, "exec_qty": exec_qty}


def test_table_matches_design() -> None:
    assert {k: v for k, v in TRANSITIONS.items() if k != CANCEL_REJECTED_KEY} == {
        k: (v,) for k, v in EXPECTED.items()
    }
    assert TRANSITIONS[CANCEL_REJECTED_KEY] == (S.ACKNOWLEDGED, S.PARTIALLY_FILLED)


def test_table_is_immutable() -> None:
    with pytest.raises(TypeError):
        TRANSITIONS[(S.FILLED, T.ACK)] = (S.ACKNOWLEDGED,)  # type: ignore[index]


def test_all_required_states_exist() -> None:
    # 9.1: creare, aprobare, transmitere, confirmare, respingere, parțial, total, anulare,
    # expirare și rezultat necunoscut (plus CANCEL_PENDING).
    assert {s.value for s in OrderState} == {
        "CREATED",
        "APPROVED",
        "REJECTED_RISK",
        "SUBMITTED",
        "ACKNOWLEDGED",
        "REJECTED_BROKER",
        "UNKNOWN",
        "PARTIALLY_FILLED",
        "FILLED",
        "CANCEL_PENDING",
        "CANCELLED",
        "EXPIRED",
    }


@pytest.mark.parametrize(("key", "expected"), list(EXPECTED.items()), ids=str)
def test_every_allowed_transition(key: tuple[OrderState, Trigger], expected: OrderState) -> None:
    state, trigger = key
    r = apply(state, trigger, **_qty_for(state, trigger))
    assert r.ok and not r.is_incident
    assert r.new_state is expected
    assert r.from_state is state and r.trigger is trigger and r.reason is None


@pytest.mark.parametrize(
    ("filled", "expected"), [(D(0), S.ACKNOWLEDGED), (D(4), S.PARTIALLY_FILLED)]
)
def test_cancel_rejected_returns_to_live_state(filled: Decimal, expected: OrderState) -> None:
    r = apply(S.CANCEL_PENDING, T.CANCEL_REJECTED, order_qty=D(10), filled_qty=filled)
    assert r.ok and r.new_state is expected


def test_cancel_rejected_without_filled_qty_is_rejected() -> None:
    r = apply(S.CANCEL_PENDING, T.CANCEL_REJECTED)
    assert not r.ok and r.reason is TransitionCode.QTY_CONTEXT_MISSING
    assert r.new_state is S.CANCEL_PENDING


@pytest.mark.parametrize(
    ("state", "trigger"),
    [p for p in ALL_PAIRS if p not in EXPECTED and p != CANCEL_REJECTED_KEY],
    ids=str,
)
def test_every_disallowed_pair_is_rejected_without_state_change(
    state: OrderState, trigger: Trigger
) -> None:
    r = apply(state, trigger, **_qty_for(state, trigger))
    assert not r.ok and r.is_incident
    assert r.new_state is state
    expected_code = (
        TransitionCode.TERMINAL_STATE
        if is_terminal(state)
        else TransitionCode.TRANSITION_NOT_ALLOWED
    )
    assert r.reason is expected_code
    assert r.detail


def test_terminal_states() -> None:
    expected = {S.FILLED, S.CANCELLED, S.EXPIRED, S.REJECTED_RISK, S.REJECTED_BROKER}
    assert set(TERMINAL_STATES) == expected
    for s in OrderState:
        assert is_terminal(s) == (s in TERMINAL_STATES)
        assert (allowed(s) == frozenset()) == is_terminal(s)
        assert (targets(s) == frozenset()) == is_terminal(s)


def test_allowed_and_targets() -> None:
    assert allowed(S.UNKNOWN) == {
        T.RECONCILED_ACKNOWLEDGED,
        T.RECONCILED_PARTIALLY_FILLED,
        T.RECONCILED_FILLED,
        T.RECONCILED_REJECTED,
        T.RECONCILED_CANCELLED,
    }
    assert targets(S.CANCEL_PENDING) == {
        S.CANCELLED,
        S.PARTIALLY_FILLED,
        S.FILLED,
        S.ACKNOWLEDGED,
    }
    assert targets(S.SUBMITTED) == {S.ACKNOWLEDGED, S.REJECTED_BROKER, S.UNKNOWN}


def test_every_state_reachable_from_created() -> None:
    seen = {S.CREATED}
    frontier = [S.CREATED]
    while frontier:
        for nxt in targets(frontier.pop()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert seen == set(OrderState)


@pytest.mark.parametrize("state", [S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.CANCEL_PENDING])
@pytest.mark.parametrize("trigger", [T.PARTIAL_FILL, T.FILL])
def test_overfill_rejected(state: OrderState, trigger: Trigger) -> None:
    r = apply(state, trigger, order_qty=D(10), filled_qty=D(4), exec_qty=D(7))
    assert not r.ok and r.reason is TransitionCode.OVERFILL and r.new_state is state


def test_partial_fill_exhausting_order_rejected() -> None:
    r = apply(S.PARTIALLY_FILLED, T.PARTIAL_FILL, order_qty=D(10), filled_qty=D(4), exec_qty=D(6))
    assert not r.ok and r.reason is TransitionCode.PARTIAL_FILL_EXHAUSTS_ORDER
    assert r.new_state is S.PARTIALLY_FILLED


def test_fill_not_completing_order_rejected() -> None:
    r = apply(S.ACKNOWLEDGED, T.FILL, order_qty=D(10), filled_qty=D(0), exec_qty=D(3))
    assert not r.ok and r.reason is TransitionCode.FILL_QTY_MISMATCH
    assert r.new_state is S.ACKNOWLEDGED


@pytest.mark.parametrize("exec_qty", [D(0), D(-1)])
def test_non_positive_exec_qty_rejected(exec_qty: Decimal) -> None:
    r = apply(S.ACKNOWLEDGED, T.PARTIAL_FILL, order_qty=D(10), filled_qty=D(0), exec_qty=exec_qty)
    assert not r.ok and r.reason is TransitionCode.EXEC_QTY_INVALID


@pytest.mark.parametrize("trigger", [T.PARTIAL_FILL, T.FILL])
def test_fill_without_qty_context_rejected(trigger: Trigger) -> None:
    r = apply(S.ACKNOWLEDGED, trigger)
    assert not r.ok and r.reason is TransitionCode.QTY_CONTEXT_MISSING
    assert r.new_state is S.ACKNOWLEDGED


@pytest.mark.parametrize(
    ("order_qty", "filled_qty"), [(D(10), D(-1)), (D(10), D(11)), (D(0), D(0))]
)
def test_invalid_qty_context_rejected(order_qty: Decimal, filled_qty: Decimal) -> None:
    r = apply(
        S.ACKNOWLEDGED, T.PARTIAL_FILL, order_qty=order_qty, filled_qty=filled_qty, exec_qty=D(1)
    )
    assert not r.ok and r.reason is TransitionCode.QTY_CONTEXT_INVALID


def test_disallowed_check_precedes_qty_checks() -> None:
    # Fără context de cantitate, o execuție dintr-o stare greșită raportează tranziția invalidă.
    r = apply(S.CREATED, T.FILL)
    assert r.reason is TransitionCode.TRANSITION_NOT_ALLOWED


def test_trigger_for_exec_covers_all_kinds() -> None:
    assert {k: trigger_for_exec(k) for k in ExecKind} == {
        ExecKind.ACK: T.ACK,
        ExecKind.REJECT: T.BROKER_REJECT,
        ExecKind.PARTIAL_FILL: T.PARTIAL_FILL,
        ExecKind.FILL: T.FILL,
        ExecKind.CANCELLED: T.CANCEL_CONFIRMED,
        ExecKind.CANCEL_REJECTED: T.CANCEL_REJECTED,
        ExecKind.EXPIRED: T.EXPIRE,
    }
