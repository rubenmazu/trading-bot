"""Mașina de stări a ordinului (Order_Management_Subsystem). Req 9.1, 9.2, 9.4.

Tranzițiile sunt definite într-un tabel static și imuabil `(stare, declanșator) -> țintă`,
conform diagramei din design. Funcțiile sunt pure (fără I/O, fără stare globală mutabilă):

- `apply(state, trigger, ...)` validează tranziția și întoarce un `TransitionResult`;
- o tranziție absentă din tabel, sau una incompatibilă cu cantitățile, este respinsă cu un cod
  stabil (`TransitionCode`), iar `new_state` rămâne starea curentă (9.4). Rezultatul respins
  este un incident pe care apelantul (`oms/manager.py`) îl jurnalizează;
- stările terminale nu acceptă niciun declanșator.

Decizii:

- `CANCEL_REJECTED` din `CANCEL_PENDING` revine la starea vie anterioară, determinată din
  `filled_qty`: `PARTIALLY_FILLED` dacă `filled_qty > 0`, altfel `ACKNOWLEDGED` (9.5).
- Execuțiile de la broker (`PARTIAL_FILL`, `FILL`) cer contextul de cantitate
  (`order_qty`, `filled_qty`, `exec_qty`) și sunt verificate strict:
  * `exec_qty > remaining` → `OVERFILL`;
  * `PARTIAL_FILL` care epuizează cantitatea rămasă → `PARTIAL_FILL_EXHAUSTS_ORDER`;
  * `FILL` care nu completează exact ordinul → `FILL_QTY_MISMATCH`.
  Normalizarea (de ex. tratarea unei execuții parțiale finale ca `FILL`) ține de manager.
- Rezultatele reconcilierii pentru `UNKNOWN` (`RECONCILED_*`) nu au verificări de cantitate:
  reconcilierea stabilește cantitatea executată absolută din snapshot-ul brokerului.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from qts.core.models import ExecKind, OrderState

__all__ = [
    "TERMINAL_STATES",
    "TRANSITIONS",
    "TransitionCode",
    "TransitionResult",
    "Trigger",
    "allowed",
    "apply",
    "is_terminal",
    "targets",
    "trigger_for_exec",
]


class Trigger(StrEnum):
    """Declanșatorii tipizați ai tranzițiilor."""

    RISK_APPROVE = "RISK_APPROVE"
    RISK_REJECT = "RISK_REJECT"
    SUBMIT = "SUBMIT"  # după scrierea write-ahead în jurnal
    ACK = "ACK"
    BROKER_REJECT = "BROKER_REJECT"
    SUBMIT_TIMEOUT = "SUBMIT_TIMEOUT"  # timeout / deconectare → UNKNOWN
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCEL_REQUEST = "CANCEL_REQUEST"
    CANCEL_CONFIRMED = "CANCEL_CONFIRMED"
    CANCEL_REJECTED = "CANCEL_REJECTED"
    EXPIRE = "EXPIRE"
    RECONCILED_ACKNOWLEDGED = "RECONCILED_ACKNOWLEDGED"
    RECONCILED_PARTIALLY_FILLED = "RECONCILED_PARTIALLY_FILLED"
    RECONCILED_FILLED = "RECONCILED_FILLED"
    RECONCILED_REJECTED = "RECONCILED_REJECTED"
    RECONCILED_CANCELLED = "RECONCILED_CANCELLED"


class TransitionCode(StrEnum):
    """Coduri stabile de respingere a unei tranziții (9.4)."""

    TERMINAL_STATE = "TERMINAL_STATE"
    TRANSITION_NOT_ALLOWED = "TRANSITION_NOT_ALLOWED"
    QTY_CONTEXT_MISSING = "QTY_CONTEXT_MISSING"
    QTY_CONTEXT_INVALID = "QTY_CONTEXT_INVALID"
    EXEC_QTY_INVALID = "EXEC_QTY_INVALID"
    OVERFILL = "OVERFILL"
    PARTIAL_FILL_EXHAUSTS_ORDER = "PARTIAL_FILL_EXHAUSTS_ORDER"
    FILL_QTY_MISMATCH = "FILL_QTY_MISMATCH"


S = OrderState
T = Trigger

TERMINAL_STATES: Final[frozenset[OrderState]] = frozenset(
    {S.FILLED, S.CANCELLED, S.EXPIRED, S.REJECTED_RISK, S.REJECTED_BROKER}
)

# Tabelul static. Pentru (CANCEL_PENDING, CANCEL_REJECTED) sunt două ținte posibile; ținta
# efectivă se alege după `filled_qty` (vezi docstring-ul modulului).
TRANSITIONS: Final[MappingProxyType[tuple[OrderState, Trigger], tuple[OrderState, ...]]] = (
    MappingProxyType(
        {
            (S.CREATED, T.RISK_APPROVE): (S.APPROVED,),
            (S.CREATED, T.RISK_REJECT): (S.REJECTED_RISK,),
            (S.APPROVED, T.SUBMIT): (S.SUBMITTED,),
            (S.SUBMITTED, T.ACK): (S.ACKNOWLEDGED,),
            (S.SUBMITTED, T.BROKER_REJECT): (S.REJECTED_BROKER,),
            (S.SUBMITTED, T.SUBMIT_TIMEOUT): (S.UNKNOWN,),
            (S.ACKNOWLEDGED, T.PARTIAL_FILL): (S.PARTIALLY_FILLED,),
            (S.ACKNOWLEDGED, T.FILL): (S.FILLED,),
            (S.ACKNOWLEDGED, T.CANCEL_REQUEST): (S.CANCEL_PENDING,),
            (S.ACKNOWLEDGED, T.EXPIRE): (S.EXPIRED,),
            (S.PARTIALLY_FILLED, T.PARTIAL_FILL): (S.PARTIALLY_FILLED,),
            (S.PARTIALLY_FILLED, T.FILL): (S.FILLED,),
            (S.PARTIALLY_FILLED, T.CANCEL_REQUEST): (S.CANCEL_PENDING,),
            (S.CANCEL_PENDING, T.CANCEL_CONFIRMED): (S.CANCELLED,),
            (S.CANCEL_PENDING, T.PARTIAL_FILL): (S.PARTIALLY_FILLED,),
            (S.CANCEL_PENDING, T.FILL): (S.FILLED,),
            (S.CANCEL_PENDING, T.CANCEL_REJECTED): (S.ACKNOWLEDGED, S.PARTIALLY_FILLED),
            (S.UNKNOWN, T.RECONCILED_ACKNOWLEDGED): (S.ACKNOWLEDGED,),
            (S.UNKNOWN, T.RECONCILED_PARTIALLY_FILLED): (S.PARTIALLY_FILLED,),
            (S.UNKNOWN, T.RECONCILED_FILLED): (S.FILLED,),
            (S.UNKNOWN, T.RECONCILED_REJECTED): (S.REJECTED_BROKER,),
            (S.UNKNOWN, T.RECONCILED_CANCELLED): (S.CANCELLED,),
        }
    )
)

_FILL_TRIGGERS: Final = frozenset({T.PARTIAL_FILL, T.FILL})

_EXEC_TRIGGERS: Final[MappingProxyType[ExecKind, Trigger]] = MappingProxyType(
    {
        ExecKind.ACK: T.ACK,
        ExecKind.REJECT: T.BROKER_REJECT,
        ExecKind.PARTIAL_FILL: T.PARTIAL_FILL,
        ExecKind.FILL: T.FILL,
        ExecKind.CANCELLED: T.CANCEL_CONFIRMED,
        ExecKind.CANCEL_REJECTED: T.CANCEL_REJECTED,
        ExecKind.EXPIRED: T.EXPIRE,
    }
)


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Rezultatul validării. La respingere `new_state == from_state` și `reason` este setat."""

    ok: bool
    from_state: OrderState
    trigger: Trigger
    new_state: OrderState
    reason: TransitionCode | None = None
    detail: str = ""

    @property
    def is_incident(self) -> bool:
        return not self.ok


def is_terminal(state: OrderState) -> bool:
    return state in TERMINAL_STATES


def allowed(state: OrderState) -> frozenset[Trigger]:
    """Declanșatorii acceptați din `state` (vid pentru stările terminale)."""
    return frozenset(t for (s, t) in TRANSITIONS if s == state)


def targets(state: OrderState) -> frozenset[OrderState]:
    """Stările în care se poate ajunge direct din `state`."""
    return frozenset(x for (s, _), tgt in TRANSITIONS.items() if s == state for x in tgt)


def trigger_for_exec(kind: ExecKind) -> Trigger:
    """Declanșatorul corespunzător unui `ExecutionEvent.kind` de la broker."""
    return _EXEC_TRIGGERS[kind]


def _reject(
    state: OrderState, trigger: Trigger, code: TransitionCode, detail: str
) -> TransitionResult:
    return TransitionResult(
        ok=False, from_state=state, trigger=trigger, new_state=state, reason=code, detail=detail
    )


def _check_fill(
    state: OrderState,
    trigger: Trigger,
    order_qty: Decimal | None,
    filled_qty: Decimal | None,
    exec_qty: Decimal | None,
) -> TransitionResult | None:
    if order_qty is None or filled_qty is None or exec_qty is None:
        return _reject(
            state,
            trigger,
            TransitionCode.QTY_CONTEXT_MISSING,
            "execuția necesită order_qty, filled_qty și exec_qty",
        )
    if exec_qty <= 0:
        return _reject(state, trigger, TransitionCode.EXEC_QTY_INVALID, f"exec_qty={exec_qty}")
    remaining = order_qty - filled_qty
    if exec_qty > remaining:
        return _reject(
            state,
            trigger,
            TransitionCode.OVERFILL,
            f"exec_qty={exec_qty} > remaining={remaining}",
        )
    if trigger is T.PARTIAL_FILL and exec_qty == remaining:
        return _reject(
            state,
            trigger,
            TransitionCode.PARTIAL_FILL_EXHAUSTS_ORDER,
            f"execuția parțială epuizează remaining={remaining}",
        )
    if trigger is T.FILL and exec_qty != remaining:
        return _reject(
            state,
            trigger,
            TransitionCode.FILL_QTY_MISMATCH,
            f"exec_qty={exec_qty} != remaining={remaining}",
        )
    return None


def apply(
    state: OrderState,
    trigger: Trigger,
    *,
    order_qty: Decimal | None = None,
    filled_qty: Decimal | None = None,
    exec_qty: Decimal | None = None,
) -> TransitionResult:
    """Validează și calculează tranziția `state --trigger--> ?` (9.2).

    Cantitățile sunt necesare numai pentru `PARTIAL_FILL`/`FILL` (toate trei) și pentru
    `CANCEL_REJECTED` din `CANCEL_PENDING` (`filled_qty`). Orice respingere lasă starea neschimbată.
    """
    if is_terminal(state):
        return _reject(
            state, trigger, TransitionCode.TERMINAL_STATE, f"{state} este o stare terminală"
        )
    tgt = TRANSITIONS.get((state, trigger))
    if tgt is None:
        return _reject(
            state,
            trigger,
            TransitionCode.TRANSITION_NOT_ALLOWED,
            f"{trigger} nu este permis din {state}",
        )
    if filled_qty is not None and (
        filled_qty < 0 or (order_qty is not None and (order_qty <= 0 or filled_qty > order_qty))
    ):
        return _reject(
            state,
            trigger,
            TransitionCode.QTY_CONTEXT_INVALID,
            f"order_qty={order_qty}, filled_qty={filled_qty}",
        )

    if trigger in _FILL_TRIGGERS:
        bad = _check_fill(state, trigger, order_qty, filled_qty, exec_qty)
        if bad is not None:
            return bad

    if len(tgt) == 1:
        new_state = tgt[0]
    else:  # (CANCEL_PENDING, CANCEL_REJECTED): revenire la starea vie anterioară
        if filled_qty is None:
            return _reject(
                state,
                trigger,
                TransitionCode.QTY_CONTEXT_MISSING,
                "CANCEL_REJECTED necesită filled_qty pentru a alege starea vie",
            )
        new_state = S.PARTIALLY_FILLED if filled_qty > 0 else S.ACKNOWLEDGED
    return TransitionResult(ok=True, from_state=state, trigger=trigger, new_state=new_state)
