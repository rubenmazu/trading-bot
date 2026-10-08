"""`FakeBroker`: broker fals scriptabil, cu defecte injectabile (Req 9.1, 10.4, 11.1).

Testul controlează brokerul: confirmări (`ack`), respingeri (`reject`), execuții (`fill`),
expirări (`expire`) și confirmarea anulărilor amânate (`confirm_cancel`). Defectele se
injectează printr-un `FaultPlan` (directive pe `client_order_id`, consumate o singură dată):

- respingere sincronă la `submit` (`SubmitFault.REJECT`) sau asincronă (`DEFER_ACK` + `reject`);
- timeout la `submit` *înainte* de acceptare (brokerul nu a văzut ordinul) sau *după* acceptare
  (ordinul există la broker, dar apelantul primește `BrokerTimeoutError`): rezultat necunoscut;
- mesaje duplicate (`duplicate`, `redeliver`), reordonate (`reorder`, determinist după
  permutare explicită sau după `seed`) și pierdute (`drop`), cu `resend(coid, seqs)` pentru
  cererile de mesaje lipsă;
- deconectare (`disconnect`/`reconnect`): apelurile ridică `BrokerDisconnectedError`, iar
  evenimentele produse între timp rămân la broker și sunt livrate după reconectare
  (opțional, cele aflate în tranzit se pierd și pot fi recuperate prin `resend`);
- snapshot incomplet (`complete=False`, ultimul ordin omis) sau indisponibil (excepție);
- execuție în timpul anulării (`CancelFault.DEFER`: anularea este acceptată, dar `CANCELLED`
  sosește abia la `confirm_cancel`, după eventuale execuții);
- respingeri pentru capabilități lipsă, prin `check_request`.

Starea reală a brokerului (ordine, execuții, poziții, numerar) este păstrată separat de
canalul de livrare și este disponibilă prin `truth()`, indiferent de defecte, pentru testele
de reconciliere. O anulare amânată nu schimbă starea raportată a ordinului (rămâne
`ACKNOWLEDGED`/`PARTIALLY_FILLED` până la `CANCELLED`); un ordin primit și neconfirmat apare
ca `SUBMITTED`. Evenimentele au `seq` pe ordin, de la 1, `broker_exec_id =
"{account_id}:{client_order_id}:{seq}"` și `broker_order_id = "FAKE-{n:08d}"`. Comisionul
este în EUR; numerarul se modifică cu `preț × cantitate × fx_rate` și comisionul. Aleatorul
este numai `random.Random(seed)` local; toate valorile sunt `Decimal`.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Context, Decimal, localcontext
from enum import StrEnum
from typing import Final

from qts.core.clock import Clock
from qts.core.models import ExecKind, ExecutionEvent, Instrument, OrderState
from qts.core.money import PRECISION, ZERO

from .adapter import (
    BrokerCapabilities,
    BrokerOrderStatus,
    BrokerSnapshot,
    CancelAck,
    Environment,
    OrderRequest,
    ReasonCode,
    SubmitAck,
    check_request,
)

__all__ = [
    "DEFAULT_FAKE_CAPABILITIES",
    "BrokerDisconnectedError",
    "BrokerTimeoutError",
    "CancelFault",
    "FakeBroker",
    "FakeFill",
    "FaultPlan",
    "SnapshotFault",
    "SnapshotUnavailableError",
    "SubmitFault",
]

DEFAULT_FAKE_CAPABILITIES: Final = BrokerCapabilities(
    order_types=frozenset({"MARKET", "LIMIT"}),
    time_in_force=frozenset({"GTC", "DAY"}),
    asset_classes=frozenset({"etf", "stock", "fx", "commodity_etp", "index_etp", "crypto"}),
    fractional=True,
    short=False,
    leverage=False,
    derivatives=False,
)

_CTX: Final = Context(prec=PRECISION)
_OPEN_STATES: Final = frozenset({OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED})


# --------------------------------------------------------------------------- erori


class BrokerTimeoutError(TimeoutError):
    """Apelul a expirat; apelantul nu știe dacă brokerul a acceptat cererea."""


class BrokerDisconnectedError(ConnectionError):
    """Conexiunea cu brokerul este întreruptă."""


class SnapshotUnavailableError(RuntimeError):
    """Brokerul nu a putut furniza snapshot-ul."""


# --------------------------------------------------------------------------- defecte


class SubmitFault(StrEnum):
    REJECT = "REJECT"  # SubmitAck respins cu cod de motiv + eveniment REJECT
    TIMEOUT_BEFORE_ACCEPT = "TIMEOUT_BEFORE_ACCEPT"  # brokerul nu a primit ordinul
    TIMEOUT_AFTER_ACCEPT = "TIMEOUT_AFTER_ACCEPT"  # ordin acceptat + ACK, apelantul vede timeout
    DEFER_ACK = "DEFER_ACK"  # primit, fără ACK până la `ack()` sau `reject()`


class CancelFault(StrEnum):
    REJECT = "REJECT"  # anulare respinsă + CANCEL_REJECTED
    DEFER = "DEFER"  # acceptată; CANCELLED abia la `confirm_cancel()`
    TIMEOUT_AFTER_ACCEPT = "TIMEOUT_AFTER_ACCEPT"  # anulare aplicată, apelantul vede timeout


class SnapshotFault(StrEnum):
    INCOMPLETE = "INCOMPLETE"  # complete=False, ultimul ordin omis
    RAISE = "RAISE"  # SnapshotUnavailableError


@dataclass(slots=True)
class FaultPlan:
    """Directive de defect. Cele pe ordin și pe mesaj se consumă la prima aplicare."""

    submit: dict[str, tuple[SubmitFault, ReasonCode]] = field(default_factory=dict)
    cancel: dict[str, tuple[CancelFault, ReasonCode]] = field(default_factory=dict)
    dropped: set[tuple[str, int]] = field(default_factory=set)
    duplicated: dict[tuple[str, int], int] = field(default_factory=dict)
    snapshot: deque[SnapshotFault] = field(default_factory=deque)

    def on_submit(
        self,
        client_order_id: str,
        fault: SubmitFault,
        reason: ReasonCode = ReasonCode.INVALID_QTY,
    ) -> FaultPlan:
        self.submit[client_order_id] = (fault, reason)
        return self

    def on_cancel(
        self,
        client_order_id: str,
        fault: CancelFault,
        reason: ReasonCode = ReasonCode.ORDER_NOT_OPEN,
    ) -> FaultPlan:
        self.cancel[client_order_id] = (fault, reason)
        return self

    def drop(self, client_order_id: str, *seqs: int) -> FaultPlan:
        """Mesajele cu aceste secvențe se pierd la emitere (recuperabile cu `resend`)."""
        self.dropped.update((client_order_id, s) for s in seqs)
        return self

    def duplicate(self, client_order_id: str, seq: int, copies: int = 1) -> FaultPlan:
        """Mesajul cu secvența dată este livrat de `1 + copies` ori."""
        if copies < 1:
            raise ValueError("copies trebuie să fie >= 1")
        self.duplicated[(client_order_id, seq)] = copies
        return self

    def fail_snapshot(self, fault: SnapshotFault, times: int = 1) -> FaultPlan:
        self.snapshot.extend([fault] * times)
        return self


# --------------------------------------------------------------------------- stare broker


@dataclass(frozen=True, slots=True)
class FakeFill:
    broker_exec_id: str
    client_order_id: str
    instrument: str
    side: str
    qty: Decimal
    price: Decimal
    commission: Decimal
    ts: datetime


@dataclass(slots=True)
class _FakeOrder:
    req: OrderRequest
    ack: SubmitAck
    state: OrderState
    next_seq: int = 1
    filled_qty: Decimal = ZERO
    notional: Decimal = ZERO
    cancel_pending: bool = False
    cancel_ack: CancelAck | None = None
    reason_code: ReasonCode | None = None
    history: dict[int, ExecutionEvent] = field(default_factory=dict)

    @property
    def remaining(self) -> Decimal:
        return self.req.qty - self.filled_qty

    def status(self) -> BrokerOrderStatus:
        avg = None
        if self.filled_qty > 0:
            with localcontext(_CTX):
                avg = self.notional / self.filled_qty
        return BrokerOrderStatus(
            client_order_id=self.req.client_order_id,
            instrument=self.req.instrument,
            side=self.req.side,
            qty=self.req.qty,
            state=self.state,
            filled_qty=self.filled_qty,
            avg_fill_price=avg,
            broker_order_id=self.ack.broker_order_id,
            reason_code=self.reason_code,
        )


class FakeBroker:
    """Broker fals pentru teste; satisface `BrokerAdapter`."""

    def __init__(
        self,
        *,
        instruments: Iterable[Instrument],
        clock: Clock,
        account_id: str = "FAKE-DEMO-1",
        environment: Environment = "demo",
        endpoint: str = "fake://local",
        capabilities: BrokerCapabilities = DEFAULT_FAKE_CAPABILITIES,
        initial_cash: Decimal = ZERO,
        faults: FaultPlan | None = None,
        seed: int = 0,
    ) -> None:
        if environment not in ("demo", "sim"):
            raise ValueError(f"FakeBroker nu poate rula în mediul {environment!r}")
        if not endpoint.lower().startswith("fake://"):
            raise ValueError(f"endpoint-ul FakeBroker trebuie să fie fake://, nu {endpoint!r}")
        self._environment: Environment = environment
        self._account_id = account_id
        self._endpoint = endpoint
        self._instruments = {i.symbol: i for i in instruments}
        self._clock = clock
        self._caps = capabilities
        self._cash = initial_cash
        self.faults = faults if faults is not None else FaultPlan()
        # Aleator local, determinist, folosit numai pentru reordonarea mesajelor în teste.
        self._rng = random.Random(seed)  # noqa: S311
        self._orders: dict[str, _FakeOrder] = {}
        self._outbox: deque[ExecutionEvent] = deque()
        self._exec_ids: list[str] = []
        self._fills: list[FakeFill] = []
        self._positions: dict[str, Decimal] = {}
        self._accepted = 0
        self._connected = True

    # ------------------------------------------------------------------ BrokerAdapter

    @property
    def environment(self) -> Environment:
        return self._environment

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def capabilities(self) -> BrokerCapabilities:
        return self._caps

    def submit(self, req: OrderRequest) -> SubmitAck:
        self._ensure_connected()
        now = self._clock.now()
        existing = self._orders.get(req.client_order_id)
        if existing is not None:
            if existing.req == req:
                return existing.ack
            return SubmitAck(
                client_order_id=req.client_order_id,
                accepted=False,
                ts=now,
                reason_code=ReasonCode.CLIENT_ORDER_ID_CONFLICT,
                detail="client_order_id refolosit cu alt conținut",
            )
        fault = self.faults.submit.pop(req.client_order_id, None)
        kind = fault[0] if fault is not None else None
        if kind is SubmitFault.TIMEOUT_BEFORE_ACCEPT:
            raise BrokerTimeoutError(f"timeout la trimiterea {req.client_order_id}")
        if fault is not None and kind is SubmitFault.REJECT:
            return self._reject_on_submit(req, now, fault[1], "respingere injectată")
        rejection = check_request(
            self._caps,
            req,
            self._instruments.get(req.instrument),
            available_qty=self._available_to_sell(req.instrument),
        )
        if rejection is not None:
            return self._reject_on_submit(req, now, rejection.code, rejection.detail)
        self._accepted += 1
        ack = SubmitAck(
            client_order_id=req.client_order_id,
            accepted=True,
            ts=now,
            broker_order_id=f"FAKE-{self._accepted:08d}",
        )
        deferred = kind is SubmitFault.DEFER_ACK
        state = OrderState.SUBMITTED if deferred else OrderState.ACKNOWLEDGED
        order = _FakeOrder(req, ack, state)
        self._orders[req.client_order_id] = order
        if not deferred:
            self._emit(order, ExecKind.ACK)
        if kind is SubmitFault.TIMEOUT_AFTER_ACCEPT:
            raise BrokerTimeoutError(f"timeout după acceptarea {req.client_order_id}")
        return ack

    def cancel(self, client_order_id: str) -> CancelAck:
        self._ensure_connected()
        now = self._clock.now()
        order = self._orders.get(client_order_id)
        if order is None:
            return CancelAck(
                client_order_id=client_order_id,
                accepted=False,
                ts=now,
                reason_code=ReasonCode.UNKNOWN_ORDER,
                detail="ordin necunoscut",
            )
        if order.cancel_ack is not None:
            return order.cancel_ack
        fault = self.faults.cancel.pop(client_order_id, None)
        kind = fault[0] if fault is not None else None
        if fault is not None and kind is CancelFault.REJECT:
            return self._reject_cancel(order, now, fault[1], "respingere injectată")
        if order.state not in _OPEN_STATES:
            return self._reject_cancel(
                order, now, ReasonCode.ORDER_NOT_OPEN, f"ordinul este {order.state.value}"
            )
        order.cancel_ack = CancelAck(client_order_id=client_order_id, accepted=True, ts=now)
        if kind is CancelFault.DEFER:
            order.cancel_pending = True
            return order.cancel_ack
        self._finish_cancel(order)
        if kind is CancelFault.TIMEOUT_AFTER_ACCEPT:
            raise BrokerTimeoutError(f"timeout după anularea {client_order_id}")
        return order.cancel_ack

    def snapshot(self) -> BrokerSnapshot:
        self._ensure_connected()
        fault = self.faults.snapshot.popleft() if self.faults.snapshot else None
        if fault is SnapshotFault.RAISE:
            raise SnapshotUnavailableError("snapshot indisponibil (defect injectat)")
        snap = self.truth()
        if fault is SnapshotFault.INCOMPLETE and snap.orders:
            prefix = f"{self._account_id}:{snap.orders[-1].client_order_id}:"
            return snap.model_copy(
                update={
                    "orders": snap.orders[:-1],
                    "execution_ids": tuple(
                        x for x in snap.execution_ids if not x.startswith(prefix)
                    ),
                    "complete": False,
                }
            )
        if fault is SnapshotFault.INCOMPLETE:
            return snap.model_copy(update={"complete": False})
        return snap

    def events(self) -> Iterator[ExecutionEvent]:
        """Livrează mesajele din tranzit; ridică `BrokerDisconnectedError` fără conexiune."""
        self._ensure_connected()
        return self._deliver()

    # ------------------------------------------------------------------ control (test)

    @property
    def connected(self) -> bool:
        return self._connected

    def disconnect(self, *, lose_in_flight: bool = False) -> None:
        """Întrerupe conexiunea; opțional pierde mesajele aflate în tranzit."""
        self._connected = False
        if lose_in_flight:
            self._outbox.clear()

    def reconnect(self) -> None:
        self._connected = True

    def ack(self, client_order_id: str) -> ExecutionEvent:
        order = self._order(client_order_id)
        if order.state is not OrderState.SUBMITTED:
            raise ValueError(f"ordinul {client_order_id} nu așteaptă confirmare")
        order.state = OrderState.ACKNOWLEDGED
        return self._emit(order, ExecKind.ACK)

    def reject(self, client_order_id: str, reason: ReasonCode) -> ExecutionEvent:
        """Respingere asincronă a unui ordin primit și încă neconfirmat."""
        order = self._order(client_order_id)
        if order.state is not OrderState.SUBMITTED:
            raise ValueError(f"ordinul {client_order_id} nu așteaptă confirmare")
        order.state = OrderState.REJECTED_BROKER
        order.reason_code = reason
        return self._emit(order, ExecKind.REJECT, reason=reason.value)

    def fill(
        self,
        client_order_id: str,
        qty: Decimal,
        price: Decimal,
        commission: Decimal = ZERO,
        *,
        fx_rate: Decimal = Decimal(1),
    ) -> ExecutionEvent:
        """Execuție la broker; `FILL` dacă epuizează cantitatea, altfel `PARTIAL_FILL`."""
        order = self._order(client_order_id)
        if order.state not in _OPEN_STATES:
            raise ValueError(f"ordinul {client_order_id} nu este deschis ({order.state.value})")
        if qty <= 0 or qty > order.remaining or price <= 0 or commission < 0 or fx_rate <= 0:
            raise ValueError(f"execuție invalidă: qty={qty} price={price} rest={order.remaining}")
        inst = order.req.instrument
        sign = Decimal(1) if order.req.side == "BUY" else Decimal(-1)
        with localcontext(_CTX):
            order.filled_qty += qty
            order.notional += price * qty
            self._positions[inst] = self._positions.get(inst, ZERO) + sign * qty
            self._cash -= sign * price * qty * fx_rate + commission
        done = order.remaining == 0
        order.state = OrderState.FILLED if done else OrderState.PARTIALLY_FILLED
        event = self._emit(
            order,
            ExecKind.FILL if done else ExecKind.PARTIAL_FILL,
            qty=qty,
            price=price,
            commission=commission,
        )
        self._fills.append(
            FakeFill(
                event.broker_exec_id,
                client_order_id,
                inst,
                order.req.side,
                qty,
                price,
                commission,
                event.ts_broker,
            )
        )
        return event

    def expire(self, client_order_id: str) -> ExecutionEvent:
        order = self._order(client_order_id)
        if order.state not in _OPEN_STATES:
            raise ValueError(f"ordinul {client_order_id} nu este deschis ({order.state.value})")
        remaining = order.remaining
        order.state = OrderState.EXPIRED
        order.cancel_pending = False
        return self._emit(order, ExecKind.EXPIRED, qty=remaining)

    def confirm_cancel(self, client_order_id: str) -> ExecutionEvent:
        """Finalizează o anulare amânată: `CANCELLED` pentru rest sau, dacă ordinul s-a
        executat complet între timp, `CANCEL_REJECTED` (`ORDER_NOT_OPEN`)."""
        order = self._order(client_order_id)
        if not order.cancel_pending:
            raise ValueError(f"ordinul {client_order_id} nu are o anulare amânată")
        if order.state is OrderState.FILLED:
            order.cancel_pending = False
            return self._emit(
                order, ExecKind.CANCEL_REJECTED, reason=ReasonCode.ORDER_NOT_OPEN.value
            )
        return self._finish_cancel(order)

    def resend(self, client_order_id: str, seqs: Iterable[int]) -> list[ExecutionEvent]:
        """Răspunde unei cereri de mesaje lipsă: retransmite mesajele cu același conținut."""
        order = self._order(client_order_id)
        out: list[ExecutionEvent] = []
        for s in sorted(set(seqs)):
            event = order.history.get(s)
            if event is None:
                raise KeyError(f"{client_order_id}: secvența {s} nu a fost emisă")
            out.append(event)
        self._outbox.extend(out)
        return out

    def redeliver(self, client_order_id: str, seq: int) -> ExecutionEvent:
        """Livrează din nou un mesaj deja emis (duplicat cu același `broker_exec_id`)."""
        return self.resend(client_order_id, (seq,))[0]

    def reorder(self, permutation: Sequence[int] | None = None) -> list[int]:
        """Permută mesajele din tranzit: `new[i] = old[permutation[i]]`; fără permutare,
        amestecare deterministă după `seed`. Întoarce permutarea aplicată."""
        pending = list(self._outbox)
        if permutation is None:
            perm = list(range(len(pending)))
            self._rng.shuffle(perm)
        else:
            perm = list(permutation)
        if sorted(perm) != list(range(len(pending))):
            raise ValueError(f"permutare invalidă pentru {len(pending)} mesaje: {perm}")
        self._outbox = deque(pending[i] for i in perm)
        return perm

    def in_flight(self) -> tuple[ExecutionEvent, ...]:
        return tuple(self._outbox)

    def truth(self) -> BrokerSnapshot:
        """Starea reală a brokerului, fără defecte și fără verificarea conexiunii."""
        return BrokerSnapshot(
            ts=self._clock.now(),
            environment=self._environment,
            account_id=self._account_id,
            orders=tuple(o.status() for o in self._orders.values()),
            execution_ids=tuple(self._exec_ids),
            positions={k: v for k, v in sorted(self._positions.items()) if v != 0},
            cash=self._cash,
            complete=True,
        )

    @property
    def fills(self) -> tuple[FakeFill, ...]:
        return tuple(self._fills)

    def next_seq(self, client_order_id: str) -> int:
        """Următoarea secvență pe care brokerul o va emite pentru ordin."""
        return self._order(client_order_id).next_seq

    def is_cancel_pending(self, client_order_id: str) -> bool:
        return self._order(client_order_id).cancel_pending

    # ------------------------------------------------------------------ intern

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise BrokerDisconnectedError(f"conexiune întreruptă la {self._endpoint}")

    def _deliver(self) -> Iterator[ExecutionEvent]:
        while self._connected and self._outbox:
            event = self._outbox.popleft()
            yield event.model_copy(update={"ts_receipt": self._clock.now()})

    def _order(self, client_order_id: str) -> _FakeOrder:
        order = self._orders.get(client_order_id)
        if order is None:
            raise KeyError(f"ordin necunoscut: {client_order_id}")
        return order

    def _available_to_sell(self, instrument: str) -> Decimal:
        reserved = sum(
            (
                o.remaining
                for o in self._orders.values()
                if o.req.instrument == instrument
                and o.req.side == "SELL"
                and o.state in (*_OPEN_STATES, OrderState.SUBMITTED)
            ),
            ZERO,
        )
        return self._positions.get(instrument, ZERO) - reserved

    def _reject_on_submit(
        self, req: OrderRequest, now: datetime, code: ReasonCode, detail: str
    ) -> SubmitAck:
        ack = SubmitAck(
            client_order_id=req.client_order_id,
            accepted=False,
            ts=now,
            reason_code=code,
            detail=detail,
        )
        order = _FakeOrder(req, ack, OrderState.REJECTED_BROKER, reason_code=code)
        self._orders[req.client_order_id] = order
        self._emit(order, ExecKind.REJECT, reason=code.value)
        return ack

    def _reject_cancel(
        self, order: _FakeOrder, now: datetime, code: ReasonCode, detail: str
    ) -> CancelAck:
        self._emit(order, ExecKind.CANCEL_REJECTED, reason=code.value)
        return CancelAck(
            client_order_id=order.req.client_order_id,
            accepted=False,
            ts=now,
            reason_code=code,
            detail=detail,
        )

    def _finish_cancel(self, order: _FakeOrder) -> ExecutionEvent:
        remaining = order.remaining
        order.state = OrderState.CANCELLED
        order.cancel_pending = False
        return self._emit(order, ExecKind.CANCELLED, qty=remaining)

    def _emit(
        self,
        order: _FakeOrder,
        kind: ExecKind,
        *,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        commission: Decimal | None = None,
        reason: str | None = None,
    ) -> ExecutionEvent:
        now = self._clock.now()
        seq = order.next_seq
        order.next_seq += 1
        coid = order.req.client_order_id
        event = ExecutionEvent(
            broker_exec_id=f"{self._account_id}:{coid}:{seq}",
            client_order_id=coid,
            kind=kind,
            qty=qty,
            price=price,
            commission=commission,
            broker_order_id=order.ack.broker_order_id,
            reason=reason,
            ts_broker=now,
            ts_receipt=now,
            seq=seq,
        )
        order.history[seq] = event
        self._exec_ids.append(event.broker_exec_id)
        key = (coid, seq)
        if key in self.faults.dropped:
            self.faults.dropped.discard(key)
            return event
        copies = self.faults.duplicated.pop(key, 0)
        self._outbox.extend([event] * (1 + copies))
        return event
