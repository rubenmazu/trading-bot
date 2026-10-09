"""Order_Management_Subsystem: aplicarea comenzilor interne și a execuțiilor brokerului.

Req 9.3, 9.5, 9.6, 10.1–10.5. Validarea tranzițiilor este delegată `qts.oms.fsm`; managerul
adaugă contextul de cantitate, idempotența, secvențierea și înghețarea.

Write-ahead
    Fiecare efect este descris într-un `OmsRecord` trimis sincron către `OmsSink` *înainte* ca
    starea internă să fie modificată. Dacă sink-ul ridică o excepție, starea rămâne neschimbată.
    `submit()` jurnalizează tranziția `APPROVED → SUBMITTED` și abia apoi întoarce
    `send=True`; apelantul (motorul, sarcina 12.1) trimite ordinul la broker numai după aceea.
    `JournalOmsSink` adaptează sink-ul la `qts.persistence.journal.Journal`.

Idempotență (10.1–10.3, 10.5)
    `client_order_id` este `idempotency_key(run_id, strategy_id, instrument, signal_seq)`.
    O a doua creare cu aceeași cheie nu creează alt ordin (`ORDER_DUPLICATE`). Execuțiile sunt
    deduplicate după `broker_exec_id`: o execuție deja procesată (aplicată, respinsă ca incident
    sau reflectată de reconciliere) ori aflată în așteptare este ignorată și jurnalizată ca
    `EXEC_RETRANSMISSION`, cu `original_record_seq` spre înregistrarea operației inițiale. Un
    `broker_exec_id` refolosit cu alt conținut este incident (`EXEC_ID_CONFLICT`).

Execuții (9.3)
    O execuție actualizează `filled_qty`, cantitatea rămasă, prețul mediu ponderat (VWAP) și
    costurile cumulate; invariantul `filled + remaining = qty` este garantat de `Order`.
    Normalizare față de FSM-ul strict: un `PARTIAL_FILL` care epuizează cantitatea rămasă se
    tratează ca `FILL`; un `FILL` cu cantitate mai mică decât restul este ambiguu și devine
    incident (`FILL_QTY_MISMATCH`), la fel ca supra-execuția (`OVERFILL`).

Costuri
    `cost_fn(order, event) -> CostBreakdown` (EUR) este injectabil; implicit `commission_only`,
    care ia `event.commission` ca valoare deja în EUR. `fx_fn(order, event) -> (monedă, curs)`
    dă moneda instrumentului și cursul spre EUR pentru `PortfolioFill`; implicit EUR cu curs 1.
    Managerul nu modifică portofoliul: fiecare execuție aplicată produce un `PortfolioFill`.

Anulări (9.5)
    `request_cancel()` duce ordinul în `CANCEL_PENDING` și marchează o anulare în curs. FSM-ul
    duce `CANCEL_PENDING + PARTIAL_FILL → PARTIALLY_FILLED`, iar din `PARTIALLY_FILLED` un
    `CANCELLED` ulterior ar fi invalid. De aceea, când anularea este încă în curs, managerul
    aplică imediat după execuția parțială o tranziție internă `CANCEL_REQUEST` înapoi în
    `CANCEL_PENDING`, jurnalizată ca `CANCEL_PENDING_RESTORED`, fără o nouă cerere la broker
    (una este deja în curs). Confirmarea sau respingerea anulării se înregistrează cu cantitatea
    rămasă executabilă. Un `CANCEL_REJECTED` sosit după ce ordinul a fost executat complet în
    timpul anulării este o cursă normală și se înregistrează ca `LATE_CANCEL_REJECT`.

Înghețare (9.6)
    Un ordin `UNKNOWN` sau cu un incident de stare nerezolvat (mesaj de broker invalid) este
    înghețat: comenzile interne sunt refuzate, execuțiile sosite sunt reținute (`EXEC_HELD`),
    iar ordinele noi pe același instrument sunt blocate (`InstrumentBlockedError`) până la
    `reconcile()`. Comenzile interne invalide (de exemplu anularea unui ordin `SUBMITTED`) sunt
    respinse și jurnalizate (`TRANSITION_REJECTED`), dar nu îngheață ordinul: ele nu indică o
    divergență față de broker.

Secvență (10.4)
    `ExecutionEvent.seq` este numărul mesajului *pe ordin* (implicit începând cu `first_seq=1`);
    `seq=None` înseamnă mesaj nesecvențiat, aplicat direct. Un mesaj cu `seq` mai mare decât cel
    așteptat suspendă numai ordinul respectiv: mesajul este pus în buffer, se emite o cerere
    `MissingMessagesRequest` pentru secvențele lipsă, iar mesajele altor ordine continuă. Când
    golul este completat, bufferul se golește în ordine. O secvență deja consumată sau dublată
    cu alt `broker_exec_id` este incident (`SEQ_CONFLICT`).

Reconciliere
    `reconcile()` aplică rezultatul brokerului: pentru `UNKNOWN`, tranziția `RECONCILED_*` cu
    cantitatea absolută din snapshot; pentru alte incidente, numai confirmarea că starea și
    cantitatea interne coincid cu ale brokerului (altfel `ReconciliationMismatchError`, ordinul
    rămâne înghețat). `reflected_exec_ids` marchează execuțiile deja incluse în snapshot, iar
    `next_seq` resetează secvența așteptată. Execuțiile reținute sunt apoi reprocesate.
    Corecțiile de portofoliu care rezultă din snapshot țin de `Reconciliation_Subsystem` (13.1).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Context, Decimal, localcontext
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable

from qts.core.models import (
    CostBreakdown,
    ExecKind,
    ExecutionEvent,
    Frozen,
    Order,
    OrderIntent,
    OrderState,
    UtcDatetime,
)
from qts.core.money import PRECISION, REPORTING_CURRENCY, ZERO
from qts.oms import fsm
from qts.oms.fsm import Trigger
from qts.oms.idempotency import (
    AppliedExec,
    IdempotencyRegistry,
    exec_fingerprint,
    idempotency_key,
)
from qts.persistence.journal import Journal
from qts.portfolio.portfolio import PortfolioFill
from qts.risk.engine import RiskDecision

__all__ = [
    "OMS_VERSION",
    "CommandResult",
    "CostFn",
    "ExecOutcome",
    "ExecStatus",
    "FxFn",
    "IncidentCode",
    "InstrumentBlockedError",
    "JournalOmsSink",
    "ListSink",
    "MissingMessagesRequest",
    "OmsRecord",
    "OmsSink",
    "OrderManager",
    "ReconciliationMismatchError",
    "RecordType",
    "UnknownOrderError",
    "commission_only",
    "eur_identity",
    "reporting_identity",
]

OMS_VERSION: Final = "1"
_CTX: Final = Context(prec=PRECISION)

CostFn = Callable[[Order, ExecutionEvent], CostBreakdown]
FxFn = Callable[[Order, ExecutionEvent], tuple[str, Decimal]]


def commission_only(order: Order, event: ExecutionEvent) -> CostBreakdown:
    """Cost implicit: numai comisionul raportat de broker, considerat deja în EUR."""
    return CostBreakdown(commission=event.commission if event.commission is not None else ZERO)


def eur_identity(order: Order, event: ExecutionEvent) -> tuple[str, Decimal]:
    """Monedă implicită: EUR, curs 1."""
    return REPORTING_CURRENCY, Decimal(1)


def reporting_identity(reporting_currency: str = REPORTING_CURRENCY) -> FxFn:
    """`FxFn` care raportează moneda de raportare a rulării, curs 1 (fără conversie).

    Pentru o rulare în moneda de raportare a instrumentului (de exemplu un Demo în USD cu
    instrument SPY/USD și cont USD), execuțiile nu necesită conversie: `PortfolioFill` poartă
    moneda de raportare cu `fx_rate_to_eur = 1`, deci numerarul proiectat coincide cu cel al
    contului și reconcilierea rămâne strictă. Implicit EUR, deci `reporting_identity()` este
    echivalent cu `eur_identity`.
    """

    def fx(order: Order, event: ExecutionEvent) -> tuple[str, Decimal]:
        return reporting_currency, Decimal(1)

    return fx


# --------------------------------------------------------------------------- înregistrări


class RecordType(StrEnum):
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_DUPLICATE = "ORDER_DUPLICATE"
    ORDER_BLOCKED = "ORDER_BLOCKED"
    TRANSITION = "TRANSITION"
    TRANSITION_REJECTED = "TRANSITION_REJECTED"
    COMMAND_DUPLICATE = "COMMAND_DUPLICATE"
    COMMAND_REFUSED_FROZEN = "COMMAND_REFUSED_FROZEN"
    EXEC_APPLIED = "EXEC_APPLIED"
    EXEC_RETRANSMISSION = "EXEC_RETRANSMISSION"
    EXEC_BUFFERED = "EXEC_BUFFERED"
    EXEC_HELD = "EXEC_HELD"
    MISSING_MESSAGES_REQUEST = "MISSING_MESSAGES_REQUEST"
    CANCEL_PENDING_RESTORED = "CANCEL_PENDING_RESTORED"
    LATE_CANCEL_REJECT = "LATE_CANCEL_REJECT"
    INCIDENT = "INCIDENT"
    RECONCILED = "RECONCILED"


class IncidentCode(StrEnum):
    TRANSITION_INVALID = "TRANSITION_INVALID"
    UNKNOWN_ORDER = "UNKNOWN_ORDER"
    SEQ_CONFLICT = "SEQ_CONFLICT"
    EXEC_ID_CONFLICT = "EXEC_ID_CONFLICT"
    COST_INVALID = "COST_INVALID"


class OmsRecord(Frozen):
    """Înregistrare de audit emisă de OMS înaintea efectului."""

    seq: int
    ts: UtcDatetime
    type: RecordType
    correlation_id: str  # client_order_id
    outcome: str
    payload: dict[str, Any]


@runtime_checkable
class OmsSink(Protocol):
    def record(self, record: OmsRecord) -> None: ...


class ListSink:
    """Sink în memorie (teste, backtest fără persistență)."""

    def __init__(self) -> None:
        self.records: list[OmsRecord] = []

    def record(self, record: OmsRecord) -> None:
        self.records.append(record)


class JournalOmsSink:
    """Adaptor spre jurnalul append-only; `oms_seq` este păstrat în payload."""

    def __init__(self, journal: Journal, actor: str = "system") -> None:
        self._journal = journal
        self._actor = actor

    def record(self, record: OmsRecord) -> None:
        self._journal.append(
            ts=record.ts,
            type=f"oms.{record.type.value}",
            correlation_id=record.correlation_id,
            component="oms",
            component_version=OMS_VERSION,
            actor=self._actor,
            outcome=record.outcome,
            payload={"oms_seq": record.seq, **record.payload},
        )


# --------------------------------------------------------------------------- rezultate


class ExecStatus(StrEnum):
    APPLIED = "APPLIED"
    RETRANSMISSION = "RETRANSMISSION"
    BUFFERED = "BUFFERED"
    HELD = "HELD"
    INCIDENT = "INCIDENT"
    IGNORED = "IGNORED"


@dataclass(frozen=True, slots=True)
class MissingMessagesRequest:
    """Cerere către broker pentru mesajele lipsă ale unui ordin (10.4)."""

    client_order_id: str
    missing_seqs: tuple[int, ...]
    ts: datetime


@dataclass(frozen=True, slots=True)
class ExecOutcome:
    event: ExecutionEvent
    status: ExecStatus
    record_seq: int
    order: Order | None
    fill: PortfolioFill | None = None
    incident: IncidentCode | None = None
    detail: str = ""
    request: MissingMessagesRequest | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Rezultatul unei comenzi interne; `send=True` cere efectul la broker (după jurnal)."""

    order: Order
    send: bool
    record_seq: int


class InstrumentBlockedError(RuntimeError):
    """Ordin nou pe un instrument înghețat (ordin `UNKNOWN` sau incident nerezolvat)."""


class UnknownOrderError(KeyError):
    """`client_order_id` necunoscut managerului."""


class ReconciliationMismatchError(ValueError):
    """Rezultatul reconcilierii nu poate fi aplicat; ordinul rămâne înghețat."""


@dataclass(slots=True)
class _Entry:
    order: Order
    next_seq: int
    cancel_outstanding: bool = False
    cancel_resolved_by_fill: bool = False
    frozen: str | None = None
    buffer: dict[int, ExecutionEvent] = field(default_factory=dict)
    held: list[ExecutionEvent] = field(default_factory=list)
    requested: set[int] = field(default_factory=set)


_RECONCILED: Final[dict[OrderState, Trigger]] = {
    OrderState.ACKNOWLEDGED: Trigger.RECONCILED_ACKNOWLEDGED,
    OrderState.PARTIALLY_FILLED: Trigger.RECONCILED_PARTIALLY_FILLED,
    OrderState.FILLED: Trigger.RECONCILED_FILLED,
    OrderState.REJECTED_BROKER: Trigger.RECONCILED_REJECTED,
    OrderState.CANCELLED: Trigger.RECONCILED_CANCELLED,
}

_FILL_KINDS: Final = frozenset({ExecKind.PARTIAL_FILL, ExecKind.FILL})


def _with(order: Order, **update: Any) -> Order:
    """Copie validată (invariantele `Order` sunt reverificate)."""
    return Order.model_validate({**order.model_dump(), **update})


def _order_json(order: Order) -> dict[str, Any]:
    data: dict[str, Any] = order.model_dump(mode="json")
    data["remaining_qty"] = str(order.remaining_qty)
    return data


# --------------------------------------------------------------------------- manager


class OrderManager:
    def __init__(
        self,
        sink: OmsSink,
        *,
        registry: IdempotencyRegistry | None = None,
        cost_fn: CostFn = commission_only,
        fx_fn: FxFn = eur_identity,
        first_seq: int = 1,
    ) -> None:
        self._sink = sink
        self._registry = registry if registry is not None else IdempotencyRegistry()
        self._cost_fn = cost_fn
        self._fx_fn = fx_fn
        self._first_seq = first_seq
        self._seq = 0
        self._orders: dict[str, _Entry] = {}
        # broker_exec_id → (record_seq, amprentă) pentru execuțiile din buffer sau reținute
        self._pending: dict[str, tuple[int, str]] = {}

    # ------------------------------------------------------------------ interogări

    def order(self, client_order_id: str) -> Order:
        return self._entry(client_order_id).order

    @property
    def orders(self) -> dict[str, Order]:
        return {k: e.order for k, e in self._orders.items()}

    def is_frozen(self, client_order_id: str) -> bool:
        return self._entry(client_order_id).frozen is not None

    def is_instrument_blocked(self, instrument: str) -> bool:
        return instrument in self.blocked_instruments()

    def blocked_instruments(self) -> frozenset[str]:
        return frozenset(e.order.instrument for e in self._orders.values() if e.frozen)

    def cancel_outstanding(self, client_order_id: str) -> bool:
        return self._entry(client_order_id).cancel_outstanding

    def missing_seqs(self, client_order_id: str) -> tuple[int, ...]:
        """Secvențele cerute și încă nesosite pentru ordin."""
        return tuple(sorted(self._entry(client_order_id).requested))

    @property
    def registry(self) -> IdempotencyRegistry:
        return self._registry

    # ------------------------------------------------------------------ comenzi interne

    def create_order(
        self,
        intent: OrderIntent,
        decision: RiskDecision,
        *,
        run_id: str,
        strategy_id: str,
        signal_seq: int,
        ts: datetime,
    ) -> CommandResult:
        """Creează ordinul pentru un Order_Intent aprobat (10.1); idempotent după cheie (10.2)."""
        if decision.intent_id != intent.intent_id:
            raise ValueError("decizia de risc aparține altui Order_Intent")
        if not decision.approved or decision.qty is None:
            raise ValueError("numai un Order_Intent aprobat de Risk_Engine devine ordin")
        coid = idempotency_key(run_id, strategy_id, intent.instrument, signal_seq)
        existing = self._orders.get(coid)
        if existing is not None:
            seq = self._emit(
                ts,
                RecordType.ORDER_DUPLICATE,
                coid,
                "ignored",
                {"intent_id": intent.intent_id, "state": existing.order.state.value},
            )
            return CommandResult(existing.order, send=False, record_seq=seq)
        if self.is_instrument_blocked(intent.instrument):
            self._emit(
                ts,
                RecordType.ORDER_BLOCKED,
                coid,
                "blocked",
                {"intent_id": intent.intent_id, "instrument": intent.instrument},
            )
            raise InstrumentBlockedError(
                f"instrumentul {intent.instrument} este înghețat până la reconciliere"
            )
        created = Order(
            client_order_id=coid,
            intent_id=intent.intent_id,
            instrument=intent.instrument,
            side=intent.side,
            qty=decision.qty,
        )
        res = fsm.apply(created.state, Trigger.RISK_APPROVE)
        order = _with(created, state=res.new_state, version=created.version + 1)
        seq = self._emit(
            ts,
            RecordType.ORDER_CREATED,
            coid,
            "ok",
            {
                "order": _order_json(order),
                "run_id": run_id,
                "strategy_id": strategy_id,
                "signal_id": intent.signal_id,
                "signal_seq": signal_seq,
                "risk_rules_version": decision.rules_version,
            },
        )
        self._registry.register_order(coid)
        self._orders[coid] = _Entry(order=order, next_seq=self._first_seq)
        return CommandResult(order, send=False, record_seq=seq)

    def submit(self, client_order_id: str, ts: datetime) -> CommandResult:
        """Write-ahead: jurnalizează `APPROVED → SUBMITTED`; trimiterea urmează după `send`."""
        entry = self._entry(client_order_id)
        if entry.frozen:
            return self._refuse(entry, "submit", ts)
        if entry.order.state is not OrderState.APPROVED:
            seq = self._emit(
                ts,
                RecordType.COMMAND_DUPLICATE,
                client_order_id,
                "ignored",
                {"command": "submit", "state": entry.order.state.value},
            )
            return CommandResult(entry.order, send=False, record_seq=seq)
        return self._command(entry, Trigger.SUBMIT, ts, send_on_ok=True)

    def submit_timeout(self, client_order_id: str, ts: datetime) -> CommandResult:
        """Timeout / deconectare după trimitere: `SUBMITTED → UNKNOWN` și înghețare (9.6)."""
        entry = self._entry(client_order_id)
        if entry.order.state is OrderState.UNKNOWN:
            seq = self._emit(
                ts,
                RecordType.COMMAND_DUPLICATE,
                client_order_id,
                "ignored",
                {"command": "submit_timeout", "state": entry.order.state.value},
            )
            return CommandResult(entry.order, send=False, record_seq=seq)
        if entry.frozen:
            return self._refuse(entry, "submit_timeout", ts)
        result = self._command(entry, Trigger.SUBMIT_TIMEOUT, ts, send_on_ok=False)
        if result.order.state is OrderState.UNKNOWN:
            entry.frozen = OrderState.UNKNOWN.value
        return result

    def request_cancel(self, client_order_id: str, ts: datetime) -> CommandResult:
        """Cerere de anulare; `send=True` numai pentru prima cerere în curs."""
        entry = self._entry(client_order_id)
        if entry.frozen:
            return self._refuse(entry, "cancel", ts)
        if entry.cancel_outstanding:
            seq = self._emit(
                ts,
                RecordType.COMMAND_DUPLICATE,
                client_order_id,
                "ignored",
                {"command": "cancel", "state": entry.order.state.value},
            )
            return CommandResult(entry.order, send=False, record_seq=seq)
        result = self._command(entry, Trigger.CANCEL_REQUEST, ts, send_on_ok=True)
        if result.send:
            entry.cancel_outstanding = True
        return result

    # ------------------------------------------------------------------ execuții

    def on_execution(self, event: ExecutionEvent) -> list[ExecOutcome]:
        """Procesează un mesaj al brokerului; poate elibera și mesaje din buffer."""
        out: list[ExecOutcome] = []
        self._process(event, out, drain=True)
        return out

    # ------------------------------------------------------------------ reconciliere

    def reconcile(
        self,
        client_order_id: str,
        *,
        ts: datetime,
        state: OrderState,
        filled_qty: Decimal,
        avg_fill_price: Decimal | None = None,
        broker_order_id: str | None = None,
        reflected_exec_ids: Iterable[str] = (),
        next_seq: int | None = None,
        note: str = "",
    ) -> list[ExecOutcome]:
        """Rezolvă înghețarea pe baza rezultatului brokerului și reprocesează mesajele reținute."""
        entry = self._entry(client_order_id)
        order = entry.order
        if entry.frozen is None:
            raise ReconciliationMismatchError(f"ordinul {client_order_id} nu este înghețat")
        if order.state is OrderState.UNKNOWN:
            trigger = _RECONCILED.get(state)
            if trigger is None:
                raise ReconciliationMismatchError(f"starea {state} nu rezultă din reconciliere")
            self._check_reconciled_qty(order, state, filled_qty, avg_fill_price)
            res = fsm.apply(order.state, trigger)
            if not res.ok:  # pragma: no cover - tabelul conține toate țintele RECONCILED_*
                raise ReconciliationMismatchError(res.detail)
            updated = _with(
                order,
                state=res.new_state,
                filled_qty=filled_qty,
                avg_fill_price=avg_fill_price if filled_qty > 0 else None,
                broker_order_id=broker_order_id or order.broker_order_id,
                version=order.version + 1,
            )
        else:
            if state is not order.state or filled_qty != order.filled_qty:
                raise ReconciliationMismatchError(
                    f"broker: {state}/{filled_qty}, intern: {order.state}/{order.filled_qty}"
                )
            updated = order
        reflected = [x for x in reflected_exec_ids if self._registry.applied(x) is None]
        seq = self._emit(
            ts,
            RecordType.RECONCILED,
            client_order_id,
            "ok",
            {
                "from_state": order.state.value,
                "to_state": updated.state.value,
                "frozen_reason": entry.frozen,
                "order": _order_json(updated),
                "reflected_exec_ids": reflected,
                "next_seq": next_seq,
                "held": len(entry.held),
                "note": note,
            },
        )
        entry.order = updated
        entry.frozen = None
        if order.state is OrderState.UNKNOWN:
            entry.cancel_outstanding = False
        for exec_id in reflected:
            self._registry.mark_applied(AppliedExec(exec_id, client_order_id, seq, None))
        replay = entry.held
        entry.held = []
        if next_seq is not None:
            stale = sorted(s for s in entry.buffer if s < next_seq)
            replay.extend(entry.buffer.pop(s) for s in stale)
            entry.next_seq = next_seq
            entry.requested = {s for s in entry.requested if s >= next_seq}
        out: list[ExecOutcome] = []
        for event in replay:
            self._pending.pop(event.broker_exec_id, None)
            self._process(event, out, drain=True)
        self._drain(entry, out)
        return out

    # ------------------------------------------------------------------ intern

    def _entry(self, client_order_id: str) -> _Entry:
        entry = self._orders.get(client_order_id)
        if entry is None:
            raise UnknownOrderError(client_order_id)
        return entry

    def _emit(
        self,
        ts: datetime,
        type_: RecordType,
        coid: str,
        outcome: str,
        payload: dict[str, Any],
    ) -> int:
        seq = self._seq + 1
        record = OmsRecord(
            seq=seq, ts=ts, type=type_, correlation_id=coid, outcome=outcome, payload=payload
        )
        self._sink.record(record)  # write-ahead: înaintea oricărei modificări interne
        self._seq = seq
        return seq

    def _refuse(self, entry: _Entry, command: str, ts: datetime) -> CommandResult:
        seq = self._emit(
            ts,
            RecordType.COMMAND_REFUSED_FROZEN,
            entry.order.client_order_id,
            "refused",
            {"command": command, "state": entry.order.state.value, "frozen": entry.frozen},
        )
        return CommandResult(entry.order, send=False, record_seq=seq)

    def _command(
        self, entry: _Entry, trigger: Trigger, ts: datetime, *, send_on_ok: bool
    ) -> CommandResult:
        order = entry.order
        res = fsm.apply(order.state, trigger, order_qty=order.qty, filled_qty=order.filled_qty)
        if not res.ok:
            seq = self._emit(
                ts,
                RecordType.TRANSITION_REJECTED,
                order.client_order_id,
                "rejected",
                {
                    "trigger": trigger.value,
                    "state": order.state.value,
                    "reason": res.reason.value if res.reason else None,
                    "detail": res.detail,
                },
            )
            return CommandResult(order, send=False, record_seq=seq)
        updated = _with(order, state=res.new_state, version=order.version + 1)
        seq = self._emit(
            ts,
            RecordType.TRANSITION,
            order.client_order_id,
            "ok",
            {
                "trigger": trigger.value,
                "from_state": order.state.value,
                "to_state": updated.state.value,
                "remaining_qty": str(updated.remaining_qty),
            },
        )
        entry.order = updated
        return CommandResult(updated, send=send_on_ok, record_seq=seq)

    def _incident(
        self,
        event: ExecutionEvent,
        entry: _Entry | None,
        code: IncidentCode,
        detail: str,
        out: list[ExecOutcome],
        *,
        mark: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "code": code.value,
            "detail": detail,
            "broker_exec_id": event.broker_exec_id,
            "kind": event.kind.value,
            "seq": event.seq,
            "state": entry.order.state.value if entry else None,
            "frozen": entry is not None,
            **(extra or {}),
        }
        seq = self._emit(
            event.ts_receipt, RecordType.INCIDENT, event.client_order_id, code, payload
        )
        if entry is not None:
            entry.frozen = code.value
        if mark:
            self._registry.mark_applied(
                AppliedExec(
                    event.broker_exec_id, event.client_order_id, seq, exec_fingerprint(event)
                )
            )
        out.append(
            ExecOutcome(
                event=event,
                status=ExecStatus.INCIDENT,
                record_seq=seq,
                order=entry.order if entry else None,
                incident=code,
                detail=detail,
            )
        )

    def _retransmission(
        self,
        event: ExecutionEvent,
        original_seq: int,
        original_fp: str | None,
        out: list[ExecOutcome],
    ) -> None:
        entry = self._orders.get(event.client_order_id)
        fp = exec_fingerprint(event)
        conflict = original_fp is not None and original_fp != fp
        seq = self._emit(
            event.ts_receipt,
            RecordType.EXEC_RETRANSMISSION,
            event.client_order_id,
            "conflict" if conflict else "ignored",
            {
                "broker_exec_id": event.broker_exec_id,
                "original_record_seq": original_seq,
                "kind": event.kind.value,
                "seq": event.seq,
                "content_matches": not conflict,
            },
        )
        if conflict:
            self._incident(
                event,
                entry,
                IncidentCode.EXEC_ID_CONFLICT,
                "broker_exec_id refolosit cu alt conținut",
                out,
                mark=False,
                extra={"original_record_seq": original_seq},
            )
            return
        out.append(
            ExecOutcome(
                event=event,
                status=ExecStatus.RETRANSMISSION,
                record_seq=seq,
                order=entry.order if entry else None,
            )
        )

    def _process(self, event: ExecutionEvent, out: list[ExecOutcome], *, drain: bool) -> None:
        applied = self._registry.applied(event.broker_exec_id)
        if applied is not None:
            self._retransmission(event, applied.record_seq, applied.fingerprint, out)
            return
        pending = self._pending.get(event.broker_exec_id)
        if pending is not None:
            self._retransmission(event, pending[0], pending[1], out)
            return
        entry = self._orders.get(event.client_order_id)
        if entry is None:
            self._incident(event, None, IncidentCode.UNKNOWN_ORDER, "ordin necunoscut", out)
            return
        if entry.frozen:
            seq = self._emit(
                event.ts_receipt,
                RecordType.EXEC_HELD,
                event.client_order_id,
                "held",
                {"broker_exec_id": event.broker_exec_id, "seq": event.seq, "frozen": entry.frozen},
            )
            entry.held.append(event)
            self._pending[event.broker_exec_id] = (seq, exec_fingerprint(event))
            out.append(ExecOutcome(event, ExecStatus.HELD, seq, entry.order))
            return
        if event.seq is None:
            self._apply(entry, event, out)
            return
        if event.seq < entry.next_seq or event.seq in entry.buffer:
            self._incident(
                event,
                entry,
                IncidentCode.SEQ_CONFLICT,
                f"secvența {event.seq} este deja consumată sau în buffer",
                out,
            )
            return
        if event.seq > entry.next_seq:
            self._buffer(entry, event, out)
            return
        entry.next_seq += 1
        entry.requested.discard(event.seq)
        self._apply(entry, event, out)
        if drain:
            self._drain(entry, out)

    def _buffer(self, entry: _Entry, event: ExecutionEvent, out: list[ExecOutcome]) -> None:
        if event.seq is None:  # pragma: no cover - apelat numai pentru mesaje secvențiate
            raise ValueError("mesaj nesecvențiat")
        missing = tuple(
            s
            for s in range(entry.next_seq, event.seq)
            if s not in entry.buffer and s not in entry.requested
        )
        seq = self._emit(
            event.ts_receipt,
            RecordType.EXEC_BUFFERED,
            event.client_order_id,
            "suspended",
            {"broker_exec_id": event.broker_exec_id, "seq": event.seq, "expected": entry.next_seq},
        )
        entry.buffer[event.seq] = event
        self._pending[event.broker_exec_id] = (seq, exec_fingerprint(event))
        request: MissingMessagesRequest | None = None
        if missing:
            request = MissingMessagesRequest(event.client_order_id, missing, event.ts_receipt)
            self._emit(
                event.ts_receipt,
                RecordType.MISSING_MESSAGES_REQUEST,
                event.client_order_id,
                "requested",
                {"missing_seqs": list(missing), "trigger_record_seq": seq},
            )
            entry.requested.update(missing)
        out.append(ExecOutcome(event, ExecStatus.BUFFERED, seq, entry.order, request=request))

    def _drain(self, entry: _Entry, out: list[ExecOutcome]) -> None:
        while entry.frozen is None and entry.next_seq in entry.buffer:
            event = entry.buffer.pop(entry.next_seq)
            self._pending.pop(event.broker_exec_id, None)
            self._process(event, out, drain=False)

    def _apply(self, entry: _Entry, event: ExecutionEvent, out: list[ExecOutcome]) -> None:
        order = entry.order
        if (
            event.kind is ExecKind.CANCEL_REJECTED
            and order.state is OrderState.FILLED
            and entry.cancel_resolved_by_fill
        ):
            seq = self._emit(
                event.ts_receipt,
                RecordType.LATE_CANCEL_REJECT,
                order.client_order_id,
                "ignored",
                {
                    "broker_exec_id": event.broker_exec_id,
                    "state": order.state.value,
                    "remaining_qty": str(order.remaining_qty),
                    "reason": event.reason,
                },
            )
            self._mark(event, seq)
            out.append(ExecOutcome(event, ExecStatus.IGNORED, seq, order))
            return

        trigger = fsm.trigger_for_exec(event.kind)
        is_fill = event.kind in _FILL_KINDS
        normalized = False
        if (
            event.kind is ExecKind.PARTIAL_FILL
            and event.qty is not None
            and event.qty == order.remaining_qty
        ):
            trigger, normalized = Trigger.FILL, True
        res = fsm.apply(
            order.state,
            trigger,
            order_qty=order.qty,
            filled_qty=order.filled_qty,
            exec_qty=event.qty if is_fill else None,
        )
        if not res.ok:
            self._incident(
                event,
                entry,
                IncidentCode.TRANSITION_INVALID,
                res.detail,
                out,
                extra={
                    "trigger": trigger.value,
                    "fsm_reason": res.reason.value if res.reason else None,
                },
            )
            return

        update: dict[str, Any] = {"state": res.new_state, "version": order.version + 1}
        fill: PortfolioFill | None = None
        if is_fill and event.qty is not None and event.price is not None:
            try:
                costs = self._cost_fn(order, event)
                currency, fx = self._fx_fn(order, event)
                fill = PortfolioFill(
                    fill_id=event.broker_exec_id,
                    instrument=order.instrument,
                    currency=currency,
                    side=order.side,
                    qty=event.qty,
                    price=event.price,
                    fx_rate_to_eur=fx,
                    ts=event.ts_broker,
                    costs=costs,
                )
            except ValueError as exc:
                self._incident(event, entry, IncidentCode.COST_INVALID, str(exc), out)
                return
            new_filled = order.filled_qty + event.qty
            with localcontext(_CTX):
                prev_value = (order.avg_fill_price or ZERO) * order.filled_qty
                avg = (prev_value + event.price * event.qty) / new_filled
            update |= {
                "filled_qty": new_filled,
                "avg_fill_price": avg,
                "costs": order.costs + costs,
            }
        if event.broker_order_id and event.kind is ExecKind.ACK:
            update["broker_order_id"] = event.broker_order_id
        updated = _with(order, **update)

        seq = self._emit(
            event.ts_receipt,
            RecordType.EXEC_APPLIED,
            order.client_order_id,
            "ok",
            {
                "broker_exec_id": event.broker_exec_id,
                "kind": event.kind.value,
                "trigger": trigger.value,
                "normalized": normalized,
                "seq": event.seq,
                "from_state": order.state.value,
                "to_state": updated.state.value,
                "exec_qty": str(event.qty) if event.qty is not None else None,
                "exec_price": str(event.price) if event.price is not None else None,
                "fill_costs": fill.costs.model_dump(mode="json") if fill else None,
                "order": _order_json(updated),
                "reason": event.reason,
            },
        )
        entry.order = updated
        self._mark(event, seq)

        restore = (
            trigger is Trigger.PARTIAL_FILL
            and order.state is OrderState.CANCEL_PENDING
            and updated.state is OrderState.PARTIALLY_FILLED
            and entry.cancel_outstanding
        )
        if trigger in (Trigger.CANCEL_CONFIRMED, Trigger.CANCEL_REJECTED, Trigger.EXPIRE):
            entry.cancel_outstanding = False
        elif updated.state is OrderState.FILLED and entry.cancel_outstanding:
            entry.cancel_outstanding = False
            entry.cancel_resolved_by_fill = True
        if restore:
            self._restore_cancel_pending(entry, event)
        out.append(ExecOutcome(event, ExecStatus.APPLIED, seq, entry.order, fill=fill))

    def _restore_cancel_pending(self, entry: _Entry, event: ExecutionEvent) -> None:
        """Revine în `CANCEL_PENDING` după o execuție parțială; nu trimite altă anulare."""
        order = entry.order
        res = fsm.apply(
            order.state, Trigger.CANCEL_REQUEST, order_qty=order.qty, filled_qty=order.filled_qty
        )
        if not res.ok:  # pragma: no cover - PARTIALLY_FILLED + CANCEL_REQUEST este permis
            raise RuntimeError(res.detail)
        updated = _with(order, state=res.new_state, version=order.version + 1)
        self._emit(
            event.ts_receipt,
            RecordType.CANCEL_PENDING_RESTORED,
            order.client_order_id,
            "ok",
            {
                "trigger": Trigger.CANCEL_REQUEST.value,
                "from_state": order.state.value,
                "to_state": updated.state.value,
                "remaining_qty": str(updated.remaining_qty),
                "broker_request_sent": False,
                "after_broker_exec_id": event.broker_exec_id,
            },
        )
        entry.order = updated

    def _mark(self, event: ExecutionEvent, record_seq: int) -> None:
        self._registry.mark_applied(
            AppliedExec(
                event.broker_exec_id, event.client_order_id, record_seq, exec_fingerprint(event)
            )
        )

    @staticmethod
    def _check_reconciled_qty(
        order: Order, state: OrderState, filled_qty: Decimal, avg: Decimal | None
    ) -> None:
        ok = {
            OrderState.ACKNOWLEDGED: filled_qty == 0,
            OrderState.REJECTED_BROKER: filled_qty == 0,
            OrderState.PARTIALLY_FILLED: 0 < filled_qty < order.qty,
            OrderState.FILLED: filled_qty == order.qty,
            OrderState.CANCELLED: 0 <= filled_qty < order.qty,
        }[state]
        if not ok:
            raise ReconciliationMismatchError(
                f"filled_qty={filled_qty} incompatibil cu {state} (qty={order.qty})"
            )
        if filled_qty > 0 and (avg is None or avg <= 0):
            raise ReconciliationMismatchError("avg_fill_price > 0 este necesar când filled_qty > 0")
