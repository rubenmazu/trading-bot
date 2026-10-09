"""Reconciliere între proiecția internă și `BrokerSnapshot` (Req 11.1–11.7).

`Reconciler` compară starea raportată de broker (`BrokerSnapshot`: poziții, numerar, ordine
deschise, execuții) cu proiecția internă (`Portfolio` + `OrderManager`) și blochează sincron
divergențele prin `Kill_Switch`, înaintea oricărui ordin nou (fail-closed).

Momente de rulare (design, 11.1–11.4)
    La pornire, după reconectare, după fiecare `Execution_Event` (pentru instrumentul afectat)
    și periodic (interval ≤ 60 s în Demo/Live). Reconciler nu pornește fire proprii: expune
    `should_run(now, mode)` și `interval_seconds(mode)`, iar motorul/bootstrap conduce timpul
    (o integrare de `Timer` externă citește acești hook-uri).

Toleranțe versionate (11.5)
    `ReconciliationTolerances` poartă o `version` stabilă, o toleranță implicită de cantitate,
    toleranțe pe instrument și o toleranță de numerar (EUR). O diferență peste toleranță este
    o divergență.

Blocare (11.5, 11.6, design)
    - O diferență de poziție sau un ordin deschis neconcordant pe un instrument blochează
      sincron **numai** acel instrument (`Kill_Switch` INSTRUMENT) și scrie un `Audit_Record`.
    - O diferență de numerar, sau una care nu poate fi atribuită unui instrument, activează
      `Kill_Switch` GLOBAL (numerarul stă la baza tuturor limitelor de risc).
    - Dacă snapshot-ul nu poate fi obținut complet (`complete == False` sau reconcilierea
      ridică o excepție), se activează `Kill_Switch` GLOBAL (11.6). Reconciler eșuează închis:
      orice excepție neașteptată devine o divergență GLOBAL auditată, nu o reconciliere reușită.

Recuperare înaintea ordinelor noi (11.1)
    OMS rămâne autoritatea stării ordinelor. Pentru ordinele interne `UNKNOWN` pe care
    snapshot-ul le rezolvă, Reconciler apelează `oms.reconcile(...)` ca să le decongeleze pe
    baza stării brokerului, înainte ca motorul să trimită ordine noi. Ordinele deschise la
    broker fără corespondent intern (sau invers) sunt raportate ca divergențe.

Reluare (11.7)
    Reluarea trece prin `KillSwitch.resume`, care cere deja cauză, corecție, aprobarea
    operatorului și o reconciliere ulterioară reușită. `request_resume` este un ajutor subțire
    care construiește `ResumeRequest` și deleagă; o reconciliere eșuată refuză reluarea.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final

from pydantic import Field, model_validator

from qts.broker.adapter import BrokerSnapshot
from qts.core.clock import Clock, ensure_utc
from qts.core.models import Dec, Frozen, OrderState, UtcDatetime
from qts.core.money import REPORTING_CURRENCY, ZERO
from qts.oms.fsm import TERMINAL_STATES
from qts.oms.manager import OrderManager, ReconciliationMismatchError
from qts.persistence.journal import Journal
from qts.portfolio.portfolio import PortfolioState
from qts.risk.context import KillSwitchScope
from qts.safety.kill_switch import KillSwitch, ResumeDecision, ResumeRequest

__all__ = [
    "COMPONENT",
    "COMPONENT_VERSION",
    "DEMO_LIVE_MAX_INTERVAL_SECONDS",
    "DifferenceCode",
    "Reconciler",
    "ReconciliationDifference",
    "ReconciliationReason",
    "ReconciliationResult",
    "ReconciliationTolerances",
]

COMPONENT: Final = "reconciliation"
COMPONENT_VERSION: Final = "1"
JOURNAL_TYPE_PREFIX: Final = "recon."
DEMO_LIVE_MAX_INTERVAL_SECONDS: Final = 60

Mode = str  # Operational_Mode: "backtest" | "sim" | "shadow" | "demo" | "live"
_DEMO_LIVE: Final = frozenset({"demo", "live"})

# Stările brokerului considerate „ordin deschis” (nefinalizate).
_OPEN_BROKER_STATES: Final = frozenset(
    s for s in OrderState if s not in TERMINAL_STATES and s is not OrderState.CREATED
)


class ReconciliationReason(StrEnum):
    """Motivul declanșării unei reconcilieri (design, 11.1–11.4)."""

    STARTUP = "STARTUP"
    RECONNECT = "RECONNECT"
    EXECUTION_EVENT = "EXECUTION_EVENT"
    PERIODIC = "PERIODIC"


class DifferenceCode(StrEnum):
    """Coduri stabile pentru divergențele detectate la reconciliere."""

    POSITION_MISMATCH = "RECON_POSITION_MISMATCH"
    CASH_MISMATCH = "RECON_CASH_MISMATCH"
    MISSING_OPEN_ORDER = "RECON_MISSING_OPEN_ORDER"  # deschis la broker, absent/închis intern
    UNEXPECTED_OPEN_ORDER = "RECON_UNEXPECTED_OPEN_ORDER"  # deschis intern, absent la broker
    SNAPSHOT_INCOMPLETE = "RECON_SNAPSHOT_INCOMPLETE"
    SNAPSHOT_UNAVAILABLE = "RECON_SNAPSHOT_UNAVAILABLE"
    INTERNAL_ERROR = "RECON_INTERNAL_ERROR"


# Divergențele care nu pot fi atribuite unui instrument individual escaladează la GLOBAL.
_GLOBAL_CODES: Final = frozenset(
    {
        DifferenceCode.CASH_MISMATCH,
        DifferenceCode.SNAPSHOT_INCOMPLETE,
        DifferenceCode.SNAPSHOT_UNAVAILABLE,
        DifferenceCode.INTERNAL_ERROR,
    }
)


class ReconciliationTolerances(Frozen):
    """Toleranțe versionate pentru reconciliere (11.5). Model imuabil.

    `reporting_currency` este moneda de raportare a rulării (implicit EUR, deci comportamentul
    Backtest/Shadow rămâne neschimbat). Reconcilierea compară `snapshot.currency` cu această
    monedă configurată, nu cu constanta globală: astfel un Demo în USD (cont Alpaca paper în USD)
    este intern coerent, iar reconcilierea rămâne strictă — orice monedă diferită de cea de
    raportare activează Kill_Switch GLOBAL (fail-closed).
    """

    version: str = "1"
    qty_default: Dec = ZERO
    qty_by_instrument: dict[str, Dec] = Field(default_factory=dict)
    cash_eur: Dec = ZERO
    reporting_currency: str = REPORTING_CURRENCY

    @model_validator(mode="after")
    def _check(self) -> ReconciliationTolerances:
        if not self.version.strip():
            raise ValueError("version este obligatoriu")
        if not self.reporting_currency.strip():
            raise ValueError("reporting_currency este obligatoriu")
        if self.qty_default < 0 or self.cash_eur < 0:
            raise ValueError("toleranțele nu pot fi negative")
        for key, value in self.qty_by_instrument.items():
            if not key.strip():
                raise ValueError("instrumentul toleranței nu poate fi gol")
            if value < 0:
                raise ValueError(f"toleranța pentru {key} nu poate fi negativă")
        return self

    def qty_tolerance(self, instrument: str) -> Decimal:
        return self.qty_by_instrument.get(instrument, self.qty_default)


class ReconciliationDifference(Frozen):
    """O divergență detectată, cu cod stabil, instrument (dacă există) și valorile comparate."""

    code: DifferenceCode
    scope: KillSwitchScope
    instrument: str | None = None
    broker_value: str | None = None
    internal_value: str | None = None
    detail: str = ""


class ReconciliationResult(Frozen):
    """Rezultatul structurat al unei reconcilieri."""

    ts: UtcDatetime
    reason: ReconciliationReason
    tolerances_version: str
    complete: bool
    differences: tuple[ReconciliationDifference, ...] = ()
    activated_global: bool = False
    blocked_instruments: tuple[str, ...] = ()
    resolved_orders: tuple[str, ...] = ()  # client_order_id decongelate pe baza snapshot-ului

    @property
    def ok(self) -> bool:
        """Reconciliere reușită: snapshot complet și nicio divergență."""
        return self.complete and not self.differences


class Reconciler:
    """Compară broker vs intern și blochează divergențele prin `Kill_Switch` (fail-closed)."""

    def __init__(
        self,
        kill_switch: KillSwitch,
        journal: Journal,
        clock: Clock,
        tolerances: ReconciliationTolerances,
        *,
        component: str = COMPONENT,
    ) -> None:
        self._ks = kill_switch
        self._journal = journal
        self._clock = clock
        self._tol = tolerances
        self._component = component

    # ------------------------------------------------------------------ programare

    def interval_seconds(self, mode: Mode) -> int:
        """Intervalul maxim de reconciliere periodică: ≤ 60 s în Demo/Live (11.3)."""
        return DEMO_LIVE_MAX_INTERVAL_SECONDS

    def requires_periodic(self, mode: Mode) -> bool:
        """Numai Demo/Live cer reconciliere periodică (11.3)."""
        return mode in _DEMO_LIVE

    def should_run(self, now: datetime, mode: Mode, *, last_run: datetime | None = None) -> bool:
        """Hook de programare: True dacă a trecut intervalul de la ultima rulare (11.3).

        Reconciler nu pornește fire proprii; motorul/bootstrap apelează acest hook pentru a decide
        când să ruleze reconcilierea periodică. Fără o rulare anterioară, răspunsul este True.
        """
        if not self.requires_periodic(mode):
            return False
        if last_run is None:
            return True
        elapsed = (ensure_utc(now) - ensure_utc(last_run)).total_seconds()
        return elapsed >= self.interval_seconds(mode)

    # ------------------------------------------------------------------ reconciliere

    def reconcile(
        self,
        snapshot: BrokerSnapshot,
        portfolio_state: PortfolioState,
        oms: OrderManager,
        *,
        reason: ReconciliationReason = ReconciliationReason.PERIODIC,
    ) -> ReconciliationResult:
        """Compară snapshot-ul brokerului cu proiecția internă și blochează divergențele.

        Fail-closed: orice excepție neașteptată devine o divergență GLOBAL auditată.
        """
        ts = ensure_utc(self._clock.now())
        try:
            return self._reconcile(snapshot, portfolio_state, oms, reason, ts)
        except Exception as exc:  # fail-closed: nimic nu rămâne nereconciliat în tăcere
            diff = ReconciliationDifference(
                code=DifferenceCode.INTERNAL_ERROR,
                scope=KillSwitchScope.GLOBAL,
                detail=f"{type(exc).__name__}: {exc}",
            )
            return self._block([diff], ts, reason, resolved=())

    def _reconcile(
        self,
        snapshot: BrokerSnapshot,
        portfolio_state: PortfolioState,
        oms: OrderManager,
        reason: ReconciliationReason,
        ts: datetime,
    ) -> ReconciliationResult:
        if not snapshot.complete:
            diff = ReconciliationDifference(
                code=DifferenceCode.SNAPSHOT_INCOMPLETE,
                scope=KillSwitchScope.GLOBAL,
                detail="snapshot-ul brokerului nu este complet (11.6)",
            )
            return self._block([diff], ts, reason, resolved=())

        differences: list[ReconciliationDifference] = []
        differences.extend(self._compare_positions(snapshot, portfolio_state))
        differences.extend(self._compare_cash(snapshot, portfolio_state))
        resolved, order_diffs = self._compare_orders(snapshot, oms, ts)
        differences.extend(order_diffs)
        return self._block(differences, ts, reason, resolved=tuple(resolved))

    # ------------------------------------------------------------------ comparații

    def _compare_positions(
        self, snapshot: BrokerSnapshot, portfolio_state: PortfolioState
    ) -> list[ReconciliationDifference]:
        """Compară cantitatea pe instrument: broker (semnat) vs portofoliu (long, qty ≥ 0)."""
        internal: dict[str, Decimal] = {
            inst: pos.qty for inst, pos in portfolio_state.positions.items()
        }
        broker: Mapping[str, Decimal] = snapshot.positions
        out: list[ReconciliationDifference] = []
        for instrument in sorted(set(internal) | set(broker)):
            broker_qty = broker.get(instrument, ZERO)
            internal_qty = internal.get(instrument, ZERO)
            tol = self._tol.qty_tolerance(instrument)
            if abs(broker_qty - internal_qty) > tol:
                out.append(
                    ReconciliationDifference(
                        code=DifferenceCode.POSITION_MISMATCH,
                        scope=KillSwitchScope.INSTRUMENT,
                        instrument=instrument,
                        broker_value=str(broker_qty),
                        internal_value=str(internal_qty),
                        detail=f"diferență de poziție peste toleranța {tol}",
                    )
                )
        return out

    def _compare_cash(
        self, snapshot: BrokerSnapshot, portfolio_state: PortfolioState
    ) -> list[ReconciliationDifference]:
        """Compară numerarul: broker (moneda contului) vs proiecția internă (moneda de raportare).

        Moneda contului brokerului trebuie să fie exact moneda de raportare *configurată* a rulării
        (`self._tol.reporting_currency`, implicit EUR). Orice altă monedă este o divergență GLOBAL
        (numerarul stă la baza tuturor limitelor de risc). Rămâne strict: un cont USD reconciliază
        numai într-o rulare configurată în USD; o rulare în EUR respinge un snapshot USD.
        """
        reporting = self._tol.reporting_currency
        if snapshot.currency != reporting:
            return [
                ReconciliationDifference(
                    code=DifferenceCode.CASH_MISMATCH,
                    scope=KillSwitchScope.GLOBAL,
                    broker_value=f"{snapshot.cash} {snapshot.currency}",
                    internal_value=f"{portfolio_state.cash_eur} {reporting}",
                    detail="moneda numerarului brokerului nu este cea de raportare",
                )
            ]
        if abs(snapshot.cash - portfolio_state.cash_eur) > self._tol.cash_eur:
            return [
                ReconciliationDifference(
                    code=DifferenceCode.CASH_MISMATCH,
                    scope=KillSwitchScope.GLOBAL,
                    broker_value=str(snapshot.cash),
                    internal_value=str(portfolio_state.cash_eur),
                    detail=f"diferență de numerar peste toleranța {self._tol.cash_eur} {reporting}",
                )
            ]
        return []

    def _compare_orders(
        self, snapshot: BrokerSnapshot, oms: OrderManager, ts: datetime
    ) -> tuple[list[str], list[ReconciliationDifference]]:
        """Reconciliază ordinele: decongelează `UNKNOWN` din snapshot, raportează neconcordanțele.

        OMS rămâne autoritatea: pentru ordinele interne `UNKNOWN` pe care snapshot-ul le descrie,
        apelează `oms.reconcile(...)`. Ordinele deschise la broker fără corespondent deschis
        intern (și invers) sunt divergențe de instrument.
        """
        resolved: list[str] = []
        out: list[ReconciliationDifference] = []
        internal_orders = oms.orders
        broker_by_coid = {o.client_order_id: o for o in snapshot.orders}

        for coid, status in broker_by_coid.items():
            order = internal_orders.get(coid)
            if order is None:
                out.append(
                    ReconciliationDifference(
                        code=DifferenceCode.MISSING_OPEN_ORDER,
                        scope=KillSwitchScope.INSTRUMENT,
                        instrument=status.instrument,
                        broker_value=status.state.value,
                        internal_value=None,
                        detail=f"ordinul {coid} există la broker, dar nu intern",
                    )
                )
                continue
            if order.state is OrderState.UNKNOWN:
                try:
                    oms.reconcile(
                        coid,
                        ts=ts,
                        state=status.state,
                        filled_qty=status.filled_qty,
                        avg_fill_price=status.avg_fill_price,
                        broker_order_id=status.broker_order_id,
                        note=f"recon:{self._tol.version}",
                    )
                    resolved.append(coid)
                except ReconciliationMismatchError as exc:
                    out.append(
                        ReconciliationDifference(
                            code=DifferenceCode.POSITION_MISMATCH,
                            scope=KillSwitchScope.INSTRUMENT,
                            instrument=status.instrument,
                            broker_value=f"{status.state.value}/{status.filled_qty}",
                            internal_value=f"{order.state.value}/{order.filled_qty}",
                            detail=f"reconcilierea OMS a eșuat: {exc}",
                        )
                    )

        for coid, order in internal_orders.items():
            if order.state in _OPEN_BROKER_STATES and coid not in broker_by_coid:
                out.append(
                    ReconciliationDifference(
                        code=DifferenceCode.UNEXPECTED_OPEN_ORDER,
                        scope=KillSwitchScope.INSTRUMENT,
                        instrument=order.instrument,
                        broker_value=None,
                        internal_value=order.state.value,
                        detail=f"ordinul {coid} este deschis intern, dar absent la broker",
                    )
                )
        return resolved, out

    # ------------------------------------------------------------------ blocare + audit

    def _block(
        self,
        differences: list[ReconciliationDifference],
        ts: datetime,
        reason: ReconciliationReason,
        *,
        resolved: tuple[str, ...],
    ) -> ReconciliationResult:
        """Activează kill switch-urile necesare, auditează divergențele și întoarce rezultatul."""
        complete = not any(d.code is DifferenceCode.SNAPSHOT_INCOMPLETE for d in differences)
        activated_global = False
        blocked: list[str] = []
        for diff in differences:
            if diff.code in _GLOBAL_CODES or diff.scope is KillSwitchScope.GLOBAL:
                self._ks.activate_automatic(
                    KillSwitchScope.GLOBAL,
                    component=self._component,
                    reason_code=diff.code.value,
                    detail=diff.detail,
                )
                activated_global = True
            else:
                instrument = diff.instrument
                if instrument is None:  # fail-closed: nealocabil → GLOBAL
                    self._ks.activate_automatic(
                        KillSwitchScope.GLOBAL,
                        component=self._component,
                        reason_code=diff.code.value,
                        detail=diff.detail,
                    )
                    activated_global = True
                else:
                    self._ks.activate_automatic(
                        KillSwitchScope.INSTRUMENT,
                        component=self._component,
                        reason_code=diff.code.value,
                        detail=diff.detail,
                        instrument=instrument,
                    )
                    blocked.append(instrument)
            self._audit(diff, ts, reason)
        return ReconciliationResult(
            ts=ts,
            reason=reason,
            tolerances_version=self._tol.version,
            complete=complete,
            differences=tuple(differences),
            activated_global=activated_global,
            blocked_instruments=tuple(dict.fromkeys(blocked)),
            resolved_orders=resolved,
        )

    def _audit(
        self, diff: ReconciliationDifference, ts: datetime, reason: ReconciliationReason
    ) -> None:
        """Scrie un `Audit_Record` pentru o divergență (11.5)."""
        self._journal.append(
            ts=ts,
            type=f"{JOURNAL_TYPE_PREFIX}difference",
            correlation_id=diff.instrument or "GLOBAL",
            component=self._component,
            component_version=COMPONENT_VERSION,
            actor=f"system:{self._component}",
            outcome=diff.code.value,
            payload={
                "reason": reason.value,
                "tolerances_version": self._tol.version,
                "code": diff.code.value,
                "scope": diff.scope.value,
                "instrument": diff.instrument,
                "broker_value": diff.broker_value,
                "internal_value": diff.internal_value,
                "detail": diff.detail,
            },
        )

    # ------------------------------------------------------------------ reluare (11.7)

    def request_resume(
        self,
        activation_id: str,
        *,
        operator: str,
        approved: bool,
        reconciliation_ok: bool,
        reconciliation_id: str,
        reason_resolved: bool,
        cause: str,
        correction: str,
        ts: datetime | None = None,
    ) -> ResumeDecision:
        """Ajutor subțire peste `KillSwitch.resume`: cauză + corecție + aprobare + recon (11.7).

        `KillSwitch` rămâne autoritatea reluării; o reconciliere eșuată (`reconciliation_ok=False`)
        este refuzată de el, indiferent de aprobare.
        """
        when = ensure_utc(ts) if ts is not None else ensure_utc(self._clock.now())
        request = ResumeRequest(
            activation_id=activation_id,
            operator=operator,
            approved=approved,
            reconciliation_ok=reconciliation_ok,
            reconciliation_id=reconciliation_id,
            reason_resolved=reason_resolved,
            cause=cause,
            correction=correction,
            ts=when,
        )
        return self._ks.resume(request)
