"""Teste de integrare cap-la-cap pentru recuperarea din căderi/defecte (Req 10.2, 11.1, 11.6, 26.4).

Spre deosebire de testele unitare, aici componentele reale sunt compuse împreună și `FakeBroker`
(`broker/fake.py`) injectează defectele care conduc fluxul: cădere între `commit` și `submit`,
deconectare/reconectare, execuție în timpul anulării, snapshot incomplet și Recovery_Point
invalid. Stiva este cea de producție:

    Journal (append-only, lanț hash)  ──  JournalOmsSink  ──  OrderManager
          │                                                        │
          ├── KillSwitch (JournalKillSwitchStore)                  │
          ├── Reconciler (recon/reconciler) ──────────────────────┘
          ├── Portfolio (proiecție event-sourced)
          └── FakeBroker învelit în FailSafeBlock (broker/fail_safe) ← ApprovedTarget demo

Căderea este simulată reutilizând `persistence/recovery.recover` + `validate_integrity`: baza de
date rămâne pe disc (WAL + synchronous=FULL), iar „repornirea” înseamnă redeschiderea aceleiași
baze și reconstruirea proiecțiilor și a kill switch-ului DIN jurnal. Evenimentele se conduc
direct (apeluri pe manager/broker), nu printr-un flux complet de date, ca să controlăm exact
punctele de cădere.

Invariante verificate cap-la-cap:
- jurnalul se verifică (`verify_journal`);
- idempotența execuțiilor (un `broker_exec_id` dublat → `RETRANSMISSION`, fără dublarea fill-ului
  în portofoliu);
- invariantul `filled + remaining = qty`;
- instrumentele înghețate sunt corecte și se deblochează abia după reconciliere;
- kill switch-ul activ supraviețuiește repornirii (reconstruit din jurnal).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from qts.broker.adapter import OrderRequest, order_request_from
from qts.broker.fail_safe import ApprovedTarget, FailSafeBlock
from qts.broker.fake import (
    BrokerDisconnectedError,
    BrokerTimeoutError,
    CancelFault,
    FakeBroker,
    FaultPlan,
    SnapshotFault,
    SubmitFault,
)
from qts.core.clock import SimClock
from qts.core.models import ExecutionEvent, Instrument, OrderIntent, OrderState
from qts.oms.manager import (
    ExecOutcome,
    ExecStatus,
    JournalOmsSink,
    OrderManager,
)
from qts.persistence.audit import verify_journal
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.persistence.recovery import (
    global_kill_on_invalid_recovery,
    recover,
    validate_integrity,
)
from qts.portfolio.portfolio import PortfolioState, apply_fill
from qts.recon.reconciler import (
    DifferenceCode,
    Reconciler,
    ReconciliationReason,
    ReconciliationTolerances,
)
from qts.risk.context import KillSwitchScope
from qts.risk.engine import RiskDecision
from qts.safety.kill_switch import JournalKillSwitchStore, KillSwitch
from qts.safety.stage import ProjectStage, StageInfo

T0 = datetime(2026, 4, 1, 9, tzinfo=UTC)
D = Decimal
S = OrderState
ACCT = "FAKE-DEMO-1"
STAGE = StageInfo(ProjectStage.INITIAL, "stage.lock")
INITIAL_CASH = Decimal(10000)


# --------------------------------------------------------------------------- helperi


def _inst(symbol: str) -> Instrument:
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


INSTRUMENTS = (_inst("XYZ"), _inst("ABC"))


def _tolerances() -> ReconciliationTolerances:
    return ReconciliationTolerances(version="recon-v1", qty_default=D(0), cash_eur=D(0))


class Stack:
    """Stiva reală compusă în jurul unui singur `Journal` pe disc, cu `FakeBroker`."""

    def __init__(
        self,
        db_path: Path,
        clock: SimClock,
        *,
        faults: FaultPlan | None = None,
        initial_cash: Decimal = INITIAL_CASH,
        registry: Any = None,
        broker: FakeBroker | None = None,
    ) -> None:
        self.conn: sqlite3.Connection = open_db(db_path)
        self.journal = Journal(self.conn)
        self.clock = clock
        self.sink = JournalOmsSink(self.journal)
        self.manager = OrderManager(self.sink, registry=registry)
        self.kill_switch = KillSwitch(JournalKillSwitchStore(self.journal), clock=clock)
        self.reconciler = Reconciler(self.kill_switch, self.journal, clock, _tolerances())
        # Brokerul este un sistem EXTERN: supraviețuiește căderii procesului. La „repornire”
        # îl refolosim pentru a reflecta starea pe care brokerul a păstrat-o (ordine acceptate).
        self.fake = (
            broker
            if broker is not None
            else FakeBroker(
                instruments=INSTRUMENTS,
                clock=clock,
                account_id=ACCT,
                environment="demo",
                endpoint="fake://local",
                initial_cash=initial_cash,
                faults=faults,
            )
        )
        # Bariera reală de siguranță în fața brokerului fals: mediu/cont aprobat demo.
        self.broker: FailSafeBlock[OrderRequest, Any] = FailSafeBlock(
            self.fake,
            approved=ApprovedTarget(environment="demo", account_id=ACCT),
            stage=STAGE,
            kill_switch=self.kill_switch,
            audit=self.journal,
            clock=clock,
        )
        # Portofoliul rulării curente (reconstruit la repornire din jurnal).
        self.portfolio = PortfolioState.initial(initial_cash)

    def close(self) -> None:
        self.conn.close()

    # ---- fluxul normal: creează un ordin aprobat și fă write-ahead la SUBMITTED ----

    def create(
        self, instrument: str = "XYZ", n: int = 1, qty: str = "10"
    ) -> tuple[str, OrderRequest]:
        intent = OrderIntent(
            intent_id=f"i-{instrument}-{n}",
            signal_id=f"s-{n}",
            instrument=instrument,
            side="BUY",
            ref_price=D(100),
        )
        decision = RiskDecision(intent_id=intent.intent_id, approved=True, qty=D(qty))
        res = self.manager.create_order(
            intent, decision, run_id="run1", strategy_id="mr", signal_seq=n, ts=self.clock.now()
        )
        coid = res.order.client_order_id
        return coid, order_request_from(self.manager.order(coid), intent)

    def pump(self) -> list[ExecOutcome]:
        """Golește evenimentele brokerului în OMS și aplică fill-urile în portofoliu."""
        out: list[ExecOutcome] = []
        for event in self.broker.events():
            for outcome in self.manager.on_execution(event):
                out.append(outcome)
                if outcome.fill is not None:
                    self.portfolio = apply_fill(self.portfolio, outcome.fill)
        return out


def _snapshot_for_reconcile(stack: Stack) -> Any:
    """Snapshot-ul brokerului prin bariera (delegare `__getattr__`)."""
    return stack.broker.snapshot()


def _reflected_for(snapshot: Any, coid: str) -> list[str]:
    return [x for x in snapshot.execution_ids if x.startswith(f"{ACCT}:{coid}:")]


# --------------------------------------------------------------------------- fixturi


@pytest.fixture
def clock() -> SimClock:
    return SimClock(T0)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "crash.db"


# =========================================================================== Scenariul 1
# Cădere ÎNTRE commit-ul jurnalului (SUBMITTED write-ahead) și `broker.submit` (Req 26.4, 10.2).


def test_crash_between_journal_commit_and_broker_submit(clock: SimClock, db_path: Path) -> None:
    # Brokerul ACCEPTĂ ordinul, dar apelantul vede un timeout: rezultatul este necunoscut.
    coid, surviving_broker = _run_until_unknown(clock, db_path)

    # „Repornire”: redeschide aceeași bază; brokerul extern supraviețuiește cu starea sa.
    reopened = Stack(db_path, SimClock(T0), broker=surviving_broker)
    try:
        verdict = validate_integrity(reopened.journal)
        assert verdict.ok  # jurnalul este intact

        rec = recover(reopened.journal, initial_cash=D(10000))
        # Ordinul rămâne în așteptare (UNKNOWN), instrumentul blocat, resume_seq setat (26.1, 26.3).
        assert coid in rec.pending_orders
        assert rec.order_states[coid] is S.UNKNOWN
        assert rec.frozen_instruments == frozenset({"XYZ"})
        assert rec.resume_seq is not None
        assert not rec.recovery_complete

        # Reconstruiește OMS cu registrul recuperat și readu ordinul în stare UNKNOWN înghețată.
        reopened.manager = OrderManager(reopened.sink, registry=rec.registry)
        _rehydrate_unknown(reopened, coid)

        # Reconciliere din snapshot: client_order_id determinist permite potrivirea; ordinul se
        # decongelează fără dublarea fill-ului (idempotență prin registrul reconstruit) (11.1).
        snap = _snapshot_for_reconcile(reopened)
        assert snap.complete
        result = reopened.reconciler.reconcile(
            snap,
            rec.portfolio,
            reopened.manager,
            reason=ReconciliationReason.STARTUP,
        )
        assert result.ok
        assert result.resolved_orders == (coid,)
        # OMS este autoritatea; ordinul a fost acceptat la broker (ACK), fără execuții.
        assert reopened.manager.order(coid).state is S.ACKNOWLEDGED
        assert not reopened.manager.is_frozen(coid)
        # Instrument deblocat după reconciliere; ordinele noi sunt din nou permise.
        assert reopened.kill_switch.allows("XYZ")
        assert not reopened.manager.is_instrument_blocked("XYZ")

        # Jurnalul se verifică și nu există dublare de fill (niciun fill, de fapt).
        assert verify_journal(reopened.journal).ok
        assert reopened.manager.order(coid).filled_qty == D(0)
    finally:
        reopened.close()


def _run_until_unknown(clock: SimClock, db_path: Path) -> tuple[str, FakeBroker]:
    stack = Stack(db_path, clock, faults=FaultPlan())
    try:
        coid, req = stack.create("XYZ", n=1, qty="10")
        # Write-ahead: jurnalizează APPROVED → SUBMITTED înaintea trimiterii.
        assert stack.manager.submit(coid, clock.now()).send
        assert stack.manager.order(coid).state is S.SUBMITTED
        # Brokerul acceptă (ACK), dar apelantul primește timeout → rezultat necunoscut.
        stack.fake.faults.on_submit(coid, SubmitFault.TIMEOUT_AFTER_ACCEPT)
        with pytest.raises(BrokerTimeoutError):
            stack.broker.submit(req)
        # OMS: SUBMITTED → UNKNOWN + înghețare (cădere între jurnal și broker).
        assert stack.manager.submit_timeout(coid, clock.now()).order.state is S.UNKNOWN
        assert stack.manager.is_frozen(coid)
        return coid, stack.fake
    finally:
        stack.close()


def _rehydrate_unknown(stack: Stack, coid: str) -> None:
    """Reface ordinul UNKNOWN înghețat în noul manager (OMS este autoritatea stării)."""
    stack.create("XYZ", n=1, qty="10")  # idempotent: aceeași cheie, niciun ordin nou
    assert stack.manager.submit(coid, stack.clock.now()).send
    stack.manager.submit_timeout(coid, stack.clock.now())
    assert stack.manager.order(coid).state is S.UNKNOWN
    assert stack.manager.is_frozen(coid)


# =========================================================================== Scenariul 2
# Reconectare (Req 11.1): ordinele noi sunt blocate cât timp conexiunea e întreruptă/nereconciliată,
# și permise după o reconciliere reușită pe baza snapshot-ului.


def test_reconnect_blocks_new_orders_until_reconciled(clock: SimClock, db_path: Path) -> None:
    stack = Stack(db_path, clock)
    try:
        coid, req = stack.create("XYZ", n=1, qty="10")
        assert stack.manager.submit(coid, clock.now()).send
        assert stack.broker.submit(req).accepted
        stack.pump()  # ACK aplicat
        assert stack.manager.order(coid).state is S.ACKNOWLEDGED

        # Deconectare: apelurile ridică erori; brokerul execută în fundal.
        stack.fake.disconnect()
        for call in (
            stack.broker.events,
            stack.broker.snapshot,
            lambda: stack.broker.submit(req),
        ):
            with pytest.raises(BrokerDisconnectedError):
                call()
        stack.fake.fill(coid, D(10), D(100), D(1))  # fill produs în timpul deconectării

        # Reconectare: evenimentele tampon sunt livrate; reconciliază ÎNAINTE de ordine noi.
        stack.fake.reconnect()
        out = stack.pump()
        assert any(o.status is ExecStatus.APPLIED for o in out)
        assert stack.manager.order(coid).state is S.FILLED

        snap = _snapshot_for_reconcile(stack)
        # Portofoliul reflectă fill-ul recepționat; reconcilierea trebuie să fie coerentă.
        result = stack.reconciler.reconcile(
            snap, stack.portfolio, stack.manager, reason=ReconciliationReason.RECONNECT
        )
        assert result.ok
        assert stack.kill_switch.allows("XYZ")

        # Invariant: filled + remaining = qty; fără dublarea fill-ului în portofoliu.
        order = stack.manager.order(coid)
        assert order.filled_qty + order.remaining_qty == order.qty
        assert order.filled_qty == D(10)
        assert stack.portfolio.positions["XYZ"].qty == D(10)
        assert verify_journal(stack.journal).ok
    finally:
        stack.close()


def test_new_order_blocked_while_global_kill_active(clock: SimClock, db_path: Path) -> None:
    """Un snapshot incomplet la reconectare activează GLOBAL.

    `FailSafeBlock` oprește ordinele noi.
    """
    stack = Stack(db_path, clock, faults=FaultPlan().fail_snapshot(SnapshotFault.INCOMPLETE))
    try:
        coid, req = stack.create("XYZ", n=1, qty="10")
        assert stack.manager.submit(coid, clock.now()).send
        assert stack.broker.submit(req).accepted
        stack.pump()

        snap = _snapshot_for_reconcile(stack)
        assert not snap.complete
        result = stack.reconciler.reconcile(
            snap, stack.portfolio, stack.manager, reason=ReconciliationReason.RECONNECT
        )
        assert result.activated_global
        assert not result.complete

        # Ordin nou pe alt instrument: bariera de siguranță îl respinge (kill switch GLOBAL).
        from qts.broker.fail_safe import FailSafeRejectedError

        coid2, req2 = stack.create("ABC", n=2, qty="5")
        assert stack.manager.submit(coid2, clock.now()).send
        with pytest.raises(FailSafeRejectedError):
            stack.broker.submit(req2)
    finally:
        stack.close()


# =========================================================================== Scenariul 3
# Execuție în timpul anulării (Req 10.2): anulare amânată, sosește un fill parțial, apoi CANCELLED.


def test_execution_during_cancel(clock: SimClock, db_path: Path) -> None:
    stack = Stack(db_path, clock, faults=FaultPlan().on_cancel("placeholder", CancelFault.DEFER))
    try:
        coid, req = stack.create("XYZ", n=1, qty="10")
        # Programează anularea amânată pe coid-ul real (determinist).
        stack.fake.faults.cancel.clear()
        stack.fake.faults.on_cancel(coid, CancelFault.DEFER)

        assert stack.manager.submit(coid, clock.now()).send
        assert stack.broker.submit(req).accepted
        stack.pump()

        # Cerere de anulare: OMS → CANCEL_PENDING, brokerul o acceptă dar o amână.
        assert stack.manager.request_cancel(coid, clock.now()).send
        assert stack.broker.cancel(coid).accepted
        # Un fill parțial sosește înaintea confirmării anulării.
        stack.fake.fill(coid, D(4), D(100), D(1))
        stack.fake.confirm_cancel(coid)
        out = stack.pump()
        assert all(o.status is ExecStatus.APPLIED for o in out)

        order = stack.manager.order(coid)
        assert order.state is S.CANCELLED
        assert order.filled_qty == D(4)
        assert order.filled_qty + order.remaining_qty == order.qty
        # Portofoliul reflectă DOAR fill-ul real, o singură dată.
        assert stack.portfolio.positions["XYZ"].qty == D(4)
        # Un singur fill de 4 unități a fost aplicat (fără dublare).
        fills = [o for o in out if o.fill is not None]
        assert len(fills) == 1 and fills[0].fill is not None and fills[0].fill.qty == D(4)
        assert verify_journal(stack.journal).ok
    finally:
        stack.close()


# =========================================================================== Scenariul 4
# Snapshot incomplet (Req 11.6): Reconciler activează Kill_Switch(GLOBAL), audit în jurnal;
# un snapshot complet ulterior + calea de reluare îl eliberează.


def test_incomplete_snapshot_activates_global_and_resume_clears(
    clock: SimClock, db_path: Path
) -> None:
    stack = Stack(db_path, clock, faults=FaultPlan().fail_snapshot(SnapshotFault.INCOMPLETE))
    try:
        coid, req = stack.create("XYZ", n=1, qty="10")
        assert stack.manager.submit(coid, clock.now()).send
        assert stack.broker.submit(req).accepted
        stack.pump()

        snap = _snapshot_for_reconcile(stack)
        assert not snap.complete
        result = stack.reconciler.reconcile(
            snap, stack.portfolio, stack.manager, reason=ReconciliationReason.PERIODIC
        )
        assert result.activated_global
        assert result.differences[0].code is DifferenceCode.SNAPSHOT_INCOMPLETE
        assert stack.kill_switch.state().global_active is True
        assert not stack.kill_switch.allows("XYZ")

        # Audit în jurnal pentru divergență.
        recon_records = [r for r in stack.journal.read() if r.type.startswith("recon.")]
        assert any(r.outcome == DifferenceCode.SNAPSHOT_INCOMPLETE.value for r in recon_records)

        activation = stack.kill_switch.active_activations()[0]

        # Un snapshot complet ulterior reconciliază curat (defectul s-a consumat).
        snap_ok = _snapshot_for_reconcile(stack)
        assert snap_ok.complete
        ok_result = stack.reconciler.reconcile(
            snap_ok, stack.portfolio, stack.manager, reason=ReconciliationReason.PERIODIC
        )
        assert ok_result.ok

        # Reluarea cere reconciliere reușită + aprobare + cauză/corecție (KillSwitch.resume, 11.7).
        decision = stack.reconciler.request_resume(
            activation.activation_id,
            operator="alice",
            approved=True,
            reconciliation_ok=True,
            reconciliation_id="recon-ok",
            reason_resolved=True,
            cause="snapshot complet obținut",
            correction="reconectare la broker",
            ts=clock.now(),
        )
        assert decision.accepted
        assert stack.kill_switch.state().global_active is False
        assert stack.kill_switch.allows("XYZ")
        assert verify_journal(stack.journal).ok
    finally:
        stack.close()


def test_snapshot_unavailable_fails_closed_global(clock: SimClock, db_path: Path) -> None:
    """Un snapshot care ridică excepție (RAISE) tot activează GLOBAL (fail-closed, 11.6)."""
    stack = Stack(db_path, clock, faults=FaultPlan().fail_snapshot(SnapshotFault.RAISE))
    try:
        coid, req = stack.create("XYZ", n=1, qty="10")
        assert stack.manager.submit(coid, clock.now()).send
        assert stack.broker.submit(req).accepted
        stack.pump()

        # Obținerea snapshot-ului ridică excepție; Reconciler eșuează închis pe GLOBAL.
        from qts.broker.fake import SnapshotUnavailableError

        try:
            _snapshot_for_reconcile(stack)
        except SnapshotUnavailableError:
            # Reconciler primește de obicei snapshot-ul; aici simulăm indisponibilitatea printr-o
            # reconciliere care eșuează închis (orice excepție devine divergență GLOBAL).
            class _RaisingSnapshot:
                complete = True

                def __getattr__(self, _name: str) -> Any:
                    raise SnapshotUnavailableError("indisponibil")

            result = stack.reconciler.reconcile(
                _RaisingSnapshot(),  # type: ignore[arg-type]
                stack.portfolio,
                stack.manager,
                reason=ReconciliationReason.PERIODIC,
            )
            assert result.activated_global
            assert stack.kill_switch.state().global_active is True
            assert not stack.kill_switch.allows("XYZ")
            return
        pytest.fail("snapshot-ul ar fi trebuit să ridice SnapshotUnavailableError")
    finally:
        stack.close()


# =========================================================================== Scenariul 5
# Recovery_Point invalid (Req 26.4): manipulează o înregistrare din jurnal, validarea eșuează,
# `global_kill_on_invalid_recovery` activează GLOBAL pe calea rapidă; ordinele sunt blocate.


def test_invalid_recovery_point_activates_global_fast_path(clock: SimClock, db_path: Path) -> None:
    # Rulare normală cu un fill, apoi „repornire” peste un jurnal manipulat.
    stack = Stack(db_path, clock)
    try:
        coid, req = stack.create("XYZ", n=1, qty="10")
        assert stack.manager.submit(coid, clock.now()).send
        assert stack.broker.submit(req).accepted
        stack.fake.fill(coid, D(10), D(100), D(1))
        stack.pump()
        assert stack.manager.order(coid).state is S.FILLED
        assert verify_journal(stack.journal).ok
    finally:
        stack.close()

    # Manipulează o înregistrare din mijloc: lanțul hash se rupe.
    conn = open_db(db_path)
    try:
        conn.execute("DROP TRIGGER journal_no_update")
        conn.execute("UPDATE journal SET outcome='tampered' WHERE seq=3")
    finally:
        conn.close()

    reopened = Stack(db_path, SimClock(T0))
    try:
        verdict = validate_integrity(reopened.journal)
        assert not verdict.ok
        assert verdict.first_bad_seq == 3

        activated = global_kill_on_invalid_recovery(reopened.kill_switch, verdict)
        assert activated
        # Blocarea este sincronă (calea rapidă) și se aplică oricărui instrument.
        assert not reopened.kill_switch.allows("XYZ")
        assert reopened.kill_switch.blocking_scope("ANY") is KillSwitchScope.GLOBAL

        # Un ordin nou este oprit de bariera reală de siguranță.
        from qts.broker.fail_safe import FailSafeRejectedError

        coid2, req2 = reopened.create("ABC", n=2, qty="5")
        assert reopened.manager.submit(coid2, reopened.clock.now()).send
        with pytest.raises(FailSafeRejectedError):
            reopened.broker.submit(req2)
    finally:
        reopened.close()


# =========================================================================== Invariante comune


def test_kill_switch_survives_simulated_restart(clock: SimClock, db_path: Path) -> None:
    """Un Kill_Switch activat înaintea căderii rămâne activ după repornire (din jurnal)."""
    stack = Stack(db_path, clock)
    try:
        stack.kill_switch.activate_automatic(
            KillSwitchScope.GLOBAL, component="recon", reason_code="BOOM"
        )
        assert not stack.kill_switch.allows("XYZ")
    finally:
        stack.close()

    reopened = Stack(db_path, SimClock(T0))
    try:
        # Reconstruit din jurnal: kill switch-ul GLOBAL este încă activ.
        assert not reopened.kill_switch.allows("XYZ")
        assert reopened.kill_switch.blocking_scope("XYZ") is KillSwitchScope.GLOBAL
        assert verify_journal(reopened.journal).ok
    finally:
        reopened.close()


def test_duplicate_broker_exec_id_is_retransmission_without_double_fill(
    clock: SimClock, db_path: Path
) -> None:
    """Idempotență cap-la-cap: un mesaj dublat → RETRANSMISSION, fără dublare în portofoliu."""
    stack = Stack(db_path, clock)
    try:
        coid, req = stack.create("XYZ", n=1, qty="10")
        assert stack.manager.submit(coid, clock.now()).send
        # FakeBroker dublează mesajul FILL (seq 2).
        stack.fake.faults.duplicate(coid, 2)
        assert stack.broker.submit(req).accepted
        stack.fake.fill(coid, D(10), D(100), D(1))
        out = stack.pump()
        statuses = [o.status for o in out]
        assert statuses == [ExecStatus.APPLIED, ExecStatus.APPLIED, ExecStatus.RETRANSMISSION]

        order = stack.manager.order(coid)
        assert order.state is S.FILLED and order.filled_qty == D(10)
        assert order.filled_qty + order.remaining_qty == order.qty
        # Portofoliul a aplicat fill-ul o singură dată.
        assert stack.portfolio.positions["XYZ"].qty == D(10)
        fills = [o for o in out if o.fill is not None]
        assert len(fills) == 1
        assert verify_journal(stack.journal).ok
    finally:
        stack.close()


def test_recovered_registry_dedups_fill_across_restart(clock: SimClock, db_path: Path) -> None:
    """La repornire, registrul reconstruit deduplică retransmisia unei execuții aplicate (26.8)."""
    stack = Stack(db_path, clock)
    try:
        coid, req = stack.create("XYZ", n=1, qty="10")
        assert stack.manager.submit(coid, clock.now()).send
        assert stack.broker.submit(req).accepted
        stack.fake.fill(coid, D(10), D(100), D(1))
        stack.pump()
        assert stack.manager.order(coid).state is S.FILLED
    finally:
        stack.close()

    reopened = Stack(db_path, SimClock(T0))
    try:
        rec = recover(reopened.journal, initial_cash=D(10000))
        assert rec.order_states[coid] is S.FILLED
        assert rec.portfolio.positions["XYZ"].qty == D(10)
        # Numerar rejucat: 10 @ 100 + comision 1 = -1001 față de 10000.
        assert rec.portfolio.cash_eur == D(10000) - D(1001)

        # OMS nou cu registrul reconstruit; o retransmisie a aceleiași execuții e deduplicată.
        fresh = OrderManager(reopened.sink, registry=rec.registry)
        event = ExecutionEvent(
            broker_exec_id=f"{ACCT}:{coid}:2",
            client_order_id=coid,
            kind=stack_fill_kind(),
            qty=D(10),
            price=D(100),
            commission=D(1),
            ts_broker=T0,
            ts_receipt=T0,
        )
        outcomes = fresh.on_execution(event)
        assert [o.status for o in outcomes] == [ExecStatus.RETRANSMISSION]
        assert verify_journal(reopened.journal).ok
    finally:
        reopened.close()


def stack_fill_kind() -> Any:
    from qts.core.models import ExecKind

    return ExecKind.FILL
