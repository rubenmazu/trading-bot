"""Teste unitare pentru `recon/reconciler.py` (Req 11.1–11.7)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from qts.broker.adapter import BrokerOrderStatus, BrokerSnapshot
from qts.core.clock import SimClock
from qts.core.models import OrderIntent, OrderState
from qts.oms.manager import ListSink, OrderManager
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.portfolio.pnl import Mark
from qts.portfolio.portfolio import PortfolioState, Position
from qts.recon.reconciler import (
    DifferenceCode,
    Reconciler,
    ReconciliationReason,
    ReconciliationTolerances,
)
from qts.risk.context import KillSwitchScope
from qts.risk.engine import RiskDecision
from qts.safety.kill_switch import JournalKillSwitchStore, KillSwitch

T0 = datetime(2026, 3, 2, 10, tzinfo=UTC)
D = Decimal
S = OrderState


# --------------------------------------------------------------------------- fixturi


@pytest.fixture
def clock() -> SimClock:
    return SimClock(T0)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return open_db(tmp_path / "recon.db")


@pytest.fixture
def journal(conn: sqlite3.Connection) -> Journal:
    return Journal(conn)


@pytest.fixture
def ks(journal: Journal, clock: SimClock) -> KillSwitch:
    # Un singur `Journal` pentru kill switch și reconciliere: lanțul append-only este partajat.
    return KillSwitch(JournalKillSwitchStore(journal), clock=clock)


def _reconciler(
    ks: KillSwitch,
    journal: Journal,
    clock: SimClock,
    *,
    qty_default: str = "0",
    cash_eur: str = "0",
    qty_by_instrument: dict[str, Decimal] | None = None,
    reporting_currency: str = "EUR",
) -> Reconciler:
    tol = ReconciliationTolerances(
        version="recon-v1",
        qty_default=D(qty_default),
        cash_eur=D(cash_eur),
        qty_by_instrument=qty_by_instrument or {},
        reporting_currency=reporting_currency,
    )
    return Reconciler(ks, journal, clock, tol)


def _state(cash: str = "1000", positions: dict[str, str] | None = None) -> PortfolioState:
    base = PortfolioState.initial(D(cash))
    if not positions:
        return base
    pos: dict[str, Position] = {}
    marks = {}
    for inst, qty in positions.items():
        pos[inst] = Position(
            instrument=inst,
            currency="EUR",
            qty=D(qty),
            cost_basis_ccy=D(qty) * D(100),
            cost_basis_eur=D(qty) * D(100),
        )
        marks[inst] = Mark(price=D(100), fx_rate_to_eur=D(1), ts=T0)
    return base.model_copy(update={"positions": pos, "marks": marks})


def _snapshot(
    *,
    cash: str = "1000",
    currency: str = "EUR",
    positions: dict[str, str] | None = None,
    orders: tuple[BrokerOrderStatus, ...] = (),
    complete: bool = True,
) -> BrokerSnapshot:
    return BrokerSnapshot(
        ts=T0,
        environment="demo",
        account_id="ACC-1",
        orders=orders,
        execution_ids=(),
        positions={k: D(v) for k, v in (positions or {}).items()},
        cash=D(cash),
        currency=currency,
        complete=complete,
    )


def _unknown_order(m: OrderManager, instrument: str = "XYZ", qty: str = "10") -> str:
    """Creează un ordin intern în stare UNKNOWN (înghețat), reconciliabil din snapshot."""
    intent = OrderIntent(
        intent_id=f"i-{instrument}",
        signal_id="s-1",
        instrument=instrument,
        side="BUY",
        ref_price=D(100),
    )
    decision = RiskDecision(intent_id=intent.intent_id, approved=True, qty=D(qty))
    res = m.create_order(intent, decision, run_id="run1", strategy_id="mr", signal_seq=1, ts=T0)
    coid = res.order.client_order_id
    assert m.submit(coid, T0).send
    m.submit_timeout(coid, T0)
    assert m.order(coid).state is S.UNKNOWN
    return coid


def _recon_records(journal: Journal) -> list[str]:
    return [r.outcome for r in journal.read() if r.type.startswith("recon.")]


# --------------------------------------------------------------------------- reconciliere OK


def test_matching_snapshot_no_activation(ks: KillSwitch, journal: Journal, clock: SimClock) -> None:
    r = _reconciler(ks, journal, clock)
    result = r.reconcile(
        _snapshot(cash="1000", positions={"XYZ": "5"}),
        _state(cash="1000", positions={"XYZ": "5"}),
        OrderManager(ListSink()),
        reason=ReconciliationReason.STARTUP,
    )
    assert result.ok
    assert not result.differences
    assert not result.activated_global
    assert result.blocked_instruments == ()
    assert ks.state().global_active is False
    assert ks.state().instruments == frozenset()
    assert _recon_records(journal) == []


def test_position_within_tolerance_ok(ks: KillSwitch, journal: Journal, clock: SimClock) -> None:
    r = _reconciler(ks, journal, clock, qty_default="0.01")
    result = r.reconcile(
        _snapshot(positions={"XYZ": "5.005"}),
        _state(positions={"XYZ": "5"}),
        OrderManager(ListSink()),
    )
    assert result.ok
    assert ks.allows("XYZ")


# --------------------------------------------------------------------------- divergențe


def test_position_diff_over_tolerance_blocks_instrument_and_audits(
    ks: KillSwitch, journal: Journal, clock: SimClock
) -> None:
    r = _reconciler(ks, journal, clock, qty_default="0.01")
    result = r.reconcile(
        _snapshot(positions={"XYZ": "7"}),
        _state(positions={"XYZ": "5"}),
        OrderManager(ListSink()),
        reason=ReconciliationReason.PERIODIC,
    )
    assert not result.ok
    assert result.blocked_instruments == ("XYZ",)
    assert not result.activated_global
    diff = result.differences[0]
    assert diff.code is DifferenceCode.POSITION_MISMATCH
    assert diff.scope is KillSwitchScope.INSTRUMENT
    # Blocare sincronă pe instrument (11.5), GLOBAL neatins.
    assert ks.blocking_scope("XYZ") is KillSwitchScope.INSTRUMENT
    assert ks.allows("OTHER")
    assert _recon_records(journal) == [DifferenceCode.POSITION_MISMATCH.value]


def test_cash_diff_activates_global(ks: KillSwitch, journal: Journal, clock: SimClock) -> None:
    r = _reconciler(ks, journal, clock, cash_eur="0.5")
    result = r.reconcile(
        _snapshot(cash="900"),
        _state(cash="1000"),
        OrderManager(ListSink()),
    )
    assert not result.ok
    assert result.activated_global
    assert result.differences[0].code is DifferenceCode.CASH_MISMATCH
    assert result.differences[0].scope is KillSwitchScope.GLOBAL
    assert ks.state().global_active is True
    assert _recon_records(journal) == [DifferenceCode.CASH_MISMATCH.value]


def test_cash_within_tolerance_ok(ks: KillSwitch, journal: Journal, clock: SimClock) -> None:
    r = _reconciler(ks, journal, clock, cash_eur="1")
    result = r.reconcile(_snapshot(cash="1000.5"), _state(cash="1000"), OrderManager(ListSink()))
    assert result.ok
    assert ks.state().global_active is False


def test_incomplete_snapshot_activates_global(
    ks: KillSwitch, journal: Journal, clock: SimClock
) -> None:
    r = _reconciler(ks, journal, clock)
    result = r.reconcile(
        _snapshot(complete=False),
        _state(),
        OrderManager(ListSink()),
        reason=ReconciliationReason.RECONNECT,
    )
    assert not result.ok
    assert result.complete is False
    assert result.activated_global
    assert result.differences[0].code is DifferenceCode.SNAPSHOT_INCOMPLETE
    assert ks.state().global_active is True
    assert _recon_records(journal) == [DifferenceCode.SNAPSHOT_INCOMPLETE.value]


def test_non_eur_cash_currency_activates_global(
    ks: KillSwitch, journal: Journal, clock: SimClock
) -> None:
    r = _reconciler(ks, journal, clock, cash_eur="100")
    result = r.reconcile(
        _snapshot(cash="1000", currency="USD"),
        _state(cash="1000"),
        OrderManager(ListSink()),
    )
    assert not result.ok
    assert result.activated_global
    assert result.differences[0].code is DifferenceCode.CASH_MISMATCH


def test_usd_snapshot_reconciles_in_usd_configured_run(
    ks: KillSwitch, journal: Journal, clock: SimClock
) -> None:
    """Un snapshot USD reconciliază curat într-o rulare configurată în USD (Demo coerent).

    Moneda contului (USD) este exact moneda de raportare configurată, deci comparația de numerar
    continuă strict (USD==USD), fără nicio activare de Kill_Switch pe calea fericită.
    """
    r = _reconciler(ks, journal, clock, cash_eur="0", reporting_currency="USD")
    result = r.reconcile(
        _snapshot(cash="1000", currency="USD", positions={"SPY": "5"}),
        _state(cash="1000", positions={"SPY": "5"}),
        OrderManager(ListSink()),
        reason=ReconciliationReason.STARTUP,
    )
    assert result.ok
    assert not result.activated_global
    assert ks.state().global_active is False
    assert _recon_records(journal) == []


def test_eur_run_still_rejects_usd_snapshot_regression(
    ks: KillSwitch, journal: Journal, clock: SimClock
) -> None:
    """Regresie: o rulare EUR (implicit) respinge în continuare un snapshot USD → GLOBAL.

    Dovada că flexibilizarea nu a slăbit reconcilierea: chiar dacă numerarul numeric coincide,
    moneda contului (USD) diferă de moneda de raportare a rulării (EUR), deci numerarul este o
    divergență GLOBAL (fail-closed), nu o reconciliere reușită.
    """
    r = _reconciler(ks, journal, clock, cash_eur="100", reporting_currency="EUR")
    result = r.reconcile(
        _snapshot(cash="1000", currency="USD"),
        _state(cash="1000"),
        OrderManager(ListSink()),
    )
    assert not result.ok
    assert result.activated_global
    assert result.differences[0].code is DifferenceCode.CASH_MISMATCH
    assert result.differences[0].scope is KillSwitchScope.GLOBAL
    assert ks.state().global_active is True


# --------------------------------------------------------------------------- ordine


def test_unknown_order_resolved_from_snapshot(
    ks: KillSwitch, journal: Journal, clock: SimClock
) -> None:
    m = OrderManager(ListSink())
    coid = _unknown_order(m, "XYZ", qty="10")
    status = BrokerOrderStatus(
        client_order_id=coid,
        instrument="XYZ",
        side="BUY",
        qty=D(10),
        state=S.FILLED,
        filled_qty=D(10),
        avg_fill_price=D(100),
        broker_order_id="FAKE-1",
    )
    r = _reconciler(ks, journal, clock)
    # Portofoliul reflectă execuția completă (10 @ 100 = 1000 numerar cheltuit).
    result = r.reconcile(
        _snapshot(cash="0", positions={"XYZ": "10"}, orders=(status,)),
        _state(cash="0", positions={"XYZ": "10"}),
        m,
        reason=ReconciliationReason.EXECUTION_EVENT,
    )
    assert result.ok
    assert result.resolved_orders == (coid,)
    # OMS este autoritatea: ordinul a fost decongelat pe baza brokerului (11.1).
    assert m.order(coid).state is S.FILLED
    assert not m.is_frozen(coid)
    assert ks.allows("XYZ")


def test_broker_open_order_missing_internally_blocks_instrument(
    ks: KillSwitch, journal: Journal, clock: SimClock
) -> None:
    status = BrokerOrderStatus(
        client_order_id="ghost-1",
        instrument="XYZ",
        side="BUY",
        qty=D(3),
        state=S.ACKNOWLEDGED,
    )
    r = _reconciler(ks, journal, clock)
    result = r.reconcile(
        _snapshot(orders=(status,)),
        _state(),
        OrderManager(ListSink()),
    )
    assert not result.ok
    assert result.differences[0].code is DifferenceCode.MISSING_OPEN_ORDER
    assert result.blocked_instruments == ("XYZ",)
    assert ks.blocking_scope("XYZ") is KillSwitchScope.INSTRUMENT


# --------------------------------------------------------------------------- reluare (11.7)


def test_resume_requires_cause_correction_approval_and_reconciliation(
    ks: KillSwitch, journal: Journal, clock: SimClock
) -> None:
    r = _reconciler(ks, journal, clock, cash_eur="0")
    r.reconcile(_snapshot(cash="900"), _state(cash="1000"), OrderManager(ListSink()))
    activation = ks.active_activations()[0]
    aid = activation.activation_id

    # O reconciliere eșuată refuză reluarea, indiferent de aprobare.
    refused = r.request_resume(
        aid,
        operator="alice",
        approved=True,
        reconciliation_ok=False,
        reconciliation_id="recon-x",
        reason_resolved=True,
        cause="cauza",
        correction="corectia",
        ts=T0 + timedelta(minutes=1),
    )
    assert not refused.accepted
    assert ks.state().global_active is True

    # Cu cauză + corecție + aprobare + reconciliere reușită, reluarea trece.
    accepted = r.request_resume(
        aid,
        operator="alice",
        approved=True,
        reconciliation_ok=True,
        reconciliation_id="recon-ok",
        reason_resolved=True,
        cause="numerar aliniat",
        correction="ajustare manuală",
        ts=T0 + timedelta(minutes=2),
    )
    assert accepted.accepted
    assert ks.state().global_active is False


def test_resume_missing_cause_refused(ks: KillSwitch, journal: Journal, clock: SimClock) -> None:
    r = _reconciler(ks, journal, clock)
    r.reconcile(_snapshot(cash="900"), _state(cash="1000"), OrderManager(ListSink()))
    aid = ks.active_activations()[0].activation_id
    decision = r.request_resume(
        aid,
        operator="alice",
        approved=True,
        reconciliation_ok=True,
        reconciliation_id="recon-ok",
        reason_resolved=True,
        cause="",
        correction="",
        ts=T0 + timedelta(minutes=1),
    )
    assert not decision.accepted
    assert ks.state().global_active is True


# --------------------------------------------------------------------------- programare


def test_interval_policy_demo_live_le_60s(
    ks: KillSwitch, journal: Journal, clock: SimClock
) -> None:
    r = _reconciler(ks, journal, clock)
    for mode in ("demo", "live"):
        assert r.interval_seconds(mode) <= 60
        assert r.requires_periodic(mode)
    for mode in ("backtest", "sim", "shadow"):
        assert not r.requires_periodic(mode)


def test_should_run_respects_interval(ks: KillSwitch, journal: Journal, clock: SimClock) -> None:
    r = _reconciler(ks, journal, clock)
    assert r.should_run(T0, "demo", last_run=None)  # prima rulare
    assert not r.should_run(T0 + timedelta(seconds=30), "demo", last_run=T0)
    assert r.should_run(T0 + timedelta(seconds=60), "demo", last_run=T0)
    assert not r.should_run(T0 + timedelta(seconds=60), "backtest", last_run=T0)


# --------------------------------------------------------------------------- fail-closed


def test_exception_during_reconcile_fails_closed_global(
    ks: KillSwitch, journal: Journal, clock: SimClock
) -> None:
    class _Boom:
        @property
        def orders(self) -> dict[str, object]:
            raise RuntimeError("eroare internă")

    r = _reconciler(ks, journal, clock)
    result = r.reconcile(_snapshot(), _state(), _Boom())  # type: ignore[arg-type]
    assert not result.ok
    assert result.activated_global
    assert result.differences[0].code is DifferenceCode.INTERNAL_ERROR
    assert ks.state().global_active is True


# --------------------------------------------------------------------------- coduri stabile


def test_difference_codes_are_stable() -> None:
    assert DifferenceCode.POSITION_MISMATCH.value == "RECON_POSITION_MISMATCH"
    assert DifferenceCode.CASH_MISMATCH.value == "RECON_CASH_MISMATCH"
    assert DifferenceCode.MISSING_OPEN_ORDER.value == "RECON_MISSING_OPEN_ORDER"
    assert DifferenceCode.UNEXPECTED_OPEN_ORDER.value == "RECON_UNEXPECTED_OPEN_ORDER"
    assert DifferenceCode.SNAPSHOT_INCOMPLETE.value == "RECON_SNAPSHOT_INCOMPLETE"
    assert DifferenceCode.SNAPSHOT_UNAVAILABLE.value == "RECON_SNAPSHOT_UNAVAILABLE"
    assert DifferenceCode.INTERNAL_ERROR.value == "RECON_INTERNAL_ERROR"


def test_reconciliation_reason_codes_are_stable() -> None:
    assert ReconciliationReason.STARTUP.value == "STARTUP"
    assert ReconciliationReason.RECONNECT.value == "RECONNECT"
    assert ReconciliationReason.EXECUTION_EVENT.value == "EXECUTION_EVENT"
    assert ReconciliationReason.PERIODIC.value == "PERIODIC"
