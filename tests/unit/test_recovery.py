"""Teste unitare pentru `persistence/recovery.py` (Req 26.1–26.5, 26.8, 26.9).

Jurnalele sunt construite cu `OrderManager` + `JournalOmsSink` peste jurnalul append-only
real, deci înregistrările `oms.*` sunt cele produse în producție. Recuperarea reconstruiește
proiecțiile din aceste înregistrări (fără re-rularea motorului) și identifică punctul de reluare
și instrumentele blocate.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from decimal import Decimal

from qts.core.clock import SimClock
from qts.core.models import ExecKind, ExecutionEvent, OrderIntent, OrderState
from qts.oms.manager import ExecStatus, JournalOmsSink, OrderManager
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.persistence.recovery import (
    PENDING_STATES,
    global_kill_on_invalid_recovery,
    latest_checkpoint,
    projections_hash,
    recover,
    validate_integrity,
    write_checkpoint,
)
from qts.risk.context import KillSwitchScope
from qts.risk.engine import RiskDecision
from qts.safety.kill_switch import JournalKillSwitchStore, KillSwitch

T0 = datetime(2026, 1, 5, 9, tzinfo=UTC)
D = Decimal
S = OrderState


# --------------------------------------------------------------------------- fixturi


def _journal() -> tuple[sqlite3.Connection, Journal]:
    conn = open_db(":memory:")
    return conn, Journal(conn)


def _intent(instrument: str = "XYZ", n: int = 1) -> OrderIntent:
    return OrderIntent(
        intent_id=f"i-{instrument}-{n}",
        signal_id=f"s-{n}",
        instrument=instrument,
        side="BUY",
        ref_price=D(100),
    )


def _create(m: OrderManager, instrument: str = "XYZ", n: int = 1, qty: str = "10") -> str:
    intent = _intent(instrument, n)
    res = m.create_order(
        intent,
        RiskDecision(intent_id=intent.intent_id, approved=True, qty=D(qty)),
        run_id="run1",
        strategy_id="mr",
        signal_seq=n,
        ts=T0,
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
    )


def _filled_order(m: OrderManager, instrument: str = "XYZ", n: int = 1, qty: str = "10") -> str:
    """Ordin dus până la FILLED cu o execuție completă (produce un fill în portofoliu)."""
    coid = _create(m, instrument, n, qty)
    m.submit(coid, T0)
    m.on_execution(_ev(coid, f"ack-{coid}", ExecKind.ACK))
    m.on_execution(_ev(coid, f"fill-{coid}", ExecKind.FILL, qty=qty, price="100", commission="1"))
    assert m.order(coid).state is S.FILLED
    return coid


def _submitted_order(m: OrderManager, instrument: str = "ABC", n: int = 2, qty: str = "5") -> str:
    """Ordin lăsat în SUBMITTED (cerere trimisă, fără confirmare: cădere între jurnal și broker)."""
    coid = _create(m, instrument, n, qty)
    assert m.submit(coid, T0).send
    assert m.order(coid).state is S.SUBMITTED
    return coid


# ----------------------------------------------------------------- checkpoint + integritate


def test_checkpoint_roundtrip_and_validate_ok() -> None:
    conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    _filled_order(m)
    rec = recover(j)
    ph = projections_hash(
        registry=rec.registry, order_states=rec.order_states, portfolio=rec.portfolio
    )
    cp = write_checkpoint(j, proj_hash=ph, ts=T0)
    assert cp.journal_seq == j.head[0]
    assert latest_checkpoint(conn) == cp
    verdict = validate_integrity(j, cp)
    assert verdict.ok and verdict.reason is None


def test_validate_ok_without_checkpoint() -> None:
    _conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    _filled_order(m)
    verdict = validate_integrity(j)
    assert verdict.ok


def test_corrupted_record_fails_integrity_with_first_bad_seq() -> None:
    conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    _filled_order(m)
    # Alterează o înregistrare din mijloc: lanțul se rupe la acel seq.
    conn.execute("DROP TRIGGER journal_no_update")
    conn.execute("UPDATE journal SET outcome='tampered' WHERE seq=3")
    verdict = validate_integrity(Journal(conn))
    assert not verdict.ok
    assert verdict.first_bad_seq == 3


def test_truncated_tail_vs_checkpoint_detected() -> None:
    conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    _filled_order(m)
    rec = recover(j)
    ph = projections_hash(
        registry=rec.registry, order_states=rec.order_states, portfolio=rec.portfolio
    )
    cp = write_checkpoint(j, proj_hash=ph, ts=T0)
    # Retează coada: șterge ultima înregistrare, care este chiar seq-ul checkpoint-ului.
    head_seq = cp.journal_seq
    conn.execute("DROP TRIGGER journal_no_delete")
    conn.execute("DELETE FROM journal WHERE seq = ?", (head_seq,))
    verdict = validate_integrity(Journal(conn), cp)
    assert not verdict.ok
    assert verdict.first_bad_seq == head_seq


def test_projection_hash_mismatch_fails() -> None:
    conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    _filled_order(m)
    # Checkpoint cu un hash de proiecții greșit → integritatea eșuează, deși lanțul e intact.
    seq, digest = j.head
    conn.execute(
        "INSERT INTO checkpoints (journal_seq, journal_hash, projections_hash, ts) "
        "VALUES (?, ?, ?, ?)",
        (seq, digest, "deadbeef", T0.isoformat()),
    )
    cp = latest_checkpoint(conn)
    assert cp is not None
    verdict = validate_integrity(j, cp)
    assert not verdict.ok and verdict.first_bad_seq == seq


# --------------------------------------------------------------------------- recuperare


def test_recover_rebuilds_order_states_and_portfolio() -> None:
    _conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    coid = _filled_order(m, qty="10")
    rec = recover(j)
    assert rec.order_states[coid] is S.FILLED
    assert rec.order_instruments[coid] == "XYZ"
    # Portofoliul rejucat: o cumpărare de 10 @ 100, comision 1 → numerar -1001.
    assert rec.portfolio.cash_eur == D(-1001)
    assert "XYZ" in rec.portfolio.positions
    assert rec.portfolio.positions["XYZ"].qty == D(10)
    assert not rec.pending_orders  # ordin terminal: nimic blocat
    assert rec.frozen_instruments == frozenset()
    assert rec.resume_seq is None
    assert rec.recovery_complete


def test_recover_rebuilt_registry_dedups_duplicate_execution() -> None:
    """O retransmisie a unei execuții deja aplicate nu dublează fill-ul (Req 26.8)."""
    _conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    coid = _filled_order(m, qty="10")

    rec = recover(j)
    # OMS nou, cu registrul reconstruit; realimentăm aceeași execuție (retransmisie broker).
    fresh = OrderManager(JournalOmsSink(j), registry=rec.registry)
    # Ordinul nu există în noul manager, dar execuția deja aplicată e deduplicată de registru.
    outcomes = fresh.on_execution(
        _ev(coid, f"fill-{coid}", ExecKind.FILL, qty="10", price="100", commission="1")
    )
    assert [o.status for o in outcomes] == [ExecStatus.RETRANSMISSION]


def test_submitted_order_freezes_instrument_and_sets_resume_seq() -> None:
    """Un ordin SUBMITTED la cădere ține instrumentul blocat; resume_seq = primul neconfirmat."""
    _conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    _filled_order(m, instrument="XYZ", n=1, qty="10")  # terminal, nu blochează
    submitted = _submitted_order(m, instrument="ABC", n=2, qty="5")

    rec = recover(j)
    assert submitted in rec.pending_orders
    assert rec.order_states[submitted] is S.SUBMITTED
    assert rec.frozen_instruments == frozenset({"ABC"})
    assert "XYZ" not in rec.frozen_instruments
    # resume_seq pointează la înregistrarea care a dus ordinul în SUBMITTED.
    assert rec.resume_seq is not None
    row = _conn.execute(
        "SELECT type, payload FROM journal WHERE seq = ?", (rec.resume_seq,)
    ).fetchone()
    assert row["type"] == "oms.TRANSITION"
    assert '"to_state":"SUBMITTED"' in row["payload"]
    assert not rec.recovery_complete  # rămân ordine în așteptare (Req 26.5)


def test_unknown_order_freezes_instrument() -> None:
    _conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    coid = _submitted_order(m, instrument="ZZZ", n=3, qty="4")
    m.submit_timeout(coid, T0)  # SUBMITTED → UNKNOWN + înghețare
    assert m.order(coid).state is S.UNKNOWN

    rec = recover(j)
    assert rec.order_states[coid] is S.UNKNOWN
    assert rec.frozen_instruments == frozenset({"ZZZ"})
    assert coid in rec.pending_orders
    assert S.UNKNOWN in PENDING_STATES


# --------------------------------------------------------------------------- kill switch


def test_invalid_recovery_triggers_global_kill_switch() -> None:
    """Recovery_Point invalid → Kill_Switch(GLOBAL) (Req 26.4)."""
    conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    _filled_order(m)
    conn.execute("DROP TRIGGER journal_no_update")
    conn.execute("UPDATE journal SET outcome='tampered' WHERE seq=2")

    ks = KillSwitch(JournalKillSwitchStore(j), clock=SimClock(T0))
    verdict = validate_integrity(Journal(conn))
    assert not verdict.ok
    activated = global_kill_on_invalid_recovery(ks, verdict)
    assert activated
    assert not ks.allows("ANY")  # GLOBAL blochează orice instrument
    assert ks.blocking_scope("ANY") is KillSwitchScope.GLOBAL


def test_valid_recovery_does_not_trigger_kill_switch() -> None:
    _conn, j = _journal()
    m = OrderManager(JournalOmsSink(j))
    _filled_order(m)
    ks = KillSwitch(JournalKillSwitchStore(j), clock=SimClock(T0))
    verdict = validate_integrity(j)
    assert not global_kill_on_invalid_recovery(ks, verdict)
    assert ks.allows("ANY")


def test_kill_switch_active_survives_recovery() -> None:
    """Un Kill_Switch activat înaintea căderii rămâne activ (reconstruit din jurnal, Req 26.1)."""
    _conn, j = _journal()
    ks = KillSwitch(JournalKillSwitchStore(j), clock=SimClock(T0))
    ks.activate_automatic(KillSwitchScope.GLOBAL, component="test", reason_code="BOOM")
    # Repornire: reconstruiește din jurnal.
    ks2 = KillSwitch(JournalKillSwitchStore(Journal(_conn)), clock=SimClock(T0))
    assert not ks2.allows("ANY")
