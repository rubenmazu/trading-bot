"""Teste unitare pentru `safety/kill_switch.py` (Req 14.1–14.8, 13.4, 13.11, 26.1)."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from qts.config.schema import KillSwitchConfig, RiskConfig
from qts.core.clock import SimClock
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.risk.context import KillSwitchScope
from qts.risk.monitor import (
    CapitalConfigChangeError,
    KillSwitchSink,
    LossMonitor,
    LossObservation,
)
from qts.safety.kill_switch import (
    JOURNAL_TYPE_PREFIX,
    JournalKillSwitchStore,
    KillSwitch,
    KillSwitchEvent,
    KillSwitchPersistenceError,
    ResumeRefusal,
    ResumeRequest,
)

T0 = datetime(2026, 3, 2, 10, tzinfo=UTC)


@dataclass(frozen=True)
class _Order:
    client_order_id: str
    instrument: str


@dataclass(frozen=True)
class _Auth:
    new_capital_config_id: str
    valid: bool = True

    def is_valid(self) -> bool:
        return self.valid


def _db(path: Path) -> sqlite3.Connection:
    return open_db(path / "ks.db")


def _ks(
    conn: sqlite3.Connection,
    clock: SimClock,
    *,
    policy: str | None = None,
    orders: Iterable[_Order] = (),
) -> KillSwitch:
    cfg = (
        KillSwitchConfig()
        if policy is None
        else KillSwitchConfig.model_validate({"open_orders_policy": policy})
    )
    items = tuple(orders)
    return KillSwitch(
        JournalKillSwitchStore(Journal(conn)), clock=clock, config=cfg, open_orders=lambda: items
    )


def _resume(activation_id: str, /, **over: object) -> ResumeRequest:
    base: dict[str, object] = {
        "activation_id": activation_id,
        "operator": "alice",
        "approved": True,
        "reconciliation_ok": True,
        "reconciliation_id": "recon-1",
        "reason_resolved": True,
        "cause": "feed întrerupt",
        "correction": "feed repornit",
        "ts": T0 + timedelta(minutes=5),
    }
    base.update(over)
    return ResumeRequest.model_validate(base)


def _obs(ts: datetime, *, daily: str = "0", equity: str = "100") -> LossObservation:
    return LossObservation(
        ts=ts,
        daily_realized_pnl_eur=Decimal(daily),
        daily_unrealized_pnl_eur=Decimal(0),
        equity_eur=Decimal(equity),
    )


# --------------------------------------------------------------------------- domenii


def test_scopes_instrument_and_global(tmp_path: Path) -> None:
    ks = _ks(_db(tmp_path), SimClock(T0))
    assert ks.allows("AAA") and ks.allows("BBB")

    ks.activate_manual(
        KillSwitchScope.INSTRUMENT, operator="alice", reason_code="R", instrument="AAA"
    )
    assert ks.blocking_scope("AAA") is KillSwitchScope.INSTRUMENT
    assert ks.allows("BBB")
    assert ks.state().instruments == frozenset({"AAA"})

    ks.activate_automatic(KillSwitchScope.GLOBAL, component="health", reason_code="CRIT")
    assert ks.blocking_scope("BBB") is KillSwitchScope.GLOBAL
    st = ks.state()
    assert st.global_active and not st.day_active and not st.capital_config_active
    assert st.blocking_scope("ZZZ") is KillSwitchScope.GLOBAL


def test_manual_capital_config_activation_is_refused(tmp_path: Path) -> None:
    ks = _ks(_db(tmp_path), SimClock(T0))
    with pytest.raises(ValueError):
        ks.activate_manual(KillSwitchScope.CAPITAL_CONFIG, operator="alice", reason_code="R")


def test_reactivation_with_same_id_is_idempotent(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    ks = _ks(conn, SimClock(T0), policy="cancel", orders=[_Order("o1", "AAA")])
    out = ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="R")
    trig = ks.active_activations()[0]
    again = ks.trigger(trig)
    assert out.newly_activated and not again.newly_activated
    assert again.cancel_commands == ()
    rows = conn.execute("SELECT COUNT(*) FROM journal WHERE type = 'kill_switch.activated'")
    assert rows.fetchone()[0] == 1


# --------------------------------------------------------------------------- persistență


def test_activation_survives_restart_and_resume_is_persisted(tmp_path: Path) -> None:
    clock = SimClock(T0)
    conn = _db(tmp_path)
    ks = _ks(conn, clock)
    ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="MANUAL")
    act_id = ks.active_activations()[0].activation_id
    conn.close()

    conn2 = _db(tmp_path)
    restored = _ks(conn2, clock)
    assert restored.blocking_scope("AAA") is KillSwitchScope.GLOBAL
    assert restored.resume(_resume(act_id)).accepted
    conn2.close()

    conn3 = _db(tmp_path)
    assert _ks(conn3, clock).allows("AAA")
    types = [r[0] for r in conn3.execute("SELECT type FROM journal ORDER BY seq")]
    assert types == ["kill_switch.activated", "kill_switch.cleared"]


class _FlakyStore:
    def __init__(self) -> None:
        self.fail = True
        self.events: list[KillSwitchEvent] = []

    def append(self, event: KillSwitchEvent) -> None:
        if self.fail:
            raise OSError("disc plin")
        self.events.append(event)

    def load(self) -> Iterable[KillSwitchEvent]:
        return list(self.events)


def test_persistence_failure_still_blocks_and_defers_cancels() -> None:
    store = _FlakyStore()
    ks = KillSwitch(
        store,
        clock=SimClock(T0),
        config=KillSwitchConfig(open_orders_policy="cancel"),
        open_orders=lambda: [_Order("o1", "AAA")],
    )
    with pytest.raises(KillSwitchPersistenceError):
        ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="R")
    assert not ks.allows("AAA")  # fail-closed: blocat înaintea persistării
    assert ks.drain_cancel_commands() == ()  # efectul extern abia după persistare
    assert len(ks.unpersisted) == 1

    store.fail = False
    (outcome,) = ks.flush()
    assert [c.client_order_id for c in outcome.cancel_commands] == ["o1"]
    assert not ks.unpersisted
    assert [e.kind for e in store.events] == ["activated"]


def test_journal_store_round_trip(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    ks = _ks(conn, SimClock(T0))
    ks.activate_manual(KillSwitchScope.INSTRUMENT, operator="a", reason_code="R", instrument="X")
    record = next(Journal(conn).read())
    assert record.type.startswith(JOURNAL_TYPE_PREFIX)
    assert record.component == "kill_switch"
    loaded = list(JournalKillSwitchStore(Journal(conn)).load())
    assert loaded[0].trigger == ks.active_activations()[0]


# --------------------------------------------------------------------------- politica ordinelor


def test_default_policy_keeps_open_orders(tmp_path: Path) -> None:
    ks = _ks(_db(tmp_path), SimClock(T0), orders=[_Order("o1", "AAA"), _Order("o2", "BBB")])
    assert ks.open_orders_policy == "keep"
    out = ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="R")
    assert out.policy == "keep"
    assert out.cancel_commands == ()
    assert out.kept_order_ids == ("o1", "o2")
    assert ks.drain_cancel_commands() == ()


def test_cancel_policy_targets_orders_in_scope(tmp_path: Path) -> None:
    orders = [_Order("o1", "AAA"), _Order("o2", "BBB"), _Order("o3", "AAA")]
    ks = _ks(_db(tmp_path), SimClock(T0), policy="cancel", orders=orders)
    out = ks.activate_manual(
        KillSwitchScope.INSTRUMENT, operator="alice", reason_code="R", instrument="AAA"
    )
    assert [c.client_order_id for c in out.cancel_commands] == ["o1", "o3"]
    assert all(c.activation_id == out.activation_id for c in out.cancel_commands)
    assert ks.drain_cancel_commands() == out.cancel_commands
    assert ks.drain_cancel_commands() == ()

    glob = ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="R")
    assert [c.client_order_id for c in glob.cancel_commands] == ["o1", "o2", "o3"]


# --------------------------------------------------------------------------- reluare


def test_resume_refused_on_failed_reconciliation_even_if_approved(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    ks = _ks(conn, SimClock(T0))
    out = ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="R")
    decision = ks.resume(_resume(out.activation_id, reconciliation_ok=False, approved=True))
    assert not decision.accepted
    assert decision.refusals == (ResumeRefusal.RECONCILIATION_FAILED,)
    assert not ks.allows("AAA")
    refused = conn.execute("SELECT COUNT(*) FROM journal WHERE type='kill_switch.resume_refused'")
    assert refused.fetchone()[0] == 1


@pytest.mark.parametrize(
    ("override", "refusal"),
    [
        ({"approved": False}, ResumeRefusal.APPROVAL_MISSING),
        ({"reason_resolved": False}, ResumeRefusal.REASON_UNRESOLVED),
        ({"correction": " "}, ResumeRefusal.CAUSE_OR_CORRECTION_MISSING),
        ({"reconciliation_id": ""}, ResumeRefusal.RECONCILIATION_FAILED),
        ({"operator": ""}, ResumeRefusal.OPERATOR_MISSING),
        ({"activation_id": "nope"}, ResumeRefusal.UNKNOWN_ACTIVATION),
    ],
)
def test_resume_requires_every_condition(
    tmp_path: Path, override: dict[str, object], refusal: ResumeRefusal
) -> None:
    ks = _ks(_db(tmp_path), SimClock(T0))
    out = ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="R")
    decision = ks.resume(_resume(out.activation_id, **override))
    assert decision.refusals == (refusal,)
    assert not ks.allows("AAA")


def test_resume_clears_only_the_resolved_activation(tmp_path: Path) -> None:
    ks = _ks(_db(tmp_path), SimClock(T0))
    a = ks.activate_manual(
        KillSwitchScope.INSTRUMENT, operator="a", reason_code="R", instrument="A"
    )
    ks.activate_manual(KillSwitchScope.INSTRUMENT, operator="a", reason_code="R", instrument="B")
    assert ks.resume(_resume(a.activation_id)).accepted
    assert ks.allows("A")
    assert not ks.allows("B")


# --------------------------------------------------------------------------- integrare LossMonitor


def test_capital_config_not_clearable_by_operator(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    clock = SimClock(T0)
    ks = _ks(conn, clock)
    sink: KillSwitchSink = ks
    monitor = LossMonitor(RiskConfig(), sink, capital_config_id="cap-1")
    monitor.observe(_obs(T0, equity="90"))

    assert ks.blocking_scope("AAA") is KillSwitchScope.CAPITAL_CONFIG
    act_id = ks.active_activations()[0].activation_id
    decision = ks.resume(_resume(act_id))
    assert ResumeRefusal.CAPITAL_CONFIG_NOT_CLEARABLE in decision.refusals
    assert ks.state().capital_config_active

    with pytest.raises(CapitalConfigChangeError):
        ks.replace_capital_config(_Auth("cap-1"), actor="operator:alice")
    with pytest.raises(CapitalConfigChangeError):
        ks.replace_capital_config(_Auth("cap-2", valid=False), actor="operator:alice")

    assert ks.replace_capital_config(_Auth("cap-2"), actor="operator:alice") == (act_id,)
    assert ks.allows("AAA")
    # Reemiterea vechii activări (de exemplu după repornirea monitorului) nu mai blochează.
    monitor.flush()
    LossMonitor(RiskConfig(), ks, capital_config_id="cap-1").observe(_obs(T0, equity="90"))
    assert ks.allows("AAA")
    conn.close()
    assert _ks(_db(tmp_path), clock).allows("AAA")


def test_day_scope_expires_next_trading_day(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    clock = SimClock(T0)
    ks = _ks(conn, clock)
    LossMonitor(RiskConfig(), ks, capital_config_id="cap-1").observe(_obs(T0, daily="-2"))

    assert ks.blocking_scope("AAA") is KillSwitchScope.DAY
    act_id = ks.active_activations()[0].activation_id
    assert ResumeRefusal.DAY_NOT_CLEARABLE in ks.resume(_resume(act_id)).refusals

    later_same_day = T0 + timedelta(hours=5)
    assert ks.blocking_scope("AAA", later_same_day) is KillSwitchScope.DAY
    assert ks.expire_days(later_same_day) == ()

    next_day = T0 + timedelta(days=1)
    assert ks.blocking_scope("AAA", next_day) is None
    assert not ks.state(next_day).day_active
    assert ks.expire_days(next_day) == (act_id,)
    assert ks.active_activations() == ()
    conn.close()
    assert _ks(_db(tmp_path), clock).active_activations() == ()


# --------------------------------------------------------------------------- latență


def test_blocking_from_another_thread_within_one_second(tmp_path: Path) -> None:
    ks = _ks(_db(tmp_path), SimClock(T0))
    started = threading.Event()
    blocked_at: list[float] = []

    def submitter() -> None:
        started.set()
        deadline = time.perf_counter() + 5
        while time.perf_counter() < deadline:
            if not ks.allows("AAA"):
                blocked_at.append(time.perf_counter())
                return

    worker = threading.Thread(target=submitter)
    worker.start()
    started.wait()
    t_activate = time.perf_counter()
    ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="R")
    worker.join()
    assert blocked_at, "kill switch-ul nu a blocat în 5 s"
    assert blocked_at[0] - t_activate < 1.0
