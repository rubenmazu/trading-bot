"""Teste unitare pentru `health/monitor.py` (Req 25.1–25.4, 26.6, 26.7)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from qts.config.schema import KillSwitchConfig
from qts.core.clock import SimClock
from qts.health.alerts import AlertQueue
from qts.health.monitor import (
    CLOCK_DEVIATION_LIMIT_MS,
    Component,
    ComponentState,
    HealthMonitor,
)
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.risk.context import KillSwitchScope
from qts.safety.kill_switch import JournalKillSwitchStore, KillSwitch
from qts.secrets.store import Redactor

T0 = datetime(2026, 3, 2, 10, tzinfo=UTC)


def _ks(conn: sqlite3.Connection, clock: SimClock) -> KillSwitch:
    return KillSwitch(JournalKillSwitchStore(Journal(conn)), clock=clock, config=KillSwitchConfig())


def _monitor(
    tmp_path: Path,
    *,
    mode: str,
    redactor: Redactor | None = None,
) -> tuple[HealthMonitor, KillSwitch, AlertQueue]:
    conn = open_db(tmp_path / "health.db")
    clock = SimClock(T0)
    ks = _ks(conn, clock)
    red = redactor if redactor is not None else Redactor()
    queue = AlertQueue(conn, redactor=red)
    monitor = HealthMonitor(clock=clock, mode=mode, kill_switch=ks, alerts=queue, redactor=red)
    return monitor, ks, queue


# --------------------------------------------------------------------------- severitate


def test_component_state_severity_ordering() -> None:
    assert ComponentState.HEALTHY < ComponentState.DEGRADED < ComponentState.UNKNOWN
    assert ComponentState.UNKNOWN < ComponentState.CRITICAL
    assert max(ComponentState.DEGRADED, ComponentState.CRITICAL) is ComponentState.CRITICAL


def test_all_components_start_unknown(tmp_path: Path) -> None:
    monitor, _ks, _q = _monitor(tmp_path, mode="backtest")
    report = monitor.report()
    for component in Component:
        assert report.status(component).state is ComponentState.UNKNOWN
    assert report.overall is ComponentState.UNKNOWN


# --------------------------------------------------------------------------- tranziții de stare


def test_state_change_records_reason_and_ts_and_emits_alert(tmp_path: Path) -> None:
    monitor, _ks, queue = _monitor(tmp_path, mode="demo")
    ts = T0 + timedelta(seconds=3)
    monitor.set_state(Component.DATA, ComponentState.HEALTHY, reason="feed proaspăt", ts=ts)
    status = monitor.report().status(Component.DATA)
    assert status.state is ComponentState.HEALTHY
    assert status.reason == "feed proaspăt"
    assert status.ts == ts
    pending = queue.pending()
    assert len(pending) == 1
    assert pending[0].component == "data"
    assert pending[0].code == "HEALTH_DATA_HEALTHY"


def test_identical_state_does_not_re_emit(tmp_path: Path) -> None:
    monitor, _ks, queue = _monitor(tmp_path, mode="demo")
    monitor.set_state(Component.BROKER, ComponentState.DEGRADED, reason="latență")
    monitor.set_state(Component.BROKER, ComponentState.DEGRADED, reason="latență")
    assert len(queue.pending()) == 1


def test_overall_is_most_severe(tmp_path: Path) -> None:
    monitor, _ks, _q = _monitor(tmp_path, mode="demo")
    for component in Component:
        monitor.set_state(component, ComponentState.HEALTHY, reason="ok")
    assert monitor.overall() is ComponentState.HEALTHY
    monitor.set_state(Component.RISK, ComponentState.DEGRADED, reason="aproape de limită")
    assert monitor.overall() is ComponentState.DEGRADED
    assert monitor.healthy() is False


# --------------------------------------------------------------------------- CRITICAL → kill switch


def test_critical_in_demo_activates_global_kill_and_alert(tmp_path: Path) -> None:
    monitor, ks, queue = _monitor(tmp_path, mode="demo")
    for component in Component:
        monitor.set_state(component, ComponentState.HEALTHY, reason="ok")
    monitor.set_state(Component.STORAGE, ComponentState.CRITICAL, reason="disc plin")

    assert ks.blocking_scope("AAA") is KillSwitchScope.GLOBAL
    assert monitor.blocks_new_orders() is True
    assert monitor.startup_blocked() is True
    codes = [a.code for a in queue.pending()]
    assert "HEALTH_STORAGE_CRITICAL" in codes


def test_critical_in_backtest_does_not_kill_switch(tmp_path: Path) -> None:
    monitor, ks, _q = _monitor(tmp_path, mode="backtest")
    monitor.set_state(Component.STORAGE, ComponentState.CRITICAL, reason="disc plin")
    assert ks.allows("AAA")  # fără kill switch în backtest
    # dar starea critică blochează oricum ordinele noi / pornirea (fail-closed)
    assert monitor.blocks_new_orders() is True


# --------------------------------------------------------------------------- ceas vs. NTP


def test_clock_deviation_over_limit_suspends_in_demo(tmp_path: Path) -> None:
    monitor, _ks, _q = _monitor(tmp_path, mode="demo")
    over = CLOCK_DEVIATION_LIMIT_MS + 1
    ntp = T0 + timedelta(milliseconds=over)
    state = monitor.check_clock(ntp_now=lambda: ntp, local_now=T0)
    assert state is ComponentState.DEGRADED
    assert monitor.blocks_new_orders() is True


def test_clock_deviation_within_limit_is_healthy(tmp_path: Path) -> None:
    monitor, _ks, _q = _monitor(tmp_path, mode="demo")
    ntp = T0 + timedelta(milliseconds=CLOCK_DEVIATION_LIMIT_MS - 1)
    state = monitor.check_clock(ntp_now=lambda: ntp, local_now=T0)
    assert state is ComponentState.HEALTHY
    assert monitor.state_of(Component.CLOCK) is ComponentState.HEALTHY


def test_clock_deviation_over_limit_in_backtest_does_not_suspend(tmp_path: Path) -> None:
    monitor, _ks, _q = _monitor(tmp_path, mode="backtest")
    ntp = T0 + timedelta(milliseconds=CLOCK_DEVIATION_LIMIT_MS + 100)
    state = monitor.check_clock(ntp_now=lambda: ntp, local_now=T0)
    assert state is ComponentState.DEGRADED
    # în backtest DEGRADED nu suspendă (numai CRITICAL ar bloca)
    assert monitor.blocks_new_orders() is False


def test_missing_ntp_leaves_clock_unknown_and_suspends_in_live(tmp_path: Path) -> None:
    monitor, _ks, _q = _monitor(tmp_path, mode="live")
    state = monitor.check_clock(ntp_now=None, local_now=T0)
    assert state is ComponentState.UNKNOWN
    assert monitor.blocks_new_orders() is True


def test_ntp_source_error_is_fail_closed(tmp_path: Path) -> None:
    monitor, _ks, _q = _monitor(tmp_path, mode="demo")

    def boom() -> datetime:
        raise RuntimeError("NTP timeout")

    state = monitor.check_clock(ntp_now=boom, local_now=T0)
    assert state is ComponentState.UNKNOWN
    assert monitor.blocks_new_orders() is True


# --------------------------------------------------------------------------- redactare


def test_alert_detail_is_redacted(tmp_path: Path) -> None:
    red = Redactor()
    red.register("super-secret-token")
    monitor, _ks, queue = _monitor(tmp_path, mode="demo", redactor=red)
    monitor.set_state(
        Component.BROKER,
        ComponentState.DEGRADED,
        reason="token super-secret-token respins",
    )
    (alert,) = queue.pending()
    assert "super-secret-token" not in alert.detail
    assert "***" in alert.detail


def test_healthy_report_does_not_block(tmp_path: Path) -> None:
    monitor, _ks, _q = _monitor(tmp_path, mode="live")
    for component in Component:
        monitor.set_state(component, ComponentState.HEALTHY, reason="ok")
    assert monitor.healthy() is True
    assert monitor.blocks_new_orders() is False
    assert monitor.startup_blocked() is False


def test_naive_ntp_datetime_is_rejected(tmp_path: Path) -> None:
    monitor, _ks, _q = _monitor(tmp_path, mode="demo")
    with pytest.raises(ValueError):
        monitor.check_clock(ntp_now=lambda: datetime(2026, 3, 2, 10), local_now=T0)  # noqa: DTZ001
