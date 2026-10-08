"""Teste unitare pentru `broker/fail_safe.py` (Req 2.1, 2.5, 2.6, 14.1, 15.9)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from qts.broker.fail_safe import (
    ApprovedTarget,
    FailSafeBlock,
    FailSafeReason,
    FailSafeRejectedError,
)
from qts.config.schema import AppConfig
from qts.core.clock import SimClock
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.risk.context import KillSwitchScope
from qts.safety.kill_switch import JournalKillSwitchStore, KillSwitch
from qts.safety.stage import ProjectStage, StageInfo
from tests.helpers import DEMO_BROKER, config_dict

T0 = datetime(2026, 3, 2, 10, tzinfo=UTC)
STAGE = StageInfo(ProjectStage.INITIAL, "stage.lock")
DEMO_ACCOUNT = "DU1234567"


@dataclass(frozen=True)
class _Req:
    client_order_id: str
    instrument: str


@dataclass
class _Adapter:
    environment: str = "demo"
    account_id: str | None = DEMO_ACCOUNT
    submitted: list[_Req] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)

    def submit(self, req: _Req, /) -> str:
        self.submitted.append(req)
        return f"ack:{req.client_order_id}"

    def cancel(self, client_order_id: str) -> str:
        self.cancelled.append(client_order_id)
        return f"cancel:{client_order_id}"


class _BrokenAudit:
    def __init__(self, fail_times: int = 10**9) -> None:
        self.fail_times = fail_times
        self.records: list[dict[str, Any]] = []

    def append(self, **kwargs: Any) -> object:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise sqlite3.OperationalError("database is locked")
        self.records.append(kwargs)
        return None


class _OpenGate:
    def is_open(self) -> bool:
        return True


@dataclass
class _Env:
    conn: sqlite3.Connection
    journal: Journal
    ks: KillSwitch
    clock: SimClock


@pytest.fixture
def env(tmp_path: Path) -> _Env:
    conn = open_db(tmp_path / "fs.db")
    journal = Journal(conn)
    clock = SimClock(T0)
    ks = KillSwitch(JournalKillSwitchStore(journal), clock=clock)
    return _Env(conn, journal, ks, clock)


def _block(
    env: _Env,
    adapter: _Adapter,
    *,
    approved: ApprovedTarget | None = None,
    audit: Any = None,
    gate: Any = None,
) -> FailSafeBlock[_Req, str]:
    return FailSafeBlock(
        adapter,
        approved=approved or ApprovedTarget(environment="demo", account_id=DEMO_ACCOUNT),
        stage=STAGE,
        kill_switch=env.ks,
        audit=audit if audit is not None else env.journal,
        clock=env.clock,
        live_gate=gate,
    )


def _rejections(conn: sqlite3.Connection, type_: str = "fail_safe.rejected") -> list[str]:
    rows = conn.execute("SELECT payload FROM journal WHERE type = ? ORDER BY seq", (type_,))
    return [r[0] for r in rows]


def test_passes_when_all_checks_hold(env: _Env) -> None:
    adapter = _Adapter()
    block = _block(env, adapter)
    assert block.submit(_Req("c1", "AAA")) == "ack:c1"
    assert adapter.submitted == [_Req("c1", "AAA")]
    assert block.environment == "demo" and block.account_id == DEMO_ACCOUNT
    assert _rejections(env.conn) == []


def test_approved_target_from_config() -> None:
    cfg = AppConfig.model_validate(config_dict(environment="demo", broker=DEMO_BROKER))
    assert ApprovedTarget.from_config(cfg) == ApprovedTarget(
        environment="demo", account_id=DEMO_ACCOUNT
    )


@pytest.mark.parametrize(
    ("adapter", "reason"),
    [
        (_Adapter(environment="shadow"), FailSafeReason.ENVIRONMENT_MISMATCH),
        (_Adapter(account_id="DU999"), FailSafeReason.ACCOUNT_MISMATCH),
        (_Adapter(account_id=None), FailSafeReason.ACCOUNT_MISMATCH),
    ],
)
def test_environment_or_account_mismatch_rejects(
    env: _Env, adapter: _Adapter, reason: FailSafeReason
) -> None:
    block = _block(env, adapter)
    with pytest.raises(FailSafeRejectedError) as info:
        block.submit(_Req("c1", "AAA"))
    assert info.value.reason is reason and info.value.audited
    assert adapter.submitted == []
    (payload,) = _rejections(env.conn)
    assert reason.value in payload
    assert "DU999" not in payload  # identitatea contului nu apare în motiv


@pytest.mark.parametrize(
    ("mode", "adapter_env", "passes"),
    [
        ("backtest", "sim", True),
        ("shadow", "sim", True),
        ("demo", "demo", True),
        ("demo", "sim", False),
        ("shadow", "demo", False),
        ("backtest", "backtest", False),  # numele modului nu este un mediu de adaptor
    ],
)
def test_mode_maps_to_adapter_environment(
    env: _Env, mode: Any, adapter_env: str, passes: bool
) -> None:
    adapter = _Adapter(environment=adapter_env)
    block = _block(env, adapter, approved=ApprovedTarget(environment=mode, account_id=DEMO_ACCOUNT))
    if passes:
        assert block.submit(_Req("c1", "AAA")) == "ack:c1"
    else:
        with pytest.raises(FailSafeRejectedError) as info:
            block.submit(_Req("c1", "AAA"))
        assert info.value.reason is FailSafeReason.ENVIRONMENT_MISMATCH
        assert adapter.submitted == []


def test_invalid_request_rejected(env: _Env) -> None:
    adapter = _Adapter()
    with pytest.raises(FailSafeRejectedError) as info:
        _block(env, adapter).submit(_Req(" ", "AAA"))
    assert info.value.reason is FailSafeReason.INVALID_REQUEST
    assert adapter.submitted == []


def test_live_rejected_in_initial_stage_even_with_open_gate(env: _Env) -> None:
    adapter = _Adapter(environment="live", account_id="U1")
    block = _block(
        env, adapter, approved=ApprovedTarget(environment="live", account_id="U1"), gate=_OpenGate()
    )
    with pytest.raises(FailSafeRejectedError) as info:
        block.submit(_Req("c1", "AAA"))
    assert info.value.reason is FailSafeReason.STAGE_FORBIDS_ENVIRONMENT
    assert adapter.submitted == []


def test_live_requires_gate_independently_of_stage(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Chiar dacă etapa ar permite Live, poarta închisă (implicit) blochează cererea.
    monkeypatch.setattr(
        "qts.broker.fail_safe.ALLOWED_ENVIRONMENTS_BY_STAGE",
        {ProjectStage.INITIAL: frozenset({"demo", "live"})},
    )
    adapter = _Adapter(environment="live", account_id="U1")
    block = _block(env, adapter, approved=ApprovedTarget(environment="live", account_id="U1"))
    with pytest.raises(FailSafeRejectedError) as info:
        block.submit(_Req("c1", "AAA"))
    assert info.value.reason is FailSafeReason.LIVE_GATE_CLOSED
    assert adapter.submitted == []


def test_kill_switch_blocks_new_orders_but_cancels_pass(env: _Env) -> None:
    adapter = _Adapter()
    block = _block(env, adapter)
    env.ks.activate_manual(
        KillSwitchScope.INSTRUMENT, operator="alice", reason_code="R", instrument="AAA"
    )
    with pytest.raises(FailSafeRejectedError) as info:
        block.submit(_Req("c1", "AAA"))
    assert info.value.reason is FailSafeReason.KILL_SWITCH_ACTIVE
    assert block.submit(_Req("c2", "BBB")) == "ack:c2"

    env.ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="R")
    with pytest.raises(FailSafeRejectedError):
        block.submit(_Req("c3", "BBB"))
    assert block.cancel("c2") == "cancel:c2"  # pass-through
    assert adapter.cancelled == ["c2"]
    assert [r.client_order_id for r in adapter.submitted] == ["c2"]


def test_audit_failure_still_blocks_and_records_failed_attempt(env: _Env) -> None:
    adapter = _Adapter(environment="shadow")
    audit = _BrokenAudit(fail_times=1)  # prima scriere eșuează, a doua reușește
    block = _block(env, adapter, audit=audit)
    with pytest.raises(FailSafeRejectedError) as info:
        block.submit(_Req("c1", "AAA"))
    assert not info.value.audited
    assert adapter.submitted == []
    assert [r["type"] for r in audit.records] == ["fail_safe.rejection_audit_failed"]
    assert block.unaudited == ()


def test_total_audit_failure_still_blocks_and_queues(env: _Env) -> None:
    adapter = _Adapter(environment="shadow")
    audit = _BrokenAudit()
    block = _block(env, adapter, audit=audit)
    with pytest.raises(FailSafeRejectedError):
        block.submit(_Req("c1", "AAA"))
    assert adapter.submitted == []
    assert len(block.unaudited) == 1
    audit.fail_times = 0
    assert block.flush_audit() == 0
    assert [r["type"] for r in audit.records] == ["fail_safe.rejection_audit_failed"]


def test_check_exception_fails_closed(env: _Env) -> None:
    class _BrokenKillSwitch:
        def blocking_scope(self, instrument: str) -> KillSwitchScope | None:
            raise RuntimeError("stare coruptă")

    adapter = _Adapter()
    block = FailSafeBlock(
        adapter,
        approved=ApprovedTarget(environment="demo", account_id=DEMO_ACCOUNT),
        stage=STAGE,
        kill_switch=_BrokenKillSwitch(),
        audit=env.journal,
        clock=env.clock,
    )
    with pytest.raises(FailSafeRejectedError) as info:
        block.submit(_Req("c1", "AAA"))
    assert info.value.reason is FailSafeReason.CHECK_FAILED
    assert adapter.submitted == []
    assert len(_rejections(env.conn)) == 1
