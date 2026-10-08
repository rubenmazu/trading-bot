"""Teste unitare pentru `safety/live_gate.py` (Req 3.6, 15.1–15.5, 15.8, 15.11, 15.12)."""

from __future__ import annotations

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
    LiveGateLike,
)
from qts.core.clock import SimClock
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.safety.kill_switch import JournalKillSwitchStore, KillSwitch
from qts.safety.live_gate import (
    CONDITION_ORDER,
    LIVE_RISK_WARNING,
    UNAVAILABLE_DETAIL,
    ConditionCheck,
    ConditionProvider,
    LiveGate,
    LiveGateCondition,
)
from qts.safety.stage import FAIL_CLOSED_SOURCE, ProjectStage, StageInfo

STAGE = StageInfo(ProjectStage.INITIAL, "stage.lock")
PROVIDED = [c for c in CONDITION_ORDER if c is not LiveGateCondition.STAGE_NOT_EXITED]


@dataclass
class _Counter:
    result: ConditionCheck = field(default_factory=lambda: ConditionCheck(True, "ok"))
    calls: int = 0

    def __call__(self) -> ConditionCheck:
        self.calls += 1
        return self.result


def _all_satisfied() -> dict[LiveGateCondition, ConditionProvider]:
    return {c: _Counter() for c in PROVIDED}


def test_default_gate_is_closed_and_lists_every_condition() -> None:
    gate = LiveGate(stage=lambda: STAGE)
    ev = gate.evaluate()
    assert ev.open is False
    assert gate.is_open() is False
    assert ev.codes == CONDITION_ORDER
    assert all(u.detail == UNAVAILABLE_DETAIL for u in ev.unsatisfied[1:])
    assert ev.unsatisfied[0].requirement == "15.1"


def test_all_providers_satisfied_but_initial_stage_stays_closed() -> None:
    gate = LiveGate(stage=lambda: STAGE, providers=_all_satisfied())
    ev = gate.evaluate()
    assert ev.open is False
    assert ev.codes == (LiveGateCondition.STAGE_NOT_EXITED,)
    assert gate.is_open() is False


def test_stage_lock_missing_is_initial_and_closed(tmp_path: Path) -> None:
    gate = LiveGate.from_stage_lock(tmp_path / "missing.lock", providers=_all_satisfied())
    ev = gate.evaluate()
    assert ev.codes == (LiveGateCondition.STAGE_NOT_EXITED,)
    assert FAIL_CLOSED_SOURCE in ev.unsatisfied[0].detail


def test_unsupported_stage_lock_is_unsatisfied(tmp_path: Path) -> None:
    lock = tmp_path / "stage.lock"
    lock.write_text('project_stage = "post_initial"\n', encoding="utf-8")
    gate = LiveGate.from_stage_lock(lock, providers=_all_satisfied())
    ev = gate.evaluate()
    assert ev.codes == (LiveGateCondition.STAGE_NOT_EXITED,)
    assert "StartupRefusedError" in ev.unsatisfied[0].detail
    assert gate.is_open() is False


def test_provider_exception_or_invalid_result_is_unsatisfied() -> None:
    def boom() -> ConditionCheck:
        raise RuntimeError("db down")

    def invalid() -> Any:
        return True  # nu este ConditionCheck

    providers = _all_satisfied()
    providers[LiveGateCondition.HEALTH] = boom
    providers[LiveGateCondition.RISK_LIMITS] = invalid
    providers[LiveGateCondition.RECONCILIATION] = _Counter(ConditionCheck(False, "diferențe"))
    ev = LiveGate(stage=lambda: STAGE, providers=providers).evaluate()
    by_code = {u.code: u.detail for u in ev.unsatisfied}
    assert set(by_code) == {
        LiveGateCondition.STAGE_NOT_EXITED,
        LiveGateCondition.HEALTH,
        LiveGateCondition.RISK_LIMITS,
        LiveGateCondition.RECONCILIATION,
    }
    assert "RuntimeError" in by_code[LiveGateCondition.HEALTH]
    assert "invalidă" in by_code[LiveGateCondition.RISK_LIMITS]
    assert by_code[LiveGateCondition.RECONCILIATION] == "diferențe"


def test_stage_provider_exception_is_unsatisfied() -> None:
    def boom() -> StageInfo:
        raise OSError("disk")

    gate = LiveGate(stage=boom, providers=_all_satisfied())
    assert gate.evaluate().codes == (LiveGateCondition.STAGE_NOT_EXITED,)
    assert gate.is_open() is False


def test_each_call_reevaluates_from_scratch() -> None:
    counters: dict[LiveGateCondition, _Counter] = {c: _Counter() for c in PROVIDED}
    stage_calls: list[int] = []

    def stage() -> StageInfo:
        stage_calls.append(1)
        return STAGE

    providers: dict[LiveGateCondition, ConditionProvider] = dict(counters)
    gate = LiveGate(stage=stage, providers=providers)
    gate.evaluate()
    counters[LiveGateCondition.HEALTH].result = ConditionCheck(False, "degradat")
    second = gate.evaluate()
    gate.is_open()
    assert len(stage_calls) == 3
    assert all(c.calls == 3 for c in counters.values())
    assert LiveGateCondition.HEALTH in second.codes


def test_stage_condition_cannot_be_overridden_by_provider() -> None:
    with pytest.raises(ValueError, match="etapă"):
        LiveGate(
            stage=lambda: STAGE,
            providers={LiveGateCondition.STAGE_NOT_EXITED: _Counter()},
        )


def test_gate_has_no_opening_api() -> None:
    public = {n for n in dir(LiveGate) if not n.startswith("_")}
    assert public == {"evaluate", "is_open", "from_stage_lock"}


def test_risk_warning_text() -> None:
    assert "pierderi" in LIVE_RISK_WARNING
    assert "nu este garantat" in LIVE_RISK_WARNING


# ---------------------------------------------------------------------- integrare Fail_Safe


@dataclass(frozen=True)
class _Req:
    client_order_id: str
    instrument: str


@dataclass
class _LiveAdapter:
    environment: str = "live"
    account_id: str | None = "U1234567"
    submitted: list[_Req] = field(default_factory=list)

    def submit(self, req: _Req, /) -> str:
        self.submitted.append(req)
        return "ack"


def test_fail_safe_rejects_live_adapter_with_live_gate(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "lg.db")
    journal = Journal(conn)
    clock = SimClock(datetime(2026, 3, 2, 10, tzinfo=UTC))
    gate: LiveGateLike = LiveGate(stage=lambda: STAGE, providers=_all_satisfied())
    adapter = _LiveAdapter()
    block: FailSafeBlock[_Req, str] = FailSafeBlock(
        adapter,
        approved=ApprovedTarget(environment="live", account_id="U1234567"),
        stage=STAGE,
        kill_switch=KillSwitch(JournalKillSwitchStore(journal), clock=clock),
        audit=journal,
        clock=clock,
        live_gate=gate,
    )
    with pytest.raises(FailSafeRejectedError) as exc:
        block.submit(_Req("c1", "AAA"))
    assert exc.value.reason in {
        FailSafeReason.STAGE_FORBIDS_ENVIRONMENT,
        FailSafeReason.LIVE_GATE_CLOSED,
    }
    assert exc.value.audited is True
    assert adapter.submitted == []
