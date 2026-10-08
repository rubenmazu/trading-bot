"""Teste pentru `promotion/promote.py` (Req 16.1–16.8, 22.3, 22.7).

Acoperă: ordinea strictă Backtest → Shadow → Demo (fără sărituri, fără mers înapoi), reluarea de la
Backtest la modificarea artefactului, blocarea când criteriile nu sunt îndeplinite, diferențele
permise vs. interzise la promovare și starea terminală `Respins`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from qts.promotion.artifact import StrategyArtifact, create_artifact
from qts.promotion.promote import (
    PromotionBlockedError,
    PromotionController,
    PromotionOrderError,
    PromotionRecord,
    PromotionStage,
    StageStatus,
    detect_disallowed_differences,
)
from qts.research.partition import DataPartitioner, Partition

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 1, tzinfo=UTC)
APPROVED_AT = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
INSTRUMENTS = ("AAA", "BBB")


def _partitions() -> tuple[Partition, ...]:
    split = DataPartitioner(oos_fraction=0.25).split(
        start_ts=START, end_ts=END, instruments=INSTRUMENTS, label="study"
    )
    return (split.development, split.out_of_sample)


def _artifact(**overrides: Any) -> StrategyArtifact:
    kwargs: dict[str, Any] = {
        "strategy_id": "mean_reversion",
        "strategy_version": "1.0.0",
        "code_hash": "code-abc",
        "params": {"z": 2, "window": 15},
        "risk_config_hash": "risk-abc",
        "cost_model_version": "cost-v1",
        "universe_version": "universe-v1",
        "partitions": _partitions(),
        "preregistration_hash": "pre-abc",
        "validation_report_hash": "report-abc",
    }
    kwargs.update(overrides)
    return create_artifact(**kwargs)


def _complete(ctrl: PromotionController, stage: PromotionStage, **kw: Any) -> PromotionRecord:
    defaults: dict[str, Any] = {
        "criteria_met": True,
        "approver": "alice",
        "approved_at": APPROVED_AT,
    }
    defaults.update(kw)
    return ctrl.complete_stage(stage, **defaults)


# --------------------------------------------------------------------------- ordine (Req 16.1)


def test_promotion_follows_strict_order() -> None:
    ctrl = PromotionController(_artifact())
    assert ctrl.next_stage() is PromotionStage.BACKTEST
    _complete(ctrl, PromotionStage.BACKTEST)
    assert ctrl.next_stage() is PromotionStage.SHADOW
    _complete(ctrl, PromotionStage.SHADOW)
    assert ctrl.next_stage() is PromotionStage.DEMO
    _complete(ctrl, PromotionStage.DEMO)
    assert ctrl.next_stage() is None
    assert ctrl.state().completed_stages() == (
        PromotionStage.BACKTEST,
        PromotionStage.SHADOW,
        PromotionStage.DEMO,
    )


def test_cannot_skip_stages() -> None:
    ctrl = PromotionController(_artifact())
    with pytest.raises(PromotionOrderError):
        _complete(ctrl, PromotionStage.SHADOW)
    with pytest.raises(PromotionOrderError):
        _complete(ctrl, PromotionStage.DEMO)


def test_cannot_go_backward() -> None:
    ctrl = PromotionController(_artifact())
    _complete(ctrl, PromotionStage.BACKTEST)
    _complete(ctrl, PromotionStage.SHADOW)
    # Backtest este deja finalizat; nu se poate relua ca „etapa următoare”.
    with pytest.raises(PromotionOrderError):
        _complete(ctrl, PromotionStage.BACKTEST)


def test_can_promote_to_only_true_for_next_stage() -> None:
    ctrl = PromotionController(_artifact())
    assert ctrl.can_promote_to(PromotionStage.BACKTEST) is True
    assert ctrl.can_promote_to(PromotionStage.SHADOW) is False


# --------------------------------------------------------------------------- reset (Req 16.3)


def test_artifact_change_restarts_from_backtest() -> None:
    ctrl = PromotionController(_artifact())
    _complete(ctrl, PromotionStage.BACKTEST)
    _complete(ctrl, PromotionStage.SHADOW)
    assert ctrl.next_stage() is PromotionStage.DEMO

    changed = _artifact(params={"z": 3, "window": 15})
    reset = ctrl.set_artifact(changed)
    assert reset is True
    assert ctrl.next_stage() is PromotionStage.BACKTEST
    assert ctrl.state().completed_stages() == ()
    assert ctrl.artifact.artifact_id == changed.artifact_id


def test_identical_artifact_does_not_reset() -> None:
    art = _artifact()
    ctrl = PromotionController(art)
    _complete(ctrl, PromotionStage.BACKTEST)
    reset = ctrl.set_artifact(_artifact())  # același conținut => același artifact_id
    assert reset is False
    assert ctrl.next_stage() is PromotionStage.SHADOW


# --------------------------------------------------------------------------- criterii (Req 16.5)


def test_unmet_criteria_blocks_next_stage_and_rejects() -> None:
    ctrl = PromotionController(_artifact())
    with pytest.raises(PromotionBlockedError):
        _complete(ctrl, PromotionStage.BACKTEST, criteria_met=False)
    state = ctrl.state()
    assert state.is_rejected() is True
    assert state.records[-1].status is StageStatus.RESPINS
    # Etapa următoare este blocată: controllerul este în stare terminală.
    assert ctrl.next_stage() is None
    with pytest.raises(PromotionOrderError):
        _complete(ctrl, PromotionStage.SHADOW)


# --------------------------------------------------------------------------- diferențe (16.6/16.7)


def test_allowed_differences_do_not_block() -> None:
    ctrl = PromotionController(_artifact())
    source = {
        "adapters": {"broker": "sim"},
        "environment": "backtest",
        "credentials": {"broker_key": "qts/backtest/key"},
        "strategy_params": {"z": 2},
        "risk_rules": "risk-abc",
    }
    target = {
        "adapters": {"broker": "demo_sim"},
        "environment": "shadow",
        "credentials": {"broker_key": "qts/shadow/key"},
        "strategy_params": {"z": 2},
        "risk_rules": "risk-abc",
    }
    record = _complete(
        ctrl, PromotionStage.BACKTEST, source_config=source, target_config=target
    )
    assert record.status is StageStatus.COMPLETED


def test_disallowed_difference_blocks_and_rejects() -> None:
    ctrl = PromotionController(_artifact())
    source = {"environment": "backtest", "strategy_params": {"z": 2}}
    target = {"environment": "shadow", "strategy_params": {"z": 3}}  # logică schimbată!
    with pytest.raises(PromotionBlockedError, match="strategy_params"):
        _complete(ctrl, PromotionStage.BACKTEST, source_config=source, target_config=target)
    assert ctrl.state().is_rejected() is True


def test_detect_disallowed_differences_reports_only_outside_allowed() -> None:
    source = {
        "adapters": "sim",
        "environment": "backtest",
        "credentials": "a",
        "universe": "u1",
        "cost_model": "c1",
    }
    target = {
        "adapters": "demo",
        "environment": "demo",
        "credentials": "b",
        "universe": "u2",
        "cost_model": "c1",
    }
    assert detect_disallowed_differences(source, target) == ("universe",)


def test_detect_disallowed_differences_flags_added_or_removed_keys() -> None:
    assert detect_disallowed_differences({}, {"extra": 1}) == ("extra",)
    assert detect_disallowed_differences({"extra": 1}, {}) == ("extra",)
    assert detect_disallowed_differences({"adapters": 1}, {"adapters": 2}) == ()


# --------------------------------------------------------------------------- Respins (16.8, 22.3)


def test_explicit_reject_sets_terminal_respins() -> None:
    ctrl = PromotionController(_artifact())
    _complete(ctrl, PromotionStage.BACKTEST)
    record = ctrl.reject_stage(
        PromotionStage.SHADOW,
        approver="bob",
        approved_at=APPROVED_AT,
        reason="incident critic nerezolvat",
    )
    assert record.status is StageStatus.RESPINS
    assert record.criteria_met is False
    assert ctrl.state().is_rejected() is True
    # Terminal: nicio altă finalizare nu mai e permisă.
    with pytest.raises(PromotionOrderError):
        _complete(ctrl, PromotionStage.SHADOW)


def test_rejected_flow_recovers_only_via_new_artifact() -> None:
    ctrl = PromotionController(_artifact())
    with pytest.raises(PromotionBlockedError):
        _complete(ctrl, PromotionStage.BACKTEST, criteria_met=False)
    assert ctrl.next_stage() is None
    # Un artefact nou (versiune nouă) resetează fluxul la Backtest (Req 16.3).
    ctrl.set_artifact(_artifact(strategy_version="1.0.1"))
    assert ctrl.next_stage() is PromotionStage.BACKTEST
    assert ctrl.state().is_rejected() is False


# --------------------------------------------------------------------------- înregistrare (16.4)


def test_completion_records_approver_and_time() -> None:
    ctrl = PromotionController(_artifact())
    record = _complete(ctrl, PromotionStage.BACKTEST, approver="carol")
    assert record.approver == "carol"
    assert record.approved_at == APPROVED_AT
    assert record.criteria_met is True
    assert record.artifact_id == ctrl.artifact.artifact_id
