"""Teste pentru evaluarea `Paper_Qualification` față de criterii preînregistrate (Req 22.4–22.6)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from qts.core.money import REPORTING_CURRENCY
from qts.research.paper_qualification import (
    PaperQualificationCriteria,
    PaperStage,
    StageObservations,
    evaluate_paper_qualification,
)
from qts.research.preregistration import (
    CorrectionMethod,
    MarketRegimeSpec,
    ParameterAxis,
    PreRegistration,
    PromotionCriteria,
    StressMultipliers,
    WalkForwardSpec,
    create_preregistration,
)

D = Decimal


def _criteria(
    *,
    min_net: str = "10",
    min_orders: int = 20,
    min_regimes: int = 2,
    required_regimes: tuple[str, ...] = ("bull", "bear"),
    max_rejections: int | None = None,
    max_disconnections: int | None = None,
    max_failed_reconciliations: int | None = None,
) -> PaperQualificationCriteria:
    return PaperQualificationCriteria(
        min_net_result_eur=D(min_net),
        min_orders=min_orders,
        min_regimes=min_regimes,
        required_regimes=required_regimes,
        max_rejections=max_rejections,
        max_disconnections=max_disconnections,
        max_failed_reconciliations=max_failed_reconciliations,
    )


def _shadow(**overrides: object) -> StageObservations:
    base: dict[str, object] = {
        "stage": PaperStage.SHADOW,
        "net_result_eur": D("7"),
        "orders": 12,
        "regimes_covered": ("bull",),
    }
    base.update(overrides)
    return StageObservations(**base)


def _demo(**overrides: object) -> StageObservations:
    base: dict[str, object] = {
        "stage": PaperStage.DEMO,
        "net_result_eur": D("6"),
        "orders": 10,
        "regimes_covered": ("bear",),
    }
    base.update(overrides)
    return StageObservations(**base)


# --------------------------------------------------------------------------- pass


def test_qualification_passes_when_all_criteria_met() -> None:
    result = evaluate_paper_qualification(_criteria(), (_shadow(), _demo()))
    assert result.passed
    assert result.reasons == ()
    assert result.aggregate_net_result_eur == D("13")
    assert result.aggregate_orders == 22
    assert result.reporting_currency == REPORTING_CURRENCY
    # evaluarea enumeră criteriile verificate (auditabil)
    assert any("Req 22.6" in e for e in result.evaluated)
    assert any("Req 22.5" in e for e in result.evaluated)
    assert any("Req 22.4" in e for e in result.evaluated)


def test_cumulative_aggregation_across_shadow_and_demo() -> None:
    # net agregat 4+8=12 ≥ prag 10; ordine 15+15=30 ≥ 20; regimuri {bull, bear} ≥ 2
    result = evaluate_paper_qualification(
        _criteria(),
        (
            _shadow(net_result_eur=D("4"), orders=15, regimes_covered=("bull",)),
            _demo(net_result_eur=D("8"), orders=15, regimes_covered=("bear",)),
        ),
    )
    assert result.passed
    assert result.aggregate_net_result_eur == D("12")
    assert result.aggregate_orders == 30


# --------------------------------------------------------------------------- fail: Req 22.5


def test_fails_when_shadow_net_is_zero_or_negative() -> None:
    result = evaluate_paper_qualification(
        _criteria(min_net="1"),
        (_shadow(net_result_eur=D("0")), _demo(net_result_eur=D("10"))),
    )
    assert not result.passed
    assert any("Req 22.5" in r and "Shadow" in r for r in result.reasons)


def test_fails_when_demo_net_is_negative() -> None:
    result = evaluate_paper_qualification(
        _criteria(min_net="1"),
        (_shadow(net_result_eur=D("20")), _demo(net_result_eur=D("-3"))),
    )
    assert not result.passed
    assert any("Req 22.5" in r and "Demo" in r for r in result.reasons)


def test_positive_aggregate_still_fails_if_one_stage_not_positive() -> None:
    # agregat 20 + (-5) = 15 > 0, dar Demo ≤ 0 => respins pe etapă (Req 22.5)
    result = evaluate_paper_qualification(
        _criteria(min_net="1"),
        (_shadow(net_result_eur=D("20")), _demo(net_result_eur=D("-5"))),
    )
    assert not result.passed
    assert any("Req 22.5" in r for r in result.reasons)


# --------------------------------------------------------------------------- fail: Req 22.6


def test_blocks_on_unresolved_critical_incident() -> None:
    result = evaluate_paper_qualification(
        _criteria(),
        (_shadow(unresolved_critical_incidents=1), _demo()),
    )
    assert not result.passed
    assert any("Req 22.6" in r and "Shadow" in r for r in result.reasons)


# --------------------------------------------------------------------------- fail: Req 22.4


def test_fails_when_net_below_threshold() -> None:
    result = evaluate_paper_qualification(
        _criteria(min_net="100"),
        (_shadow(), _demo()),
    )
    assert not result.passed
    assert any("Req 22.4" in r and "net agregat" in r for r in result.reasons)


def test_fails_when_order_count_below_threshold() -> None:
    result = evaluate_paper_qualification(
        _criteria(min_orders=100),
        (_shadow(), _demo()),
    )
    assert not result.passed
    assert any("Req 22.4" in r and "ordine" in r for r in result.reasons)


def test_fails_when_regime_coverage_below_threshold() -> None:
    result = evaluate_paper_qualification(
        _criteria(min_regimes=2),
        # ambele etape acoperă doar "bull" => o singură acoperire
        (_shadow(regimes_covered=("bull",)), _demo(regimes_covered=("bull",))),
    )
    assert not result.passed
    assert any("Req 22.4" in r and "regimuri" in r for r in result.reasons)


def test_fails_when_rejections_exceed_limit() -> None:
    result = evaluate_paper_qualification(
        _criteria(max_rejections=3),
        (_shadow(rejections=2), _demo(rejections=5)),
    )
    assert not result.passed
    assert any("respingeri" in r for r in result.reasons)


def test_fails_when_disconnections_exceed_limit() -> None:
    result = evaluate_paper_qualification(
        _criteria(max_disconnections=1),
        (_shadow(disconnections=1), _demo(disconnections=2)),
    )
    assert not result.passed
    assert any("deconectări" in r for r in result.reasons)


def test_fails_when_failed_reconciliations_exceed_limit() -> None:
    result = evaluate_paper_qualification(
        _criteria(max_failed_reconciliations=0),
        (_shadow(failed_reconciliations=1), _demo()),
    )
    assert not result.passed
    assert any("reconcilieri" in r for r in result.reasons)


def test_optional_operational_limits_tolerant_when_unset() -> None:
    # fără limite fixate, numere mari de respingeri/deconectări nu respinge calificarea
    result = evaluate_paper_qualification(
        _criteria(),
        (
            _shadow(rejections=99, disconnections=50, failed_reconciliations=7),
            _demo(),
        ),
    )
    assert result.passed


def test_multiple_reasons_accumulate() -> None:
    result = evaluate_paper_qualification(
        _criteria(min_net="100", min_orders=1000),
        (_shadow(net_result_eur=D("-1"), unresolved_critical_incidents=1), _demo()),
    )
    assert not result.passed
    # Req 22.6 + Req 22.5 (Shadow) + Req 22.4 (net) + Req 22.4 (ordine)
    assert len(result.reasons) >= 4


# --------------------------------------------------------------------------- determinism


def test_evaluation_is_deterministic() -> None:
    args = (_criteria(), (_shadow(), _demo()))
    first = evaluate_paper_qualification(*args)
    second = evaluate_paper_qualification(*args)
    assert first == second
    assert first.reasons == second.reasons
    assert first.evaluated == second.evaluated


# --------------------------------------------------------------------------- validări structură


def test_requires_both_stages() -> None:
    with pytest.raises(ValueError, match="fiecare etapă"):
        evaluate_paper_qualification(_criteria(), (_shadow(),))


def test_rejects_duplicate_stage() -> None:
    with pytest.raises(ValueError, match="fiecare etapă"):
        evaluate_paper_qualification(_criteria(), (_shadow(), _shadow()))


def test_stage_observations_reject_duplicate_regimes() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _shadow(regimes_covered=("bull", "bull"))


def test_criteria_reject_duplicate_required_regimes() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        PaperQualificationCriteria(
            min_net_result_eur=D("1"), required_regimes=("bull", "bull")
        )


def test_criteria_reject_min_regimes_above_required() -> None:
    with pytest.raises(ValueError, match="min_regimes"):
        PaperQualificationCriteria(
            min_net_result_eur=D("1"), min_regimes=3, required_regimes=("bull", "bear")
        )


# --------------------------------------------------------------------------- preînregistrare


def _preregistration() -> PreRegistration:
    return create_preregistration(
        strategy_id="s1",
        primary_metric="deflated_sharpe",
        promotion_criteria=PromotionCriteria(
            min_net_result_eur=D("5"), min_trades=20, min_regimes=2
        ),
        parameter_space=(ParameterAxis(name="lookback", values=(D("10"), D("20"))),),
        variant_count=2,
        correction_method=CorrectionMethod.DEFLATED_SHARPE,
        regimes=(
            MarketRegimeSpec(name="bull"),
            MarketRegimeSpec(name="bear"),
        ),
        stress=StressMultipliers(),
        walk_forward=WalkForwardSpec(
            windows=5, train_bars=100, test_bars=20, step_bars=20, recalibrate=True
        ),
    )


def test_criteria_from_preregistration_reuses_promotion_criteria() -> None:
    pre = _preregistration()
    criteria = PaperQualificationCriteria.from_preregistration(pre)
    assert criteria.min_net_result_eur == D("5")
    assert criteria.min_orders == 20
    assert criteria.min_regimes == 2
    assert criteria.required_regimes == ("bear", "bull")


def test_evaluation_using_preregistered_criteria_passes() -> None:
    pre = _preregistration()
    criteria = PaperQualificationCriteria.from_preregistration(pre)
    result = evaluate_paper_qualification(
        criteria,
        (
            _shadow(net_result_eur=D("4"), orders=12, regimes_covered=("bull",)),
            _demo(net_result_eur=D("4"), orders=10, regimes_covered=("bear",)),
        ),
    )
    assert result.passed
