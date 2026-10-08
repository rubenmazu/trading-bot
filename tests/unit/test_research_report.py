"""Teste pentru raportul de evaluare al cercetării (Req 21.3, 21.7, 29.1–29.5)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from qts.core.models import CostBreakdown
from qts.core.money import REPORTING_CURRENCY
from qts.research.bootstrap import Distribution
from qts.research.report import (
    PROFIT_NOT_GUARANTEED_WARNING,
    REQUIRED_METRICS,
    CostAssumptions,
    EvaluationReport,
    UncertaintyReport,
    VariantReport,
    build_evaluation_report,
)

D = Decimal


def _distribution(mean: str = "0") -> Distribution:
    return Distribution(
        count=10_000,
        mean=D(mean),
        minimum=D("-5"),
        maximum=D("5"),
        quantiles={"0.50": D(mean)},
    )


def _assumptions() -> CostAssumptions:
    return CostAssumptions(
        latency="200 ms la decizie",
        liquidity="ADV configurat per instrument",
        spread="spread configurat per instrument și oră",
        slippage="k·σ·sqrt(qty/ADV) + minim în tick-uri",
        commissions="tabel per broker, versionat",
        fx_conversion="spread de conversie al brokerului",
        taxes="valoare configurată, confirmată de operator",
    )


def _uncertainty() -> UncertaintyReport:
    return UncertaintyReport(
        seed=7,
        resamples=10_000,
        net_pnl=_distribution("12.5"),
        max_drawdown=_distribution("3.0"),
        longest_losing_streak=_distribution("4"),
        prob_hit_total_loss_limit=D("0.02"),
    )


def _variants() -> tuple[VariantReport, ...]:
    return (
        VariantReport(label="v1", net_result_eur=D("12.5"), accepted=True),
        VariantReport(
            label="v2",
            net_result_eur=D("-3"),
            accepted=False,
            rejection_reason="net OOS <= 0 (Req 21.4)",
        ),
    )


def _complete_report() -> EvaluationReport:
    return build_evaluation_report(
        strategy_id="s1",
        preregistration_id="hash-1",
        assumptions=_assumptions(),
        variants=_variants(),
        period="2023-01-01/2023-06-30",
        universe=("ETF_A", "ETF_B"),
        trade_count=42,
        gross_result_eur=D("20"),
        net_result_eur=D("12.5"),
        costs=CostBreakdown(commission=D("7.5")),
        max_drawdown_eur=D("3"),
        uncertainty=_uncertainty(),
    )


# --------------------------------------------------------------------------- raport complet


def test_complete_report_is_not_incomplete() -> None:
    report = _complete_report()
    assert report.missing_metrics == ()
    assert not report.incomplete
    assert not report.promotion_blocked


def test_report_presents_gross_costs_net() -> None:
    report = _complete_report()
    assert report.gross_result_eur == D("20")
    assert report.costs is not None and report.costs.total == D("7.5")
    assert report.net_result_eur == D("12.5")
    assert report.reporting_currency == REPORTING_CURRENCY


def test_report_always_includes_profit_warning() -> None:
    report = _complete_report()
    assert report.warning == PROFIT_NOT_GUARANTEED_WARNING
    assert "profitul nu este garantat" in report.warning
    assert "pierderile sunt posibile" in report.warning


def test_report_lists_all_variants_including_rejected() -> None:
    report = _complete_report()
    labels = {v.label for v in report.variants}
    assert labels == {"v1", "v2"}
    rejected = report.rejected_variants
    assert len(rejected) == 1
    assert rejected[0].label == "v2"
    assert rejected[0].rejection_reason is not None


def test_report_reports_uncertainty_distributions() -> None:
    report = _complete_report()
    assert report.uncertainty is not None
    assert report.uncertainty.net_pnl.quantiles["0.50"] == D("12.5")
    assert report.uncertainty.prob_hit_total_loss_limit == D("0.02")


# --------------------------------------------------------------------------- marcarea incomplet


@pytest.mark.parametrize("missing", list(REQUIRED_METRICS))
def test_missing_any_required_metric_marks_incomplete(missing: str) -> None:
    kwargs: dict[str, object] = {
        "period": "2023",
        "universe": ("ETF_A",),
        "trade_count": 10,
        "gross_result_eur": D("20"),
        "net_result_eur": D("12.5"),
        "costs": CostBreakdown(commission=D("7.5")),
        "max_drawdown_eur": D("3"),
    }
    kwargs[missing] = None
    report = build_evaluation_report(
        strategy_id="s1",
        preregistration_id="hash-1",
        assumptions=_assumptions(),
        variants=_variants(),
        uncertainty=_uncertainty(),
        **kwargs,  # type: ignore[arg-type]
    )
    assert report.incomplete
    assert missing in report.missing_metrics
    assert report.promotion_blocked
    # Avertismentul rămâne prezent chiar și în raportul incomplet.
    assert report.warning == PROFIT_NOT_GUARANTEED_WARNING


def test_empty_universe_counts_as_missing_metric() -> None:
    report = build_evaluation_report(
        strategy_id="s1",
        preregistration_id="hash-1",
        assumptions=_assumptions(),
        variants=_variants(),
        period="2023",
        universe=(),
        trade_count=10,
        gross_result_eur=D("20"),
        net_result_eur=D("12.5"),
        costs=CostBreakdown(commission=D("7.5")),
        max_drawdown_eur=D("3"),
    )
    assert "universe" in report.missing_metrics
    assert report.incomplete


# --------------------------------------------------------------------------- validări


def test_rejected_variant_requires_reason() -> None:
    with pytest.raises(ValueError, match="motiv de respingere"):
        VariantReport(label="v3", net_result_eur=D("-1"), accepted=False)


def test_accepted_variant_cannot_have_reason() -> None:
    with pytest.raises(ValueError, match="nu poate avea motiv"):
        VariantReport(
            label="v4", net_result_eur=D("1"), accepted=True, rejection_reason="x"
        )


def test_warning_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="avertismentul"):
        EvaluationReport(
            strategy_id="s1",
            preregistration_id="h",
            assumptions=_assumptions(),
            variants=_variants(),
            warning="profit garantat",
        )


def test_missing_cost_assumption_is_rejected() -> None:
    with pytest.raises(ValueError, match="ipoteze de cost lipsă"):
        CostAssumptions(
            latency="x",
            liquidity="x",
            spread="x",
            slippage="x",
            commissions="x",
            fx_conversion="x",
            taxes="   ",
        )


def test_duplicate_variant_labels_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_evaluation_report(
            strategy_id="s1",
            preregistration_id="h",
            assumptions=_assumptions(),
            variants=(
                VariantReport(label="v1", net_result_eur=D("1"), accepted=True),
                VariantReport(label="v1", net_result_eur=D("2"), accepted=True),
            ),
        )


def test_report_requires_at_least_one_variant() -> None:
    with pytest.raises(ValueError, match="cel puțin varianta"):
        build_evaluation_report(
            strategy_id="s1",
            preregistration_id="h",
            assumptions=_assumptions(),
            variants=(),
        )
