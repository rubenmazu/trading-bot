from decimal import Decimal

from qts.costs.stress import StressScenario
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
from qts.research.stress import (
    StressResult,
    StressSuite,
    scenarios_from_preregistration,
)

D = Decimal


def _pre(
    *,
    min_net: str = "0",
    cost_multipliers: tuple[Decimal, ...] = (D(1), D("1.5"), D(2)),
    latency_multipliers: tuple[Decimal, ...] = (D(1), D(2)),
) -> PreRegistration:
    return create_preregistration(
        strategy_id="s1",
        primary_metric="net_eur",
        promotion_criteria=PromotionCriteria(min_net_result_eur=D(min_net)),
        parameter_space=(ParameterAxis(name="z", values=(D(1), D(2))),),
        variant_count=1,
        correction_method=CorrectionMethod.DEFLATED_SHARPE,
        regimes=(MarketRegimeSpec(name="r1"), MarketRegimeSpec(name="r2")),
        stress=StressMultipliers(
            cost_multipliers=cost_multipliers, latency_multipliers=latency_multipliers
        ),
        walk_forward=WalkForwardSpec(
            windows=5, train_bars=10, test_bars=5, step_bars=5, recalibrate=True
        ),
    )


def test_scenarios_cover_cost_and_latency_grid() -> None:
    pre = _pre()
    scenarios = scenarios_from_preregistration(pre, base_latency_ms=1000)
    # 3 cost multipliers x (base latency + 1 stress latency at mult 2 -> 2000ms) = 6
    assert len(scenarios) == 6
    pairs = {(s.cost_multiplier, s.latency_ms) for s in scenarios}
    assert pairs == {
        (m, lat) for m in (D(1), D("1.5"), D(2)) for lat in (None, 2000)
    }


def test_baseline_scenario_present() -> None:
    pre = _pre()
    scenarios = scenarios_from_preregistration(pre, base_latency_ms=500)
    baseline = [s for s in scenarios if s.is_baseline]
    assert len(baseline) == 1


def test_latency_multiplier_one_only_yields_base_latency() -> None:
    pre = _pre(latency_multipliers=(D(1),))
    scenarios = scenarios_from_preregistration(pre, base_latency_ms=800)
    assert {s.latency_ms for s in scenarios} == {None}


def test_suite_passes_when_all_scenarios_acceptable() -> None:
    pre = _pre(min_net="0")
    suite = StressSuite(pre, base_latency_ms=1000)

    def evaluate(scenario: StressScenario) -> Decimal:
        # Net degrades with cost but stays positive.
        return D(100) - (scenario.cost_multiplier - 1) * D(10)

    result = suite.run(evaluate)
    assert isinstance(result, StressResult)
    assert result.passed
    assert result.baseline.net_result_eur == D(100)
    assert result.failing() == ()


def test_suite_fails_when_a_stress_scenario_below_threshold() -> None:
    pre = _pre(min_net="50")
    suite = StressSuite(pre, base_latency_ms=1000)

    def evaluate(scenario: StressScenario) -> Decimal:
        # At x2.0 cost the net drops below the 50 EUR threshold.
        return D(100) - (scenario.cost_multiplier - 1) * D(80)

    result = suite.run(evaluate)
    assert not result.passed
    failing = result.failing()
    assert all(not o.is_baseline for o in failing)
    assert any(o.cost_multiplier == D(2) for o in failing)


def test_outcome_classification_uses_threshold() -> None:
    pre = _pre(min_net="100")
    suite = StressSuite(pre, base_latency_ms=0)
    result = suite.run(lambda _s: D(100))
    assert result.passed  # exactly at threshold counts as acceptable
    result2 = suite.run(lambda _s: D("99.99"))
    assert not result2.passed
