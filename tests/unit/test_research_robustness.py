from collections.abc import Mapping
from decimal import Decimal

import pytest

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
from qts.research.robustness import (
    drop_top_n_wins,
    neighbors_of,
    robustness_check,
)

D = Decimal


def _axis(name: str) -> ParameterAxis:
    return ParameterAxis(name=name, values=(D(1), D(2), D(3)), neighbor_step=D(1))


def _pre(min_net: str = "0", fraction: str = "0.5") -> PreRegistration:
    return create_preregistration(
        strategy_id="s1",
        primary_metric="net_eur",
        promotion_criteria=PromotionCriteria(
            min_net_result_eur=D(min_net), min_robust_neighbor_fraction=D(fraction)
        ),
        parameter_space=(_axis("a"), _axis("b")),
        variant_count=1,
        correction_method=CorrectionMethod.DEFLATED_SHARPE,
        regimes=(MarketRegimeSpec(name="r1"), MarketRegimeSpec(name="r2")),
        stress=StressMultipliers(),
        walk_forward=WalkForwardSpec(
            windows=5, train_bars=10, test_bars=5, step_bars=5, recalibrate=True
        ),
    )


def test_neighbors_at_center_has_four_for_two_axes() -> None:
    pre = _pre()
    chosen = {"a": D(2), "b": D(2)}
    neighbors = neighbors_of(pre.parameter_space, chosen)
    # a: {1,3}, b: {1,3} -> 4 neighbors
    assert len(neighbors) == 4
    assert {"a": D(1), "b": D(2)} in neighbors
    assert {"a": D(3), "b": D(2)} in neighbors
    assert {"a": D(2), "b": D(1)} in neighbors
    assert {"a": D(2), "b": D(3)} in neighbors


def test_neighbors_at_edge_fewer() -> None:
    pre = _pre()
    chosen = {"a": D(1), "b": D(1)}
    neighbors = neighbors_of(pre.parameter_space, chosen)
    # Only +1 step exists on each axis -> 2 neighbors.
    assert len(neighbors) == 2


def test_neighbors_rejects_unknown_axis() -> None:
    pre = _pre()
    with pytest.raises(ValueError, match="necunoscuți"):
        neighbors_of(pre.parameter_space, {"a": D(2), "b": D(2), "x": D(1)})


def test_neighbors_rejects_point_off_axis() -> None:
    pre = _pre()
    with pytest.raises(ValueError, match="nu se află pe axa"):
        neighbors_of(pre.parameter_space, {"a": D(9), "b": D(2)})


def test_robustness_passes_when_enough_neighbors_acceptable() -> None:
    pre = _pre(min_net="0", fraction="0.5")
    chosen = {"a": D(2), "b": D(2)}

    def evaluate(point: Mapping[str, Decimal]) -> Decimal:
        # All points profitable.
        return D(10)

    result = robustness_check(pre, chosen, evaluate)
    assert result.passed
    assert result.acceptable_fraction == D(1)
    assert result.center.is_center


def test_robustness_rejects_isolated_peak() -> None:
    pre = _pre(min_net="0", fraction="0.5")
    chosen = {"a": D(2), "b": D(2)}

    def evaluate(point: Mapping[str, Decimal]) -> Decimal:
        # Only the center is profitable; every neighbor loses.
        if dict(point) == chosen:
            return D(100)
        return D(-1)

    result = robustness_check(pre, chosen, evaluate)
    assert result.center.acceptable
    assert result.acceptable_fraction == D(0)
    assert not result.passed


def test_robustness_boundary_exactly_half() -> None:
    pre = _pre(min_net="0", fraction="0.5")
    chosen = {"a": D(2), "b": D(2)}
    acceptable_points = [{"a": D(1), "b": D(2)}, {"a": D(3), "b": D(2)}]

    def evaluate(point: Mapping[str, Decimal]) -> Decimal:
        return D(10) if dict(point) in acceptable_points else D(-1)

    result = robustness_check(pre, chosen, evaluate)
    # 2 of 4 neighbors acceptable -> fraction 0.5 >= required 0.5
    assert result.acceptable_fraction == D("0.5")
    assert result.passed


def test_drop_top_n_wins_zero_returns_total() -> None:
    assert drop_top_n_wins([D(10), D(-3), D(5)], 0) == D(12)


def test_drop_top_n_wins_removes_largest_gains() -> None:
    nets = [D(10), D(8), D(-3), D(2), D(-1)]
    # total = 16; drop top 2 wins (10, 8) -> -2
    assert drop_top_n_wins(nets, 2) == D(-2)


def test_drop_top_n_wins_only_drops_positive() -> None:
    nets = [D(-5), D(-2), D(1)]
    # total = -6; only one positive win (1); dropping 3 removes just that gain -> -7.
    assert drop_top_n_wins(nets, 3) == D(-7)


def test_drop_top_n_negative_rejected() -> None:
    with pytest.raises(ValueError, match="negativ"):
        drop_top_n_wins([D(1)], -1)
