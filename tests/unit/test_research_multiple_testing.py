import math
from decimal import Decimal

import pytest

from qts.research.multiple_testing import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    holm_bonferroni,
    multiple_testing_correction,
    normal_cdf,
    normal_ppf,
    sharpe_ratio,
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


def test_normal_cdf_known_values() -> None:
    assert math.isclose(normal_cdf(0.0), 0.5, abs_tol=1e-12)
    assert math.isclose(normal_cdf(1.96), 0.975, abs_tol=1e-3)


def test_normal_ppf_is_inverse_of_cdf() -> None:
    for p in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert math.isclose(normal_cdf(normal_ppf(p)), p, abs_tol=1e-6)


def test_normal_ppf_domain() -> None:
    with pytest.raises(ValueError, match="p în"):
        normal_ppf(0.0)
    with pytest.raises(ValueError, match="p în"):
        normal_ppf(1.0)


def test_sharpe_ratio_sign() -> None:
    assert sharpe_ratio([D(1), D(2), D(3)]) > 0
    assert sharpe_ratio([D(-1), D(-2), D(-3)]) < 0


def test_sharpe_ratio_zero_std_rejected() -> None:
    with pytest.raises(ValueError, match="deviația standard zero"):
        sharpe_ratio([D(5), D(5), D(5)])


def test_expected_max_sharpe_increases_with_trials() -> None:
    low = expected_max_sharpe(0.25, 2)
    high = expected_max_sharpe(0.25, 100)
    assert high > low > 0


def test_expected_max_sharpe_single_trial_zero() -> None:
    assert expected_max_sharpe(0.25, 1) == 0.0


def test_dsr_passes_for_strong_single_trial_strategy() -> None:
    returns = [D("0.02")] * 50 + [D("0.01")] * 50  # consistently positive
    result = deflated_sharpe_ratio(returns, [D("3.0")], alpha=D("0.05"))
    assert result.dsr >= result.threshold
    assert result.passed


def test_dsr_fails_when_many_trials_inflate_benchmark() -> None:
    # Noisy returns, high variance across many trials -> benchmark high, DSR low.
    returns = [D("0.5"), D("-0.4")] * 25
    sharpes = [D(str(s)) for s in (0.1, 0.2, -0.1, 0.3, -0.2, 0.25, 0.05, -0.3)]
    result = deflated_sharpe_ratio(returns, sharpes, alpha=D("0.05"))
    assert not result.passed


def test_dsr_alpha_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="alpha"):
        deflated_sharpe_ratio([D(1), D(2)], [D(1)], alpha=D(0))


def test_holm_bonferroni_step_down() -> None:
    p = {"v1": D("0.001"), "v2": D("0.02"), "v3": D("0.5")}
    result = holm_bonferroni(p, alpha=D("0.05"))
    by_label = {t.label: t for t in result.tests}
    # v1: 0.001 <= 0.05/3; v2: 0.02 <= 0.05/2; v3: 0.5 > 0.05/1
    assert by_label["v1"].rejected_null
    assert by_label["v2"].rejected_null
    assert not by_label["v3"].rejected_null
    assert result.any_significant


def test_holm_bonferroni_stops_after_first_failure() -> None:
    # Sorted ascending: v1(0.001), v2(0.03), v3(0.04). m=3.
    # v1 <= 0.05/3; v2 (0.03) > 0.05/2=0.025 -> stop; v3 cannot be rejected afterwards.
    p = {"v1": D("0.001"), "v2": D("0.03"), "v3": D("0.04")}
    result = holm_bonferroni(p, alpha=D("0.05"))
    by_label = {t.label: t for t in result.tests}
    assert by_label["v1"].rejected_null
    assert not by_label["v2"].rejected_null
    assert not by_label["v3"].rejected_null


def test_holm_bonferroni_rejects_bad_p_value() -> None:
    with pytest.raises(ValueError, match="afara"):
        holm_bonferroni({"v1": D("1.5")})


def test_holm_bonferroni_empty_rejected() -> None:
    with pytest.raises(ValueError, match="cel puțin o p-valoare"):
        holm_bonferroni({})


# --------------------------------------------------------------------------- orchestration


def _pre(method: CorrectionMethod) -> PreRegistration:
    return create_preregistration(
        strategy_id="s1",
        primary_metric="sharpe",
        promotion_criteria=PromotionCriteria(min_net_result_eur=D(0)),
        parameter_space=(ParameterAxis(name="z", values=(D(1), D(2))),),
        variant_count=3,
        correction_method=method,
        regimes=(MarketRegimeSpec(name="r1"), MarketRegimeSpec(name="r2")),
        stress=StressMultipliers(),
        walk_forward=WalkForwardSpec(
            windows=5, train_bars=10, test_bars=5, step_bars=5, recalibrate=True
        ),
    )


def test_correction_deflated_sharpe_only() -> None:
    pre = _pre(CorrectionMethod.DEFLATED_SHARPE)
    returns = [D("0.02")] * 60 + [D("0.015")] * 40
    result = multiple_testing_correction(
        pre,
        chosen_label="v1",
        chosen_returns=returns,
        all_variant_sharpes=[D("3.0")],
        bootstrap_p_values={},
    )
    assert result.method == CorrectionMethod.DEFLATED_SHARPE
    assert result.holm is None
    assert result.deflated_sharpe is not None
    assert result.passed == result.deflated_sharpe.passed


def test_correction_holm_requires_chosen_p_value() -> None:
    pre = _pre(CorrectionMethod.HOLM_BONFERRONI)
    with pytest.raises(ValueError, match="lipsește p-valoarea"):
        multiple_testing_correction(
            pre,
            chosen_label="v1",
            chosen_returns=[D(1), D(2)],
            all_variant_sharpes=[D(1)],
            bootstrap_p_values={"v2": D("0.01")},
        )


def test_correction_holm_passes_when_chosen_significant() -> None:
    pre = _pre(CorrectionMethod.HOLM_BONFERRONI)
    result = multiple_testing_correction(
        pre,
        chosen_label="v1",
        chosen_returns=[D(1), D(2)],
        all_variant_sharpes=[D(1)],
        bootstrap_p_values={"v1": D("0.001"), "v2": D("0.4"), "v3": D("0.6")},
    )
    assert result.deflated_sharpe is None
    assert result.holm is not None
    assert result.passed


def test_correction_holm_fails_when_chosen_not_significant() -> None:
    pre = _pre(CorrectionMethod.HOLM_BONFERRONI)
    result = multiple_testing_correction(
        pre,
        chosen_label="v3",
        chosen_returns=[D(1), D(2)],
        all_variant_sharpes=[D(1)],
        bootstrap_p_values={"v1": D("0.001"), "v2": D("0.4"), "v3": D("0.6")},
    )
    assert not result.passed


def test_correction_combined_requires_both() -> None:
    pre = _pre(CorrectionMethod.DEFLATED_SHARPE_HOLM)
    returns = [D("0.02")] * 60 + [D("0.015")] * 40
    result = multiple_testing_correction(
        pre,
        chosen_label="v1",
        chosen_returns=returns,
        all_variant_sharpes=[D("3.0")],
        bootstrap_p_values={"v1": D("0.001"), "v2": D("0.4")},
    )
    assert result.deflated_sharpe is not None
    assert result.holm is not None
    expected = result.deflated_sharpe.passed and any(
        t.rejected_null and t.label == "v1" for t in result.holm.tests
    )
    assert result.passed == expected
