from decimal import Decimal

import pytest

from qts.research.regimes import (
    RegimeLabel,
    TrendBand,
    VolBand,
    classify_regimes,
    fit_regime_model,
    realized_volatility,
    regime_report,
    trend_slope_normalized,
)

D = Decimal


def test_realized_volatility_zero_for_flat_prices() -> None:
    assert realized_volatility([D(100), D(100), D(100)]) == D(0)


def test_realized_volatility_positive_for_moving_prices() -> None:
    vol = realized_volatility([D(100), D(110), D(100), D(110)])
    assert vol > 0


def test_trend_slope_positive_for_rising_prices() -> None:
    assert trend_slope_normalized([D(100), D(101), D(102), D(103)]) > 0


def test_trend_slope_negative_for_falling_prices() -> None:
    assert trend_slope_normalized([D(103), D(102), D(101), D(100)]) < 0


def test_trend_slope_near_zero_for_flat_prices() -> None:
    assert abs(trend_slope_normalized([D(100), D(100), D(100)])) < D("1e-9")


def _dev_prices() -> list[Decimal]:
    # Mix of calm and volatile windows so terciles are non-degenerate.
    prices: list[Decimal] = []
    value = Decimal(100)
    for i in range(60):
        step = Decimal("0.1") if i % 2 == 0 else Decimal("0.2")
        if i % 10 < 5:
            step *= 5  # volatile stretch
        value += step if (i // 3) % 2 == 0 else -step
        prices.append(value)
    return prices


def test_fit_regime_model_orders_thresholds() -> None:
    model = fit_regime_model(_dev_prices(), window=5, trend_threshold=D("0.001"))
    assert model.vol_low_threshold <= model.vol_high_threshold
    assert model.window == 5
    assert model.trend_threshold == D("0.001")


def test_classify_prefixes_are_none_until_window_filled() -> None:
    model = fit_regime_model(_dev_prices(), window=5, trend_threshold=D("0.001"))
    prices = _dev_prices()
    labels = classify_regimes(model, prices)
    assert len(labels) == len(prices)
    assert all(label is None for label in labels[:4])
    assert labels[4] is not None


def test_classify_assigns_bands_by_thresholds() -> None:
    model = fit_regime_model(_dev_prices(), window=5, trend_threshold=D("0.001"))
    # Below low threshold and flat trend -> LOW vol, SIDEWAYS trend.
    low = model.classify(model.vol_low_threshold - D("0.0001"), D(0))
    assert low.vol == VolBand.LOW
    assert low.trend == TrendBand.SIDEWAYS
    high = model.classify(model.vol_high_threshold + D("0.1"), model.trend_threshold + D("0.01"))
    assert high.vol == VolBand.HIGH
    assert high.trend == TrendBand.UP
    down = model.classify(model.vol_low_threshold, -(model.trend_threshold + D("0.01")))
    assert down.trend == TrendBand.DOWN


def test_regime_report_aggregates_net_per_regime() -> None:
    a = RegimeLabel(vol=VolBand.LOW, trend=TrendBand.UP)
    b = RegimeLabel(vol=VolBand.HIGH, trend=TrendBand.DOWN)
    labels = [a, b, a, None, b]
    nets = [D(10), D(-5), D(4), D(999), D(-1)]
    report = regime_report(labels, nets)
    assert report.distinct_regimes == 2
    by_name = {s.regime: s for s in report.per_regime}
    assert by_name[a.name].net_result_eur == D(14)
    assert by_name[a.name].trade_count == 2
    assert by_name[b.name].net_result_eur == D(-6)
    assert not report.promotion_blocked


def test_regime_report_blocks_promotion_with_single_regime() -> None:
    a = RegimeLabel(vol=VolBand.MID, trend=TrendBand.SIDEWAYS)
    report = regime_report([a, a, a], [D(1), D(2), D(3)])
    assert report.distinct_regimes == 1
    assert report.promotion_blocked


def test_regime_report_blocks_promotion_with_zero_regimes() -> None:
    report = regime_report([None, None], [D(1), D(2)])
    assert report.distinct_regimes == 0
    assert report.promotion_blocked


def test_report_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="aceeași lungime"):
        regime_report([None], [D(1), D(2)])


def test_fit_rejects_short_development_set() -> None:
    with pytest.raises(ValueError, match="prea scurt"):
        fit_regime_model([D(100), D(101)], window=5, trend_threshold=D("0.001"))
