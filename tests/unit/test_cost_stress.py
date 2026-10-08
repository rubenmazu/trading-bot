from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from qts.core.models import Instrument, Quote
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import (
    CompleteCostModel,
    CostContext,
    CostModelConfig,
    CostModelIncomplete,
    FxConfig,
    LatencyConfig,
    OrderSpec,
    SlippageConfig,
    SpreadSchedule,
    TaxConfig,
)
from qts.costs.stress import (
    BASELINE,
    STRESS_LEVELS,
    StressScenario,
    stress_scenarios,
    stressed_config,
    stressed_model,
)

T0 = datetime(2025, 1, 2, 9, 30, tzinfo=UTC)
D = Decimal


def _inst(currency: str = "USD") -> Instrument:
    return Instrument(
        symbol="XYZ",
        venue="XETR",
        asset_class="etf",
        currency=currency,
        tick_size=D("0.01"),
        qty_step=D("0.001"),
        min_qty=D("0.001"),
        calendar_id="XETR",
    )


def _config(**overrides: Any) -> CostModelConfig:
    table = CommissionTable(
        broker="sim",
        version="v1",
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        currency="EUR",
        percent=D("0.001"),
        minimum=D(1),
        maximum=D(10),
        exchange_fee_percent=D("0.0001"),
        exchange_fee_fixed=D("0.10"),
    )
    data: dict[str, Any] = {
        "version": "costs-v1",
        "commissions": CommissionSchedule(tables=(table,)),
        "spreads": {"XYZ": SpreadSchedule(by_hour_utc={9: D("0.002")}, default=D("0.004"))},
        "slippage": SlippageConfig(k=D(1), min_ticks=D(1)),
        "latency": LatencyConfig(latency_ms=1000),
        "fx": FxConfig(conversion_spread=D("0.002")),
        "taxes": TaxConfig(approved=True, rate_on_notional=D("0.0001"), reference="operator"),
    }
    data.update(overrides)
    return CostModelConfig(**data)


def _order(qty: str = "100") -> OrderSpec:
    return OrderSpec.model_validate(
        {"instrument": _inst(), "side": "BUY", "qty": qty, "ts": T0, "ref_price": "10"}
    )


def _ctx(**overrides: Any) -> CostContext:
    data: dict[str, Any] = {
        "broker": "sim",
        "fx_rate": "0.9",
        "sigma_bar": "0.03",  # latență exactă: 0.03 × sqrt(1 s / 900 s) = 0.001
        "adv": "10000",
        "bar_interval_min": 15,
    }
    data.update(overrides)
    return CostContext.model_validate(data)


def _quote() -> Quote:
    return Quote(instrument="XYZ", ts=T0, bid=D("9.99"), ask=D("10.01"))


def test_predefined_levels_and_version_suffix() -> None:
    assert [s.cost_multiplier for s in STRESS_LEVELS] == [D("1.0"), D("1.5"), D("2.0")]
    base = _config()
    versions = [stressed_config(base, s).version for s in STRESS_LEVELS]
    assert versions == ["costs-v1", "costs-v1+stress1.5", "costs-v1+stress2.0"]
    lat = StressScenario(cost_multiplier=D("1.5"), latency_ms=3000)
    assert stressed_config(base, lat).version == "costs-v1+stress1.5+lat3000ms"
    assert stressed_config(base, BASELINE) is base


@pytest.mark.parametrize("m", ["1.5", "2.0"])
def test_each_stressed_component_scales_and_taxes_do_not(m: str) -> None:
    mult = D(m)
    base_model = CompleteCostModel(_config())
    stressed = stressed_model(_config(), StressScenario(cost_multiplier=mult))
    for quote in (None, _quote()):
        b = base_model.estimate(_order(), quote, _ctx())
        s = stressed.estimate(_order(), quote, _ctx())
        assert s.spread == b.spread * mult
        assert s.commission == b.commission * mult
        assert s.slippage == b.slippage * mult
        assert s.fx_conversion == b.fx_conversion * mult
        assert s.taxes == b.taxes
        assert s.latency == b.latency  # latența se stresează separat


def test_commission_scales_at_minimum_and_maximum() -> None:
    base = _config().commissions
    stressed = stressed_config(_config(), StressScenario(cost_multiplier=D(2))).commissions
    assert base is not None and stressed is not None
    tb, ts = base.tables[0], stressed.tables[0]
    for notional in (D(10), D(5000), D(1_000_000)):  # minim, interior, maxim
        assert ts.commission(notional) == tb.commission(notional) * 2


def test_stress_latency_increases_estimated_latency_cost() -> None:
    base = CompleteCostModel(_config())
    stressed = stressed_model(_config(), StressScenario(latency_ms=4000))
    assert stressed.config.latency == LatencyConfig(latency_ms=4000)
    b = base.estimate(_order(), _quote(), _ctx())
    s = stressed.estimate(_order(), _quote(), _ctx())
    assert s.latency == b.latency * 2  # sqrt(4000/1000)
    assert s.model_copy(update={"latency": b.latency}) == b


def test_stress_latency_below_baseline_or_missing_is_rejected() -> None:
    with pytest.raises(ValueError, match="sub latența de bază"):
        stressed_config(_config(), StressScenario(latency_ms=500))
    with pytest.raises(CostModelIncomplete):
        stressed_config(_config(latency=None), StressScenario(latency_ms=2000))


@pytest.mark.parametrize("bad", ["0.99", "0", "-1", "NaN", "Infinity"])
def test_multiplier_below_one_or_non_finite_is_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        StressScenario(cost_multiplier=D(bad))


def test_float_multiplier_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StressScenario.model_validate({"cost_multiplier": 1.5})


def test_missing_components_stay_missing() -> None:
    cfg = stressed_config(_config(fx=None, slippage=None), StressScenario(cost_multiplier=D(2)))
    assert cfg.fx is None and cfg.slippage is None
    with pytest.raises(CostModelIncomplete):
        CompleteCostModel(cfg).estimate(_order(), _quote(), _ctx())


def test_totals_are_monotonic_in_multiplier() -> None:
    base = _config()
    multipliers = [D("1.0"), D("1.25"), D("1.5"), D("2.0"), D("3.0")]
    for quote in (None, _quote()):
        totals = [
            stressed_model(base, StressScenario(cost_multiplier=m))
            .estimate(_order(), quote, _ctx())
            .total
            for m in multipliers
        ]
        assert totals == sorted(totals)
        assert totals[0] < totals[-1]


def test_stress_scenarios_grid() -> None:
    grid = stress_scenarios([2000, 5000, 2000])
    assert len(grid) == 9
    assert {(s.cost_multiplier, s.latency_ms) for s in grid} == {
        (m, lat) for m in (D("1.0"), D("1.5"), D("2.0")) for lat in (None, 2000, 5000)
    }


def test_quote_spread_multiplier_below_one_rejected_in_config() -> None:
    with pytest.raises(ValidationError):
        _config(quote_spread_multiplier=D("0.5"))
