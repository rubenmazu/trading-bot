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
    CostModel,
    CostModelConfig,
    CostModelIncomplete,
    Fill,
    FxConfig,
    LatencyConfig,
    OrderSpec,
    SlippageConfig,
    SpreadSchedule,
    TaxConfig,
)

T0 = datetime(2025, 1, 2, 9, 30, tzinfo=UTC)
D = Decimal


def _inst(currency: str = "EUR", symbol: str = "XYZ") -> Instrument:
    return Instrument(
        symbol=symbol,
        venue="XETR",
        asset_class="etf",
        currency=currency,
        tick_size=D("0.01"),
        qty_step=D("0.001"),
        min_qty=D("0.001"),
        calendar_id="XETR",
    )


def _table(**overrides: Any) -> CommissionTable:
    data: dict[str, Any] = {
        "broker": "sim",
        "version": "v1",
        "valid_from": datetime(2024, 1, 1, tzinfo=UTC),
        "currency": "EUR",
        "percent": "0.001",
        "minimum": "1",
        "maximum": "10",
        "exchange_fee_fixed": "0.10",
    }
    data.update(overrides)
    return CommissionTable.model_validate(data)


def _config(**overrides: Any) -> CostModelConfig:
    data: dict[str, Any] = {
        "version": "costs-v1",
        "commissions": CommissionSchedule(tables=(_table(),)),
        "spreads": {"XYZ": SpreadSchedule(by_hour_utc={9: D("0.002")})},
        "slippage": SlippageConfig(k=D(1), min_ticks=D(1)),
        "latency": LatencyConfig(latency_ms=1000),
        "fx": FxConfig(conversion_spread=D("0.002")),
        "taxes": TaxConfig(approved=True, rate_on_notional=D("0.0001"), reference="operator"),
    }
    data.update(overrides)
    return CostModelConfig(**data)


def _order(side: str = "BUY", qty: str = "100", currency: str = "EUR", **kw: Any) -> OrderSpec:
    return OrderSpec.model_validate(
        {"instrument": _inst(currency), "side": side, "qty": qty, "ts": T0, "ref_price": "10", **kw}
    )


def _quote(bid: str = "9.99", ask: str = "10.01") -> Quote:
    return Quote(instrument="XYZ", ts=T0, bid=D(bid), ask=D(ask))


def _ctx(**overrides: Any) -> CostContext:
    data: dict[str, Any] = {
        "broker": "sim",
        "sigma_bar": "0.01",
        "adv": "10000",
        "bar_interval_min": 15,
        "price_after_latency": "10.02",
    }
    data.update(overrides)
    return CostContext.model_validate(data)


def test_estimate_breaks_down_each_component() -> None:
    model: CostModel = CompleteCostModel(_config())
    cost = model.estimate(_order(), _quote(), _ctx())
    assert model.version == "costs-v1"
    assert cost.spread == D("1.00")  # 0.01 jumătate de spread × 100
    assert cost.commission == D("1.10")  # max(1000 × 0.001, 1) + 0.10
    assert cost.slippage == D("2.00")  # (1 × 0.01 × sqrt(0.01) × 10 + 0.01) × 100
    assert cost.latency == D("2.00")  # (10.02 − 10) × 100
    assert cost.fx_conversion == D(0)
    assert cost.taxes == D("0.10")
    assert cost.total == D("6.20")


def test_estimate_is_deterministic() -> None:
    model = CompleteCostModel(_config())
    assert model.estimate(_order(), _quote(), _ctx()) == model.estimate(_order(), _quote(), _ctx())


def test_sell_latency_sign_is_adverse_when_price_falls() -> None:
    cost = CompleteCostModel(_config()).estimate(
        _order(side="SELL"), _quote(), _ctx(price_after_latency="9.98")
    )
    assert cost.latency == D("2.00")
    favorable = CompleteCostModel(_config()).estimate(
        _order(side="SELL"), _quote(), _ctx(price_after_latency="10.02")
    )
    assert favorable.latency == D("-2.00")


@pytest.mark.parametrize(
    ("qty", "expected"),
    [("10", "1.10"), ("100000", "10.10")],  # minim 1, maxim 10, plus taxa de bursă fixă
)
def test_commission_min_and_max(qty: str, expected: str) -> None:
    cost = CompleteCostModel(_config()).estimate(_order(qty=qty), _quote(), _ctx(adv="100000000"))
    assert cost.commission == D(expected)


def test_commission_table_versioning_and_venue_priority() -> None:
    newer = _table(version="v2", valid_from=datetime(2025, 1, 2, tzinfo=UTC), percent="0.002")
    future = _table(version="v3", valid_from=datetime(2026, 1, 1, tzinfo=UTC), percent="0.009")
    venue = _table(version="v-xetr", venue="XETR", percent="0.003", maximum=None)
    schedule = CommissionSchedule(tables=(_table(), newer, future))
    assert schedule.lookup("sim", "XETR", T0).version == "v2"
    with_venue = CommissionSchedule(tables=(_table(), newer, venue))
    assert with_venue.lookup("sim", "XETR", T0).version == "v-xetr"
    assert with_venue.lookup("sim", "XLON", T0).version == "v2"
    cost = CompleteCostModel(_config(commissions=schedule)).estimate(_order(), _quote(), _ctx())
    assert cost.commission == D("2.10")
    with pytest.raises(CostModelIncomplete):
        schedule.lookup("other", "XETR", T0)


def test_commission_table_validation() -> None:
    with pytest.raises(ValidationError):
        _table(minimum="5", maximum="1")
    with pytest.raises(ValidationError):
        _table(percent="-0.001")
    with pytest.raises(ValidationError):
        CommissionSchedule(tables=(_table(), _table()))


def test_configured_spread_used_without_quote() -> None:
    model = CompleteCostModel(_config())
    assert model.estimate(_order(), None, _ctx()).spread == D("1.00")  # 0.002 × 10 / 2 × 100
    late = _order(ts=datetime(2025, 1, 2, 10, 0, tzinfo=UTC))
    with pytest.raises(CostModelIncomplete) as exc:
        model.estimate(late, None, _ctx())
    assert exc.value.component == "spread"


def test_fx_conversion_for_non_eur_instrument() -> None:
    quote = Quote(instrument="XYZ", ts=T0, bid=D("99.95"), ask=D("100.05"))
    order = _order(qty="10", currency="USD")
    cost = CompleteCostModel(_config()).estimate(
        order, quote, _ctx(fx_rate="0.9", price_after_latency="100")
    )
    assert cost.spread == D("0.45")  # 0.05 × 10 × 0.9
    assert cost.fx_conversion == D("1.80")  # 1000 USD × 0.9 × 0.002
    assert cost.commission == D("1.10")  # tabel în EUR: max(900 × 0.001, 1) + 0.10
    with pytest.raises(CostModelIncomplete) as exc:
        CompleteCostModel(_config()).estimate(order, quote, _ctx(price_after_latency="100"))
    assert exc.value.component == "fx_conversion"
    with pytest.raises(CostModelIncomplete):
        CompleteCostModel(_config(fx=None)).estimate(order, quote, _ctx(fx_rate="0.9"))


def test_latency_estimate_without_next_price_uses_sigma() -> None:
    model = CompleteCostModel(_config(latency=LatencyConfig(latency_ms=225_000)))
    cost = model.estimate(_order(), _quote(), _ctx(price_after_latency=None))
    assert cost.latency == D("5.00")  # 0.01 × sqrt(225 s / 900 s) × 10 × 100
    zero = CompleteCostModel(_config(latency=LatencyConfig(latency_ms=0)))
    assert zero.estimate(_order(), _quote(), _ctx(price_after_latency=None)).latency == D(0)


@pytest.mark.parametrize(
    ("config_overrides", "ctx_overrides", "component"),
    [
        ({"commissions": None}, {}, "commission"),
        ({"slippage": None}, {}, "slippage"),
        ({"latency": None}, {}, "latency"),
        ({"taxes": None}, {}, "taxes"),
        ({"taxes": TaxConfig(approved=False)}, {}, "taxes"),
        ({}, {"sigma_bar": None}, "slippage"),
        ({}, {"adv": None}, "slippage"),
        ({}, {"price_after_latency": None, "bar_interval_min": None}, "latency"),
    ],
)
def test_missing_component_raises_incomplete(
    config_overrides: dict[str, Any], ctx_overrides: dict[str, Any], component: str
) -> None:
    model = CompleteCostModel(_config(**config_overrides))
    with pytest.raises(CostModelIncomplete) as exc:
        model.estimate(_order(), _quote(), _ctx(**ctx_overrides))
    assert exc.value.component == component


def test_taxes_apply_only_to_configured_sides() -> None:
    taxes = TaxConfig(approved=True, rate_on_notional=D("0.001"), sides=("SELL",))
    model = CompleteCostModel(_config(taxes=taxes))
    assert model.estimate(_order(), _quote(), _ctx()).taxes == D(0)
    assert model.estimate(_order(side="SELL"), _quote(), _ctx()).taxes == D("1.00")


def test_invalid_quote_rejected() -> None:
    model = CompleteCostModel(_config())
    with pytest.raises(ValueError, match="cotație invalidă"):
        model.estimate(_order(), _quote(bid="10.02", ask="10.01"), _ctx())
    other = Quote(instrument="ABC", ts=T0, bid=D("9.99"), ask=D("10.01"))
    with pytest.raises(ValueError, match="nu aparține"):
        model.estimate(_order(), other, _ctx())


def test_realize_decomposes_shortfall() -> None:
    fill = Fill(instrument=_inst(), side="BUY", qty=D(100), price=D("10.05"), ts=T0)
    model = CompleteCostModel(_config())
    cost = model.realize(fill, _quote(), _ctx())
    assert cost.spread == D("1.00")
    assert cost.latency == D("2.00")
    assert cost.slippage == D("2.00")  # 5.00 shortfall − spread − latență
    assert cost.commission == D("1.105")  # din tabel: 1005 × 0.001 + 0.10
    assert cost.taxes == D("0.1005")
    reported = fill.model_copy(update={"commission": D(2)})
    assert model.realize(reported, _quote(), _ctx()).commission == D("2.00")


def test_floats_rejected_in_cost_inputs() -> None:
    with pytest.raises(ValidationError):
        CostContext.model_validate({"broker": "sim", "sigma_bar": 0.01})
