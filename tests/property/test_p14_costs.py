"""P14: costurile scad rezultatul (Req 8.4, 20.3).

Pentru orice set de tranzacții, rezultatul net (brut − Σ costuri estimate) ≤ rezultatul brut,
iar multiplicarea costurilor (×1,0 ≤ m1 ≤ m2, inclusiv ×1,5 și ×2,0) și creșterea latenței de
stres nu cresc rezultatul net. Costurile sunt cele estimate de `CompleteCostModel.estimate`;
latența folosește estimarea 1σ (fără `price_after_latency`), deci este nenegativă.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from qts.core.models import CostBreakdown, Instrument, Quote, Side
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import (
    CompleteCostModel,
    CostContext,
    CostModelConfig,
    FxConfig,
    LatencyConfig,
    OrderSpec,
    SlippageConfig,
    SpreadSchedule,
    TaxConfig,
)
from qts.costs.stress import STRESS_LEVELS, STRESS_MULTIPLIERS, StressScenario, stressed_model

BROKER = "sim"
VENUE = "XETR"
SYMBOLS = {"EUR": "EURX", "USD": "USDX", "GBP": "GBPX"}


def _dec(lo: str, hi: str, places: int) -> st.SearchStrategy[Decimal]:
    return st.decimals(
        min_value=Decimal(lo),
        max_value=Decimal(hi),
        places=places,
        allow_nan=False,
        allow_infinity=False,
    )


def _instrument(currency: str, tick: Decimal) -> Instrument:
    return Instrument(
        symbol=SYMBOLS[currency],
        venue=VENUE,
        asset_class="etf",
        currency=currency,
        tick_size=tick,
        qty_step=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        calendar_id=VENUE,
        fractional=True,
    )


@dataclass(frozen=True)
class TradeCase:
    order: OrderSpec
    quote: Quote | None
    ctx: CostContext
    gross: Decimal  # rezultatul brut al tranzacției, EUR (poate fi negativ)


@st.composite
def commission_tables(draw: st.DrawFn, currency: str) -> CommissionTable:
    minimum = draw(_dec("0", "5", 2))
    maximum = draw(st.none() | _dec("0", "50", 2).map(lambda d: d + minimum))
    return CommissionTable(
        broker=BROKER,
        version="t1",
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        currency=currency,
        venue=draw(st.sampled_from([None, VENUE])),
        percent=draw(_dec("0", "0.01", 5)),
        minimum=minimum,
        maximum=maximum,
        exchange_fee_percent=draw(_dec("0", "0.001", 6)),
        exchange_fee_fixed=draw(_dec("0", "1", 2)),
    )


@st.composite
def base_configs(draw: st.DrawFn) -> CostModelConfig:
    # Un singur tabel de comision, în EUR sau USD; instrumentele generate sunt doar cele
    # compatibile cu moneda tabelului (vezi `_allowed_currencies`).
    table_ccy = draw(st.sampled_from(["EUR", "USD"]))
    spreads = {
        sym: SpreadSchedule(
            by_hour_utc=draw(st.dictionaries(st.integers(0, 23), _dec("0", "0.01", 5), max_size=4)),
            default=draw(_dec("0", "0.01", 5)),
        )
        for sym in SYMBOLS.values()
    }
    return CostModelConfig(
        version="base",
        commissions=CommissionSchedule(tables=(draw(commission_tables(table_ccy)),)),
        spreads=spreads,
        slippage=SlippageConfig(k=draw(_dec("0", "2", 3)), min_ticks=draw(_dec("0", "3", 1))),
        latency=LatencyConfig(latency_ms=draw(st.integers(0, 2000))),
        fx=FxConfig(conversion_spread=draw(_dec("0", "0.005", 5))),
        taxes=TaxConfig(
            approved=True,
            rate_on_notional=draw(_dec("0", "0.003", 5)),
            fixed_per_trade=draw(_dec("0", "1", 2)),
            sides=draw(st.sampled_from([("BUY", "SELL"), ("BUY",), ("SELL",), ()])),
            reference="test",
        ),
    )


def _allowed_currencies(cfg: CostModelConfig) -> list[str]:
    assert cfg.commissions is not None
    table_ccy = cfg.commissions.tables[0].currency
    # Tabelul se aplică instrumentelor în moneda sa sau, dacă este în EUR, oricărui instrument.
    return list(SYMBOLS) if table_ccy == "EUR" else [table_ccy]


@st.composite
def trade_cases(draw: st.DrawFn, currencies: list[str]) -> TradeCase:
    currency = draw(st.sampled_from(currencies))
    inst = _instrument(currency, draw(st.sampled_from([Decimal("0.01"), Decimal("0.001")])))
    side: Side = draw(st.sampled_from(["BUY", "SELL"]))
    ts = datetime(2026, 3, 2, draw(st.integers(0, 23)), draw(st.integers(0, 59)), tzinfo=UTC)
    ref_price = draw(_dec("0.5", "2000", 3))
    quote = None
    if draw(st.booleans()):
        bid = draw(_dec("0.5", "2000", 3))
        quote = Quote(instrument=inst.symbol, ts=ts, bid=bid, ask=bid + draw(_dec("0", "5", 3)))
    ctx = CostContext(
        broker=BROKER,
        fx_rate=None if currency == "EUR" else draw(_dec("0.5", "1.5", 4)),
        sigma_bar=draw(_dec("0", "0.05", 5)),
        adv=draw(_dec("1", "1000000", 0)),
        bar_interval_min=draw(st.sampled_from([5, 15, 30, 60])),
    )
    order = OrderSpec(
        instrument=inst, side=side, qty=draw(_dec("0.001", "500", 3)), ts=ts, ref_price=ref_price
    )
    return TradeCase(order=order, quote=quote, ctx=ctx, gross=draw(_dec("-500", "500", 4)))


@st.composite
def scenarios_and_trades(
    draw: st.DrawFn,
) -> tuple[CostModelConfig, list[TradeCase], StressScenario, StressScenario]:
    cfg = draw(base_configs())
    trades = draw(st.lists(trade_cases(_allowed_currencies(cfg)), min_size=0, max_size=8))
    mult = st.sampled_from(STRESS_MULTIPLIERS) | _dec("1", "4", 2)
    m1, m2 = sorted((draw(mult), draw(mult)))
    assert cfg.latency is not None
    base_lat = cfg.latency.latency_ms
    lat1 = base_lat + draw(st.integers(0, 3000))
    lat2 = lat1 + draw(st.integers(0, 3000))
    # `None` = latența de bază; echivalent cu lat1 doar când lat1 == latența de bază.
    s1_lat = draw(st.sampled_from([None, lat1])) if lat1 == base_lat else lat1
    s1 = StressScenario(cost_multiplier=m1, latency_ms=s1_lat)
    s2 = StressScenario(cost_multiplier=m2, latency_ms=lat2)
    return cfg, trades, s1, s2


def _costs(model: CompleteCostModel, trades: list[TradeCase]) -> list[CostBreakdown]:
    return [model.estimate(t.order, t.quote, t.ctx) for t in trades]


def _net(trades: list[TradeCase], costs: list[CostBreakdown]) -> Decimal:
    return sum((t.gross - c.total for t, c in zip(trades, costs, strict=True)), Decimal(0))


def _components(c: CostBreakdown) -> list[Decimal]:
    return [c.spread, c.commission, c.slippage, c.latency, c.fx_conversion, c.taxes]


@given(scenarios_and_trades())
def test_property_14_net_le_gross_and_monotone_in_multiplier(
    case: tuple[CostModelConfig, list[TradeCase], StressScenario, StressScenario],
) -> None:
    """**Validates: Requirements 8.4, 20.3**"""
    cfg, trades, s1, s2 = case
    gross = sum((t.gross for t in trades), Decimal(0))

    base_costs = _costs(CompleteCostModel(cfg), trades)
    costs1 = _costs(stressed_model(cfg, s1), trades)
    costs2 = _costs(stressed_model(cfg, s2), trades)

    # Fiecare componentă estimată este nenegativă și nu scade sub stres mai puternic.
    for b, c1, c2 in zip(base_costs, costs1, costs2, strict=True):
        for vb, v1, v2 in zip(_components(b), _components(c1), _components(c2), strict=True):
            assert vb >= 0
            assert vb <= v1 <= v2

    net_base, net1, net2 = _net(trades, base_costs), _net(trades, costs1), _net(trades, costs2)
    assert net_base <= gross
    assert net2 <= net1 <= net_base


@given(st.data())
def test_property_14_stress_levels_do_not_increase_net(data: st.DataObject) -> None:
    """**Validates: Requirements 8.4, 20.3**"""
    cfg = data.draw(base_configs())
    trades = data.draw(st.lists(trade_cases(_allowed_currencies(cfg)), min_size=1, max_size=8))
    gross = sum((t.gross for t in trades), Decimal(0))
    nets = [_net(trades, _costs(stressed_model(cfg, s), trades)) for s in STRESS_LEVELS]
    assert [s.cost_multiplier for s in STRESS_LEVELS] == list(STRESS_MULTIPLIERS)
    assert nets[0] <= gross
    assert nets[2] <= nets[1] <= nets[0]
