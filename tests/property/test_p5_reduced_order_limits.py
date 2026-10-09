"""P5: un ordin redus respectă toate limitele (Req 12.7).

Dacă Risk_Engine reduce cantitatea (față de cererea strategiei sau după verificarea exactă a
costurilor neliniare), ordinul aprobat cu cantitatea redusă respectă toate limitele. Testul nu
se bazează pe valorile raportate de engine: pentru fiecare aprobare recalculează independent,
cu `CompleteCostModel` pe cantitatea finală, riscul dus-întors și numerarul, apoi verifică:

- risc per tranzacție ≤ `risk_per_trade_max_eur`;
- numerar: nominal + cost intrare ≤ numerar disponibil;
- `qty` ≥ `min_qty`, multiplu de `qty_step`, nominal ≥ `min_notional`, `qty` ≤ cererea;
- pierdere zilnică + risc deschis + risc tranzacție ≤ limita zilnică;
- pierdere totală + risc deschis + risc tranzacție ≤ limita totală;
- numărul de poziții deschise nu depășește maximul.

Maximalitatea cantității nu este cerută (doar siguranța). Scenariile forțează reduceri:
cantități cerute mari, comisioane plafonate (concave) și minime ridicate, slippage √q cu ADV
mic, numerar limitat, poziții deschise și pierderi apropiate de limite.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from typing import Any

from hypothesis import event, given, settings
from hypothesis import strategies as st

from qts.config.schema import RiskConfig
from qts.core.models import Instrument, OrderIntent, Quote
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
from qts.risk import MarketSnapshot, PositionRisk, RiskContext, RiskDecision, RiskEngine

D = Decimal
T0 = datetime(2025, 1, 2, 9, 30, tzinfo=UTC)
SYMBOL = "XYZ"
OTHER_SYMBOLS = ("AAA", "BBB", "CCC", "DDD")


def _dec(lo: str, hi: str, places: int) -> st.SearchStrategy[Decimal]:
    return st.decimals(
        min_value=D(lo), max_value=D(hi), places=places, allow_nan=False, allow_infinity=False
    )


@dataclass(frozen=True)
class Scenario:
    config: RiskConfig
    cost_config: CostModelConfig
    instrument: Instrument
    snapshot: MarketSnapshot
    ctx: RiskContext
    intent: OrderIntent


# --------------------------------------------------------------------------- strategii


@st.composite
def cost_configs(draw: st.DrawFn, currency: str, *, friendly: bool = False) -> CostModelConfig:
    # `friendly`: costuri fixe mici, ca dimensionarea să nu fie respinsă din start.
    fixed_hi = "0.02" if friendly else "0.08"
    minimum = draw(
        st.sampled_from([D(0), D("0.01")]) | _dec("0", fixed_hi if friendly else "0.2", 2)
    )
    # Plafonul face comisionul concav în qty: liniarizarea subestimează costul candidatului.
    maximum = draw(st.none() | _dec("0", "0.1", 3).map(lambda d: d + minimum))
    table = CommissionTable(
        broker="sim",
        version="t1",
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        currency=draw(st.sampled_from(["EUR", currency])),
        percent=draw(st.sampled_from([D(0), D("0.05")]) | _dec("0", "0.05", 4)),
        minimum=minimum,
        maximum=maximum,
        exchange_fee_percent=draw(_dec("0", "0.001", 5)),
        exchange_fee_fixed=draw(_dec("0", fixed_hi, 3)),
    )
    return CostModelConfig(
        version="p5",
        commissions=CommissionSchedule(tables=(table,)),
        spreads={SYMBOL: SpreadSchedule(default=draw(_dec("0", "0.01", 4)))},
        # k mare și ADV mic (în CostContext): impactul √(q/ADV) devine dominant.
        slippage=SlippageConfig(
            k=draw(_dec("0", "1" if friendly else "3", 2)),
            min_ticks=draw(_dec("0", "0.5" if friendly else "2", 1)),
        ),
        latency=LatencyConfig(latency_ms=draw(st.sampled_from([0, 250, 2000]))),
        fx=FxConfig(conversion_spread=draw(_dec("0", "0.005", 4))),
        taxes=TaxConfig(
            approved=True,
            rate_on_notional=draw(_dec("0", "0.003", 4)),
            fixed_per_trade=draw(_dec("0", "0.01" if friendly else "0.03", 3)),
            reference="test",
        ),
    )


@st.composite
def risk_configs(draw: st.DrawFn) -> RiskConfig:
    max_trade = draw(_dec("0.25", "0.5", 2))
    return RiskConfig(
        risk_per_trade_target_eur=draw(_dec("0.25", str(max_trade), 2)),
        risk_per_trade_max_eur=max_trade,
        daily_loss_limit_eur=draw(_dec("0.3", "2", 2)),
        max_open_positions=draw(st.integers(1, 4)),
    )


@st.composite
def positions(draw: st.DrawFn, symbol: str) -> PositionRisk:
    mark = draw(_dec("1", "50", 2))
    stop = draw(st.none() | _dec("0.80", "0.999", 3).map(lambda f: (mark * f).quantize(D("0.01"))))
    return PositionRisk(
        instrument=symbol,
        qty=draw(_dec("0.001", "0.05", 3)),
        mark_price=mark,
        stop_price=stop,
        fx_rate_to_eur=draw(st.sampled_from([D(1), D("0.9"), D("1.1")])),
        exit_cost_eur=draw(_dec("0", "0.05", 3)),
    )


@st.composite
def scenarios(draw: st.DrawFn, *, friendly: bool = False) -> Scenario:
    """`friendly=True`: pierderi și riscuri deschise mici, astfel încât aprobările să domine."""
    currency = draw(st.sampled_from(["EUR", "USD"]))
    steps = [D("0.001"), D("0.01")] if friendly else [D("0.001"), D("0.01"), D("0.1"), D(1)]
    step = draw(st.sampled_from(steps))
    min_qty = draw(st.sampled_from([D(0), step, step * 3]))
    inst = Instrument(
        symbol=SYMBOL,
        venue="XETR",
        asset_class="etf",
        currency=currency,
        tick_size=D("0.01"),
        qty_step=step,
        min_qty=min_qty,
        min_notional=D(0) if friendly else draw(st.sampled_from([D(0), D(0), D("0.5"), D(2)])),
        calendar_id="XETR",
        fractional=step < 1,
    )
    mid = draw(_dec("0.5", "60", 2))
    half = draw(_dec("0", "0.05", 2))
    quote = None
    if mid - half > 0 and draw(st.booleans() if not friendly else st.just(True)):
        quote = Quote(instrument=SYMBOL, ts=T0, bid=mid - half, ask=mid + half)
    entry_guess = max(mid, quote.ask) if quote is not None else mid
    stop = (entry_guess * draw(_dec("0.80", "0.998", 3))).quantize(D("0.01"))
    if stop <= 0:
        stop = D("0.01")
    cost_ctx = CostContext(
        broker="sim",
        fx_rate=None if currency == "EUR" else draw(_dec("0.8", "1.2", 3)),
        sigma_bar=draw(_dec("0", "0.05", 4)),
        adv=draw(st.sampled_from([D(1), D(5), D(50)]) | _dec("1", "100000", 0)),
        bar_interval_min=15,
    )
    snapshot = MarketSnapshot(instrument=inst, data_fresh=True, quote=quote, cost_ctx=cost_ctx)

    cfg = draw(risk_configs())
    pos_symbols = draw(
        st.lists(
            st.sampled_from((SYMBOL, *OTHER_SYMBOLS)), unique=True, max_size=0 if friendly else 3
        )
    )
    pos = {s: draw(positions(s)) for s in pos_symbols}
    if friendly:
        daily, total = D(0), D(0)
    else:
        daily = draw(
            st.sampled_from([D(0), -cfg.daily_loss_limit_eur + D("0.3")]) | _dec("-2", "1", 2)
        )
        total = draw(st.sampled_from([D(0), D("-9.6")]) | _dec("-10", "1", 2))
    ctx = RiskContext(
        ts=T0,
        mode="backtest",
        cash_eur=draw(st.sampled_from([D(100)]) | _dec("0.2", "150", 2)),
        positions=pos,
        daily_realized_pnl_eur=daily,
        total_pnl_eur=total,
        market={SYMBOL: snapshot},
    )
    big = st.sampled_from([D(1000), D(50), D(5)])
    requested = draw(st.none() | big if friendly else st.none() | big | _dec("0.001", "1000", 3))
    intent = OrderIntent(
        intent_id="i1",
        signal_id="s1",
        instrument=SYMBOL,
        side="BUY",
        ref_price=mid,
        stop_price=stop,
        requested_qty=requested,
    )
    return Scenario(
        config=cfg,
        cost_config=draw(cost_configs(currency, friendly=friendly)),
        instrument=inst,
        snapshot=snapshot,
        ctx=ctx,
        intent=intent,
    )


# ------------------------------------------------------------------ verificare independentă


def _exit_quote(quote: Quote | None, stop: Decimal) -> Quote | None:
    if quote is None:
        return None
    half = (quote.ask - quote.bid) / 2
    if stop - half <= 0:
        return None
    return Quote(instrument=quote.instrument, ts=quote.ts, bid=stop - half, ask=stop + half)


def _leg_cost(
    model: CompleteCostModel,
    sc: Scenario,
    side: str,
    qty: Decimal,
    price: Decimal,
    quote: Quote | None,
) -> Decimal:
    spec = OrderSpec.model_validate(
        {"instrument": sc.instrument, "side": side, "qty": qty, "ts": T0, "ref_price": price}
    )
    # Un cost estimat negativ nu reduce riscul (fail-closed).
    return max(model.estimate(spec, quote, sc.snapshot.cost_ctx).total, D(0))


def assert_all_limits(sc: Scenario, decision: RiskDecision) -> None:
    """Recalculează independent riscul pe cantitatea finală și verifică toate limitele."""
    assert decision.approved and decision.qty is not None
    qty = decision.qty
    inst, cfg, ctx, intent = sc.instrument, sc.config, sc.ctx, sc.intent
    assert intent.stop_price is not None
    model = CompleteCostModel(sc.cost_config)

    quote = sc.snapshot.quote
    entry = max(intent.ref_price, quote.ask) if quote is not None else intent.ref_price
    stop = intent.stop_price
    fx = D(1) if inst.currency == "EUR" else sc.snapshot.cost_ctx.fx_rate
    assert fx is not None

    with localcontext() as lc:
        lc.prec = 60
        entry_cost = _leg_cost(model, sc, "BUY", qty, entry, quote)
        exit_cost = _leg_cost(model, sc, "SELL", qty, stop, _exit_quote(quote, stop))
        trade_risk = qty * (entry - stop) * fx + entry_cost + exit_cost
        cash_required = qty * entry * fx + entry_cost
        open_risk = sum(
            (
                max(D(0), p.mark_price - (p.stop_price or D(0))) * p.qty * p.fx_rate_to_eur
                + p.exit_cost_eur
                for p in ctx.positions.values()
            ),
            D(0),
        )
        daily_loss = max(D(0), -(ctx.daily_realized_pnl_eur + ctx.daily_unrealized_pnl_eur))
        total_loss = max(D(0), -ctx.total_pnl_eur)

        # Cantitate: minim, pas, nominal minim, nu peste cerere.
        assert qty > 0
        assert qty >= inst.min_qty
        assert qty % inst.qty_step == 0
        assert qty * entry >= inst.min_notional
        if intent.requested_qty is not None:
            assert qty <= intent.requested_qty
        # Per tranzacție (13.2, 13.9) și numerar după costuri.
        assert trade_risk <= cfg.risk_per_trade_max_eur, (trade_risk, qty)
        assert cash_required <= ctx.cash_eur, (cash_required, qty)
        # Zilnic și total, cu riscul deschis cel mai defavorabil (13.3-13.5).
        assert daily_loss + open_risk + trade_risk <= cfg.daily_loss_limit_eur
        assert total_loss + open_risk + trade_risk <= cfg.total_loss_limit_eur
        # Expunere agregată: o poziție nouă nu depășește numărul maxim.
        if intent.instrument not in ctx.positions:
            assert len(ctx.positions) + 1 <= cfg.max_open_positions

    # Valorile raportate descriu cantitatea finală (12.3, 12.7).
    assert decision.metrics["qty"] == qty


def _classify(decision: RiskDecision) -> str:
    if not decision.approved:
        return "rejected"
    if not decision.reduced:
        return "approved"
    candidate = decision.metrics.get("candidate_qty")
    assert decision.qty is not None
    if candidate is not None and decision.qty < candidate:
        return "reduced: exact-cost"
    return "reduced: requested"


def _evaluate(sc: Scenario) -> RiskDecision:
    return RiskEngine(sc.config, CompleteCostModel(sc.cost_config)).evaluate(sc.intent, sc.ctx)


# --------------------------------------------------------------------------- proprietăți


@given(st.one_of(scenarios(), scenarios(friendly=True)))
def test_property_5_reduced_order_satisfies_all_limits(sc: Scenario) -> None:
    """**Validates: Requirements 12.7**"""
    decision = _evaluate(sc)
    kind = _classify(decision)
    event(kind)
    if decision.approved:
        # Orice aprobare (redusă sau nu) respectă toate limitele, pe cantitatea finală.
        assert_all_limits(sc, decision)


def test_property_5_reductions_are_exercised() -> None:
    """**Validates: Requirements 12.7**

    Non-vacuitate: în scenarii favorabile aprobării, reducerile apar efectiv (atât prin
    plafonarea cererii, cât și prin verificarea exactă a costurilor) și respectă limitele.
    """
    stats: Counter[str] = Counter()

    # Buget de exemple fix pentru acoperire: pragul de non-vacuitate (≥ 20 de reduceri) nu trebuie
    # să depindă de profilul hypothesis activ (`fast`/`ci`/`deep`), care schimbă implicitul.
    @settings(database=None, max_examples=250, deadline=None)
    @given(scenarios(friendly=True))
    def run(sc: Scenario) -> None:
        decision = _evaluate(sc)
        kind = _classify(decision)
        stats[kind] += 1
        event(kind)
        if decision.approved:
            assert_all_limits(sc, decision)

    run()
    reduced = stats["reduced: requested"] + stats["reduced: exact-cost"]
    assert reduced >= 20, stats
    assert stats["reduced: exact-cost"] >= 1, stats


# --------------------------------------------------------------------------- exemplu determinist


def _sanity_scenario(table: dict[str, Any], requested: Decimal | None) -> Scenario:
    inst = Instrument(
        symbol=SYMBOL,
        venue="XETR",
        asset_class="etf",
        currency="EUR",
        tick_size=D("0.01"),
        qty_step=D("0.001"),
        min_qty=D("0.001"),
        calendar_id="XETR",
        fractional=True,
    )
    cost_config = CostModelConfig(
        version="p5-sanity",
        commissions=CommissionSchedule(
            tables=(
                CommissionTable.model_validate(
                    {
                        "broker": "sim",
                        "version": "v1",
                        "valid_from": datetime(2024, 1, 1, tzinfo=UTC),
                        "currency": "EUR",
                        "percent": "0",
                        "exchange_fee_fixed": "0.05",
                        **table,
                    }
                ),
            )
        ),
        slippage=SlippageConfig(k=D(0)),
        latency=LatencyConfig(latency_ms=0),
        fx=FxConfig(conversion_spread=D(0)),
        taxes=TaxConfig(approved=True, reference="test"),
    )
    snapshot = MarketSnapshot(
        instrument=inst,
        data_fresh=True,
        quote=Quote(instrument=SYMBOL, ts=T0, bid=D("9.99"), ask=D("10.01")),
        cost_ctx=CostContext(broker="sim", sigma_bar=D("0.01"), adv=D(1000)),
    )
    ctx = RiskContext(ts=T0, mode="backtest", cash_eur=D(100), market={SYMBOL: snapshot})
    intent = OrderIntent(
        intent_id="i1",
        signal_id="s1",
        instrument=SYMBOL,
        side="BUY",
        ref_price=D(10),
        stop_price=D("9.5"),
        requested_qty=requested,
    )
    return Scenario(RiskConfig(), cost_config, inst, snapshot, ctx, intent)


def test_property_5_sanity_examples_are_reduced_and_safe() -> None:
    """**Validates: Requirements 12.7**"""
    # Cererea strategiei (5 unități) depășește bugetul: redusă la 0.283.
    sc = _sanity_scenario({}, D(5))
    d = _evaluate(sc)
    assert d.approved and d.reduced and d.qty == D("0.283")
    assert _classify(d) == "reduced: requested"
    assert_all_limits(sc, d)
    # Comision plafonat: liniarizarea subestimează costul, căutarea exactă reduce cantitatea.
    sc = _sanity_scenario({"percent": "0.05", "maximum": "0.02"}, None)
    d = _evaluate(sc)
    assert d.approved and d.reduced and d.qty == D("0.207")
    assert _classify(d) == "reduced: exact-cost"
    assert_all_limits(sc, d)
