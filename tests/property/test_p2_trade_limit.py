"""P2: orice ordin aprobat respectă limita per tranzacție și numerarul (Req 13.2, 13.8, 13.9).

Pentru orice `Order_Intent` BUY aprobat de `RiskEngine`:

    qty × (entry − stop) × fx + cost_intrare + cost_ieșire_la_stop ≤ risk_per_trade_max_eur ≤ 0,50
    qty × entry × fx + cost_intrare ≤ numerar

Costurile sunt recalculate independent cu `CompleteCostModel.estimate` pe ambele picioare
(intrare la prețul de intrare al engine-ului, ieșire la stop), nu preluate din
`decision.metrics`. Costurile negative (latență favorabilă) sunt tratate ca zero, ceea ce face
verificarea mai strictă decât suma brută. Cantitatea aprobată trebuie să fie multiplu de
`qty_step`, cel puțin `min_qty` și cel mult cantitatea cerută.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, localcontext

from hypothesis import event, given, settings
from hypothesis import strategies as st

from qts.config.schema import MAX_RISK_PER_TRADE_EUR, RiskConfig
from qts.core.models import Instrument, OrderIntent, Quote
from qts.core.money import ZERO
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
BROKER = "sim"
VENUE = "XETR"
SYMBOL = "TGT"
T0 = datetime(2026, 3, 2, 10, 15, tzinfo=UTC)


def _dec(lo: str, hi: str, places: int) -> st.SearchStrategy[Decimal]:
    return st.decimals(
        min_value=D(lo), max_value=D(hi), places=places, allow_nan=False, allow_infinity=False
    )


# Ramurile de respingere rămân acoperite, dar rar, ca majoritatea cazurilor să fie aprobabile.
_rarely_false = st.sampled_from([True] * 9 + [False])


@dataclass(frozen=True)
class Case:
    config: RiskConfig
    cost_config: CostModelConfig
    intent: OrderIntent
    ctx: RiskContext

    @property
    def snapshot(self) -> MarketSnapshot:
        return self.ctx.market[SYMBOL]


# --------------------------------------------------------------------------- generatori


@st.composite
def instruments(draw: st.DrawFn) -> Instrument:
    step = draw(st.sampled_from([D("0.001"), D("0.001"), D("0.01"), D("0.01"), D("0.1"), D("1")]))
    min_units = draw(st.sampled_from([0, 1, 1, 2, 5]))
    return Instrument(
        symbol=SYMBOL,
        venue=VENUE,
        asset_class=draw(st.sampled_from(["etf", "stock", "index_etp"])),
        currency=draw(st.sampled_from(["EUR", "EUR", "USD", "GBP"])),
        tick_size=draw(st.sampled_from([D("0.001"), D("0.01"), D("0.05")])),
        qty_step=step,
        min_qty=step * min_units,
        min_notional=draw(st.sampled_from([ZERO, ZERO, ZERO, D("0.5"), D("1"), D("5")])),
        fractional=step < 1,
        calendar_id=VENUE,
    )


@st.composite
def cost_configs(draw: st.DrawFn, inst: Instrument) -> CostModelConfig:
    minimum = draw(st.sampled_from([ZERO, ZERO, D("0.01"), D("0.05"), D("0.1"), D("0.3")]))
    maximum = draw(st.none() | _dec("0", "1", 2).map(lambda d: d + minimum))
    table = CommissionTable(
        broker=BROKER,
        version="t1",
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        currency=draw(st.sampled_from(["EUR", inst.currency])),
        venue=draw(st.sampled_from([None, VENUE])),
        percent=draw(_dec("0", "0.005", 5)),
        minimum=minimum,
        maximum=maximum,
        exchange_fee_percent=draw(_dec("0", "0.0005", 6)),
        exchange_fee_fixed=draw(st.sampled_from([ZERO, ZERO, D("0.01"), D("0.05"), D("0.2")])),
    )
    spreads: dict[str, SpreadSchedule] = {}
    if draw(_rarely_false):  # rareori lipsește: fără cotație → cost incomplet
        spreads[SYMBOL] = SpreadSchedule(
            by_hour_utc=draw(st.dictionaries(st.integers(0, 23), _dec("0", "0.01", 5), max_size=3)),
            default=draw(_dec("0", "0.01", 5)),
        )
    return CostModelConfig(
        version="costs-p2",
        commissions=CommissionSchedule(tables=(table,)),
        spreads=spreads,
        slippage=SlippageConfig(k=draw(_dec("0", "1", 3)), min_ticks=draw(_dec("0", "2", 1))),
        latency=LatencyConfig(latency_ms=draw(st.integers(0, 2000))),
        fx=FxConfig(conversion_spread=draw(_dec("0", "0.005", 5))),
        taxes=TaxConfig(
            approved=True,
            rate_on_notional=draw(_dec("0", "0.002", 5)),
            fixed_per_trade=draw(st.sampled_from([ZERO, ZERO, D("0.01"), D("0.1")])),
            sides=draw(st.sampled_from([("BUY", "SELL"), ("BUY",), ("SELL",), ()])),
            reference="test",
        ),
    )


@st.composite
def risk_configs(draw: st.DrawFn) -> RiskConfig:
    hard = draw(st.sampled_from([MAX_RISK_PER_TRADE_EUR, D("0.40"), D("0.30")]))
    target = draw(_dec("0.25", str(hard), 2))
    return RiskConfig(
        risk_per_trade_target_eur=target,
        risk_per_trade_max_eur=hard,
        daily_loss_limit_eur=draw(st.sampled_from([D("2"), D("1.5"), D("1")])),
        max_open_positions=draw(st.sampled_from([1, 2, 3, 3, 4, 4, 4])),
    )


@st.composite
def positions(draw: st.DrawFn) -> dict[str, PositionRisk]:
    symbols = draw(st.lists(st.sampled_from(["P1", "P2", "P3", SYMBOL]), unique=True, max_size=2))
    out: dict[str, PositionRisk] = {}
    for sym in symbols:
        mark = draw(_dec("1", "100", 2))
        out[sym] = PositionRisk(
            instrument=sym,
            qty=draw(_dec("0.001", "0.5", 3)),
            mark_price=mark,
            stop_price=mark * draw(_dec("0.98", "0.999", 3)),
            fx_rate_to_eur=draw(st.sampled_from([D(1), D("0.9")])),
            exit_cost_eur=draw(_dec("0", "0.05", 3)),
        )
    return out


@st.composite
def cases(draw: st.DrawFn) -> Case:
    inst = draw(instruments())
    cost_cfg = draw(cost_configs(inst))
    risk_cfg = draw(risk_configs())

    ref = draw(_dec("1", "200", 2))
    quote = None
    if draw(st.booleans()):
        bid = ref * (1 - draw(_dec("0", "0.01", 4)))
        quote = Quote(
            instrument=SYMBOL, ts=T0, bid=bid, ask=bid + ref * draw(_dec("0", "0.005", 4))
        )
    stop = ref * (1 - draw(_dec("0.001", "0.2", 4)))
    order_type = draw(st.sampled_from(["MARKET", "MARKET", "LIMIT"]))
    limit_price = ref * (1 + draw(_dec("-0.005", "0.01", 4))) if order_type == "LIMIT" else None
    requested = draw(st.none() | st.none() | _dec("0.001", "50", 3))
    intent = OrderIntent(
        intent_id="i1",
        signal_id="s1",
        instrument=SYMBOL,
        side="BUY",
        ref_price=ref,
        stop_price=stop,
        order_type=order_type,
        limit_price=limit_price,
        requested_qty=requested,
    )

    # Pentru EUR cursul este ignorat; pentru alte monede lipsește rar (→ respingere FX).
    fx = draw(_dec("0.5", "1.5", 4)) if draw(_rarely_false) else None
    pal = None
    if draw(st.integers(0, 3)) == 0:
        pal = ref * (1 + draw(_dec("-0.01", "0.01", 4)))
    cost_ctx = CostContext(
        broker=BROKER,
        fx_rate=fx,
        sigma_bar=draw(_dec("0", "0.02", 5)),
        adv=draw(_dec("100", "1000000", 0)),
        bar_interval_min=draw(st.sampled_from([5, 15, 30, 60])),
        price_after_latency=pal,
    )
    snapshot = MarketSnapshot(instrument=inst, data_fresh=True, quote=quote, cost_ctx=cost_ctx)
    ctx = RiskContext(
        ts=T0,
        mode=draw(st.sampled_from(["backtest", "shadow", "demo"])),
        cash_eur=draw(_dec("20", "200", 2) if draw(_rarely_false) else _dec("0.5", "20", 2)),
        positions=draw(positions()),
        daily_realized_pnl_eur=draw(_dec("-0.8", "1", 2)),
        daily_unrealized_pnl_eur=draw(_dec("-0.3", "0.3", 2)),
        total_pnl_eur=draw(_dec("-5", "5", 2)),
        market={SYMBOL: snapshot},
    )
    return Case(config=risk_cfg, cost_config=cost_cfg, intent=intent, ctx=ctx)


# --------------------------------------------------------------------------- recalcul independent


def _entry_price(intent: OrderIntent, quote: Quote | None) -> Decimal:
    """Prețul de intrare: limita pentru LIMIT, altfel cel mai defavorabil dintre ref și ask."""
    if intent.order_type == "LIMIT" and intent.limit_price is not None:
        return intent.limit_price
    return max(intent.ref_price, quote.ask) if quote is not None else intent.ref_price


def _exit_quote(quote: Quote | None, stop: Decimal) -> Quote | None:
    """Cotația la ieșire: același spread, centrată pe stop (None dacă bid-ul ar fi ≤ 0)."""
    if quote is None:
        return None
    half = (quote.ask - quote.bid) / 2
    if stop - half <= 0:
        return None
    return Quote(instrument=quote.instrument, ts=quote.ts, bid=stop - half, ask=stop + half)


@dataclass(frozen=True)
class Recomputed:
    trade_risk_eur: Decimal
    cash_required_eur: Decimal
    notional_ccy: Decimal


def _recompute(case: Case, qty: Decimal) -> Recomputed:
    snap = case.snapshot
    inst = snap.instrument
    intent = case.intent
    assert intent.stop_price is not None
    model = CompleteCostModel(case.cost_config)
    entry = _entry_price(intent, snap.quote)
    stop = intent.stop_price
    fx = D(1) if inst.currency == "EUR" else snap.cost_ctx.fx_rate
    assert fx is not None, "aprobat fără curs FX"

    def leg(side: str, price: Decimal, quote: Quote | None) -> Decimal:
        spec = OrderSpec.model_validate(
            {"instrument": inst, "side": side, "qty": qty, "ts": case.ctx.ts, "ref_price": price}
        )
        return max(model.estimate(spec, quote, snap.cost_ctx).total, ZERO)

    entry_cost = leg("BUY", entry, snap.quote)
    exit_cost = leg("SELL", stop, _exit_quote(snap.quote, stop))
    with localcontext() as c:
        c.prec = 50
        return Recomputed(
            trade_risk_eur=qty * (entry - stop) * fx + entry_cost + exit_cost,
            cash_required_eur=qty * entry * fx + entry_cost,
            notional_ccy=qty * entry,
        )


def _evaluate(case: Case) -> RiskDecision:
    engine = RiskEngine(case.config, CompleteCostModel(case.cost_config))
    return engine.evaluate(case.intent, case.ctx)


def _assert_p2(case: Case, decision: RiskDecision) -> None:
    inst = case.snapshot.instrument
    qty = decision.qty
    assert qty is not None and qty > 0
    assert qty % inst.qty_step == 0, f"qty {qty} nu este multiplu de {inst.qty_step}"
    assert qty >= inst.min_qty
    if case.intent.requested_qty is not None:
        assert qty <= case.intent.requested_qty

    r = _recompute(case, qty)
    limit = case.config.risk_per_trade_max_eur
    assert limit <= MAX_RISK_PER_TRADE_EUR
    assert r.trade_risk_eur <= limit, f"risc {r.trade_risk_eur} > {limit}"
    assert r.cash_required_eur <= case.ctx.cash_eur, (
        f"numerar necesar {r.cash_required_eur} > {case.ctx.cash_eur}"
    )
    assert r.notional_ccy >= inst.min_notional


# --------------------------------------------------------------------------- proprietăți


@given(cases())
def test_property_2_approved_order_within_trade_limit_and_cash(case: Case) -> None:
    """**Validates: Requirements 13.2, 13.8, 13.9**"""
    decision = _evaluate(case)
    event("approved" if decision.approved else f"rejected:{decision.reason}")
    if decision.approved:
        event("approved:reduced" if decision.reduced else "approved:not_reduced")
        _assert_p2(case, decision)


def test_property_2_generator_produces_approvals() -> None:
    """Testul de proprietate nu este vacuu: o parte semnificativă a cazurilor este aprobată.

    **Validates: Requirements 13.2, 13.8, 13.9**
    """
    approved: list[bool] = []
    reduced: list[bool] = []

    @settings(max_examples=300, derandomize=True, database=None)
    @given(cases())
    def run(case: Case) -> None:
        d = _evaluate(case)
        approved.append(d.approved)
        if d.approved:
            reduced.append(d.reduced)
            _assert_p2(case, d)

    run()
    assert len(approved) >= 100
    assert sum(approved) >= len(approved) // 5, f"{sum(approved)} aprobări din {len(approved)}"
    assert any(reduced), "nicio aprobare cu cantitate redusă"
