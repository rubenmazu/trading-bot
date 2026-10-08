"""Teste unitare pentru proiecția portofoliului și PnL (Req 8.4, 29.1).

Valorile așteptate sunt calculate manual în comentarii.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from qts.core.models import CostBreakdown, Side, canonical_json
from qts.portfolio.pnl import Mark, release_average_cost
from qts.portfolio.portfolio import (
    CashEvent,
    OversellError,
    Portfolio,
    PortfolioError,
    PortfolioFill,
    PortfolioState,
    apply_fill,
    equity_eur,
    pnl_report,
    to_loss_observation,
    to_position_risk,
    to_risk_accounting,
    unrealized_pnl_eur,
)
from qts.risk.context import RiskContext

D = Decimal
T0 = datetime(2024, 3, 4, 10, 0, tzinfo=UTC)  # luni
DAY1 = date(2024, 3, 4)
DAY2 = date(2024, 3, 5)


def fill(
    fid: str,
    side: Side,
    qty: str,
    price: str,
    *,
    inst: str = "ETF",
    ts: datetime = T0,
    fx: str = "1",
    currency: str = "EUR",
    **costs: str,
) -> PortfolioFill:
    return PortfolioFill(
        fill_id=fid,
        instrument=inst,
        currency=currency,
        side=side,
        qty=D(qty),
        price=D(price),
        fx_rate_to_eur=D(fx),
        ts=ts,
        costs=CostBreakdown(**{k: D(v) for k, v in costs.items()}),
    )


def mark(price: str, ts: datetime = T0, fx: str = "1") -> Mark:
    return Mark(price=D(price), fx_rate_to_eur=D(fx), ts=ts)


def run(cash: str, *fills: PortfolioFill) -> PortfolioState:
    pf = Portfolio.with_cash(D(cash))
    for f in fills:
        pf.apply_fill(f)
    return pf.snapshot()


def test_buy_updates_position_cash_and_costs() -> None:
    # 5 × 100 = 500; costuri 1 + 0,25 → numerar 1000 − 500 − 1,25 = 498,75
    st = run("1000", fill("f1", "BUY", "5", "100", commission="1", spread="0.25"))
    pos = st.positions["ETF"]
    assert (pos.qty, pos.cost_basis_eur, pos.avg_price) == (D(5), D(500), D(100))
    assert st.cash_eur == D("498.75")
    assert st.realized.gross_eur == 0
    assert st.realized.costs.total == D("1.25")
    assert st.realized.net_eur == D("-1.25")
    assert st.incidents == ()


def test_average_cost_partial_and_full_exit() -> None:
    st = run(
        "10000",
        fill("b1", "BUY", "10", "100", commission="1"),
        fill("b2", "BUY", "10", "120", commission="1"),  # 20 @ medie 110, bază 2200
    )
    assert st.positions["ETF"].avg_price == D(110)

    # vânzare 5 @ 130: 650 − 5 × 110 = 100 brut
    st = apply_fill(st, fill("s1", "SELL", "5", "130", commission="1", slippage="0.5"))
    pos = st.positions["ETF"]
    assert (pos.qty, pos.cost_basis_eur, pos.avg_price) == (D(15), D(1650), D(110))
    assert st.realized.gross_eur == D(100)

    # ieșire totală 15 @ 100: 1500 − 1650 = −150 brut
    st = apply_fill(st, fill("s2", "SELL", "15", "100", commission="1"))
    assert "ETF" not in st.positions
    assert st.realized.gross_eur == D(-50)
    assert st.realized.costs == CostBreakdown(commission=D(4), slippage=D("0.5"))
    assert st.realized.net_eur == D("-54.5")
    # 10000 − 2200 + 2150 − 4,5
    assert st.cash_eur == D("9945.5")
    assert equity_eur(st) == D(10000) + st.realized.net_eur
    assert st.realized_by_instrument["ETF"].net_eur == D("-54.5")


def test_oversell_and_sell_without_position_are_rejected() -> None:
    st = run("1000", fill("b1", "BUY", "2", "100"))
    with pytest.raises(OversellError):
        apply_fill(st, fill("s1", "SELL", "3", "100"))
    with pytest.raises(OversellError):
        apply_fill(st, fill("s2", "SELL", "1", "10", inst="OTHER"))
    assert st.positions["ETF"].qty == D(2)  # starea inițială nu este modificată


def test_multiple_instruments_unrealized_at_marks() -> None:
    st = run(
        "2000",
        fill("a", "BUY", "10", "100", inst="A"),
        fill("b", "BUY", "4", "50", inst="B"),
    )
    marks = {"A": mark("105"), "B": mark("45")}
    # A: 10 × 5 = 50; B: 4 × −5 = −20
    assert unrealized_pnl_eur(st, marks) == D(30)
    # numerar 2000 − 1000 − 200 = 800; valoare 1050 + 180
    assert equity_eur(st, marks) == D(2030)
    # fără marcaje noi se folosesc prețurile ultimelor execuții
    assert unrealized_pnl_eur(st) == 0


def test_fx_instrument_gross_includes_currency_effect() -> None:
    st = run(
        "1000",
        fill("b", "BUY", "10", "50", inst="US", currency="USD", fx="0.9", fx_conversion="0.45"),
    )
    pos = st.positions["US"]
    assert (pos.cost_basis_ccy, pos.cost_basis_eur) == (D(500), D(450))
    assert (pos.avg_price, pos.avg_fx_rate_to_eur) == (D(50), D("0.9"))
    assert st.cash_eur == D("549.55")
    # 10 × 52 × 0,85 = 442 → nerealizat −8
    assert unrealized_pnl_eur(st, {"US": mark("52", fx="0.85")}) == D(-8)

    # încasare 10 × 55 × 0,8 = 440 → brut −10 (câștig de preț anulat de curs)
    st = apply_fill(
        st,
        fill("s", "SELL", "10", "55", inst="US", currency="USD", fx="0.8", fx_conversion="0.44"),
    )
    assert st.realized.gross_eur == D(-10)
    assert st.realized.costs.fx_conversion == D("0.89")
    assert st.realized.net_eur == D("-10.89")
    assert st.cash_eur == D("989.11")


def test_currency_validation() -> None:
    with pytest.raises(ValidationError):
        fill("x", "BUY", "1", "1", fx="0.9")  # EUR cu fx ≠ 1
    with pytest.raises(ValidationError):
        fill("x", "BUY", "1", "1", commission="-1")
    st = run("1000", fill("b", "BUY", "1", "10", inst="US", currency="USD", fx="0.9"))
    with pytest.raises(PortfolioError):
        apply_fill(st, fill("s", "SELL", "1", "10", inst="US"))


def test_report_separates_gross_each_cost_category_and_net() -> None:
    all_costs = {
        "spread": "0.1",
        "commission": "0.2",
        "slippage": "0.3",
        "latency": "0.4",
        "fx_conversion": "0.5",
        "taxes": "0.6",
    }
    st = run(
        "1000",
        fill("b", "BUY", "10", "10"),
        PortfolioFill(  # brut 4 × 2 = 8
            fill_id="s",
            instrument="ETF",
            side="SELL",
            qty=D(4),
            price=D(12),
            ts=T0,
            costs=CostBreakdown(**{k: D(v) for k, v in all_costs.items()}),
        ),
    )
    report = pnl_report(st, {"ETF": mark("11")})  # nerealizat 6 × 1 = 6
    rows = dict(report.as_rows())
    assert rows["gross_realized"] == D(8)
    assert rows["gross_unrealized"] == D(6)
    assert rows["gross"] == D(14)
    assert [rows[f"cost_{k}"] for k in all_costs] == [D(v) for v in all_costs.values()]
    assert rows["cost_total"] == D("2.1")
    assert rows["net"] == D("11.9")
    assert report.net_eur <= report.gross_eur


def test_daily_rollover_and_loss_observation() -> None:
    pf = Portfolio.with_cash(D(2000))
    pf.apply_fill(fill("b", "BUY", "10", "100", commission="1"))
    pf.apply_mark("ETF", mark("110", T0 + timedelta(hours=6)))

    obs1 = pf.loss_observation(T0 + timedelta(hours=6))
    assert obs1.daily_realized_pnl_eur == D(-1)
    assert obs1.daily_unrealized_pnl_eur == D(100)
    assert obs1.equity_eur == D(2099)  # 999 + 1100

    t2 = T0 + timedelta(days=1)
    st = pf.apply_fill(fill("s", "SELL", "10", "105", ts=t2, commission="1"))
    (day1,) = st.closed_days
    assert day1.trading_day == DAY1
    assert (day1.realized.net_eur, day1.unrealized_change_eur) == (D(-1), D(100))
    assert st.today is not None and st.today.trading_day == DAY2

    obs2 = pf.loss_observation(t2)
    # 110 → 105 pe 10 unități și comision 1: −51 = 49 realizat − 100 nerealizat
    assert obs2.daily_realized_pnl_eur == D(49)
    assert obs2.daily_unrealized_pnl_eur == D(-100)
    assert obs2.daily_loss_eur == D(51)
    assert obs2.equity_eur == D(2048)

    obs3 = to_loss_observation(st, t2 + timedelta(days=1))  # zi nouă fără evenimente
    assert (obs3.daily_realized_pnl_eur, obs3.daily_unrealized_pnl_eur) == (D(0), D(0))


def test_observation_on_new_day_uses_previous_marks_as_baseline() -> None:
    st = run("2000", fill("b", "BUY", "10", "100"))
    pf = Portfolio(st)
    pf.apply_mark("ETF", mark("110", T0 + timedelta(hours=6)))
    t2 = T0 + timedelta(days=1)
    obs = to_loss_observation(pf.snapshot(), t2, {"ETF": mark("108", t2)})
    assert obs.daily_realized_pnl_eur == 0
    assert obs.daily_unrealized_pnl_eur == D(-20)


def test_deposits_withdrawals_and_negative_cash_incident() -> None:
    pf = Portfolio.with_cash(D(100))
    pf.apply_cash(CashEvent(event_id="d1", kind="DEPOSIT", amount_eur=D(50), ts=T0))
    pf.apply_cash(CashEvent(event_id="w1", kind="WITHDRAWAL", amount_eur=D(30), ts=T0))
    st = pf.snapshot()
    assert st.cash_eur == D(120)
    assert (st.cumulative_deposits_eur, st.cumulative_withdrawals_eur) == (D(50), D(30))
    obs = pf.loss_observation(T0)
    assert obs.adjusted_capital_eur == D(100)  # 120 + 30 − 50
    assert obs.daily_realized_pnl_eur == 0  # fluxurile nu sunt PnL

    st = pf.apply_fill(fill("b", "BUY", "1", "120", commission="1"))
    assert st.cash_eur == D(-1)  # se înregistrează realitatea brokerului
    (incident,) = st.incidents
    assert (incident.code, incident.ref_id, incident.cash_eur) == ("CASH_NEGATIVE", "b", D(-1))


def test_risk_adapters() -> None:
    st = run("1000", fill("b", "BUY", "2", "100", commission="1"))
    positions = to_position_risk(
        st, exit_costs_eur={"ETF": D(1)}, marks={"ETF": mark("90")}, stops={"ETF": D(95)}
    )
    pr = positions["ETF"]
    assert (pr.qty, pr.mark_price, pr.stop_price, pr.exit_cost_eur) == (D(2), D(90), D(95), D(1))
    with pytest.raises(PortfolioError):
        to_position_risk(st, exit_costs_eur={})

    acc = to_risk_accounting(st, T0, exit_costs_eur={"ETF": D(1)}, marks={"ETF": mark("90")})
    assert acc.cash_eur == D(799)
    assert acc.daily_realized_pnl_eur == D(-1)
    assert acc.daily_unrealized_pnl_eur == D(-20)
    assert acc.total_pnl_eur == D(-21)
    ctx = RiskContext(ts=T0, mode="backtest", **acc.context_fields())
    assert ctx.daily_loss_eur == D(21)


def test_canonical_serialization_is_deterministic_and_round_trips() -> None:
    events = (
        fill("b1", "BUY", "3", "10.5", commission="0.1"),
        fill("b2", "BUY", "2", "11", inst="B", ts=T0 + timedelta(hours=1)),
        fill("s1", "SELL", "1", "12", ts=T0 + timedelta(days=1), taxes="0.05"),
    )
    a, b = run("500", *events), run("500", *events)
    assert canonical_json(a) == canonical_json(b)
    assert PortfolioState.model_validate_json(canonical_json(a)) == a


def test_release_average_cost_three_way_split_conserves_basis() -> None:
    # bază 100 pentru 3 unități; trei vânzări de câte 1 eliberează exact 100 în total
    qty, basis = D(3), D(100)
    released = D(0)
    for _ in range(3):
        rel = release_average_cost(
            qty=qty, basis_ccy=basis, basis_eur=basis, sell_qty=D(1), proceeds_eur=D(0)
        )
        released += rel.released_basis_eur
        qty, basis = qty - 1, basis - rel.released_basis_eur
    assert released == D(100)
    assert basis == 0
