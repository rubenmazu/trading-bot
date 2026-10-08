"""Teste unitare pentru Risk_Engine (task 8.1): fiecare verificare și ordinea lor.

Scenariul de bază (EUR, fără FX):
- entry = max(ref 10, ask 10.01) = 10.01, stop = 9.5 → risc de preț 0.51 / unitate;
- cost pe picior: jumătate de spread 0.01 / unitate + taxă fixă de bursă 0.05;
- cost dus-întors = 0.02·q + 0.10, deci risk_per_unit = 0.53;
- buget țintă 0.25 → q = (0.25 − 0.10) / 0.53 = 0.2830… → 0.283.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from qts.config.schema import RiskConfig
from qts.core.models import Instrument, OrderIntent, Quote
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import (
    CompleteCostModel,
    CostContext,
    CostModelConfig,
    FxConfig,
    LatencyConfig,
    SlippageConfig,
    TaxConfig,
)
from qts.risk import (
    ExposureApproval,
    KillSwitchState,
    MarketSnapshot,
    PositionRisk,
    RejectReason,
    RiskContext,
    RiskEngine,
)

D = Decimal
T0 = datetime(2025, 1, 2, 9, 30, tzinfo=UTC)


def _inst(**kw: Any) -> Instrument:
    data: dict[str, Any] = {
        "symbol": "XYZ",
        "venue": "XETR",
        "asset_class": "etf",
        "currency": "EUR",
        "tick_size": "0.01",
        "qty_step": "0.001",
        "min_qty": "0.001",
        "calendar_id": "XETR",
        "fractional": True,
    }
    data.update(kw)
    return Instrument.model_validate(data)


def _cost_model(table: dict[str, Any] | None = None, **cfg: Any) -> CompleteCostModel:
    t: dict[str, Any] = {
        "broker": "sim",
        "version": "v1",
        "valid_from": datetime(2024, 1, 1, tzinfo=UTC),
        "currency": "EUR",
        "percent": "0",
        "exchange_fee_fixed": "0.05",
    }
    t.update(table or {})
    data: dict[str, Any] = {
        "version": "costs-test",
        "commissions": CommissionSchedule(tables=(CommissionTable.model_validate(t),)),
        "slippage": SlippageConfig(k=D(0)),
        "latency": LatencyConfig(latency_ms=0),
        "fx": FxConfig(conversion_spread=D(0)),
        "taxes": TaxConfig(approved=True, reference="test"),
    }
    data.update(cfg)
    return CompleteCostModel(CostModelConfig(**data))


def _snapshot(inst: Instrument | None = None, **kw: Any) -> MarketSnapshot:
    inst = inst or _inst()
    data: dict[str, Any] = {
        "instrument": inst,
        "data_fresh": True,
        "quote": Quote(instrument=inst.symbol, ts=T0, bid=D("9.99"), ask=D("10.01")),
        "cost_ctx": CostContext(broker="sim", sigma_bar=D("0.01"), adv=D(1000)),
    }
    data.update(kw)
    return MarketSnapshot.model_validate(data)


def _ctx(snapshot: MarketSnapshot | None = None, **kw: Any) -> RiskContext:
    snap = snapshot or _snapshot()
    data: dict[str, Any] = {
        "ts": T0,
        "mode": "backtest",
        "cash_eur": "100",
        "market": {snap.instrument.symbol: snap},
    }
    data.update(kw)
    return RiskContext.model_validate(data)


def _buy(**kw: Any) -> OrderIntent:
    data: dict[str, Any] = {
        "intent_id": "i1",
        "signal_id": "s1",
        "instrument": "XYZ",
        "side": "BUY",
        "ref_price": "10",
        "stop_price": "9.5",
    }
    data.update(kw)
    return OrderIntent.model_validate(data)


def _sell(**kw: Any) -> OrderIntent:
    return _buy(side="SELL", stop_price=None, **kw)


def _position(qty: str = "1", **kw: Any) -> PositionRisk:
    data: dict[str, Any] = {
        "instrument": "XYZ",
        "qty": qty,
        "mark_price": "10",
        "stop_price": "9.5",
        "fx_rate_to_eur": "1",
        "exit_cost_eur": "0.05",
    }
    data.update(kw)
    return PositionRisk.model_validate(data)


def _engine(cost_model: CompleteCostModel | None = None, **kw: Any) -> RiskEngine:
    return RiskEngine(RiskConfig(), cost_model or _cost_model(), **kw)


# --------------------------------------------------------------------------- aprobare


def test_approves_sized_on_stop_and_costs() -> None:
    d = _engine().evaluate(_buy(), _ctx())
    assert d.approved
    assert d.qty == D("0.283")
    assert not d.reduced
    assert d.metrics["entry_price"] == D("10.01")
    assert d.metrics["fixed_costs_eur"] == D("0.1")
    assert d.metrics["trade_risk_eur"] == D("0.283") * D("0.53") + D("0.10")
    assert d.metrics["trade_risk_eur"] <= D("0.25")
    assert d.metrics["cash_required_eur"] <= D(100)
    assert d.rules_version


def test_requested_qty_caps_size_and_marks_reduction() -> None:
    small = _engine().evaluate(_buy(requested_qty="0.1"), _ctx())
    assert small.approved and small.qty == D("0.1") and not small.reduced
    big = _engine().evaluate(_buy(requested_qty="5"), _ctx())
    assert big.approved and big.qty == D("0.283") and big.reduced


def test_reduction_after_exact_cost_reevaluates_limits() -> None:
    # Comision plafonat (concav în qty): liniarizarea subestimează costul candidatului.
    model = _cost_model({"percent": "0.05", "maximum": "0.02"})
    d = _engine(model).evaluate(_buy(), _ctx())
    assert d.approved and d.reduced
    assert d.metrics["candidate_qty"] == D("0.279")
    assert d.qty == D("0.207")
    assert d.metrics["trade_risk_eur"] <= D("0.25")
    # Pașii 6-7 rulează pe cantitatea redusă: aceeași reducere, dar limita zilnică respinge.
    rejected = _engine(model).evaluate(_buy(), _ctx(daily_realized_pnl_eur="-1.8"))
    assert rejected.reason is RejectReason.DAILY_LOSS_LIMIT
    assert rejected.metrics["qty"] == D("0.207")


def test_fx_converts_to_eur() -> None:
    inst = _inst(currency="USD")
    snap = _snapshot(
        inst, cost_ctx=CostContext(broker="sim", fx_rate=D("0.5"), sigma_bar=D("0.01"), adv=D(1))
    )
    # Taxa fixă de 0.05 este în EUR, spreadul și riscul de preț se înjumătățesc în EUR.
    d = _engine(_cost_model({"currency": "EUR"})).evaluate(_buy(), _ctx(snap))
    assert d.approved
    assert d.metrics["fx_rate_to_eur"] == D("0.5")
    assert d.metrics["trade_risk_eur"] <= D("0.25")
    assert d.qty is not None and d.qty > D("0.283")


# --------------------------------------------------------------------------- pasul 1


def test_kill_switch_rejects_before_any_other_check() -> None:
    snap = _snapshot(_inst(is_derivative=True), data_fresh=False)
    ctx = _ctx(snap, mode="live", kill_switch=KillSwitchState(day_active=True))
    d = _engine().evaluate(_buy(stop_price=None), ctx)
    assert d.reason is RejectReason.KILL_SWITCH_ACTIVE and "DAY" in d.detail


def test_instrument_kill_switch_scope_applies_only_to_that_instrument() -> None:
    blocked = KillSwitchState(instruments=frozenset({"XYZ"}))
    assert _engine().evaluate(_buy(), _ctx(kill_switch=blocked)).reason is (
        RejectReason.KILL_SWITCH_ACTIVE
    )
    other = KillSwitchState(instruments=frozenset({"ABC"}))
    assert _engine().evaluate(_buy(), _ctx(kill_switch=other)).approved


def test_kill_switch_also_blocks_exits() -> None:
    ctx = _ctx(positions={"XYZ": _position()}, kill_switch=KillSwitchState(global_active=True))
    assert _engine().evaluate(_sell(), ctx).reason is RejectReason.KILL_SWITCH_ACTIVE


def test_live_mode_not_permitted_in_initial_stage() -> None:
    d = _engine().evaluate(_buy(), _ctx(mode="live", project_stage="initial"))
    assert d.reason is RejectReason.MODE_NOT_PERMITTED


def test_missing_and_stale_data_rejected_before_exposure_checks() -> None:
    missing = _ctx(_snapshot(_inst(symbol="ABC")))
    assert _engine().evaluate(_buy(), missing).reason is RejectReason.DATA_MISSING
    stale = _ctx(_snapshot(_inst(is_derivative=True), data_fresh=False, freshness_reason="X"))
    d = _engine().evaluate(_buy(), stale)
    assert d.reason is RejectReason.DATA_STALE and "X" in d.detail


# --------------------------------------------------------------------------- pasul 2


@pytest.mark.parametrize(
    ("flags", "reason"),
    [
        ({"is_derivative": True}, RejectReason.DERIVATIVE_FORBIDDEN),
        ({"requires_leverage": True}, RejectReason.LEVERAGE_FORBIDDEN),
        ({"requires_short": True}, RejectReason.SHORT_FORBIDDEN),
    ],
)
def test_forbidden_instruments_rejected(flags: dict[str, bool], reason: RejectReason) -> None:
    d = _engine().evaluate(_buy(stop_price=None), _ctx(_snapshot(_inst(**flags))))
    assert d.reason is reason


def test_sell_beyond_holdings_is_short() -> None:
    over = _engine().evaluate(_sell(requested_qty="2"), _ctx(positions={"XYZ": _position("1")}))
    assert over.reason is RejectReason.SHORT_FORBIDDEN
    assert over.value == D(2) and over.limit == D(1)


def test_sell_without_position_is_short() -> None:
    d = _engine().evaluate(_sell(requested_qty="1"), _ctx())
    assert d.reason is RejectReason.SHORT_FORBIDDEN
    nothing = _engine().evaluate(_sell(), _ctx())
    assert nothing.reason is RejectReason.INVALID_QTY


def test_exit_of_held_qty_bypasses_loss_limits() -> None:
    ctx = _ctx(
        positions={"XYZ": _position("1.5")},
        daily_realized_pnl_eur="-5",
        total_pnl_eur="-20",
        cash_eur="0",
    )
    d = _engine().evaluate(_sell(), ctx)
    assert d.approved and d.is_exit and d.qty == D("1.5")


def test_exposure_approval_only_after_initial_stage() -> None:
    approval = ExposureApproval(approved=True, approval_ref="RC-1", allow_derivatives=True)
    snap = _snapshot(_inst(is_derivative=True))
    initial = _engine(exposure_approval=approval).evaluate(_buy(), _ctx(snap))
    assert initial.reason is RejectReason.DERIVATIVE_FORBIDDEN
    post = _engine(exposure_approval=approval).evaluate(
        _buy(), _ctx(snap, project_stage="post_initial")
    )
    assert post.approved
    unapproved = ExposureApproval(approved=False, approval_ref="", allow_derivatives=True)
    d = _engine(exposure_approval=unapproved).evaluate(
        _buy(), _ctx(snap, project_stage="post_initial")
    )
    assert d.reason is RejectReason.DERIVATIVE_FORBIDDEN


def test_approved_short_is_still_unsupported() -> None:
    approval = ExposureApproval(approved=True, approval_ref="RC-1", allow_short=True)
    d = _engine(exposure_approval=approval).evaluate(
        _sell(requested_qty="1"), _ctx(project_stage="post_initial")
    )
    assert d.reason is RejectReason.SHORT_UNSUPPORTED


def test_max_open_positions() -> None:
    positions = {s: _position(instrument=s) for s in ("A", "B", "C")}
    d = _engine().evaluate(_buy(), _ctx(positions=positions))
    assert d.reason is RejectReason.MAX_OPEN_POSITIONS
    assert d.value == D(4) and d.limit == D(3)


# --------------------------------------------------------------------------- pasul 3


def test_stop_required_and_below_entry() -> None:
    assert _engine().evaluate(_buy(stop_price=None), _ctx()).reason is RejectReason.STOP_MISSING
    d = _engine().evaluate(_buy(stop_price="10.01"), _ctx())
    assert d.reason is RejectReason.STOP_NOT_BELOW_ENTRY
    assert d.value == D("10.01") and d.limit == D("10.01")


def test_limit_order_uses_limit_price_as_entry() -> None:
    d = _engine().evaluate(_buy(order_type="LIMIT", limit_price="9.4"), _ctx())
    assert d.reason is RejectReason.STOP_NOT_BELOW_ENTRY and d.limit == D("9.4")


def test_missing_fx_rate_rejected() -> None:
    d = _engine().evaluate(_buy(), _ctx(_snapshot(_inst(currency="USD"))))
    assert d.reason is RejectReason.FX_RATE_MISSING


def test_incomplete_cost_model_rejects_fail_closed() -> None:
    d = _engine(_cost_model(taxes=None)).evaluate(_buy(), _ctx())
    assert d.reason is RejectReason.COST_MODEL_INCOMPLETE and "taxes" in d.detail


# --------------------------------------------------------------------------- pasul 5


def test_fixed_costs_above_trade_limit_reject() -> None:
    # Taxă fixă 0.30 / ordin → 0.60 dus-întors > 0.50, chiar la cantitatea minimă.
    d = _engine(_cost_model({"exchange_fee_fixed": "0.30"})).evaluate(_buy(), _ctx())
    assert d.reason is RejectReason.TRADE_RISK_LIMIT
    assert d.limit == D("0.50")
    assert d.value is not None and d.value > D("0.60")
    assert d.metrics["qty"] == D("0.001")


def test_min_qty_risk_within_hard_limit_is_approved() -> None:
    # Minim 0.5 unități: risc 0.5·0.53 + 0.10 = 0.365 > țintă 0.25, dar ≤ 0.50 (13.2).
    d = _engine().evaluate(_buy(), _ctx(_snapshot(_inst(min_qty="0.5"))))
    assert d.approved and d.qty == D("0.5")


def test_min_qty_above_hard_limit_rejected() -> None:
    d = _engine().evaluate(_buy(), _ctx(_snapshot(_inst(min_qty="1", qty_step="1"))))
    assert d.reason is RejectReason.TRADE_RISK_LIMIT
    assert d.value == D("0.63") and d.limit == D("0.50")


def test_requested_below_min_qty_rejected() -> None:
    d = _engine().evaluate(_buy(requested_qty="0.2"), _ctx(_snapshot(_inst(min_qty="0.5"))))
    assert d.reason is RejectReason.QTY_BELOW_MIN
    assert d.value == D("0.2") and d.limit == D("0.5")


def test_min_notional_rejected() -> None:
    d = _engine().evaluate(_buy(), _ctx(_snapshot(_inst(min_notional="50"))))
    assert d.reason is RejectReason.MIN_NOTIONAL and d.limit == D(50)


def test_cash_after_costs_rejected() -> None:
    d = _engine().evaluate(_buy(), _ctx(cash_eur="0.05"))
    assert d.reason is RejectReason.CASH_INSUFFICIENT
    assert d.value is not None and d.value > D("0.05") and d.limit == D("0.05")


def test_cash_limits_size_below_risk_budget() -> None:
    d = _engine().evaluate(_buy(), _ctx(cash_eur="1.10"))
    assert d.approved and d.qty is not None
    assert d.metrics["cash_required_eur"] <= D("1.10")
    assert d.qty < D("0.283")


# --------------------------------------------------------------------------- pașii 6-7


def test_daily_limit_includes_realized_unrealized_and_trade_risk() -> None:
    d = _engine().evaluate(
        _buy(), _ctx(daily_realized_pnl_eur="-1", daily_unrealized_pnl_eur="-0.9")
    )
    assert d.reason is RejectReason.DAILY_LOSS_LIMIT
    assert d.limit == D(2)
    assert d.value == D("1.9") + d.metrics["trade_risk_eur"]


def test_daily_limit_includes_worst_case_open_risk() -> None:
    # Risc deschis: 3·(10 − 9.5) + 0.05 = 1.55; 1.55 + 0.25 ≤ 2, dar cu 0.3 pierdere > 2.
    ctx = _ctx(positions={"ABC": _position("3", instrument="ABC")}, daily_realized_pnl_eur="-0.3")
    d = _engine().evaluate(_buy(), ctx)
    assert d.reason is RejectReason.DAILY_LOSS_LIMIT
    assert d.metrics["open_risk_eur"] == D("1.55")


def test_position_without_stop_counts_full_value_at_risk() -> None:
    ctx = _ctx(positions={"ABC": _position("1", instrument="ABC", stop_price=None)})
    d = _engine().evaluate(_buy(), ctx)
    assert d.reason is RejectReason.DAILY_LOSS_LIMIT
    assert d.metrics["open_risk_eur"] == D("10.05")


def test_daily_gains_do_not_create_negative_loss() -> None:
    d = _engine().evaluate(_buy(), _ctx(daily_realized_pnl_eur="5"))
    assert d.approved and d.metrics["daily_loss_eur"] == D(0)


def test_total_limit() -> None:
    d = _engine().evaluate(_buy(), _ctx(total_pnl_eur="-9.8"))
    assert d.reason is RejectReason.TOTAL_LOSS_LIMIT
    assert d.limit == D(10)
    assert d.value == D("9.8") + d.metrics["trade_risk_eur"]


def test_daily_checked_before_total() -> None:
    d = _engine().evaluate(_buy(), _ctx(daily_realized_pnl_eur="-1.9", total_pnl_eur="-9.9"))
    assert d.reason is RejectReason.DAILY_LOSS_LIMIT


def test_per_trade_checked_before_daily() -> None:
    model = _cost_model({"exchange_fee_fixed": "0.30"})
    d = _engine(model).evaluate(_buy(), _ctx(daily_realized_pnl_eur="-1.9"))
    assert d.reason is RejectReason.TRADE_RISK_LIMIT


def test_decision_is_deterministic() -> None:
    a = _engine().evaluate(_buy(), _ctx())
    b = _engine().evaluate(_buy(), _ctx())
    assert a == b
