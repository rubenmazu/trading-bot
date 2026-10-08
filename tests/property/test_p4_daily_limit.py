"""P4: după atingerea limitei zilnice nu se mai aprobă ordine în aceeași zi (Req 13.3, 13.4).

Se generează traiectorii intrazilnice peste una sau mai multe zile de tranzacționare: observații
`LossObservation` cu PnL zilnic realizat/nerealizat arbitrar (inclusiv revenire după depășire),
`Order_Intent` BUY intercalate în contexte altfel aprobabile, reporniri ale monitorului prin
export/import JSON al stării și, opțional, `Trading_Day` într-un fus cu decalaj fix și oră de
trecere (rollover). Atât `monitor.kill_switch_state(ts)` cât și PnL-ul zilnic curent intră în
`RiskContext`.

Oracolul (model independent):
- ziua D este „atinsă” după prima observație din D cu pierderea zilnică ≥ limita;
- orice `evaluate()` ulterior cu `ts` în D este respins (`KILL_SWITCH_ACTIVE`), chiar dacă PnL
  revine sau monitorul a repornit;
- în zilele neatinse monitorul nu influențează decizia (identică cu cea fără Kill_Switch), deci
  aprobările redevin posibile în ziua următoare.

Separat: Risk_Engine singur (fără monitor) nu aprobă niciun BUY pentru care
pierdere zilnică + risc deschis + risc tranzacție > limita zilnică; riscul tranzacției este
recalculat independent din scenariul fix de costuri (jumătate de spread 0,01/unitate + taxă fixă
de bursă 0,05 pe fiecare picior).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any, Literal

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
    SlippageConfig,
    TaxConfig,
)
from qts.risk import (
    KillSwitchState,
    MarketSnapshot,
    PositionRisk,
    RejectReason,
    RiskContext,
    RiskDecision,
    RiskEngine,
)
from qts.risk.limits import open_risk_eur
from qts.risk.monitor import (
    KillSwitchActivation,
    LossMonitor,
    LossMonitorState,
    LossObservation,
    TradingDayFn,
    utc_trading_day,
    zoned_trading_day,
)

D = Decimal
SYMBOL = "XYZ"
CC_ID = "cc-1"
T0 = datetime(2026, 3, 2, 6, 0, tzinfo=UTC)
BID, ASK = D("9.99"), D("10.01")
HALF_SPREAD = (ASK - BID) / 2
FIXED_FEE = D("0.05")
ZERO = D(0)
DEFAULT_STOP = D("9.5")


def _dec(lo: str, hi: str, places: int = 2) -> st.SearchStrategy[Decimal]:
    return st.decimals(
        min_value=D(lo), max_value=D(hi), places=places, allow_nan=False, allow_infinity=False
    )


# --------------------------------------------------------------------------- scenariu fix


INSTRUMENT = Instrument.model_validate(
    {
        "symbol": SYMBOL,
        "venue": "XETR",
        "asset_class": "etf",
        "currency": "EUR",
        "tick_size": "0.01",
        "qty_step": "0.001",
        "min_qty": "0.001",
        "calendar_id": "XETR",
        "fractional": True,
    }
)

COST_MODEL = CompleteCostModel(
    CostModelConfig(
        version="costs-p4",
        commissions=CommissionSchedule(
            tables=(
                CommissionTable.model_validate(
                    {
                        "broker": "sim",
                        "version": "v1",
                        "valid_from": datetime(2024, 1, 1, tzinfo=UTC),
                        "currency": "EUR",
                        "percent": "0",
                        "exchange_fee_fixed": str(FIXED_FEE),
                    }
                ),
            )
        ),
        slippage=SlippageConfig(k=D(0)),
        latency=LatencyConfig(latency_ms=0),
        fx=FxConfig(conversion_spread=D(0)),
        taxes=TaxConfig(approved=True, reference="test"),
    )
)


def _snapshot(ts: datetime) -> MarketSnapshot:
    return MarketSnapshot(
        instrument=INSTRUMENT,
        data_fresh=True,
        quote=Quote(instrument=SYMBOL, ts=ts, bid=BID, ask=ASK),
        cost_ctx=CostContext(broker="sim", sigma_bar=D("0.01"), adv=D(1000)),
    )


def _buy(intent_id: str, stop: Decimal) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id,
        signal_id=f"s-{intent_id}",
        instrument=SYMBOL,
        side="BUY",
        ref_price=D("10"),
        stop_price=stop,
    )


def _ctx(
    ts: datetime,
    *,
    realized: Decimal,
    unrealized: Decimal,
    positions: dict[str, PositionRisk],
    kill_switch: KillSwitchState,
) -> RiskContext:
    return RiskContext(
        ts=ts,
        mode="backtest",
        cash_eur=D("100"),
        positions=positions,
        daily_realized_pnl_eur=realized,
        daily_unrealized_pnl_eur=unrealized,
        total_pnl_eur=D(0),
        market={SYMBOL: _snapshot(ts)},
        kill_switch=kill_switch,
    )


def _engine(config: RiskConfig) -> RiskEngine:
    return RiskEngine(config, COST_MODEL)


@dataclass
class ListSink:
    received: list[KillSwitchActivation] = field(default_factory=list)

    def activate(self, activation: KillSwitchActivation) -> None:
        self.received.append(activation)


# --------------------------------------------------------------------------- generatori


@st.composite
def risk_configs(draw: st.DrawFn) -> RiskConfig:
    return RiskConfig(daily_loss_limit_eur=draw(st.sampled_from([D("2"), D("1.5"), D("1")])))


@st.composite
def trading_day_fns(draw: st.DrawFn) -> tuple[str, TradingDayFn]:
    if draw(st.booleans()):
        return "utc", utc_trading_day
    # Numai fusuri cu decalaj fix (tzdata nu este instalat).
    hours = draw(st.integers(-10, 12))
    rollover = time(draw(st.integers(0, 23)), draw(st.sampled_from([0, 30])))
    return f"UTC{hours:+d}@{rollover}", zoned_trading_day(
        timezone(timedelta(hours=hours)), rollover
    )


Kind = Literal["obs", "order", "restart"]


@dataclass(frozen=True)
class Step:
    gap_min: int  # minute de la pasul anterior (≥ 0)
    kind: Kind
    realized: Decimal = ZERO
    unrealized: Decimal = ZERO
    stop: Decimal = DEFAULT_STOP


# PnL zilnic: în mare parte în interiorul limitei, uneori dincolo (depășire), alteori pozitiv.
_pnl = st.one_of(
    _dec("-0.8", "0.5"),
    _dec("-0.8", "0.5"),
    _dec("-3", "0.5"),
    _dec("-3", "-1"),
    st.just(D("-2")),
    st.just(D("-1")),
)


@st.composite
def steps(draw: st.DrawFn) -> Step:
    gap = draw(st.one_of(st.integers(0, 90), st.integers(0, 90), st.integers(300, 1500)))
    kind = draw(st.sampled_from(["obs", "obs", "order", "order", "order", "restart"]))
    if kind == "obs":
        return Step(gap, "obs", realized=draw(_pnl), unrealized=draw(_dec("-0.5", "0.5")))
    if kind == "order":
        return Step(gap, "order", stop=draw(st.sampled_from([D("9.5"), D("9.6"), D("9.8")])))
    return Step(gap, "restart")


@st.composite
def open_positions(draw: st.DrawFn) -> dict[str, PositionRisk]:
    if not draw(st.booleans()):
        return {}
    mark = D("20")
    return {
        "P1": PositionRisk(
            instrument="P1",
            qty=draw(_dec("0.01", "1", 2)),
            mark_price=mark,
            stop_price=mark - draw(_dec("0", "0.5", 2)),
            fx_rate_to_eur=D(1),
            exit_cost_eur=draw(_dec("0", "0.05", 2)),
        )
    }


@dataclass(frozen=True)
class Trajectory:
    config: RiskConfig
    day_label: str
    day_fn: TradingDayFn
    start: datetime
    positions: dict[str, PositionRisk]
    steps: tuple[Step, ...]


@st.composite
def trajectories(draw: st.DrawFn) -> Trajectory:
    label, fn = draw(trading_day_fns())
    start = T0 + timedelta(minutes=draw(st.integers(0, 24 * 60)))
    return Trajectory(
        config=draw(risk_configs()),
        day_label=label,
        day_fn=fn,
        start=start,
        positions=draw(open_positions()),
        steps=tuple(draw(st.lists(steps(), min_size=1, max_size=40))),
    )


# --------------------------------------------------------------------------- simulare


@dataclass
class Stats:
    orders_on_breached_day: int = 0
    blocked_only_by_monitor: int = 0  # PnL revenit: engine-ul singur ar fi aprobat
    approvals_before_breach: int = 0
    approvals_after_rollover: int = 0  # aprobări într-o zi ulterioară unei zile atinse
    restarts_on_breached_day: int = 0
    days: set[date] = field(default_factory=set)


def _restart(monitor: LossMonitor, traj: Trajectory, sink: ListSink) -> LossMonitor:
    blob = monitor.export_state().model_dump_json()
    state = LossMonitorState.model_validate_json(blob)
    return LossMonitor(
        traj.config, sink, capital_config_id=CC_ID, trading_day=traj.day_fn, state=state
    )


def run_trajectory(traj: Trajectory) -> Stats:
    """Rulează traiectoria și verifică P4 la fiecare ordin."""
    sink = ListSink()
    monitor = LossMonitor(traj.config, sink, capital_config_id=CC_ID, trading_day=traj.day_fn)
    engine = _engine(traj.config)
    limit = traj.config.daily_loss_limit_eur
    stats = Stats()

    breached: set[date] = set()  # modelul: zilele în care limita zilnică a fost atinsă
    pnl_day: date | None = None  # ziua ultimei observații (PnL-ul zilnic se resetează la rollover)
    realized, unrealized = D(0), D(0)
    ts = traj.start

    for i, step in enumerate(traj.steps):
        ts += timedelta(minutes=step.gap_min)
        day = traj.day_fn(ts)
        stats.days.add(day)
        if pnl_day != day:
            pnl_day, realized, unrealized = day, D(0), D(0)

        if step.kind == "obs":
            realized, unrealized = step.realized, step.unrealized
            monitor.observe(
                LossObservation(
                    ts=ts,
                    daily_realized_pnl_eur=realized,
                    daily_unrealized_pnl_eur=unrealized,
                    equity_eur=D("100"),
                )
            )
            if -(realized + unrealized) >= limit:
                breached.add(day)
            continue

        if step.kind == "restart":
            if day in breached:
                stats.restarts_on_breached_day += 1
            monitor = _restart(monitor, traj, sink)
            continue

        intent = _buy(f"i{i}", step.stop)
        ctx = _ctx(
            ts,
            realized=realized,
            unrealized=unrealized,
            positions=traj.positions,
            kill_switch=monitor.kill_switch_state(ts),
        )
        decision = engine.evaluate(intent, ctx)
        alone = engine.evaluate(intent, ctx.model_copy(update={"kill_switch": KillSwitchState()}))

        if day in breached:
            stats.orders_on_breached_day += 1
            assert not decision.approved, (
                f"aprobat în ziua atinsă {day} la {ts.isoformat()} ({traj.day_label}): {decision}"
            )
            assert decision.reason is RejectReason.KILL_SWITCH_ACTIVE, decision
            if alone.approved:
                stats.blocked_only_by_monitor += 1
        else:
            # Domeniul DAY al altor zile nu influențează decizia (expiră la rollover).
            assert decision == alone, f"monitorul a blocat ziua neatinsă {day}: {decision}"
            if decision.approved:
                if any(d < day for d in breached):
                    stats.approvals_after_rollover += 1
                else:
                    stats.approvals_before_breach += 1

    # Cel mult o activare DAY per Trading_Day, exact pentru zilele atinse.
    day_acts = [a.trading_day for a in sink.received]
    assert len(day_acts) == len(set(day_acts))
    assert set(day_acts) == breached
    return stats


def _record_events(stats: Stats) -> None:
    event(f"days={min(len(stats.days), 4)}")
    if stats.orders_on_breached_day:
        event("orders on breached day")
    if stats.blocked_only_by_monitor:
        event("blocked only by monitor (PnL recovered)")
    if stats.restarts_on_breached_day:
        event("restart on breached day")
    if stats.approvals_after_rollover:
        event("approval after day rollover")
    if stats.approvals_before_breach:
        event("approval before breach")


# --------------------------------------------------------------------------- proprietăți


@given(trajectories())
def test_property_4_no_approval_after_daily_limit_same_day(traj: Trajectory) -> None:
    """**Validates: Requirements 13.3, 13.4**"""
    _record_events(run_trajectory(traj))


def test_property_4_trajectories_are_not_vacuous() -> None:
    """Generatorul atinge efectiv cazurile relevante; proprietatea nu este vacuă.

    **Validates: Requirements 13.3, 13.4**
    """
    totals: dict[str, int] = {
        "orders_on_breached_day": 0,
        "blocked_only_by_monitor": 0,
        "approvals_before_breach": 0,
        "approvals_after_rollover": 0,
        "restarts_on_breached_day": 0,
        "zoned": 0,
    }

    @settings(max_examples=300, derandomize=True, database=None)
    @given(trajectories())
    def run(traj: Trajectory) -> None:
        s = run_trajectory(traj)
        for key in totals:
            if key != "zoned":
                totals[key] += getattr(s, key)
        if traj.day_label != "utc" and s.orders_on_breached_day:
            totals["zoned"] += 1

    run()
    assert all(v > 0 for v in totals.values()), totals


def test_property_4_breach_then_recovery_then_rollover_example() -> None:
    """Exemplu explicit: depășire, revenire, repornire, apoi aprobare după rollover local.

    **Validates: Requirements 13.3, 13.4**
    """
    tz = timezone(timedelta(hours=2))
    traj = Trajectory(
        config=RiskConfig(daily_loss_limit_eur=D("2")),
        day_label="UTC+2@08:00",
        day_fn=zoned_trading_day(tz, time(8)),
        start=datetime(2026, 3, 2, 6, 0, tzinfo=UTC),  # 08:00 local → ziua 2026-03-02
        positions={},
        steps=(
            Step(0, "order"),  # aprobat înainte de depășire
            Step(10, "obs", realized=D("-2"), unrealized=D("0")),  # atinge limita
            Step(10, "obs", realized=D("0.5"), unrealized=D("0")),  # PnL revine
            Step(5, "order"),  # respins de Kill_Switch DAY
            Step(5, "restart"),
            Step(5, "order"),  # respins și după repornire
            Step(23 * 60, "order"),  # 05:35 UTC a doua zi = 07:35 local: tot ziua atinsă
            Step(30, "order"),  # 06:05 UTC = 08:05 local → ziua următoare: aprobat
        ),
    )
    stats = run_trajectory(traj)
    assert stats.approvals_before_breach == 1
    assert stats.orders_on_breached_day == 3
    assert stats.blocked_only_by_monitor == 3
    assert stats.approvals_after_rollover == 1


# --------------------------------------------------------------------------- engine singur


@dataclass(frozen=True)
class EngineCase:
    config: RiskConfig
    intent: OrderIntent
    ctx: RiskContext


@st.composite
def engine_cases(draw: st.DrawFn) -> EngineCase:
    ts = T0 + timedelta(minutes=draw(st.integers(0, 600)))
    requested = draw(st.none() | _dec("0.001", "2", 3))
    stop = draw(_dec("9", "9.95", 2))
    intent = _buy("e1", stop).model_copy(update={"requested_qty": requested})
    ctx = _ctx(
        ts,
        realized=draw(_dec("-2.5", "1")),
        unrealized=draw(_dec("-1", "0.5")),
        positions=draw(open_positions()),
        kill_switch=KillSwitchState(),
    )
    return EngineCase(config=draw(risk_configs()), intent=intent, ctx=ctx)


def _trade_risk(qty: Decimal, stop: Decimal) -> Decimal:
    """Riscul dus-întors recalculat: preț până la stop + spread și taxa fixă pe două picioare."""
    entry = max(D("10"), ASK)
    return qty * (entry - stop) + 2 * (qty * HALF_SPREAD + FIXED_FEE)


def _check_engine_alone(case: EngineCase) -> RiskDecision:
    d = _engine(case.config).evaluate(case.intent, case.ctx)
    if d.approved:
        assert d.qty is not None and case.intent.stop_price is not None
        trade = _trade_risk(d.qty, case.intent.stop_price)
        total = case.ctx.daily_loss_eur + open_risk_eur(case.ctx.positions.values()) + trade
        assert total <= case.config.daily_loss_limit_eur, (
            f"aprobat cu {total} > {case.config.daily_loss_limit_eur}: {d}"
        )
    return d


@given(engine_cases())
def test_property_4_engine_alone_never_exceeds_daily_limit(case: EngineCase) -> None:
    """**Validates: Requirements 13.3**"""
    d = _check_engine_alone(case)
    event("approved" if d.approved else f"rejected:{d.reason}")


def test_property_4_engine_alone_not_vacuous() -> None:
    """Atât aprobări, cât și respingeri DAILY_LOSS_LIMIT apar în cazurile generate.

    **Validates: Requirements 13.3**
    """
    reasons: list[Any] = []

    @settings(max_examples=300, derandomize=True, database=None)
    @given(engine_cases())
    def run(case: EngineCase) -> None:
        d = _check_engine_alone(case)
        reasons.append("approved" if d.approved else d.reason)

    run()
    assert reasons.count("approved") >= 30, reasons.count("approved")
    assert reasons.count(RejectReason.DAILY_LOSS_LIMIT) >= 30
