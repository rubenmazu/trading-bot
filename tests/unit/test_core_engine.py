"""Teste pentru `core/engine.py`: calea unică, write-ahead, audit (Req 1.1, 7.1, 12.1)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from qts.broker.adapter import (
    BrokerCapabilities,
    BrokerSnapshot,
    CancelAck,
    Environment,
    OrderRequest,
    SubmitAck,
)
from qts.broker.fail_safe import ApprovedTarget, FailSafeBlock
from qts.broker.fake import FakeBroker
from qts.broker.sim import SimBarContext, SimBroker
from qts.config.schema import KillSwitchConfig, RiskConfig
from qts.core.bus import CommandEvent, TimerEvent
from qts.core.clock import SimClock
from qts.core.engine import TIMER_EXPIRE_DAYS, EngineConfig, TradingEngine
from qts.core.models import (
    Bar,
    ExecutionEvent,
    Instrument,
    MarketEvent,
    OrderState,
    Signal,
    SignalAction,
)
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import (
    CompleteCostModel,
    CostContext,
    CostModelConfig,
    FxConfig,
    LatencyConfig,
    SlippageConfig,
    SpreadSchedule,
    TaxConfig,
)
from qts.data.freshness import FreshnessTracker
from qts.oms.manager import JournalOmsSink, OrderManager
from qts.persistence.audit import reconstruct_decision_chain, verify_journal
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.portfolio.portfolio import Portfolio
from qts.risk.context import KillSwitchScope
from qts.risk.engine import RiskEngine
from qts.risk.monitor import LossMonitor
from qts.safety.kill_switch import JournalKillSwitchStore, KillSwitch
from qts.safety.stage import ProjectStage, StageInfo
from qts.strategy.base import StrategyState, build_signal
from qts.strategy.history_view import HistoryView
from tests.fixtures.synthetic import SyntheticSpec, generate_bars

D = Decimal
ONE = D(1)
SIGMA = D("0.002")
ADV = D(100_000)
OPEN_STATES = frozenset(
    {
        OrderState.SUBMITTED,
        OrderState.ACKNOWLEDGED,
        OrderState.PARTIALLY_FILLED,
        OrderState.CANCEL_PENDING,
    }
)

INSTRUMENT = Instrument(
    symbol="XYZ",
    venue="XETR",
    asset_class="etf",
    currency="EUR",
    tick_size=D("0.01"),
    qty_step=D("0.001"),
    min_qty=D("0.001"),
    calendar_id="XETR",
    fractional=True,
)


def _cost_model() -> CompleteCostModel:
    table = CommissionTable.model_validate(
        {
            "broker": "sim",
            "version": "v1",
            "valid_from": datetime(2024, 1, 1, tzinfo=UTC),
            "currency": "EUR",
            "percent": "0",
            "exchange_fee_fixed": "0.05",
        }
    )
    return CompleteCostModel(
        CostModelConfig(
            version="costs-test",
            commissions=CommissionSchedule(tables=(table,)),
            spreads={"XYZ": SpreadSchedule(default=D("0.0002"))},
            slippage=SlippageConfig(k=D(0)),
            latency=LatencyConfig(latency_ms=0),
            fx=FxConfig(conversion_spread=D(0)),
            taxes=TaxConfig(approved=True, reference="test"),
        )
    )


def _market_context(inst: Instrument, bar: Bar) -> CostContext:
    return CostContext(broker="sim", sigma_bar=SIGMA, adv=ADV, bar_interval_min=bar.interval_min)


# --------------------------------------------------------------------------- strategie


class _Count(StrategyState):
    n: int = 0


class ScriptedStrategy:
    """Emite acțiunea din scenariu la bara cu numărul dat (1-based)."""

    strategy_id = "scripted"
    version = "1"

    def __init__(self, script: dict[int, SignalAction], stop_offset: Decimal = ONE) -> None:
        self.script = script
        self.stop_offset = stop_offset

    def initial_state(self) -> StrategyState:
        return _Count()

    def on_bar(
        self, bar: Bar, view: HistoryView, state: StrategyState
    ) -> tuple[Signal, StrategyState]:
        assert isinstance(state, _Count)
        assert view.current(bar.instrument) == bar  # fără look-ahead, bara curentă vizibilă
        n = state.n + 1
        action = self.script.get(n, "NONE")
        signal = build_signal(
            self,
            bar,
            action=action,
            reason_code=f"SCRIPT_{action}",
            config_snapshot_id="cfg-test",
            stop_price=bar.close - self.stop_offset if action == "ENTER_LONG" else None,
        )
        return signal, _Count(n=n)


# --------------------------------------------------------------------------- compunere


def _bars(n: int) -> list[Bar]:
    return generate_bars(SyntheticSpec(scenario="random_walk", seed=7, n_bars=n))


def _event(bar: Bar) -> MarketEvent:
    return MarketEvent(
        source_id="synthetic",
        instrument=bar.instrument,
        ts_source=bar.ts_close,
        ts_receipt=bar.ts_close,
        kind="bar",
        payload=bar,
    )


@dataclass
class ListData:
    events: list[MarketEvent]
    source_id: str = "synthetic"

    def stream(self) -> Iterator[MarketEvent]:
        return iter(self.events)


class SpyBroker:
    """Delegă la `SimBroker` și verifică write-ahead la fiecare `submit`."""

    def __init__(self, inner: SimBroker, check: Callable[[OrderRequest], None]) -> None:
        self.inner = inner
        self.check = check
        self.submitted: list[OrderRequest] = []

    @property
    def environment(self) -> Environment:
        return self.inner.environment

    @property
    def account_id(self) -> str:
        return self.inner.account_id

    def capabilities(self) -> BrokerCapabilities:
        return self.inner.capabilities()

    def submit(self, req: OrderRequest) -> SubmitAck:
        self.check(req)
        self.submitted.append(req)
        return self.inner.submit(req)

    def cancel(self, client_order_id: str) -> CancelAck:
        return self.inner.cancel(client_order_id)

    def snapshot(self) -> BrokerSnapshot:
        return self.inner.snapshot()

    def events(self) -> Iterator[ExecutionEvent]:
        return self.inner.events()

    def on_bar(self, bar: Bar, ctx: SimBarContext | None = None) -> None:
        self.inner.on_bar(bar, ctx)


@dataclass
class Harness:
    engine: TradingEngine
    journal: Journal
    clock: SimClock
    oms: OrderManager
    portfolio: Portfolio
    kill_switch: KillSwitch
    inner: Any
    trace: list[str] = field(default_factory=list)


def _harness(
    bars: list[Bar],
    script: dict[int, SignalAction],
    *,
    broker: Literal["sim", "fake", "spy"] = "sim",
    stop_offset: Decimal = ONE,
    use_data: bool = True,
    approved_account: str | None = None,
    policy: Literal["keep", "cancel"] = "keep",
    spy_check: Callable[[Journal, OrderManager, OrderRequest], None] | None = None,
) -> Harness:
    clock = SimClock(bars[0].ts_open)
    journal = Journal(open_db(":memory:"))
    oms = OrderManager(JournalOmsSink(journal))
    kill_switch = KillSwitch(
        JournalKillSwitchStore(journal),
        clock=clock,
        config=KillSwitchConfig(open_orders_policy=policy),
        open_orders=lambda: [o for o in oms.orders.values() if o.state in OPEN_STATES],
    )
    costs = _cost_model()
    inner: Any
    listeners: list[Callable[[MarketEvent], None]] = []
    mode: Literal["backtest", "demo"] = "backtest"
    if broker == "fake":
        inner = FakeBroker(instruments=[INSTRUMENT], clock=clock, environment="demo")
        mode = "demo"
    else:
        sim = SimBroker(
            account_id="SIM-LOCAL", instruments=[INSTRUMENT], cost_model=costs, clock=clock
        )
        inner = sim
        if broker == "spy":
            check = spy_check
            assert check is not None
            inner = SpyBroker(sim, lambda req: check(journal, oms, req))

        def feed(event: MarketEvent, target: Any = inner) -> None:
            if isinstance(event.payload, Bar):
                target.on_bar(event.payload, SimBarContext(sigma_bar=SIGMA, adv=ADV))

        listeners.append(feed)
    guarded: FailSafeBlock[OrderRequest, SubmitAck] = FailSafeBlock(
        inner,
        approved=ApprovedTarget(
            environment=mode,
            account_id=approved_account if approved_account is not None else inner.account_id,
        ),
        stage=StageInfo(stage=ProjectStage.INITIAL, source="test"),
        kill_switch=kill_switch,
        audit=journal,
        clock=clock,
    )
    portfolio = Portfolio.with_cash(D(100))
    trace: list[str] = []
    engine = TradingEngine(
        config=EngineConfig(run_id="run-1", mode=mode, instruments=(INSTRUMENT,)),
        clock=clock,
        broker=guarded,
        journal=journal,
        strategy=ScriptedStrategy(script, stop_offset),
        risk=RiskEngine(RiskConfig(), costs),
        oms=oms,
        portfolio=portfolio,
        kill_switch=kill_switch,
        cost_model=costs,
        freshness=FreshnessTracker(clock, timedelta(minutes=30)),
        loss_monitor=LossMonitor(RiskConfig(), kill_switch, capital_config_id="cap-1"),
        data=ListData([_event(b) for b in bars]) if use_data else None,
        market_context=_market_context,
        market_listeners=listeners,
        clock_driver=clock,
        trace=trace.append,
    )
    return Harness(engine, journal, clock, oms, portfolio, kill_switch, inner, trace)


def _types(journal: Journal) -> list[str]:
    return [r.type for r in journal.read()]


# --------------------------------------------------------------------------- teste


def test_backtest_round_trip_entry_and_exit_through_single_path() -> None:
    bars = _bars(7)
    h = _harness(bars, {3: "ENTER_LONG", 5: "EXIT"})
    processed = h.engine.run()
    assert processed >= len(bars)
    assert h.clock.now() == bars[-1].ts_close

    entry, exit_ = (d for d in h.engine.decisions if d.action != "NONE")
    assert entry.approved and exit_.approved
    assert entry.client_order_id and exit_.client_order_id
    buy = h.oms.order(entry.client_order_id)
    sell = h.oms.order(exit_.client_order_id)
    assert buy.state is OrderState.FILLED and sell.state is OrderState.FILLED
    assert sell.qty == buy.qty

    state = h.portfolio.snapshot()
    assert state.positions == {}
    assert state.realized.costs.commission == D("0.10")  # comision fix pe fiecare ordin
    assert state.cash_eur == D(100) + state.realized.net_eur

    # Umplerea la deschiderea barei 4 este aplicată înaintea strategiei pe bara 4.
    on_bar = [i for i, s in enumerate(h.trace) if s == "strategy.on_bar"]
    first_fill = h.trace.index("portfolio.apply_fill")
    assert on_bar[2] < first_fill < on_bar[3]

    for d in (entry, exit_):
        chain = reconstruct_decision_chain(h.journal, d.correlation_id)
        assert chain.complete and not chain.risk_rejected
        assert [r.type for r in chain.records][:5] == [
            "market_event",
            "signal",
            "order_intent",
            "risk_decision",
            "broker_request",
        ]
        assert chain.records[-1].type == "execution_event"
    assert verify_journal(h.journal).ok


def test_write_ahead_journal_precedes_broker_submit() -> None:
    seen: list[str] = []

    def check(journal: Journal, oms: OrderManager, req: OrderRequest) -> None:
        records = list(journal.read())
        sent = [r for r in records if r.type == "broker_request"]
        assert sent and sent[-1].payload["client_order_id"] == req.client_order_id
        transitions = [
            r
            for r in records
            if r.type == "oms.TRANSITION"
            and r.correlation_id == req.client_order_id
            and r.payload["to_state"] == "SUBMITTED"
        ]
        assert transitions and transitions[-1].seq < sent[-1].seq
        assert oms.order(req.client_order_id).state is OrderState.SUBMITTED
        seen.append(req.client_order_id)

    bars = _bars(4)
    h = _harness(bars, {2: "ENTER_LONG"}, broker="spy", spy_check=check)
    h.engine.run()
    assert len(seen) == 1
    assert h.inner.submitted[0].client_order_id == seen[0]


def test_risk_rejection_is_journaled_and_nothing_is_sent() -> None:
    bars = _bars(4)
    h = _harness(bars, {2: "ENTER_LONG"}, stop_offset=D(-1))  # stop peste intrare
    h.engine.run()
    (decision,) = (d for d in h.engine.decisions if d.action != "NONE")
    assert decision.approved is False and decision.reason == "RISK_STOP_NOT_BELOW_ENTRY"
    chain = reconstruct_decision_chain(h.journal, decision.correlation_id)
    assert chain.risk_rejected and chain.complete
    assert "broker_request" not in _types(h.journal)
    assert h.inner.snapshot().orders == ()


def test_operator_kill_switch_blocks_new_orders() -> None:
    bars = _bars(4)
    h = _harness(bars, {3: "ENTER_LONG"})
    h.engine.submit_event(
        CommandEvent(
            command="kill_switch.activate",
            actor="alice",
            ts=bars[0].ts_close,
            scope=KillSwitchScope.GLOBAL,
            reason_code="MANUAL",
        )
    )
    h.engine.run()
    (decision,) = (d for d in h.engine.decisions if d.action != "NONE")
    assert decision.approved is False and decision.reason == "RISK_KILL_SWITCH_ACTIVE"
    assert h.inner.snapshot().orders == ()
    types = _types(h.journal)
    assert types.index("kill_switch.activated") < types.index("order_intent")


def test_fail_safe_rejection_is_applied_as_local_reject() -> None:
    bars = _bars(3)
    h = _harness(bars, {2: "ENTER_LONG"}, approved_account="OTHER")
    h.engine.run()
    (decision,) = (d for d in h.engine.decisions if d.action != "NONE")
    assert decision.approved and decision.client_order_id
    assert h.oms.order(decision.client_order_id).state is OrderState.REJECTED_BROKER
    assert h.inner.snapshot().orders == ()  # brokerul nu a văzut cererea
    types = _types(h.journal)
    assert "fail_safe.rejected" in types and "engine.fail_safe_blocked" in types


def test_same_step_sequence_with_sim_and_fake_brokers() -> None:
    bars = _bars(3)
    traces = []
    kinds: tuple[Literal["sim", "fake"], ...] = ("sim", "fake")
    for kind in kinds:
        h = _harness(bars, {3: "ENTER_LONG"}, broker=kind)
        h.engine.run()
        traces.append(h.trace)
    assert traces[0] == traces[1]
    assert "broker.submit" in traces[0] and "oms.on_execution" in traces[0]


def test_fake_broker_execution_updates_portfolio() -> None:
    bars = _bars(3)
    h = _harness(bars, {3: "ENTER_LONG"}, broker="fake", use_data=False)
    for bar in bars:
        h.engine.submit_event(_event(bar))
    h.engine.run(timeout=0)
    (decision,) = (d for d in h.engine.decisions if d.action != "NONE")
    assert decision.client_order_id
    order = h.oms.order(decision.client_order_id)
    assert order.state is OrderState.ACKNOWLEDGED

    fake: FakeBroker = h.inner
    fake.fill(order.client_order_id, order.qty, D("100.00"), D("0.05"))
    assert h.engine.pump_broker() == 1
    h.engine.run(timeout=0)
    assert h.oms.order(order.client_order_id).state is OrderState.FILLED
    position = h.portfolio.snapshot().positions["XYZ"]
    assert position.qty == order.qty
    assert h.portfolio.snapshot().cash_eur == D(100) - order.qty * D("100.00") - D("0.05")
    chain = reconstruct_decision_chain(h.journal, decision.correlation_id)
    assert chain.complete


def test_duplicate_and_invalid_bars_are_excluded_before_strategy() -> None:
    bars = _bars(2)
    h = _harness(bars, {}, use_data=False)
    h.engine.submit_event(_event(bars[0]))
    h.engine.submit_event(_event(bars[0]))  # duplicat
    bad = bars[1].model_copy(update={"high": bars[1].low - D("0.01")})
    h.engine.submit_event(_event(bad))
    h.engine.run(timeout=0)
    assert h.trace.count("strategy.on_bar") == 1
    types = _types(h.journal)
    assert types.count("engine.market_event_ignored") == 1
    assert types.count("engine.bar_rejected") == 1


def test_timer_expires_day_scope_and_shutdown_stops_loop() -> None:
    bars = _bars(1)
    h = _harness(bars, {}, use_data=False)
    t0 = bars[0].ts_close
    h.engine.submit_event(
        CommandEvent(
            command="kill_switch.activate",
            actor="alice",
            ts=t0,
            scope=KillSwitchScope.DAY,
            reason_code="MANUAL",
        )
    )
    next_day = t0 + timedelta(days=1)
    h.engine.submit_event(TimerEvent(name=TIMER_EXPIRE_DAYS, ts=next_day))
    h.engine.submit_event(CommandEvent(command="shutdown", actor="alice", ts=next_day))
    late = TimerEvent(name=TIMER_EXPIRE_DAYS, ts=next_day + timedelta(hours=1))
    h.engine.submit_event(late)  # după shutdown: rămâne neprocesat
    assert h.kill_switch.state(t0).day_active is False
    h.engine.run(timeout=0)
    assert h.engine.stopped
    assert len(h.engine.bus) == 1
    assert h.kill_switch.state(next_day).day_active is False
    types = _types(h.journal)
    assert types.index("kill_switch.activated") < types.index("kill_switch.cleared")
    assert types.count("engine.timer") == 1
