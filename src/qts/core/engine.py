"""Trading_Engine: bucla de evenimente cu un singur fir, identică în toate modurile (Req 1.1, 12.1).

Motorul nu își cunoaște modul. Primește prin constructor porturile (`Clock`, adaptorul broker
deja învelit în `FailSafeBlock`, opțional `DataAdapter`) și componentele nucleului, și nu
importă niciun adaptor concret. Compunerea per mod aparține `bootstrap.py` (sarcina 12.2).

Calea unică de decizie (fără scurtături):

    Market_Event → validare → prospețime → istoric → Strategy → Signal → Order_Intent
        → Risk_Engine → OMS (create, submit write-ahead) → Fail_Safe → Broker_Adapter
    Execution_Event → OMS → Portofoliu → LossMonitor → Kill_Switch

Pașii unui `MarketEvent` de tip bară, în ordine:

1. deduplicare pe cheia canonică și validarea barei (barele invalide sunt jurnalizate și
   excluse; nu reîmprospătează datele);
2. `market_event` în jurnal (write-ahead), apoi prospețime, istoric și marcajul portofoliului;
3. ascultătorii de piață injectați (de exemplu `SimBroker.on_bar`, adăugat de bootstrap în
   Backtest/Shadow), apoi preluarea evenimentelor brokerului în coadă. Execuțiile cu cheie mai
   mică decât evenimentul curent (de exemplu umplerile la deschiderea barei) sunt aplicate
   imediat, înaintea strategiei, conform priorității `Execution_Event < Market_Event`;
4. `LossMonitor` (pierdere zilnică / totală) → `Kill_Switch`;
5. Strategy → `signal` în jurnal; `NONE` se oprește aici;
6. Order_Intent (`ENTER_LONG` → BUY cu stop; `EXIT` → SELL cu cantitatea deținută) →
   `order_intent` → Risk_Engine → `risk_decision`; respingerile se opresc aici;
7. OMS `create_order` și `submit` (tranziția `SUBMITTED` este jurnalizată înainte), apoi
   `broker_request` în jurnal și abia apoi `broker.submit`.

Corelare: fiecare lanț de decizie folosește ca `correlation_id` identificatorul determinist al
evenimentului de piață (`market_event_id`), comun înregistrărilor `market_event`, `signal`,
`order_intent`, `risk_decision`, `broker_request` și `execution_event`, deci lanțul se
reconstruiește cu `persistence.audit.reconstruct_decision_chain`. `signal_id` și
`client_order_id` apar în payload; înregistrările OMS folosesc `client_order_id`.

Transmitere:
- `FailSafeRejectedError`: brokerul nu a văzut ordinul; motorul aplică în OMS un
  `ExecutionEvent` local `REJECT` (`broker_exec_id = "local:fail_safe:<coid>"`), jurnalizat;
- `TimeoutError` / `ConnectionError`: rezultat necunoscut → `OrderManager.submit_timeout`
  (`UNKNOWN`, ordin înghețat până la reconciliere).

Timp: motorul nu citește ceasul de perete. Dacă primește `clock_driver` (în Backtest
`SimClock`), îl avansează la cheia fiecărui eveniment, numai înainte (timpii mai vechi, de
exemplu execuțiile la deschiderea barei, nu întorc ceasul).

`run()` cu `DataAdapter`: fluxul (ordonat temporal) este interclasat cu coada, ținând mereu
exact următorul eveniment de date în coadă; rularea se termină când fluxul și coada sunt goale.
Fără `DataAdapter`, `run()` consumă coada (alimentată de alte fire prin `submit_event`) până la
închiderea ei sau comanda `shutdown`.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Any, Final, Protocol

from qts.broker.adapter import CancelAck, OrderRequest, SubmitAck, TimeInForce, order_request_from
from qts.broker.fail_safe import FailSafeRejectedError
from qts.config.schema import Environment
from qts.core.bus import (
    BusKey,
    CommandEvent,
    Envelope,
    EventBus,
    TimerEvent,
)
from qts.core.clock import Clock
from qts.core.models import (
    Bar,
    ExecKind,
    ExecutionEvent,
    Frozen,
    Instrument,
    MarketEvent,
    OrderIntent,
    Quote,
    Signal,
    SignalAction,
)
from qts.core.money import REPORTING_CURRENCY, ZERO
from qts.costs.errors import CostModelIncomplete
from qts.costs.model import CostContext, CostModel, OrderSpec
from qts.data.adapter import DataAdapter
from qts.data.freshness import FreshnessTracker
from qts.data.gaps import GapPolicy
from qts.data.normalize import Deduplicator, DedupOutcome
from qts.data.universe_history import UniverseHistory
from qts.data.validate import BarValidator
from qts.oms.manager import InstrumentBlockedError, OrderManager, UnknownOrderError
from qts.persistence.journal import Journal
from qts.portfolio.pnl import Mark
from qts.portfolio.portfolio import Portfolio, PortfolioError, PortfolioFill
from qts.risk.context import KillSwitchScope, MarketSnapshot, ProjectStageName, RiskContext
from qts.risk.engine import RiskDecision, RiskEngine
from qts.risk.limits import RejectReason
from qts.risk.monitor import LossMonitor
from qts.safety.kill_switch import KillSwitch
from qts.strategy.base import Strategy, StrategyState
from qts.strategy.history_view import HistoryView

__all__ = [
    "COMPONENT",
    "ENGINE_VERSION",
    "TIMER_EXPIRE_DAYS",
    "AdvanceableClock",
    "Decision",
    "EngineBroker",
    "EngineConfig",
    "MarketContextFn",
    "MarketListener",
    "TradingEngine",
    "market_event_id",
]

ENGINE_VERSION: Final = "1"
COMPONENT: Final = "engine"
ACTOR: Final = "system:engine"
TIMER_EXPIRE_DAYS: Final = "kill_switch.expire_days"

MarketListener = Callable[[MarketEvent], None]
MarketContextFn = Callable[[Instrument, Bar], CostContext]
StepTrace = Callable[[str], None]


class EngineBroker(Protocol):
    """Ce folosește motorul din `BrokerAdapter` (învelit de `FailSafeBlock`)."""

    def submit(self, req: OrderRequest, /) -> SubmitAck: ...

    def cancel(self, client_order_id: str, /) -> CancelAck: ...

    def events(self) -> Iterator[ExecutionEvent]: ...


class AdvanceableClock(Protocol):
    def now(self) -> datetime: ...

    def advance_to(self, ts: datetime) -> None: ...


class EngineConfig(Frozen):
    run_id: str
    mode: Environment  # transmis numai Risk_Engine (regula modurilor permise pe etapă)
    project_stage: ProjectStageName = "initial"
    broker_name: str = "sim"  # cheia tabelelor de comisioane în `CostContext`
    instruments: tuple[Instrument, ...]
    time_in_force: TimeInForce = "GTC"


@dataclass(frozen=True, slots=True)
class Decision:
    """Rezumatul unui lanț de decizie, pentru inspecție și audit."""

    correlation_id: str
    signal_id: str
    instrument: str
    action: SignalAction
    intent_id: str | None = None
    approved: bool | None = None
    reason: str | None = None
    client_order_id: str | None = None


@dataclass(frozen=True, slots=True)
class _OrderLink:
    correlation_id: str
    intent: OrderIntent


def market_event_id(event: MarketEvent) -> str:
    """Identificator determinist al evenimentului de piață (cheia canonică, Req 5.7)."""
    raw = f"{event.source_id}|{event.instrument}|{event.ts_source.isoformat()}|{event.seq}"
    return "mkt-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _json(model: Frozen) -> dict[str, Any]:
    data: dict[str, Any] = model.model_dump(mode="json")
    return data


class TradingEngine:
    """Motorul event-driven; aceeași succesiune de pași în Backtest, Shadow și Demo."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        clock: Clock,
        broker: EngineBroker,
        journal: Journal,
        strategy: Strategy,
        risk: RiskEngine,
        oms: OrderManager,
        portfolio: Portfolio,
        kill_switch: KillSwitch,
        cost_model: CostModel,
        freshness: FreshnessTracker,
        loss_monitor: LossMonitor | None = None,
        data: DataAdapter | None = None,
        bus: EventBus | None = None,
        validator: BarValidator | None = None,
        dedup: Deduplicator | None = None,
        market_context: MarketContextFn | None = None,
        market_listeners: Sequence[MarketListener] = (),
        clock_driver: AdvanceableClock | None = None,
        trace: StepTrace | None = None,
        gap_policy: GapPolicy | None = None,
        universe_history: UniverseHistory | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._broker = broker
        self._journal = journal
        self._strategy = strategy
        self._risk = risk
        self._oms = oms
        self._portfolio = portfolio
        self._kill_switch = kill_switch
        self._costs = cost_model
        self._freshness = freshness
        self._loss_monitor = loss_monitor
        self._data = data
        self._bus = bus if bus is not None else EventBus()
        self._validator = validator if validator is not None else BarValidator()
        self._dedup = dedup if dedup is not None else Deduplicator()
        self._market_context = market_context or self._default_market_context
        self._listeners = tuple(market_listeners)
        self._clock_driver = clock_driver
        self._trace = trace
        self._gap_policy = gap_policy
        self._universe_history = universe_history
        self._instruments = {i.symbol: i for i in config.instruments}
        self._history: HistoryView | None = None
        self._last_bar: dict[str, Bar] = {}
        self._cost_ctx: dict[str, CostContext] = {}
        self._quotes: dict[str, Quote] = {}
        self._strategy_state: StrategyState = strategy.initial_state()
        self._signal_seq = 0
        self._orders: dict[str, _OrderLink] = {}
        self._stops: dict[str, Decimal] = {}
        self._decisions: list[Decision] = []
        self._processed = 0
        self._stopped = False
        self._failed: Exception | None = None

    # ------------------------------------------------------------------ interogări

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def decisions(self) -> tuple[Decision, ...]:
        return tuple(self._decisions)

    @property
    def strategy_state(self) -> StrategyState:
        return self._strategy_state

    @property
    def processed(self) -> int:
        return self._processed

    @property
    def stopped(self) -> bool:
        return self._stopped

    def correlation_for(self, client_order_id: str) -> str | None:
        link = self._orders.get(client_order_id)
        return link.correlation_id if link is not None else None

    def active_universe(self, ts: datetime) -> frozenset[str] | None:
        """Componența universului la timpul simulat `ts`, dacă există istoric (Req 7.5).

        Întoarce `None` când nu s-a injectat `UniverseHistory` (universul static din
        configurație rămâne în vigoare). Seam consultat de drivere/`bootstrap` atunci când
        sursa de date oferă istoricul componenței, inclusiv instrumente delistate valabile la
        timpul simulat.
        """
        if self._universe_history is None:
            return None
        return self._universe_history.active_universe(ts)

    # ------------------------------------------------------------------ intrări

    def submit_event(self, event: MarketEvent | ExecutionEvent | TimerEvent | CommandEvent) -> None:
        """Pune un eveniment în coadă; sigur din orice fir (adaptoare I/O, operator)."""
        self._bus.put(event)

    def pump_broker(self) -> int:
        """Preia evenimentele brokerului în coadă; întoarce numărul lor."""
        count = 0
        try:
            for event in self._broker.events():
                self._bus.put(event, internal=True)
                count += 1
        except ConnectionError as exc:
            self._record(
                "engine.broker_events_unavailable",
                "broker",
                "unavailable",
                {"error": type(exc).__name__, "detail": str(exc)},
                self._clock.now(),
            )
        return count

    # ------------------------------------------------------------------ buclă

    def run(self, *, max_events: int | None = None, timeout: float | None = None) -> int:
        """Procesează evenimente; întoarce câte au fost procesate în acest apel."""
        start = self._processed
        stream = iter(self._data.stream()) if self._data is not None else None
        sync = self._data is not None
        feed: Envelope | None = None
        while not self._stopped:
            if max_events is not None and self._processed - start >= max_events:
                break
            if stream is not None and feed is None:
                nxt = next(stream, None)
                if nxt is None:
                    stream = None
                else:
                    feed = self._bus.put(nxt, internal=True)
            env = self._bus.get_nowait() if sync else self._bus.get(timeout)
            if env is None:
                break
            if env is feed:
                feed = None
            self.step(env)
        return self._processed - start

    def step(self, env: Envelope) -> None:
        """Aplică un eveniment, atomic din punctul de vedere al firului motorului."""
        self._advance(env.ts)
        event = env.event
        try:
            if isinstance(event, MarketEvent):
                self._on_market(event, env.key)
            elif isinstance(event, ExecutionEvent):
                self._on_execution(event)
            elif isinstance(event, TimerEvent):
                self._on_timer(event)
            else:
                self._on_command(event)
            now = self._clock.now()
            self._apply_cancels(now)
            self.pump_broker()
        except Exception as exc:
            self._fail_closed(env, exc)
            raise
        self._processed += 1

    # ------------------------------------------------------------------ Market_Event

    def _on_market(self, event: MarketEvent, key: BusKey) -> None:
        self._step("market.receive")
        now = self._clock.now()
        chain = market_event_id(event)
        outcome = self._dedup.offer(event)
        if outcome is not DedupOutcome.NEW:
            self._record("engine.market_event_ignored", chain, outcome.value, _json(event), now)
            return
        inst = self._instruments.get(event.instrument)
        if inst is None:
            self._record("engine.market_event_ignored", chain, "UNKNOWN_INSTRUMENT", {}, now)
            return
        payload = event.payload
        if not isinstance(payload, Bar):
            self._record("market_event", chain, event.kind, _json(event), now)
            self._freshness.record(event)
            if isinstance(payload, Quote):
                self._quotes[inst.symbol] = payload
            return

        self._step("market.validate")
        verdict = self._validator.validate(payload)
        if verdict.bar is None:
            reason = verdict.reason.value if verdict.reason is not None else "BAR_INVALID"
            rejected = {"detail": verdict.detail, **_json(event)}
            self._record("engine.bar_rejected", chain, reason, rejected, now)
            return
        bar = verdict.bar
        self._step("market.journal")
        self._record("market_event", chain, "accepted", _json(event), now)
        self._freshness.record(event)
        boundary = self._on_gap_boundary(bar, chain, now)
        self._history = self._extend_history(inst.symbol, bar, reset=boundary)
        self._last_bar[inst.symbol] = bar
        ctx = self._market_context(inst, bar)
        self._cost_ctx[inst.symbol] = ctx
        self._mark(inst, bar, ctx)

        self._step("market.listeners")
        for listener in self._listeners:
            listener(event)
        self.pump_broker()
        self._drain_before(key)

        self._step("risk.monitor")
        self._observe_losses(now)

        self._step("strategy.on_bar")
        signal, self._strategy_state = self._strategy.on_bar(
            bar, self._history, self._strategy_state
        )
        self._step("signal.journal")
        self._record("signal", chain, signal.action, _json(signal), now)
        # Prima bară de după un gol peste prag deschide o subperioadă nouă: starea și istoricul
        # instrumentului au fost deja resetate, iar generarea de ordine este suspendată pentru
        # această bară, astfel încât niciun ordin să nu traverseze golul (Req 7.4).
        if boundary:
            self._record("engine.subperiod_order_suppressed", chain, signal.action, {}, now)
            self._decisions.append(
                Decision(chain, signal.signal_id, signal.instrument, signal.action)
            )
            return
        self._act(signal, bar, chain, now)

    def _on_gap_boundary(self, bar: Bar, chain: str, now: datetime) -> bool:
        """Clasifică bara cu `GapPolicy`; la o graniță peste prag resetează starea strategiei.

        Întoarce `True` numai pentru prima bară a unei subperioade deschise de un gol peste
        prag. Fără `gap_policy` sau pentru goluri sub prag întoarce `False` și nu schimbă nimic
        (Req 7.7). La graniță jurnalizează invalidarea subperioadei încheiate (Req 7.4) și
        readuce starea strategiei la cea inițială, ca niciun semnal să nu traverseze golul.
        """
        policy = self._gap_policy
        if policy is None:
            return False
        mark = policy.observe(bar)
        if not mark.boundary:
            return False
        self._record(
            "engine.subperiod_invalidated",
            chain,
            "invalidated",
            {
                "instrument": mark.instrument,
                "invalidated_subperiod_id": mark.subperiod_id - 1,
                "new_subperiod_id": mark.subperiod_id,
                "missing_bars": mark.missing_before,
                "max_gap_bars": policy.max_gap_bars,
            },
            now,
        )
        self._strategy_state = self._strategy.initial_state()
        return True

    def _extend_history(self, symbol: str, bar: Bar, *, reset: bool) -> HistoryView:
        """Adaugă bara la istoric; la o graniță resetează seria instrumentului (fără cross-gap)."""
        if self._history is None:
            return HistoryView({bar.instrument: [bar]}, bar.ts_close)
        if not reset:
            return self._history.append(bar)
        # Reconstruiește istoricul păstrând celelalte instrumente și repornind seria afectată.
        series: dict[str, list[Bar]] = {
            sym: list(self._history.last(sym, self._history.count(sym)))
            for sym in self._history.instruments()
            if sym != symbol
        }
        series[symbol] = [bar]
        return HistoryView(series, bar.ts_close)

    def _mark(self, inst: Instrument, bar: Bar, ctx: CostContext) -> None:
        fx = Decimal(1) if inst.currency == REPORTING_CURRENCY else ctx.fx_rate
        if fx is not None:
            self._portfolio.apply_mark(
                inst.symbol, Mark(price=bar.close, fx_rate_to_eur=fx, ts=bar.ts_close)
            )

    def _drain_before(self, key: BusKey) -> None:
        """Aplică evenimentele din coadă cu cheie mai mică decât evenimentul curent."""
        while (head := self._bus.peek_key()) is not None and head < key:
            env = self._bus.get_nowait()
            if env is None:  # pragma: no cover - un singur consumator
                return
            self.step(env)

    # ------------------------------------------------------------------ Signal → ordin

    def _act(self, signal: Signal, bar: Bar, chain: str, now: datetime) -> None:
        decision = Decision(chain, signal.signal_id, signal.instrument, signal.action)
        intent = self._intent_for(signal, bar)
        if intent is None:
            self._decisions.append(decision)
            return
        self._step("intent.journal")
        self._record("order_intent", chain, "created", _json(intent), now)

        self._step("risk.evaluate")
        verdict = self._evaluate(intent, now)
        self._step("risk.journal")
        self._record(
            "risk_decision",
            chain,
            "approved" if verdict.approved else "rejected",
            {"signal_id": signal.signal_id, **_json(verdict)},
            now,
        )
        decision = replace(
            decision,
            intent_id=intent.intent_id,
            approved=verdict.approved,
            reason=verdict.reason.value if verdict.reason is not None else None,
        )
        if not verdict.approved:
            self._decisions.append(decision)
            return

        self._step("oms.create_order")
        self._signal_seq += 1
        try:
            created = self._oms.create_order(
                intent,
                verdict,
                run_id=self._config.run_id,
                strategy_id=signal.strategy_id,
                signal_seq=self._signal_seq,
                ts=now,
            )
        except InstrumentBlockedError as exc:
            self._record("engine.order_blocked", chain, "blocked", {"detail": str(exc)}, now)
            self._decisions.append(decision)
            return
        coid = created.order.client_order_id
        self._orders.setdefault(coid, _OrderLink(chain, intent))
        self._decisions.append(replace(decision, client_order_id=coid))
        self._send(coid, now)

    def _intent_for(self, signal: Signal, bar: Bar) -> OrderIntent | None:
        intent_id = f"{signal.signal_id}:intent"
        if signal.action == "ENTER_LONG":
            return OrderIntent(
                intent_id=intent_id,
                signal_id=signal.signal_id,
                instrument=signal.instrument,
                side="BUY",
                ref_price=bar.close,
                stop_price=signal.stop_price,
            )
        if signal.action == "EXIT":
            position = self._portfolio.snapshot().positions.get(signal.instrument)
            return OrderIntent(
                intent_id=intent_id,
                signal_id=signal.signal_id,
                instrument=signal.instrument,
                side="SELL",
                ref_price=bar.close,
                requested_qty=position.qty if position is not None else ZERO,
            )
        return None

    def _evaluate(self, intent: OrderIntent, now: datetime) -> RiskDecision:
        try:
            ctx = self._risk_context(intent, now)
        except CostModelIncomplete as exc:
            reason, detail = RejectReason.COST_MODEL_INCOMPLETE, f"{exc.component}: {exc.detail}"
        except PortfolioError as exc:
            reason, detail = RejectReason.DATA_MISSING, str(exc)
        else:
            return self._risk.evaluate(intent, ctx)
        # Contextul nu poate fi construit complet: respingere (fail-closed).
        return RiskDecision(
            intent_id=intent.intent_id,
            approved=False,
            reason=reason,
            detail=detail,
            is_exit=intent.side == "SELL",
        )

    def _risk_context(self, intent: OrderIntent, now: datetime) -> RiskContext:
        state = self._portfolio.snapshot()
        market: dict[str, MarketSnapshot] = {}
        for sym in sorted({intent.instrument, *state.positions}):
            inst = self._instruments.get(sym)
            if inst is None or sym not in self._cost_ctx:
                continue
            fresh = self._freshness.check(sym)
            market[sym] = MarketSnapshot(
                instrument=inst,
                data_fresh=fresh.ok,
                freshness_reason=fresh.reason.value if fresh.reason is not None else None,
                quote=self._quotes.get(sym),
                cost_ctx=self._cost_ctx[sym],
            )
        exit_costs: dict[str, Decimal] = {}
        for sym, pos in sorted(state.positions.items()):
            inst = self._instruments.get(sym)
            if inst is None:
                continue  # fără cost de ieșire → PortfolioError (fail-closed)
            spec = OrderSpec(
                instrument=inst, side="SELL", qty=pos.qty, ts=now, ref_price=state.marks[sym].price
            )
            ctx = self._cost_ctx.get(sym) or self._default_market_context(inst, None)
            exit_costs[sym] = self._costs.estimate(spec, self._quotes.get(sym), ctx).total
        accounting = self._portfolio.risk_accounting(
            now, exit_costs_eur=exit_costs, stops=dict(self._stops)
        )
        ks = self._kill_switch.state(now)
        if self._loss_monitor is not None:
            ks = self._loss_monitor.kill_switch_state(now, ks)
        return RiskContext(
            ts=now,
            mode=self._config.mode,
            project_stage=self._config.project_stage,
            cash_eur=accounting.cash_eur,
            positions=accounting.positions,
            daily_realized_pnl_eur=accounting.daily_realized_pnl_eur,
            daily_unrealized_pnl_eur=accounting.daily_unrealized_pnl_eur,
            total_pnl_eur=accounting.total_pnl_eur,
            market=market,
            kill_switch=ks,
        )

    def _send(self, coid: str, now: datetime) -> None:
        self._step("oms.submit")
        result = self._oms.submit(coid, now)  # write-ahead: SUBMITTED jurnalizat
        if not result.send:
            return
        link = self._orders[coid]
        req = order_request_from(
            result.order, link.intent, time_in_force=self._config.time_in_force
        )
        self._step("broker.journal")
        self._record("broker_request", link.correlation_id, "sent", _json(req), now)
        self._step("broker.submit")
        try:
            ack = self._broker.submit(req)
        except FailSafeRejectedError as exc:
            self._record(
                "engine.fail_safe_blocked",
                link.correlation_id,
                "blocked",
                {"client_order_id": coid, "reason": str(exc.reason), "detail": exc.detail},
                now,
            )
            self._on_execution(
                ExecutionEvent(
                    broker_exec_id=f"local:fail_safe:{coid}",
                    client_order_id=coid,
                    kind=ExecKind.REJECT,
                    reason=str(exc.reason),
                    ts_broker=now,
                    ts_receipt=now,
                )
            )
            return
        except (TimeoutError, ConnectionError) as exc:
            self._record(
                "engine.broker_submit_unknown",
                link.correlation_id,
                "unknown",
                {"client_order_id": coid, "error": type(exc).__name__},
                now,
            )
            self._oms.submit_timeout(coid, now)
            return
        self._record(
            "engine.broker_ack",
            link.correlation_id,
            "accepted" if ack.accepted else "rejected",
            _json(ack),
            now,
        )
        self.pump_broker()

    # ------------------------------------------------------------------ Execution_Event

    def _on_execution(self, event: ExecutionEvent) -> None:
        now = self._clock.now()
        link = self._orders.get(event.client_order_id)
        chain = link.correlation_id if link is not None else event.client_order_id
        self._step("execution.journal")
        self._record("execution_event", chain, event.kind.value, _json(event), now)
        self._step("oms.on_execution")
        for outcome in self._oms.on_execution(event):
            if outcome.request is not None:
                self._record(
                    "engine.missing_messages",
                    outcome.request.client_order_id,
                    "requested",
                    {"missing_seqs": list(outcome.request.missing_seqs)},
                    now,
                )
            if outcome.fill is not None:
                self._apply_fill(outcome.fill, outcome.event.client_order_id, now)
        self._step("risk.monitor")
        self._observe_losses(now)

    def _apply_fill(self, fill: PortfolioFill, coid: str, now: datetime) -> None:
        self._step("portfolio.apply_fill")
        try:
            state = self._portfolio.apply_fill(fill)
        except PortfolioError as exc:
            self._record(
                "engine.portfolio_rejected",
                coid,
                "rejected",
                {"fill": _json(fill), "detail": str(exc)},
                now,
            )
            return
        link = self._orders.get(coid)
        if fill.side == "BUY" and link is not None and link.intent.stop_price is not None:
            self._stops[fill.instrument] = link.intent.stop_price
        if fill.instrument not in state.positions:
            self._stops.pop(fill.instrument, None)

    def _observe_losses(self, now: datetime) -> None:
        if self._loss_monitor is not None:
            self._loss_monitor.observe(self._portfolio.loss_observation(now))

    # ------------------------------------------------------------------ Timer / Command

    def _on_timer(self, event: TimerEvent) -> None:
        self._step("timer")
        known = event.name == TIMER_EXPIRE_DAYS
        self._record(
            "engine.timer",
            f"timer:{event.name}",
            "applied" if known else "ignored",
            _json(event),
            event.ts,
        )
        if known:
            self._kill_switch.expire_days(event.ts)

    def _on_command(self, event: CommandEvent) -> None:
        self._step("command")
        correlation = f"command:{event.command}"
        if event.command == "shutdown":
            self._record("engine.command", correlation, "applied", _json(event), event.ts)
            self._stopped = True
            return
        if event.scope is None or event.scope is KillSwitchScope.CAPITAL_CONFIG:
            self._record("engine.command", correlation, "rejected", _json(event), event.ts)
            return
        self._record("engine.command", correlation, "applied", _json(event), event.ts)
        self._kill_switch.activate_manual(
            event.scope,
            operator=event.actor,
            reason_code=event.reason_code,
            detail=event.detail,
            instrument=event.instrument,
        )

    def _apply_cancels(self, now: datetime) -> None:
        """Politica `cancel` a Kill_Switch: OMS întâi (write-ahead), apoi brokerul."""
        for cmd in self._kill_switch.drain_cancel_commands():
            coid = cmd.client_order_id
            correlation = self.correlation_for(coid) or coid
            self._step("oms.request_cancel")
            try:
                result = self._oms.request_cancel(coid, now)
            except UnknownOrderError:
                self._record("engine.cancel_unknown_order", coid, "ignored", _json(cmd), now)
                continue
            if not result.send:
                continue
            self._step("broker.cancel")
            self._record("engine.broker_cancel", correlation, "sent", _json(cmd), now)
            try:
                ack = self._broker.cancel(coid)
            except (TimeoutError, ConnectionError) as exc:
                self._record(
                    "engine.broker_cancel_unknown",
                    correlation,
                    "unknown",
                    {"client_order_id": coid, "error": type(exc).__name__},
                    now,
                )
                continue
            self._record(
                "engine.cancel_ack",
                correlation,
                "accepted" if ack.accepted else "rejected",
                _json(ack),
                now,
            )

    # ------------------------------------------------------------------ intern

    def _default_market_context(self, inst: Instrument, bar: Bar | None) -> CostContext:
        return CostContext(
            broker=self._config.broker_name,
            bar_interval_min=bar.interval_min if bar is not None else None,
        )

    def _advance(self, ts: datetime) -> None:
        driver = self._clock_driver
        if driver is not None and ts > driver.now():
            driver.advance_to(ts)

    def _step(self, name: str) -> None:
        if self._trace is not None:
            self._trace(name)

    def _record(
        self,
        type_: str,
        correlation_id: str,
        outcome: str,
        payload: dict[str, Any],
        ts: datetime,
        *,
        actor: str = ACTOR,
    ) -> None:
        self._journal.append(
            ts=ts,
            type=type_,
            correlation_id=correlation_id,
            component=COMPONENT,
            component_version=ENGINE_VERSION,
            actor=actor,
            outcome=outcome,
            payload=payload,
        )

    def _fail_closed(self, env: Envelope, exc: Exception) -> None:
        """Eroare neprevăzută: blochează ordinele noi (GLOBAL) și jurnalizează, best effort."""
        if exc is self._failed:  # deja tratată într-un pas imbricat
            return
        self._failed = exc
        detail = f"{type(exc).__name__}: {exc}"
        # Blocarea are loc în memorie înaintea persistenței; jurnalul poate fi chiar cauza.
        with contextlib.suppress(Exception):
            self._kill_switch.activate_automatic(
                KillSwitchScope.GLOBAL, component=COMPONENT, reason_code="ENGINE_ERROR"
            )
        with contextlib.suppress(Exception):
            self._record(
                "engine.error",
                f"engine:{env.seq}",
                "error",
                {"kind": env.kind.name, "detail": detail[:500]},
                self._clock.now(),
            )
