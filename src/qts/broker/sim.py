"""`SimBroker`: execuție simulată locală pentru Backtest și Shadow (Req 3.1, 3.4, 7.3).

Modelul de execuție (versiunea `SIM_EXECUTION_MODEL`)
    - Un ordin acceptat la momentul `t_sub` (închiderea barei *t*) devine eligibil la prima bară
      a instrumentului cu `ts_open >= t_sub`, adică bara *t+1*. Barele anterioare nu îl ating.
    - Latența: prețul după latență este deschiderea barei eligibile (design, „Latență”);
      `latency_ms` din modelul de costuri nu întârzie suplimentar execuția simulată.
    - Prețul MARKET = `open ± (jumătate de spread + slippage pe unitate)`, rotunjit advers la
      tick (BUY în sus, SELL în jos). Jumătatea de spread vine din cotația barei, dacă există
      (înmulțită cu `quote_spread_multiplier`), altfel din `SpreadSchedule` la ora UTC a
      deschiderii. Slippage = `k × σ_bar × sqrt(qty / ADV) × open + min_ticks × tick`, ca în
      `CompleteCostModel`. O componentă lipsă ridică `CostModelIncomplete` (Req 8.3).
    - LIMIT: BUY se execută la prețul MARKET ajustat dacă acesta este ≤ limită; altfel la limită
      numai dacă `low < limită` (strict, atingerea nu ajunge). SELL simetric cu `high > limită`.
    - Execuții parțiale: cantitatea pe bară este limitată de `max_participation × volum` și de
      `max_fill_qty_per_bar`, rotunjită în jos la `qty_step`; restul continuă la barele
      următoare. Fără limite, ordinul se execută integral la prima bară eligibilă.
    - Anularea are efect imediat (`CANCELLED` pentru restul neexecutat). Anularea unui ordin
      terminal produce `CANCEL_REJECTED` (`ORDER_NOT_OPEN`), iar repetarea unei anulări
      confirmate întoarce aceeași confirmare, fără eveniment nou.

Evenimente
    `ExecutionEvent.seq` este numerotat *pe ordin*, de la 1 (ACK sau REJECT), apoi execuții și
    confirmări de anulare. `broker_exec_id = "{account_id}:{client_order_id}:{seq}"`, iar
    `broker_order_id = "SIM-{n:08d}"` în ordinea acceptării: totul este determinist.

Comision și numerar
    Comisionul raportat în `ExecutionEvent.commission` este în EUR (moneda de raportare), așa
    cum presupune `qts.oms.manager.commission_only`. Se calculează din tabelul de comisioane
    pe valoarea nominală *cumulată* a ordinului, iar fiecare execuție raportează diferența față
    de comisionul deja raportat; astfel minimul și taxa fixă se aplică o singură dată per ordin.
    Numerarul (EUR) se modifică cu nominalul convertit și comisionul; spreadul de conversie FX
    și taxele sunt contabilizate de modelul de costuri (`realize`), nu de numerarul simulat.
    Nu există verificare de marjă: suficiența capitalului ține de Risk_Engine.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext
from typing import Final

from pydantic import model_validator

from qts.core.clock import Clock
from qts.core.models import (
    Bar,
    Dec,
    ExecKind,
    ExecutionEvent,
    Frozen,
    Instrument,
    OrderState,
    Quote,
)
from qts.core.money import (
    REPORTING_CURRENCY,
    ZERO,
    ceil_to_step,
    floor_to_step,
    quantize_money,
)
from qts.costs.errors import CostModelIncomplete, cost_context
from qts.costs.model import CompleteCostModel

from .adapter import (
    BrokerCapabilities,
    BrokerOrderStatus,
    BrokerSnapshot,
    CancelAck,
    Environment,
    OrderRequest,
    ReasonCode,
    SubmitAck,
    check_request,
)

__all__ = [
    "DEFAULT_SIM_CAPABILITIES",
    "SIM_EXECUTION_MODEL",
    "SimBarContext",
    "SimBroker",
    "SimBrokerConfig",
]

SIM_EXECUTION_MODEL: Final = "sim-exec-v1"

DEFAULT_SIM_CAPABILITIES: Final = BrokerCapabilities(
    order_types=frozenset({"MARKET", "LIMIT"}),
    # EXPIRED este valid în FSM numai din ACKNOWLEDGED, deci DAY/IOC (expirare după execuții
    # parțiale) nu sunt oferite de simulator.
    time_in_force=frozenset({"GTC"}),
    asset_classes=frozenset({"etf", "stock", "fx", "commodity_etp", "index_etp", "crypto"}),
    fractional=True,
    short=False,
    leverage=False,
    derivatives=False,
)

_OPEN_STATES: Final = frozenset({OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED})


class SimBrokerConfig(Frozen):
    broker: str = "sim"  # cheia tabelelor de comisioane
    initial_cash: Dec = ZERO  # EUR
    max_participation: Dec | None = None  # fracție din volumul barei, în (0, 1]
    max_fill_qty_per_bar: Dec | None = None
    capabilities: BrokerCapabilities = DEFAULT_SIM_CAPABILITIES

    @model_validator(mode="after")
    def _check(self) -> SimBrokerConfig:
        if self.max_participation is not None and not 0 < self.max_participation <= 1:
            raise ValueError("max_participation trebuie să fie în (0, 1]")
        if self.max_fill_qty_per_bar is not None and self.max_fill_qty_per_bar <= 0:
            raise ValueError("max_fill_qty_per_bar trebuie să fie > 0")
        return self


class SimBarContext(Frozen):
    """Date de piață auxiliare pentru execuția la o bară."""

    quote: Quote | None = None  # cotația la deschidere; altfel spreadul configurat
    sigma_bar: Dec | None = None
    adv: Dec | None = None
    fx_rate: Dec | None = None  # EUR per unitate din moneda instrumentului


@dataclass(slots=True)
class _SimOrder:
    req: OrderRequest
    ack: SubmitAck
    submitted_at: datetime
    state: OrderState
    next_seq: int = 1
    filled_qty: Decimal = ZERO
    notional_ccy: Decimal = ZERO  # Σ preț × cantitate, moneda instrumentului
    notional_eur: Decimal = ZERO
    commission_eur: Decimal = ZERO  # comisionul deja raportat
    cancel_ack: CancelAck | None = None

    @property
    def remaining(self) -> Decimal:
        return self.req.qty - self.filled_qty

    def status(self) -> BrokerOrderStatus:
        avg = None
        if self.filled_qty > 0:
            with localcontext(cost_context()):
                avg = self.notional_ccy / self.filled_qty
        return BrokerOrderStatus(
            client_order_id=self.req.client_order_id,
            instrument=self.req.instrument,
            side=self.req.side,
            qty=self.req.qty,
            state=self.state,
            filled_qty=self.filled_qty,
            avg_fill_price=avg,
            broker_order_id=self.ack.broker_order_id,
            reason_code=self.ack.reason_code,
        )


class SimBroker:
    """Broker simulat determinist, condus de bare (`on_bar`) și de un `Clock` injectat."""

    def __init__(
        self,
        *,
        account_id: str,
        instruments: Iterable[Instrument],
        cost_model: CompleteCostModel,
        clock: Clock,
        config: SimBrokerConfig | None = None,
    ) -> None:
        self._account_id = account_id
        self._instruments = {i.symbol: i for i in instruments}
        self._cost_model = cost_model
        self._clock = clock
        self._config = config if config is not None else SimBrokerConfig()
        self._orders: dict[str, _SimOrder] = {}
        self._queue: deque[ExecutionEvent] = deque()
        self._exec_ids: list[str] = []
        self._positions: dict[str, Decimal] = {}
        self._cash: Decimal = self._config.initial_cash
        self._accepted = 0

    # ------------------------------------------------------------------ BrokerAdapter

    @property
    def environment(self) -> Environment:
        return "sim"

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def execution_model(self) -> str:
        return SIM_EXECUTION_MODEL

    def capabilities(self) -> BrokerCapabilities:
        return self._config.capabilities

    def submit(self, req: OrderRequest) -> SubmitAck:
        existing = self._orders.get(req.client_order_id)
        if existing is not None:
            if existing.req == req:
                return existing.ack
            return SubmitAck(
                client_order_id=req.client_order_id,
                accepted=False,
                ts=self._clock.now(),
                reason_code=ReasonCode.CLIENT_ORDER_ID_CONFLICT,
                detail="client_order_id refolosit cu alt conținut",
            )
        now = self._clock.now()
        rejection = check_request(
            self._config.capabilities,
            req,
            self._instruments.get(req.instrument),
            available_qty=self._available_to_sell(req.instrument),
        )
        if rejection is not None:
            ack = SubmitAck(
                client_order_id=req.client_order_id,
                accepted=False,
                ts=now,
                reason_code=rejection.code,
                detail=rejection.detail,
            )
            order = _SimOrder(req, ack, now, OrderState.REJECTED_BROKER)
            self._orders[req.client_order_id] = order
            self._emit(order, ExecKind.REJECT, now, reason=rejection.code.value)
            return ack
        self._accepted += 1
        ack = SubmitAck(
            client_order_id=req.client_order_id,
            accepted=True,
            ts=now,
            broker_order_id=f"SIM-{self._accepted:08d}",
        )
        order = _SimOrder(req, ack, now, OrderState.ACKNOWLEDGED)
        self._orders[req.client_order_id] = order
        self._emit(order, ExecKind.ACK, now)
        return ack

    def cancel(self, client_order_id: str) -> CancelAck:
        now = self._clock.now()
        order = self._orders.get(client_order_id)
        if order is None:
            return CancelAck(
                client_order_id=client_order_id,
                accepted=False,
                ts=now,
                reason_code=ReasonCode.UNKNOWN_ORDER,
                detail="ordin necunoscut",
            )
        if order.cancel_ack is not None:
            return order.cancel_ack
        if order.state not in _OPEN_STATES:
            detail = f"ordinul este în starea {order.state.value}"
            self._emit(order, ExecKind.CANCEL_REJECTED, now, reason=ReasonCode.ORDER_NOT_OPEN.value)
            return CancelAck(
                client_order_id=client_order_id,
                accepted=False,
                ts=now,
                reason_code=ReasonCode.ORDER_NOT_OPEN,
                detail=detail,
            )
        remaining = order.remaining
        order.state = OrderState.CANCELLED
        order.cancel_ack = CancelAck(client_order_id=client_order_id, accepted=True, ts=now)
        self._emit(order, ExecKind.CANCELLED, now, qty=remaining)
        return order.cancel_ack

    def snapshot(self) -> BrokerSnapshot:
        return BrokerSnapshot(
            ts=self._clock.now(),
            environment=self.environment,
            account_id=self._account_id,
            orders=tuple(o.status() for o in self._orders.values()),
            execution_ids=tuple(self._exec_ids),
            positions={k: v for k, v in sorted(self._positions.items()) if v != 0},
            cash=self._cash,
            complete=True,
        )

    def events(self) -> Iterator[ExecutionEvent]:
        """Golește coada de evenimente, în ordinea emiterii."""
        while self._queue:
            yield self._queue.popleft()

    # ------------------------------------------------------------------ simulare

    def on_bar(self, bar: Bar, ctx: SimBarContext | None = None) -> None:
        """Execută ordinele deschise eligibile la deschiderea barei (bara *t+1*)."""
        ctx = ctx if ctx is not None else SimBarContext()
        if ctx.quote is not None and ctx.quote.instrument != bar.instrument:
            raise ValueError(f"cotația {ctx.quote.instrument} nu aparține {bar.instrument}")
        eligible = [
            o
            for o in self._orders.values()
            if o.req.instrument == bar.instrument
            and o.state in _OPEN_STATES
            and bar.ts_open >= o.submitted_at
        ]
        if not eligible:
            return
        inst = self._instruments[bar.instrument]
        volume_left = (
            floor_to_step(bar.volume * self._config.max_participation, inst.qty_step)
            if self._config.max_participation is not None
            else None
        )
        for order in eligible:  # ordinea acceptării
            qty = self._fill_qty(order, inst, volume_left)
            if qty <= 0:
                continue
            price = self._fill_price(order.req, inst, bar, qty, ctx)
            if price is None:
                continue
            self._apply_fill(order, inst, bar, qty, price, ctx)
            if volume_left is not None:
                volume_left -= qty

    # ------------------------------------------------------------------ intern

    def _available_to_sell(self, instrument: str) -> Decimal:
        reserved = sum(
            (
                o.remaining
                for o in self._orders.values()
                if o.req.instrument == instrument
                and o.req.side == "SELL"
                and o.state in _OPEN_STATES
            ),
            ZERO,
        )
        return self._positions.get(instrument, ZERO) - reserved

    def _fill_qty(self, order: _SimOrder, inst: Instrument, volume_left: Decimal | None) -> Decimal:
        cap = order.remaining
        if volume_left is not None:
            cap = min(cap, volume_left)
        if self._config.max_fill_qty_per_bar is not None:
            cap = min(cap, self._config.max_fill_qty_per_bar)
        if cap <= 0:
            return ZERO
        qty = floor_to_step(cap, inst.qty_step)
        if qty < order.remaining and qty < inst.min_qty:
            return ZERO
        return qty

    def _half_spread(self, inst: Instrument, bar: Bar, ctx: SimBarContext) -> Decimal:
        cfg = self._cost_model.config
        if ctx.quote is not None:
            q = ctx.quote
            if q.bid <= 0 or q.ask < q.bid:
                raise ValueError(f"cotație invalidă: bid={q.bid} ask={q.ask}")
            return (q.ask - q.bid) / 2 * cfg.quote_spread_multiplier
        schedule = cfg.spreads.get(inst.symbol)
        hour = bar.ts_open.hour
        fraction = schedule.spread_for(hour) if schedule is not None else None
        if fraction is None:
            raise CostModelIncomplete(
                "spread",
                f"fără cotație și fără spread configurat pentru {inst.symbol} la ora {hour} UTC",
            )
        return fraction * bar.open / 2

    def _slippage(self, inst: Instrument, bar: Bar, qty: Decimal, ctx: SimBarContext) -> Decimal:
        cfg = self._cost_model.config.slippage
        if cfg is None:
            raise CostModelIncomplete("slippage", "parametrii de slippage nu sunt configurați")
        if ctx.sigma_bar is None:
            raise CostModelIncomplete("slippage", "lipsește sigma_bar")
        if ctx.adv is None or ctx.adv <= 0:
            raise CostModelIncomplete("slippage", "lipsește ADV sau ADV <= 0")
        impact = cfg.k * ctx.sigma_bar * (qty / ctx.adv).sqrt() * bar.open
        return impact + cfg.min_ticks * inst.tick_size

    def _fill_price(
        self, req: OrderRequest, inst: Instrument, bar: Bar, qty: Decimal, ctx: SimBarContext
    ) -> Decimal | None:
        with localcontext(cost_context()):
            adj = self._half_spread(inst, bar, ctx) + self._slippage(inst, bar, qty, ctx)
            if req.side == "BUY":
                price = ceil_to_step(bar.open + adj, inst.tick_size)
            else:
                price = max(floor_to_step(bar.open - adj, inst.tick_size), inst.tick_size)
        if req.order_type == "MARKET":
            return price
        limit = req.limit_price
        if limit is None:  # pragma: no cover - garantat de OrderRequest
            raise ValueError("ordinul LIMIT necesită limit_price")
        if req.side == "BUY":
            if price <= limit:
                return price
            return limit if bar.low < limit else None
        if price >= limit:
            return price
        return limit if bar.high > limit else None

    def _fx_rate(self, inst: Instrument, ctx: SimBarContext) -> Decimal:
        if inst.currency == REPORTING_CURRENCY:
            return Decimal(1)
        if ctx.fx_rate is None or ctx.fx_rate <= 0:
            raise CostModelIncomplete("fx_conversion", f"lipsește cursul {inst.currency}→EUR")
        return ctx.fx_rate

    def _commission_total(
        self,
        inst: Instrument,
        ts: datetime,
        notional_ccy: Decimal,
        notional_eur: Decimal,
        rate: Decimal,
    ) -> Decimal:
        schedule = self._cost_model.config.commissions
        if schedule is None:
            raise CostModelIncomplete("commission", "niciun tabel de comisioane configurat")
        table = schedule.lookup(self._config.broker, inst.venue, ts)
        if table.currency == REPORTING_CURRENCY:
            return table.commission(notional_eur)
        if table.currency == inst.currency:
            return table.commission(notional_ccy) * rate
        raise CostModelIncomplete(
            "commission",
            f"moneda tabelului {table.currency} nu este nici {inst.currency}, nici EUR",
        )

    def _apply_fill(
        self,
        order: _SimOrder,
        inst: Instrument,
        bar: Bar,
        qty: Decimal,
        price: Decimal,
        ctx: SimBarContext,
    ) -> None:
        ts = bar.ts_open
        with localcontext(cost_context()):
            rate = self._fx_rate(inst, ctx)
            fill_ccy = price * qty
            fill_eur = fill_ccy * rate
            total = quantize_money(
                self._commission_total(
                    inst, ts, order.notional_ccy + fill_ccy, order.notional_eur + fill_eur, rate
                )
            )
            commission = max(total - order.commission_eur, ZERO)
            order.filled_qty += qty
            order.notional_ccy += fill_ccy
            order.notional_eur += fill_eur
            order.commission_eur += commission
            sign = Decimal(1) if order.req.side == "BUY" else Decimal(-1)
            pos = self._positions.get(inst.symbol, ZERO) + sign * qty
            self._positions[inst.symbol] = pos
            self._cash -= sign * fill_eur + commission
        done = order.remaining == 0
        order.state = OrderState.FILLED if done else OrderState.PARTIALLY_FILLED
        self._emit(
            order,
            ExecKind.FILL if done else ExecKind.PARTIAL_FILL,
            ts,
            qty=qty,
            price=price,
            commission=commission,
        )

    def _emit(
        self,
        order: _SimOrder,
        kind: ExecKind,
        ts: datetime,
        *,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        commission: Decimal | None = None,
        reason: str | None = None,
    ) -> None:
        seq = order.next_seq
        order.next_seq += 1
        coid = order.req.client_order_id
        exec_id = f"{self._account_id}:{coid}:{seq}"
        event = ExecutionEvent(
            broker_exec_id=exec_id,
            client_order_id=coid,
            kind=kind,
            qty=qty,
            price=price,
            commission=commission,
            broker_order_id=order.ack.broker_order_id,
            reason=reason,
            ts_broker=ts,
            ts_receipt=ts,
            seq=seq,
        )
        self._exec_ids.append(exec_id)
        self._queue.append(event)
