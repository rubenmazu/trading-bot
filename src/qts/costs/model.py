"""Complete_Cost_Model: spread, comision, slippage, latență, conversie FX și taxe (Req 8.1-8.3).

Toate costurile sunt calculate separat, în moneda de raportare (EUR), numai cu `Decimal`.
Un cost pozitiv reduce rezultatul; latența și slippage-ul realizat pot fi negative (favorabile).

Convenții:
- `fx_rate` din `CostContext` = unități EUR pentru o unitate din moneda instrumentului;
- spreadurile configurate și ratele sunt fracții din preț/nominal (0.001 = 10 bps);
- `sigma_bar` este deviația standard a randamentului pe bară, ca fracție din preț;
- ora spreadului configurat este ora UTC a deciziei.

O componentă obligatorie lipsă ridică `CostModelIncomplete`; apelantul invalidează evaluarea.
Multiplicatorii de stres (6.2) se aplică construind o configurație derivată (`costs.stress`).
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Annotated, Protocol

from pydantic import Field, model_validator

from qts.core.models import CostBreakdown, Dec, Frozen, Instrument, Quote, Side, UtcDatetime
from qts.core.money import REPORTING_CURRENCY, ZERO, quantize_money

from .commission_tables import CommissionSchedule
from .errors import CostModelIncomplete, cost_context

__all__ = [
    "CompleteCostModel",
    "CostContext",
    "CostModel",
    "CostModelConfig",
    "CostModelIncomplete",
    "Fill",
    "FxConfig",
    "LatencyConfig",
    "OrderSpec",
    "SlippageConfig",
    "SpreadSchedule",
    "TaxConfig",
]

_SECONDS_PER_MIN = Decimal(60)
_MS_PER_SECOND = Decimal(1000)


def _non_negative(*values: Decimal) -> None:
    if any(v < 0 for v in values):
        raise ValueError("valorile de cost configurate nu pot fi negative")


# --------------------------------------------------------------------------- configurație


class SpreadSchedule(Frozen):
    """Spread complet (fracție din preț) per oră UTC, folosit în lipsa cotației bid/ask."""

    by_hour_utc: dict[int, Dec] = Field(default_factory=dict)
    default: Dec | None = None

    @model_validator(mode="after")
    def _check(self) -> SpreadSchedule:
        if any(not 0 <= h <= 23 for h in self.by_hour_utc):
            raise ValueError("orele spreadului trebuie să fie în [0, 23]")
        _non_negative(*self.by_hour_utc.values())
        if self.default is not None:
            _non_negative(self.default)
        return self

    def spread_for(self, hour: int) -> Decimal | None:
        return self.by_hour_utc.get(hour, self.default)


class SlippageConfig(Frozen):
    """`k × σ_bar × sqrt(qty / ADV) × preț + min_ticks × tick` pe unitate."""

    k: Dec
    min_ticks: Dec = ZERO

    @model_validator(mode="after")
    def _check(self) -> SlippageConfig:
        _non_negative(self.k, self.min_ticks)
        return self


class LatencyConfig(Frozen):
    latency_ms: Annotated[int, Field(ge=0)]


class FxConfig(Frozen):
    """Spreadul de conversie al brokerului, fracție din nominalul convertit."""

    conversion_spread: Dec

    @model_validator(mode="after")
    def _check(self) -> FxConfig:
        _non_negative(self.conversion_spread)
        return self


class TaxConfig(Frozen):
    """Taxe configurate și aprobate de operator; nu există valori implicite hardcodate."""

    approved: bool
    rate_on_notional: Dec = ZERO
    fixed_per_trade: Dec = ZERO  # în EUR
    sides: tuple[Side, ...] = ("BUY", "SELL")
    reference: str = ""  # sursa/aprobarea valorii

    @model_validator(mode="after")
    def _check(self) -> TaxConfig:
        _non_negative(self.rate_on_notional, self.fixed_per_trade)
        return self


class CostModelConfig(Frozen):
    """Configurația versionată. Componentele lipsă sunt detectate la evaluare (8.3)."""

    version: str
    commissions: CommissionSchedule | None = None
    spreads: dict[str, SpreadSchedule] = Field(default_factory=dict)
    slippage: SlippageConfig | None = None
    latency: LatencyConfig | None = None
    fx: FxConfig | None = None
    taxes: TaxConfig | None = None
    # Multiplicator aplicat jumătății de spread observate în cotația bid/ask (doar `estimate`).
    # Rămâne 1 în configurația de bază; îl setează doar scenariile de stres (6.2, Req 20.3).
    quote_spread_multiplier: Dec = Decimal(1)

    @model_validator(mode="after")
    def _check_multiplier(self) -> CostModelConfig:
        if self.quote_spread_multiplier < 1:
            raise ValueError("quote_spread_multiplier trebuie să fie >= 1")
        return self


# --------------------------------------------------------------------------- intrări


class OrderSpec(Frozen):
    instrument: Instrument
    side: Side
    qty: Dec
    ts: UtcDatetime  # momentul deciziei
    ref_price: Dec  # prețul de decizie când nu există cotație

    @model_validator(mode="after")
    def _check(self) -> OrderSpec:
        if self.qty <= 0 or self.ref_price <= 0:
            raise ValueError("qty și ref_price trebuie să fie > 0")
        return self


class Fill(Frozen):
    instrument: Instrument
    side: Side
    qty: Dec
    price: Dec
    ts: UtcDatetime
    commission: Dec | None = None  # raportat de broker, în moneda instrumentului

    @model_validator(mode="after")
    def _check(self) -> Fill:
        if self.qty <= 0 or self.price <= 0:
            raise ValueError("qty și price trebuie să fie > 0")
        if self.commission is not None and self.commission < 0:
            raise ValueError("comisionul raportat nu poate fi negativ")
        return self


class CostContext(Frozen):
    broker: str
    fx_rate: Dec | None = None  # EUR per unitate din moneda instrumentului
    sigma_bar: Dec | None = None
    adv: Dec | None = None  # volum mediu zilnic, în unități
    bar_interval_min: int | None = None
    price_after_latency: Dec | None = None  # Backtest: deschiderea barei t+1

    @model_validator(mode="after")
    def _check(self) -> CostContext:
        if self.fx_rate is not None and self.fx_rate <= 0:
            raise ValueError("fx_rate trebuie să fie > 0")
        if self.sigma_bar is not None and self.sigma_bar < 0:
            raise ValueError("sigma_bar nu poate fi negativ")
        if self.price_after_latency is not None and self.price_after_latency <= 0:
            raise ValueError("price_after_latency trebuie să fie > 0")
        return self


class CostModel(Protocol):
    @property
    def version(self) -> str: ...

    def estimate(
        self, order: OrderSpec, quote: Quote | None, ctx: CostContext
    ) -> CostBreakdown: ...

    def realize(self, fill: Fill, quote_at_decision: Quote, ctx: CostContext) -> CostBreakdown: ...


# --------------------------------------------------------------------------- implementare


def _sign(side: Side) -> Decimal:
    return Decimal(1) if side == "BUY" else Decimal(-1)


def _check_quote(quote: Quote, instrument: Instrument) -> None:
    if quote.instrument != instrument.symbol:
        raise ValueError(f"cotația {quote.instrument} nu aparține {instrument.symbol}")
    if quote.bid <= 0 or quote.ask < quote.bid:
        raise ValueError(f"cotație invalidă: bid={quote.bid} ask={quote.ask}")


def _mid(quote: Quote) -> Decimal:
    return (quote.bid + quote.ask) / 2


class CompleteCostModel:
    """Implementarea `CostModel` folosită de SimBroker în Backtest și Shadow."""

    def __init__(self, config: CostModelConfig) -> None:
        self._config = config

    @property
    def version(self) -> str:
        return self._config.version

    @property
    def config(self) -> CostModelConfig:
        return self._config

    # ----------------------------------------------------------------- API public

    def estimate(self, order: OrderSpec, quote: Quote | None, ctx: CostContext) -> CostBreakdown:
        inst = order.instrument
        with localcontext(cost_context()):
            rate = self._fx_rate(inst, ctx)
            if quote is not None:
                _check_quote(quote, inst)
                price = _mid(quote)
                half = (quote.ask - quote.bid) / 2 * self._config.quote_spread_multiplier
            else:
                price = order.ref_price
                half = self._configured_spread(inst, order.ts.hour) * price / 2
            notional_eur = price * order.qty * rate
            return self._breakdown(
                spread=half * order.qty * rate,
                commission=self._commission(inst, order, price, rate, ctx),
                slippage=self._slippage(inst, order.qty, price, ctx) * order.qty * rate,
                latency=self._latency(order, price, ctx) * rate,
                fx_conversion=self._fx_cost(inst, notional_eur),
                taxes=self._taxes(order.side, notional_eur),
            )

    def realize(self, fill: Fill, quote_at_decision: Quote, ctx: CostContext) -> CostBreakdown:
        """Descompune costul efectiv al unei execuții față de mijlocul cotației la decizie."""
        inst = fill.instrument
        _check_quote(quote_at_decision, inst)
        with localcontext(cost_context()):
            rate = self._fx_rate(inst, ctx)
            sign = _sign(fill.side)
            mid = _mid(quote_at_decision)
            shortfall = sign * (fill.price - mid) * fill.qty
            spread = (quote_at_decision.ask - quote_at_decision.bid) / 2 * fill.qty
            latency = (
                sign * (ctx.price_after_latency - mid) * fill.qty
                if ctx.price_after_latency is not None
                else ZERO
            )
            notional_eur = fill.price * fill.qty * rate
            if fill.commission is not None:
                commission = fill.commission * rate
            else:
                commission = self._table_commission(
                    inst, ctx.broker, fill.ts, fill.price * fill.qty, rate
                )
            return self._breakdown(
                spread=spread * rate,
                commission=commission,
                slippage=(shortfall - spread - latency) * rate,
                latency=latency * rate,
                fx_conversion=self._fx_cost(inst, notional_eur),
                taxes=self._taxes(fill.side, notional_eur),
            )

    # ----------------------------------------------------------------- componente

    @staticmethod
    def _breakdown(**parts: Decimal) -> CostBreakdown:
        return CostBreakdown(**{k: quantize_money(v) for k, v in parts.items()})

    @staticmethod
    def _fx_rate(inst: Instrument, ctx: CostContext) -> Decimal:
        if inst.currency == REPORTING_CURRENCY:
            return Decimal(1)
        if ctx.fx_rate is None:
            raise CostModelIncomplete("fx_conversion", f"lipsește cursul {inst.currency}→EUR")
        return ctx.fx_rate

    def _configured_spread(self, inst: Instrument, hour: int) -> Decimal:
        schedule = self._config.spreads.get(inst.symbol)
        value = schedule.spread_for(hour) if schedule is not None else None
        if value is None:
            raise CostModelIncomplete(
                "spread",
                f"fără cotație bid/ask și fără spread configurat pentru {inst.symbol}"
                f" la ora {hour} UTC",
            )
        return value

    def _commission(
        self, inst: Instrument, order: OrderSpec, price: Decimal, rate: Decimal, ctx: CostContext
    ) -> Decimal:
        return self._table_commission(inst, ctx.broker, order.ts, price * order.qty, rate)

    def _table_commission(
        self,
        inst: Instrument,
        broker: str,
        ts: UtcDatetime,
        notional_ccy: Decimal,
        rate: Decimal,
    ) -> Decimal:
        if self._config.commissions is None:
            raise CostModelIncomplete("commission", "niciun tabel de comisioane configurat")
        table = self._config.commissions.lookup(broker, inst.venue, ts)
        if table.currency == inst.currency:
            return table.commission(notional_ccy) * rate
        if table.currency == REPORTING_CURRENCY:
            return table.commission(notional_ccy * rate)
        raise CostModelIncomplete(
            "commission",
            f"moneda tabelului {table.currency} nu este nici {inst.currency}, nici EUR",
        )

    def _slippage(
        self, inst: Instrument, qty: Decimal, price: Decimal, ctx: CostContext
    ) -> Decimal:
        """Slippage estimat pe unitate, în moneda instrumentului."""
        cfg = self._config.slippage
        if cfg is None:
            raise CostModelIncomplete("slippage", "parametrii de slippage nu sunt configurați")
        if ctx.sigma_bar is None:
            raise CostModelIncomplete("slippage", "lipsește sigma_bar")
        if ctx.adv is None or ctx.adv <= 0:
            raise CostModelIncomplete("slippage", "lipsește ADV sau ADV <= 0")
        impact = cfg.k * ctx.sigma_bar * (qty / ctx.adv).sqrt() * price
        return impact + cfg.min_ticks * inst.tick_size

    def _latency(self, order: OrderSpec, price: Decimal, ctx: CostContext) -> Decimal:
        """Efectul latenței (total, moneda instrumentului).

        Cu `price_after_latency` (Backtest: open-ul barei următoare) se folosește mișcarea
        efectivă, cu semn. Altfel, o estimare conservatoare de 1σ scalată la durata latenței.
        """
        cfg = self._config.latency
        if cfg is None:
            raise CostModelIncomplete("latency", "latența nu este configurată")
        if ctx.price_after_latency is not None:
            return _sign(order.side) * (ctx.price_after_latency - price) * order.qty
        if cfg.latency_ms == 0:
            return ZERO
        if ctx.sigma_bar is None or ctx.bar_interval_min is None or ctx.bar_interval_min <= 0:
            raise CostModelIncomplete(
                "latency", "lipsesc price_after_latency sau sigma_bar și bar_interval_min"
            )
        latency_s = Decimal(cfg.latency_ms) / _MS_PER_SECOND
        bar_s = Decimal(ctx.bar_interval_min) * _SECONDS_PER_MIN
        return ctx.sigma_bar * (latency_s / bar_s).sqrt() * price * order.qty

    def _fx_cost(self, inst: Instrument, notional_eur: Decimal) -> Decimal:
        if inst.currency == REPORTING_CURRENCY:
            return ZERO
        if self._config.fx is None:
            raise CostModelIncomplete(
                "fx_conversion", f"spreadul de conversie {inst.currency}→EUR nu este configurat"
            )
        return notional_eur * self._config.fx.conversion_spread

    def _taxes(self, side: Side, notional_eur: Decimal) -> Decimal:
        cfg = self._config.taxes
        if cfg is None:
            raise CostModelIncomplete("taxes", "taxele aplicabile nu sunt configurate")
        if not cfg.approved:
            raise CostModelIncomplete("taxes", "taxele configurate nu sunt aprobate")
        if side not in cfg.sides:
            return ZERO
        return notional_eur * cfg.rate_on_notional + cfg.fixed_per_trade
