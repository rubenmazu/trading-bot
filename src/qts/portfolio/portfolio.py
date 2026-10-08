"""Proiecția portofoliului: poziții, numerar, PnL realizat și nerealizat (Req 8.4, 29.1).

Portofoliul este o proiecție *event-sourced*: o stare imuabilă (`PortfolioState`) reconstruită
din evenimentele jurnalului prin funcții pure:

- `apply_fill(state, fill)`     — execuție normalizată (`PortfolioFill`);
- `apply_cash(state, event)`    — depunere sau retragere explicită (`CashEvent`);
- `apply_mark(state, inst, m)`  — marcaj de piață (preț + curs), folosit pentru nerealizat.

Aceeași secvență de evenimente produce aceeași stare și aceeași serializare canonică
(`qts.core.models.canonical_json`). Deduplicarea după `broker_exec_id` / `fill_id` este
responsabilitatea OMS (sarcina 10.2); funcțiile de aici aplică fiecare eveniment primit.

Reguli:

- Numai poziții long (Initial_Stage). O vânzare mai mare decât cantitatea deținută este
  respinsă cu `OversellError`, fără modificarea stării.
- Numerarul este în EUR. O cumpărare scade `qty × price × fx + costuri`; o vânzare adaugă
  `qty × price × fx − costuri`. Costurile sunt deja în EUR, pe categorii (`CostBreakdown`).
- Portofoliul înregistrează realitatea raportată de broker: dacă o execuție sau o retragere
  aduce numerarul sub zero, evenimentul **se aplică** (nu se ajustează și nu se ignoră) și se
  adaugă un `PortfolioIncident(CASH_NEGATIVE)`, care trebuie tratat de reconciliere/kill switch.
- Costul mediu ponderat și recunoașterea costurilor sunt descrise în `qts.portfolio.pnl`.
- `Trading_Day` se obține printr-o funcție injectabilă (`TradingDayFn`, implicit data UTC).
  La primul eveniment dintr-o zi nouă, ziua anterioară se închide într-un `DailyPnLRecord`,
  iar nerealizatul la ultimele marcaje devine baza zilei noi. Evenimentele întârziate dintr-o
  zi deja închisă se contabilizează în ziua curentă (zilele închise nu se redeschid).

Identitate verificabilă: `equity = initial_cash + depuneri − retrageri + net realizat +
nerealizat brut`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from qts.core.models import CostBreakdown, Dec, Frozen, Side, UtcDatetime
from qts.core.money import REPORTING_CURRENCY, ZERO, notional
from qts.risk.context import PositionRisk
from qts.risk.monitor import LossObservation, TradingDayFn, utc_trading_day

from .pnl import (
    DailyPnL,
    DailyPnLRecord,
    Mark,
    PnLBreakdown,
    PnLReport,
    market_value_eur,
    release_average_cost,
    unrealized_eur,
)

__all__ = [
    "CashEvent",
    "OversellError",
    "Portfolio",
    "PortfolioError",
    "PortfolioFill",
    "PortfolioIncident",
    "PortfolioState",
    "Position",
    "RiskAccounting",
    "apply_cash",
    "apply_fill",
    "apply_mark",
    "equity_eur",
    "pnl_report",
    "roll_to",
    "to_loss_observation",
    "to_position_risk",
    "to_risk_accounting",
    "unrealized_pnl_eur",
]


class PortfolioError(ValueError):
    """Eveniment incompatibil cu starea portofoliului; starea rămâne nemodificată."""


class OversellError(PortfolioError):
    """Vânzare mai mare decât cantitatea deținută (vânzarea în lipsă este interzisă)."""


# --------------------------------------------------------------------------- evenimente


def _non_negative_costs(costs: CostBreakdown) -> None:
    for name, value in costs.model_dump().items():
        if value < 0:
            raise ValueError(f"costul {name} nu poate fi negativ: {value}")


class PortfolioFill(Frozen):
    """Execuție normalizată pentru portofoliu; costurile sunt în EUR, pe categorii."""

    fill_id: str
    instrument: str
    currency: str = REPORTING_CURRENCY
    side: Side
    qty: Dec
    price: Dec  # moneda instrumentului
    fx_rate_to_eur: Dec = Decimal(1)
    ts: UtcDatetime
    costs: CostBreakdown = CostBreakdown()

    @model_validator(mode="after")
    def _check(self) -> PortfolioFill:
        if not self.fill_id.strip() or not self.instrument.strip():
            raise ValueError("fill_id și instrument sunt obligatorii")
        if self.qty <= 0 or self.price <= 0 or self.fx_rate_to_eur <= 0:
            raise ValueError("qty, price și fx_rate_to_eur trebuie să fie > 0")
        if self.currency == REPORTING_CURRENCY and self.fx_rate_to_eur != 1:
            raise ValueError("un instrument în EUR are fx_rate_to_eur = 1")
        _non_negative_costs(self.costs)
        return self

    @property
    def notional_eur(self) -> Decimal:
        return notional(self.price, self.qty) * self.fx_rate_to_eur


class CashEvent(Frozen):
    """Depunere sau retragere explicită, în EUR."""

    event_id: str
    kind: Literal["DEPOSIT", "WITHDRAWAL"]
    amount_eur: Dec
    ts: UtcDatetime

    @model_validator(mode="after")
    def _check(self) -> CashEvent:
        if self.amount_eur <= 0:
            raise ValueError("amount_eur trebuie să fie > 0")
        return self


class PortfolioIncident(Frozen):
    code: Literal["CASH_NEGATIVE"]
    ref_id: str  # fill_id sau event_id
    ts: UtcDatetime
    cash_eur: Dec


# --------------------------------------------------------------------------- stare


class Position(Frozen):
    """Poziție long deschisă, cu baza de cost în moneda instrumentului și în EUR."""

    instrument: str
    currency: str
    qty: Dec
    cost_basis_ccy: Dec
    cost_basis_eur: Dec

    @model_validator(mode="after")
    def _check(self) -> Position:
        if self.qty <= 0:
            raise ValueError("qty poziției trebuie să fie > 0 (numai long)")
        return self

    @property
    def avg_price(self) -> Decimal:
        """Prețul mediu de achiziție, în moneda instrumentului."""
        return self.cost_basis_ccy / self.qty

    @property
    def avg_fx_rate_to_eur(self) -> Decimal:
        return self.cost_basis_eur / self.cost_basis_ccy


class PortfolioState(Frozen):
    initial_cash_eur: Dec
    cash_eur: Dec
    positions: dict[str, Position] = Field(default_factory=dict)
    marks: dict[str, Mark] = Field(default_factory=dict)
    realized: PnLBreakdown = PnLBreakdown()  # brut realizat + toate costurile suportate
    realized_by_instrument: dict[str, PnLBreakdown] = Field(default_factory=dict)
    cumulative_deposits_eur: Dec = ZERO
    cumulative_withdrawals_eur: Dec = ZERO
    today: DailyPnL | None = None
    closed_days: tuple[DailyPnLRecord, ...] = ()
    incidents: tuple[PortfolioIncident, ...] = ()

    @classmethod
    def initial(cls, cash_eur: Decimal) -> PortfolioState:
        return cls(initial_cash_eur=cash_eur, cash_eur=cash_eur)

    @model_validator(mode="after")
    def _check(self) -> PortfolioState:
        for key, pos in self.positions.items():
            if key != pos.instrument:
                raise ValueError(f"cheia poziției {key} diferă de instrument {pos.instrument}")
            if key not in self.marks:
                raise ValueError(f"poziția {key} nu are marcaj")
        return self


# --------------------------------------------------------------------------- calcule


def _effective_marks(state: PortfolioState, marks: Mapping[str, Mark] | None) -> dict[str, Mark]:
    merged = dict(state.marks)
    if marks:
        merged.update(marks)
    return merged


def unrealized_pnl_eur(state: PortfolioState, marks: Mapping[str, Mark] | None = None) -> Decimal:
    """PnL nerealizat brut la `marks` (cu revenire la ultimele marcaje din stare)."""
    holdings = {k: (p.qty, p.cost_basis_eur) for k, p in state.positions.items()}
    return unrealized_eur(holdings, _effective_marks(state, marks))


def equity_eur(state: PortfolioState, marks: Mapping[str, Mark] | None = None) -> Decimal:
    """Numerar + valoarea pozițiilor marcate, în EUR."""
    eff = _effective_marks(state, marks)
    value = state.cash_eur
    for key in sorted(state.positions):
        value += market_value_eur(state.positions[key].qty, eff[key])
    return value


def pnl_report(state: PortfolioState, marks: Mapping[str, Mark] | None = None) -> PnLReport:
    """Brut realizat, brut nerealizat, fiecare categorie de cost și net (Req 8.4, 29.1)."""
    return PnLReport(
        realized_gross_eur=state.realized.gross_eur,
        unrealized_gross_eur=unrealized_pnl_eur(state, marks),
        costs=state.realized.costs,
    )


# --------------------------------------------------------------------------- tranziții


def roll_to(state: PortfolioState, day: date) -> PortfolioState:
    """Trece starea în `Trading_Day` `day` (fără efect dacă ziua nu este mai nouă)."""
    today = state.today
    if today is not None and day <= today.trading_day:
        return state
    current_unrealized = unrealized_pnl_eur(state)
    closed = state.closed_days
    if today is not None:
        record = DailyPnLRecord(
            trading_day=today.trading_day,
            realized=today.realized,
            unrealized_change_eur=current_unrealized - today.start_unrealized_eur,
        )
        closed = (*closed, record)
    return state.model_copy(
        update={
            "today": DailyPnL(trading_day=day, start_unrealized_eur=current_unrealized),
            "closed_days": closed,
        }
    )


def _today(state: PortfolioState) -> DailyPnL:
    if state.today is None:  # imposibil după roll_to
        raise PortfolioError("starea nu are Trading_Day curent")
    return state.today


def _with_mark(marks: dict[str, Mark], instrument: str, mark: Mark) -> dict[str, Mark]:
    existing = marks.get(instrument)
    if existing is not None and mark.ts < existing.ts:
        return marks  # un marcaj mai vechi nu înlocuiește unul mai nou
    return {**marks, instrument: mark}


def _cash_incident(cash: Decimal, ref_id: str, ts: datetime) -> tuple[PortfolioIncident, ...]:
    if cash < 0:
        return (PortfolioIncident(code="CASH_NEGATIVE", ref_id=ref_id, ts=ts, cash_eur=cash),)
    return ()


def apply_mark(
    state: PortfolioState,
    instrument: str,
    mark: Mark,
    *,
    trading_day: TradingDayFn = utc_trading_day,
) -> PortfolioState:
    """Înregistrează un marcaj (după trecerea în ziua marcajului, cu baza la marcajele vechi)."""
    rolled = roll_to(state, trading_day(mark.ts))
    return rolled.model_copy(update={"marks": _with_mark(rolled.marks, instrument, mark)})


def apply_fill(
    state: PortfolioState,
    fill: PortfolioFill,
    *,
    trading_day: TradingDayFn = utc_trading_day,
) -> PortfolioState:
    """Aplică o execuție; ridică `PortfolioError` (stare nemodificată) dacă este incompatibilă."""
    pos = state.positions.get(fill.instrument)
    if pos is not None and pos.currency != fill.currency:
        raise PortfolioError(
            f"moneda execuției {fill.currency} diferă de moneda poziției {pos.currency}"
        )
    if fill.side == "SELL":
        held = pos.qty if pos is not None else ZERO
        if fill.qty > held:
            raise OversellError(
                f"vânzare {fill.qty} {fill.instrument} peste cantitatea deținută {held}"
            )

    rolled = roll_to(state, trading_day(fill.ts))
    positions = dict(rolled.positions)
    notional_eur = fill.notional_eur
    notional_ccy = notional(fill.price, fill.qty)
    gross = ZERO

    if fill.side == "BUY":
        if pos is None:
            positions[fill.instrument] = Position(
                instrument=fill.instrument,
                currency=fill.currency,
                qty=fill.qty,
                cost_basis_ccy=notional_ccy,
                cost_basis_eur=notional_eur,
            )
        else:
            positions[fill.instrument] = pos.model_copy(
                update={
                    "qty": pos.qty + fill.qty,
                    "cost_basis_ccy": pos.cost_basis_ccy + notional_ccy,
                    "cost_basis_eur": pos.cost_basis_eur + notional_eur,
                }
            )
        cash = rolled.cash_eur - notional_eur - fill.costs.total
    else:
        if pos is None:  # imposibil după verificarea vânzării în lipsă
            raise OversellError(f"nicio poziție deschisă pe {fill.instrument}")
        release = release_average_cost(
            qty=pos.qty,
            basis_ccy=pos.cost_basis_ccy,
            basis_eur=pos.cost_basis_eur,
            sell_qty=fill.qty,
            proceeds_eur=notional_eur,
        )
        gross = release.realized_gross_eur
        if fill.qty == pos.qty:
            del positions[fill.instrument]
        else:
            positions[fill.instrument] = pos.model_copy(
                update={
                    "qty": pos.qty - fill.qty,
                    "cost_basis_ccy": pos.cost_basis_ccy - release.released_basis_ccy,
                    "cost_basis_eur": pos.cost_basis_eur - release.released_basis_eur,
                }
            )
        cash = rolled.cash_eur + notional_eur - fill.costs.total

    by_inst = dict(rolled.realized_by_instrument)
    by_inst[fill.instrument] = by_inst.get(fill.instrument, PnLBreakdown()).add(gross, fill.costs)
    current = _today(rolled)
    today = current.model_copy(update={"realized": current.realized.add(gross, fill.costs)})
    mark = Mark(price=fill.price, fx_rate_to_eur=fill.fx_rate_to_eur, ts=fill.ts)
    return rolled.model_copy(
        update={
            "positions": positions,
            "marks": _with_mark(rolled.marks, fill.instrument, mark),
            "cash_eur": cash,
            "realized": rolled.realized.add(gross, fill.costs),
            "realized_by_instrument": by_inst,
            "today": today,
            "incidents": (*rolled.incidents, *_cash_incident(cash, fill.fill_id, fill.ts)),
        }
    )


def apply_cash(
    state: PortfolioState,
    event: CashEvent,
    *,
    trading_day: TradingDayFn = utc_trading_day,
) -> PortfolioState:
    """Aplică o depunere sau o retragere; fluxurile de numerar nu sunt PnL."""
    rolled = roll_to(state, trading_day(event.ts))
    if event.kind == "DEPOSIT":
        cash = rolled.cash_eur + event.amount_eur
        update: dict[str, object] = {
            "cumulative_deposits_eur": rolled.cumulative_deposits_eur + event.amount_eur
        }
    else:
        cash = rolled.cash_eur - event.amount_eur
        update = {
            "cumulative_withdrawals_eur": rolled.cumulative_withdrawals_eur + event.amount_eur
        }
    update["cash_eur"] = cash
    update["incidents"] = (*rolled.incidents, *_cash_incident(cash, event.event_id, event.ts))
    return rolled.model_copy(update=update)


# --------------------------------------------------------------------------- adaptoare risc


def _daily_figures(
    state: PortfolioState, ts: datetime, marks: Mapping[str, Mark] | None, fn: TradingDayFn
) -> tuple[PortfolioState, Decimal, Decimal]:
    rolled = roll_to(state, fn(ts))
    today = _today(rolled)
    daily_unrealized = unrealized_pnl_eur(rolled, marks) - today.start_unrealized_eur
    return rolled, today.realized.net_eur, daily_unrealized


def to_loss_observation(
    state: PortfolioState,
    ts: datetime,
    marks: Mapping[str, Mark] | None = None,
    *,
    trading_day: TradingDayFn = utc_trading_day,
) -> LossObservation:
    """Instantaneul pentru `LossMonitor`: PnL zilnic net realizat + nerealizat, capital, fluxuri."""
    rolled, daily_realized, daily_unrealized = _daily_figures(state, ts, marks, trading_day)
    return LossObservation(
        ts=ts,
        daily_realized_pnl_eur=daily_realized,
        daily_unrealized_pnl_eur=daily_unrealized,
        equity_eur=equity_eur(rolled, marks),
        cumulative_withdrawals_eur=rolled.cumulative_withdrawals_eur,
        cumulative_deposits_eur=rolled.cumulative_deposits_eur,
    )


def to_position_risk(
    state: PortfolioState,
    *,
    exit_costs_eur: Mapping[str, Decimal],
    marks: Mapping[str, Mark] | None = None,
    stops: Mapping[str, Decimal | None] | None = None,
) -> dict[str, PositionRisk]:
    """Pozițiile pentru `RiskContext`. Costul de ieșire este obligatoriu (fail-closed)."""
    eff = _effective_marks(state, marks)
    stops = stops or {}
    result: dict[str, PositionRisk] = {}
    for key in sorted(state.positions):
        if key not in exit_costs_eur:
            raise PortfolioError(f"lipsește costul estimat al ieșirii pentru {key}")
        mark = eff[key]
        result[key] = PositionRisk(
            instrument=key,
            qty=state.positions[key].qty,
            mark_price=mark.price,
            stop_price=stops.get(key),
            fx_rate_to_eur=mark.fx_rate_to_eur,
            exit_cost_eur=exit_costs_eur[key],
        )
    return result


class RiskAccounting(Frozen):
    """Câmpurile contabile ale `RiskContext`, derivate din portofoliu."""

    cash_eur: Dec
    positions: dict[str, PositionRisk]
    daily_realized_pnl_eur: Dec
    daily_unrealized_pnl_eur: Dec
    total_pnl_eur: Dec  # net realizat + nerealizat brut

    def context_fields(self) -> dict[str, object]:
        """Argumente pentru `RiskContext(**fields, ts=..., mode=..., ...)`."""
        return {
            "cash_eur": self.cash_eur,
            "positions": dict(self.positions),
            "daily_realized_pnl_eur": self.daily_realized_pnl_eur,
            "daily_unrealized_pnl_eur": self.daily_unrealized_pnl_eur,
            "total_pnl_eur": self.total_pnl_eur,
        }


def to_risk_accounting(
    state: PortfolioState,
    ts: datetime,
    *,
    exit_costs_eur: Mapping[str, Decimal],
    marks: Mapping[str, Mark] | None = None,
    stops: Mapping[str, Decimal | None] | None = None,
    trading_day: TradingDayFn = utc_trading_day,
) -> RiskAccounting:
    rolled, daily_realized, daily_unrealized = _daily_figures(state, ts, marks, trading_day)
    return RiskAccounting(
        cash_eur=rolled.cash_eur,
        positions=to_position_risk(rolled, exit_costs_eur=exit_costs_eur, marks=marks, stops=stops),
        daily_realized_pnl_eur=daily_realized,
        daily_unrealized_pnl_eur=daily_unrealized,
        total_pnl_eur=rolled.realized.net_eur + unrealized_pnl_eur(rolled, marks),
    )


# --------------------------------------------------------------------------- fațadă


class Portfolio:
    """Fațadă cu stare peste funcțiile pure; `snapshot()` întoarce starea imuabilă curentă."""

    def __init__(
        self,
        state: PortfolioState,
        *,
        trading_day: TradingDayFn = utc_trading_day,
    ) -> None:
        self._state = state
        self._trading_day = trading_day

    @classmethod
    def with_cash(
        cls, cash_eur: Decimal, *, trading_day: TradingDayFn = utc_trading_day
    ) -> Portfolio:
        return cls(PortfolioState.initial(cash_eur), trading_day=trading_day)

    def snapshot(self) -> PortfolioState:
        return self._state

    def apply_fill(self, fill: PortfolioFill) -> PortfolioState:
        self._state = apply_fill(self._state, fill, trading_day=self._trading_day)
        return self._state

    def apply_cash(self, event: CashEvent) -> PortfolioState:
        self._state = apply_cash(self._state, event, trading_day=self._trading_day)
        return self._state

    def apply_mark(self, instrument: str, mark: Mark) -> PortfolioState:
        self._state = apply_mark(self._state, instrument, mark, trading_day=self._trading_day)
        return self._state

    def loss_observation(
        self, ts: datetime, marks: Mapping[str, Mark] | None = None
    ) -> LossObservation:
        return to_loss_observation(self._state, ts, marks, trading_day=self._trading_day)

    def risk_accounting(
        self,
        ts: datetime,
        *,
        exit_costs_eur: Mapping[str, Decimal],
        marks: Mapping[str, Mark] | None = None,
        stops: Mapping[str, Decimal | None] | None = None,
    ) -> RiskAccounting:
        """Câmpurile contabile ale `RiskContext`, cu `Trading_Day` al portofoliului."""
        return to_risk_accounting(
            self._state,
            ts,
            exit_costs_eur=exit_costs_eur,
            marks=marks,
            stops=stops,
            trading_day=self._trading_day,
        )
