"""Dimensionarea pe stop și costuri (design Risk_Engine, pașii 3-4 și 8; Req 12.3, 12.7, 13.9).

Riscul unei intrări long de cantitate `q`, în EUR:

    trade_risk(q) = q × (entry − stop) × fx + cost_intrare(q) + cost_ieșire_la_stop(q)

Costurile vin exclusiv din `CostModel.estimate`, pe ambele picioare (dus-întors). Ele nu sunt
liniare (comision minim, slippage ∝ √q), așa că:

1. pasul 3-4 liniarizează costul dus-întors între `q_min` și plafonul de numerar, obținând
   `fixed_costs` și `cost_per_unit_roundtrip`, apoi aplică formula din design:
   `qty = floor_to_step(min((budget − fixed) / risk_per_unit, (cash − fixed_entry) / entry))`;
2. cantitatea candidată este verificată exact cu modelul de costuri; dacă depășește bugetul
   sau numerarul, este redusă (căutare binară pe pași de cantitate) până la cea mai mare
   cantitate care se încadrează. Verificările limitelor rulează apoi pe cantitatea finală (12.7).

Dacă formula dă sub minimul tranzacționabil, se propune `q_min`; limitele dure (0,50 EUR,
numerar) decid apoi aprobarea sau respingerea (13.8).
"""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Final

from qts.core.models import Dec, Frozen, Instrument, Quote
from qts.core.money import PRECISION, ZERO, ceil_to_step, floor_to_step
from qts.costs.model import CostContext, CostModel, OrderSpec

from .context import MarketSnapshot

__all__ = ["SizingResult", "TradeRisk", "TradeRiskEstimator", "min_tradable_qty", "size_entry"]

_CTX: Final = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)


class TradeRisk(Frozen):
    """Valorile calculate pentru o cantitate dată (Req 12.3); toate sumele în EUR."""

    qty: Dec
    entry_price: Dec
    stop_price: Dec
    fx_rate_to_eur: Dec
    notional_ccy: Dec
    notional_eur: Dec
    price_risk_eur: Dec
    entry_cost_eur: Dec
    exit_cost_eur: Dec
    trade_risk_eur: Dec
    cash_required_eur: Dec

    @property
    def cost_eur(self) -> Decimal:
        return self.entry_cost_eur + self.exit_cost_eur


def _quote_at(quote: Quote | None, price: Decimal) -> Quote | None:
    """Cotația mutată la `price`, cu același spread; None dacă bid-ul ar deveni ≤ 0."""
    if quote is None:
        return None
    half = (quote.ask - quote.bid) / 2
    bid = price - half
    if bid <= 0:
        return None
    return Quote(instrument=quote.instrument, ts=quote.ts, bid=bid, ask=price + half)


class TradeRiskEstimator:
    """Calculează `TradeRisk` exact pentru orice cantitate, prin modelul de costuri."""

    def __init__(
        self,
        cost_model: CostModel,
        snapshot: MarketSnapshot,
        *,
        entry_price: Decimal,
        stop_price: Decimal,
        fx_rate_to_eur: Decimal,
        ts_decision: datetime,
    ) -> None:
        if not entry_price > stop_price > 0:
            raise ValueError("este necesar entry > stop > 0")
        self._model = cost_model
        self._instrument: Instrument = snapshot.instrument
        self._quote = snapshot.quote
        self._exit_quote = _quote_at(snapshot.quote, stop_price)
        self._cost_ctx: CostContext = snapshot.cost_ctx
        self._entry = entry_price
        self._stop = stop_price
        self._fx = fx_rate_to_eur
        self._ts = ts_decision

    @property
    def entry_price(self) -> Decimal:
        return self._entry

    @property
    def stop_price(self) -> Decimal:
        return self._stop

    @property
    def fx_rate_to_eur(self) -> Decimal:
        return self._fx

    def _spec(self, side: str, qty: Decimal, price: Decimal) -> OrderSpec:
        return OrderSpec.model_validate(
            {
                "instrument": self._instrument,
                "side": side,
                "qty": qty,
                "ts": self._ts,
                "ref_price": price,
            }
        )

    def at(self, qty: Decimal) -> TradeRisk:
        """Riscul exact pentru `qty` > 0; ridică `CostModelIncomplete` dacă lipsesc costuri."""
        if qty <= 0:
            raise ValueError("qty trebuie să fie > 0")
        entry_cost = self._model.estimate(
            self._spec("BUY", qty, self._entry), self._quote, self._cost_ctx
        ).total
        exit_cost = self._model.estimate(
            self._spec("SELL", qty, self._stop), self._exit_quote, self._cost_ctx
        ).total
        with localcontext(_CTX):
            notional_ccy = qty * self._entry
            notional_eur = notional_ccy * self._fx
            price_risk = qty * (self._entry - self._stop) * self._fx
            # Costurile negative estimate (latență favorabilă) nu reduc riscul: fail-closed.
            entry_cost = max(entry_cost, ZERO)
            exit_cost = max(exit_cost, ZERO)
            return TradeRisk(
                qty=qty,
                entry_price=self._entry,
                stop_price=self._stop,
                fx_rate_to_eur=self._fx,
                notional_ccy=notional_ccy,
                notional_eur=notional_eur,
                price_risk_eur=price_risk,
                entry_cost_eur=entry_cost,
                exit_cost_eur=exit_cost,
                trade_risk_eur=price_risk + entry_cost + exit_cost,
                cash_required_eur=notional_eur + entry_cost,
            )


class SizingResult(Frozen):
    qty: Dec  # cantitatea finală propusă verificărilor (poate fi sub minim → respingere)
    candidate_qty: Dec  # rezultatul formulei din pasul 4
    fixed_costs_eur: Dec
    cost_per_unit_roundtrip_eur: Dec
    risk_per_unit_eur: Dec
    reduced: bool  # cantitatea a fost redusă față de candidat sau față de cererea strategiei


def min_tradable_qty(instrument: Instrument) -> Decimal:
    """Cea mai mică cantitate > 0, multiplu de `qty_step`, cel puțin `min_qty`."""
    return ceil_to_step(max(instrument.min_qty, instrument.qty_step), instrument.qty_step)


def _linearize(c1: Decimal, c2: Decimal, q1: Decimal, q2: Decimal) -> tuple[Decimal, Decimal]:
    """(fix, pe unitate) din două puncte; fără pantă dacă q2 == q1."""
    if q2 > q1:
        per_unit = max(ZERO, (c2 - c1) / (q2 - q1))
        return max(ZERO, c1 - per_unit * q1), per_unit
    return ZERO, c1 / q1


def size_entry(
    est: TradeRiskEstimator,
    instrument: Instrument,
    *,
    budget_eur: Decimal,
    cash_eur: Decimal,
    requested_qty: Decimal | None = None,
) -> SizingResult:
    """Pașii 3-4 (formula) și reducerea exactă; nu aplică limitele (le aplică engine-ul)."""
    step = instrument.qty_step
    q_min = min_tradable_qty(instrument)
    with localcontext(_CTX):
        entry_eur = est.entry_price * est.fx_rate_to_eur
        q_cash_cap = floor_to_step(max(cash_eur, ZERO) / entry_eur, step)
        q2 = max(q_cash_cap, q_min)

        r1, r2 = est.at(q_min), est.at(q2)
        fixed, per_unit = _linearize(r1.cost_eur, r2.cost_eur, q_min, q2)
        e_fixed, e_per_unit = _linearize(r1.entry_cost_eur, r2.entry_cost_eur, q_min, q2)
        risk_per_unit = (est.entry_price - est.stop_price) * est.fx_rate_to_eur + per_unit

        q_risk = (budget_eur - fixed) / risk_per_unit
        q_cash = (cash_eur - e_fixed) / (entry_eur + e_per_unit)
        raw = min(q_risk, q_cash)
        if requested_qty is not None:
            raw = min(raw, requested_qty)
        candidate = floor_to_step(max(raw, ZERO), step)

    def result(qty: Decimal, reduced: bool) -> SizingResult:
        if requested_qty is not None and qty < requested_qty:
            reduced = True
        return SizingResult(
            qty=qty,
            candidate_qty=candidate,
            fixed_costs_eur=fixed,
            cost_per_unit_roundtrip_eur=per_unit,
            risk_per_unit_eur=risk_per_unit,
            reduced=reduced,
        )

    if requested_qty is not None and requested_qty < q_min:
        return result(floor_to_step(max(requested_qty, ZERO), step), reduced=False)
    if candidate < q_min:
        # Formula nu acoperă minimul: propunem minimul, limitele dure decid (13.8).
        return result(q_min, reduced=False)

    def fits(qty: Decimal) -> bool:
        r = est.at(qty)
        return r.trade_risk_eur <= budget_eur and r.cash_required_eur <= cash_eur

    if fits(candidate):
        return result(candidate, reduced=False)

    # Liniarizarea a subestimat costul: reducere exactă la cea mai mare cantitate încadrată.
    lo, hi = int(q_min / step), int(candidate / step) - 1
    best: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid * step):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return result(best * step if best is not None else q_min, reduced=True)
