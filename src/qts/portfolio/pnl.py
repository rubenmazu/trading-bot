"""Calculul PnL: metoda costului mediu, rezultat brut vs. net pe categorii de cost (Req 8.4, 29.1).

Convenții (documentate, folosite de `qts.portfolio.portfolio`):

- **Costul mediu ponderat**: baza de cost a unei poziții se ține atât în moneda instrumentului,
  cât și în EUR (la cursul fiecărei cumpărări). La o ieșire parțială se eliberează fracțiunea
  `sell_qty / qty` din ambele baze; la ieșirea totală se eliberează exact baza rămasă, astfel
  încât suma bazelor eliberate este egală cu suma cumpărărilor (fără reziduuri de rotunjire).
- **Rezultatul brut realizat** = încasarea în EUR (`qty × price × fx`) − baza EUR eliberată.
  Include deci și efectul de curs valutar al instrumentelor în altă monedă.
- **Costurile** (`CostBreakdown`, în EUR) nu se capitalizează în baza de cost. Ele se
  recunosc în rezultatul net realizat la momentul în care sunt suportate (când se plătesc din
  numerar), pe categorii: spread, comision, slippage, latență, conversie valutară, taxe.
- **Rezultatul nerealizat** se calculează brut, la marcaje: `qty × mark × fx_mark − baza EUR`.
  Costul estimat al ieșirii nu este inclus aici; Risk_Engine îl primește separat
  (`PositionRisk.exit_cost_eur`).
- **Net** = brut − total costuri. Cu costuri nenegative, net ≤ brut (P14).

Toate calculele folosesc `Decimal` cu precizie 28 și rotunjire half-even; nu se folosesc float.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Final

from pydantic import model_validator

from qts.core.models import CostBreakdown, Dec, Frozen, UtcDatetime
from qts.core.money import PRECISION, ZERO

__all__ = [
    "AverageCostRelease",
    "DailyPnL",
    "DailyPnLRecord",
    "Mark",
    "PnLBreakdown",
    "PnLReport",
    "market_value_eur",
    "release_average_cost",
    "unrealized_eur",
]

_CTX: Final = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)


class Mark(Frozen):
    """Prețul de marcare al unui instrument (moneda instrumentului) și cursul către EUR."""

    price: Dec
    fx_rate_to_eur: Dec
    ts: UtcDatetime

    @model_validator(mode="after")
    def _check(self) -> Mark:
        if self.price <= 0 or self.fx_rate_to_eur <= 0:
            raise ValueError("price și fx_rate_to_eur trebuie să fie > 0")
        return self


class PnLBreakdown(Frozen):
    """Rezultat brut + costuri pe categorii; netul este derivat (Req 8.4)."""

    gross_eur: Dec = ZERO
    costs: CostBreakdown = CostBreakdown()

    @property
    def net_eur(self) -> Decimal:
        return self.gross_eur - self.costs.total

    def add(self, gross_eur: Decimal = ZERO, costs: CostBreakdown | None = None) -> PnLBreakdown:
        return PnLBreakdown(
            gross_eur=self.gross_eur + gross_eur,
            costs=self.costs if costs is None else self.costs + costs,
        )


class DailyPnL(Frozen):
    """Acumulatorul `Trading_Day` curent.

    `start_unrealized_eur` este PnL-ul nerealizat la ultimele marcaje cunoscute în momentul
    trecerii în această zi. PnL-ul nerealizat zilnic = nerealizat curent − această valoare;
    astfel, realizat zilnic + nerealizat zilnic = variația capitalului în zi (fără fluxuri).
    """

    trading_day: date
    realized: PnLBreakdown = PnLBreakdown()
    start_unrealized_eur: Dec = ZERO


class DailyPnLRecord(Frozen):
    """Rezultatul unui `Trading_Day` închis."""

    trading_day: date
    realized: PnLBreakdown
    unrealized_change_eur: Dec

    @property
    def gross_eur(self) -> Decimal:
        return self.realized.gross_eur + self.unrealized_change_eur

    @property
    def net_eur(self) -> Decimal:
        return self.realized.net_eur + self.unrealized_change_eur


class PnLReport(Frozen):
    """Raportul separat: brut, fiecare categorie de cost, net (Req 8.4, 29.1)."""

    realized_gross_eur: Dec
    unrealized_gross_eur: Dec
    costs: CostBreakdown

    @property
    def gross_eur(self) -> Decimal:
        return self.realized_gross_eur + self.unrealized_gross_eur

    @property
    def total_costs_eur(self) -> Decimal:
        return self.costs.total

    @property
    def net_eur(self) -> Decimal:
        return self.gross_eur - self.costs.total

    def as_rows(self) -> tuple[tuple[str, Decimal], ...]:
        """Rânduri stabile pentru afișare: brut, fiecare categorie de cost, total costuri, net."""
        c = self.costs
        return (
            ("gross_realized", self.realized_gross_eur),
            ("gross_unrealized", self.unrealized_gross_eur),
            ("gross", self.gross_eur),
            ("cost_spread", c.spread),
            ("cost_commission", c.commission),
            ("cost_slippage", c.slippage),
            ("cost_latency", c.latency),
            ("cost_fx_conversion", c.fx_conversion),
            ("cost_taxes", c.taxes),
            ("cost_total", c.total),
            ("net", self.net_eur),
        )


class AverageCostRelease(Frozen):
    """Efectul unei vânzări asupra bazei de cost, după metoda costului mediu."""

    released_basis_ccy: Dec
    released_basis_eur: Dec
    realized_gross_eur: Dec


def release_average_cost(
    *,
    qty: Decimal,
    basis_ccy: Decimal,
    basis_eur: Decimal,
    sell_qty: Decimal,
    proceeds_eur: Decimal,
) -> AverageCostRelease:
    """Eliberează baza de cost pentru `sell_qty` din `qty` (0 < sell_qty ≤ qty)."""
    if qty <= 0 or sell_qty <= 0 or sell_qty > qty:
        raise ValueError(f"vânzare invalidă: sell_qty={sell_qty}, qty={qty}")
    with localcontext(_CTX):
        if sell_qty == qty:
            rel_ccy, rel_eur = basis_ccy, basis_eur
        else:
            rel_ccy = basis_ccy * sell_qty / qty
            rel_eur = basis_eur * sell_qty / qty
        return AverageCostRelease(
            released_basis_ccy=rel_ccy,
            released_basis_eur=rel_eur,
            realized_gross_eur=proceeds_eur - rel_eur,
        )


def market_value_eur(qty: Decimal, mark: Mark) -> Decimal:
    with localcontext(_CTX):
        return qty * mark.price * mark.fx_rate_to_eur


def unrealized_eur(
    holdings: Mapping[str, tuple[Decimal, Decimal]], marks: Mapping[str, Mark]
) -> Decimal:
    """Σ (qty × mark × fx − baza EUR) pentru `holdings[instrument] = (qty, basis_eur)`.

    Un instrument fără marcaj ridică `KeyError` (fail-closed: nu se presupune valoare zero).
    """
    total = ZERO
    with localcontext(_CTX):
        for instrument in sorted(holdings):
            qty, basis_eur = holdings[instrument]
            total += market_value_eur(qty, marks[instrument]) - basis_eur
    return total
