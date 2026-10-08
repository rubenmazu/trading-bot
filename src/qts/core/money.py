"""Aritmetică monetară deterministă bazată pe Decimal.

Reguli:
- valorile `float` sunt refuzate, pentru a evita erorile de rotunjire binară;
- prețurile se rotunjesc la `tick_size`, iar cantitățile se rotunjesc mereu în jos la `qty_step`,
  astfel încât rotunjirea să nu poată crește expunerea sau riscul.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Final

PRECISION: Final = 28
REPORTING_CURRENCY: Final = "EUR"
MONEY_QUANTUM: Final = Decimal("0.00000001")
ZERO: Final = Decimal(0)

_CTX: Final = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)


class FloatNotAllowedError(ValueError):
    """Ridicată când o valoare monetară este construită dintr-un float."""


def dec(value: Decimal | int | str) -> Decimal:
    """Construiește un Decimal finit; refuză float, NaN și infinit."""
    if isinstance(value, float):
        raise FloatNotAllowedError("valorile monetare nu pot fi float; folosiți str sau Decimal")
    if isinstance(value, bool):
        raise TypeError("bool nu este o valoare monetară")
    result = value if isinstance(value, Decimal) else Decimal(value)
    if not result.is_finite():
        raise ValueError(f"valoare nefinită: {value!r}")
    return result


def _require_positive(step: Decimal, name: str) -> None:
    if step <= 0:
        raise ValueError(f"{name} trebuie să fie > 0, primit {step}")


def floor_to_step(qty: Decimal, step: Decimal) -> Decimal:
    """Rotunjește în jos (spre -inf) la multiplu de `step`."""
    _require_positive(step, "step")
    with localcontext(_CTX) as ctx:
        ctx.rounding = ROUND_FLOOR
        units = (qty / step).to_integral_value(rounding=ROUND_FLOOR)
        return units * step


def ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Rotunjește în sus (spre +inf) la multiplu de `step` (de exemplu pentru costuri)."""
    _require_positive(step, "step")
    with localcontext(_CTX):
        units = (value / step).to_integral_value(rounding=ROUND_CEILING)
        return units * step


def round_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    """Rotunjește prețul la cel mai apropiat tick (half-even)."""
    _require_positive(tick, "tick")
    with localcontext(_CTX):
        units = (price / tick).to_integral_value(rounding=ROUND_HALF_EVEN)
        return units * tick


def notional(price: Decimal, qty: Decimal) -> Decimal:
    """Valoarea nominală `price × qty`, în moneda instrumentului."""
    with localcontext(_CTX):
        return price * qty


def convert(amount: Decimal, rate: Decimal) -> Decimal:
    """Convertește `amount` din moneda sursă în moneda țintă.

    `rate` este numărul de unități ale monedei țintă pentru o unitate a monedei sursă
    (de exemplu, USD→EUR = 0.92). Spreadul de conversie al brokerului ține de modelul de costuri.
    """
    if rate <= 0:
        raise ValueError(f"rate trebuie să fie > 0, primit {rate}")
    with localcontext(_CTX):
        return amount * rate


def quantize_money(amount: Decimal) -> Decimal:
    """Normalizează o sumă la cuantumul intern de raportare."""
    with localcontext(_CTX):
        return amount.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_EVEN)
