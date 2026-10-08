"""Erori și contextul zecimal comun modelului de costuri."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Context
from typing import Final

from qts.core.money import PRECISION

_COST_CTX: Final = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)


def cost_context() -> Context:
    """Copie a contextului zecimal folosit de calculele de cost (precizie fixă, half-even)."""
    return _COST_CTX.copy()


class CostModelIncomplete(Exception):  # noqa: N818 - numele este fixat de design
    """O componentă obligatorie a costului lipsește; evaluarea strategiei este invalidată (8.3)."""

    def __init__(self, component: str, detail: str) -> None:
        super().__init__(f"componenta de cost '{component}' lipsește: {detail}")
        self.component = component
        self.detail = detail
