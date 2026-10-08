"""Contextul imuabil primit de Risk_Engine pentru fiecare `Order_Intent` (Req 12.3, 12.6).

Contextul este un instantaneu: numerar, poziții, PnL zilnic și total, date de piață per
instrument (cotație, context de cost, prospețime) și starea `Kill_Switch`. Risk_Engine nu
modifică și nu citește altă stare. Toate sumele sunt `Decimal`, în EUR, cu excepția prețurilor,
care sunt în moneda instrumentului și se convertesc cu `fx_rate_to_eur`.

Kill switch-ul nu este implementat aici (Req 11.3, 14): contextul doar transportă starea
domeniilor active, iar Risk_Engine o consumă.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from qts.config.schema import Environment
from qts.core.models import Dec, Frozen, Instrument, Quote, UtcDatetime
from qts.core.money import ZERO
from qts.costs.model import CostContext

__all__ = [
    "KillSwitchScope",
    "KillSwitchState",
    "MarketSnapshot",
    "PositionRisk",
    "ProjectStageName",
    "RiskContext",
]

# Orice valoare diferită de "post_initial" este tratată ca Initial_Stage (fail-closed).
ProjectStageName = Literal["initial", "post_initial"]


class KillSwitchScope(StrEnum):
    INSTRUMENT = "INSTRUMENT"
    DAY = "DAY"
    GLOBAL = "GLOBAL"
    CAPITAL_CONFIG = "CAPITAL_CONFIG"


class KillSwitchState(Frozen):
    """Domeniile `Kill_Switch` active la momentul instantaneului."""

    global_active: bool = False
    day_active: bool = False
    capital_config_active: bool = False
    instruments: frozenset[str] = frozenset()

    def blocking_scope(self, instrument: str) -> KillSwitchScope | None:
        """Primul domeniu activ aplicabil instrumentului (ordine stabilă), sau None."""
        if self.capital_config_active:
            return KillSwitchScope.CAPITAL_CONFIG
        if self.global_active:
            return KillSwitchScope.GLOBAL
        if self.day_active:
            return KillSwitchScope.DAY
        if instrument in self.instruments:
            return KillSwitchScope.INSTRUMENT
        return None


class MarketSnapshot(Frozen):
    """Datele de piață pentru un instrument, așa cum le vede Risk_Engine."""

    instrument: Instrument
    data_fresh: bool  # verdictul `FreshnessTracker` (Req 5.5)
    freshness_reason: str | None = None
    quote: Quote | None = None
    cost_ctx: CostContext

    @model_validator(mode="after")
    def _quote_matches(self) -> MarketSnapshot:
        if self.quote is not None and self.quote.instrument != self.instrument.symbol:
            raise ValueError("cotația nu aparține instrumentului instantaneului")
        return self


class PositionRisk(Frozen):
    """O poziție long deschisă, cu datele necesare riscului deschis cel mai defavorabil."""

    instrument: str
    qty: Dec
    mark_price: Dec  # moneda instrumentului
    stop_price: Dec | None = None  # None: riscul deschis este întreaga valoare (fail-closed)
    fx_rate_to_eur: Dec
    exit_cost_eur: Dec  # costul estimat al ieșirii, inclus în limite (Req 13.9)

    @model_validator(mode="after")
    def _check(self) -> PositionRisk:
        if self.qty <= 0:
            raise ValueError("qty poziției trebuie să fie > 0 (numai long în Initial_Stage)")
        if self.mark_price <= 0 or self.fx_rate_to_eur <= 0:
            raise ValueError("mark_price și fx_rate_to_eur trebuie să fie > 0")
        if self.exit_cost_eur < 0:
            raise ValueError("exit_cost_eur nu poate fi negativ")
        return self


class RiskContext(Frozen):
    ts: UtcDatetime
    mode: Environment
    project_stage: ProjectStageName = "initial"
    cash_eur: Dec
    positions: dict[str, PositionRisk] = Field(default_factory=dict)
    daily_realized_pnl_eur: Dec = ZERO
    daily_unrealized_pnl_eur: Dec = ZERO
    total_pnl_eur: Dec = ZERO  # realizat + nerealizat în configurația de capital curentă
    market: dict[str, MarketSnapshot] = Field(default_factory=dict)
    kill_switch: KillSwitchState = KillSwitchState()

    @model_validator(mode="after")
    def _check(self) -> RiskContext:
        for key, pos in self.positions.items():
            if key != pos.instrument:
                raise ValueError(f"cheia poziției {key} diferă de instrument {pos.instrument}")
        for key, snap in self.market.items():
            if key != snap.instrument.symbol:
                raise ValueError(f"cheia instantaneului {key} diferă de instrument")
        return self

    @property
    def daily_loss_eur(self) -> Dec:
        """Pierderea zilnică realizată + nerealizată (≥ 0) (Req 13.3, 13.4)."""
        return max(ZERO, -(self.daily_realized_pnl_eur + self.daily_unrealized_pnl_eur))

    @property
    def total_loss_eur(self) -> Dec:
        return max(ZERO, -self.total_pnl_eur)
