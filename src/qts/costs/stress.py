"""Scenarii de stres pentru Complete_Cost_Model (Req 20.3, 20.4).

Un `StressScenario` derivă o configurație nouă, versionată, din configurația de bază:

- spread: spreadurile configurate (per oră și implicit) și jumătatea de spread din cotație
  (`quote_spread_multiplier`) sunt înmulțite cu `cost_multiplier`;
- comision: procentul, minimul, maximul și taxele de bursă (procent și fix) sunt înmulțite,
  deci comisionul rezultat este exact `cost_multiplier ×` comisionul de bază;
- slippage: `k` și `min_ticks` sunt înmulțite (slippage-ul estimat este liniar în ambele);
- conversie FX: spreadul de conversie al brokerului este tot un spread de execuție, deci este
  stresat conservator împreună cu celelalte;
- taxe: NU sunt stresate. Sunt valori statutare, aprobate de operator (Req 8), nu ipoteze de
  piață; înmulțirea lor ar introduce o valoare neaprobată în configurație;
- latență: `latency_ms` este înlocuită cu valoarea de stres preînregistrată, care trebuie să fie
  >= latența de bază. Când `CostContext.price_after_latency` este furnizat, efectul latenței
  vine din mișcarea efectivă a prețului; apelantul (Backtest) alege atunci prețul corespunzător
  `config.latency.latency_ms` al configurației stresate.

Comisioanele raportate de broker în `realize` sunt valori observate și nu sunt stresate.
Stresul nu poate reduce costurile: `cost_multiplier >= 1`. Componentele lipsă rămân lipsă,
astfel încât incompletitudinea este detectată la evaluare (Req 8.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, localcontext
from typing import Annotated, Final

from pydantic import Field, model_validator

from qts.core.models import Dec, Frozen

from .commission_tables import CommissionSchedule, CommissionTable
from .errors import CostModelIncomplete, cost_context
from .model import CompleteCostModel, CostModelConfig, LatencyConfig, SpreadSchedule

__all__ = [
    "BASELINE",
    "STRESS_LEVELS",
    "STRESS_MULTIPLIERS",
    "StressScenario",
    "stress_scenarios",
    "stressed_config",
    "stressed_model",
]

STRESS_MULTIPLIERS: Final[tuple[Decimal, ...]] = (Decimal("1.0"), Decimal("1.5"), Decimal("2.0"))


def _label(value: Decimal) -> str:
    text = f"{value.normalize():f}"
    return text if "." in text else f"{text}.0"


class StressScenario(Frozen):
    """Multiplicatorul costurilor și, opțional, latența de stres (None = latența de bază)."""

    cost_multiplier: Dec = Decimal(1)
    latency_ms: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def _check(self) -> StressScenario:
        if not self.cost_multiplier.is_finite() or self.cost_multiplier < 1:
            raise ValueError("cost_multiplier trebuie să fie finit și >= 1")
        return self

    @property
    def is_baseline(self) -> bool:
        return self.cost_multiplier == 1 and self.latency_ms is None

    @property
    def suffix(self) -> str:
        """Sufixul de versiune, de ex. `+stress1.5` sau `+stress2.0+lat500ms`."""
        parts = [f"+stress{_label(self.cost_multiplier)}"] if self.cost_multiplier != 1 else []
        if self.latency_ms is not None:
            parts.append(f"+lat{self.latency_ms}ms")
        return "".join(parts)


BASELINE: Final = StressScenario()
STRESS_LEVELS: Final[tuple[StressScenario, ...]] = tuple(
    StressScenario(cost_multiplier=m) for m in STRESS_MULTIPLIERS
)


def stress_scenarios(
    stress_latencies_ms: Sequence[int] = (),
    multipliers: Sequence[Decimal] = STRESS_MULTIPLIERS,
) -> tuple[StressScenario, ...]:
    """Grila completă: fiecare multiplicator × (latența de bază + fiecare latență de stres)."""
    latencies: tuple[int | None, ...] = (None, *dict.fromkeys(stress_latencies_ms))
    return tuple(
        StressScenario(cost_multiplier=m, latency_ms=lat) for m in multipliers for lat in latencies
    )


# --------------------------------------------------------------------------- derivare


def _scale_table(table: CommissionTable, m: Decimal) -> CommissionTable:
    return table.model_copy(
        update={
            "percent": table.percent * m,
            "minimum": table.minimum * m,
            "maximum": None if table.maximum is None else table.maximum * m,
            "exchange_fee_percent": table.exchange_fee_percent * m,
            "exchange_fee_fixed": table.exchange_fee_fixed * m,
        }
    )


def _scale_spread(schedule: SpreadSchedule, m: Decimal) -> SpreadSchedule:
    return SpreadSchedule(
        by_hour_utc={h: v * m for h, v in schedule.by_hour_utc.items()},
        default=None if schedule.default is None else schedule.default * m,
    )


def _stressed_latency(base: LatencyConfig | None, latency_ms: int | None) -> LatencyConfig | None:
    if latency_ms is None:
        return base
    if base is None:
        raise CostModelIncomplete(
            "latency", "latența de bază lipsește; stresul nu poate fi aplicat"
        )
    if latency_ms < base.latency_ms:
        raise ValueError(
            f"latența de stres {latency_ms} ms este sub latența de bază {base.latency_ms} ms"
        )
    return LatencyConfig(latency_ms=latency_ms)


def stressed_config(base: CostModelConfig, scenario: StressScenario) -> CostModelConfig:
    """Configurația derivată pentru `scenario`; scenariul de bază întoarce `base` neschimbată."""
    if scenario.is_baseline:
        return base
    m = scenario.cost_multiplier
    with localcontext(cost_context()):
        commissions = (
            None
            if base.commissions is None
            else CommissionSchedule(
                tables=tuple(_scale_table(t, m) for t in base.commissions.tables)
            )
        )
        slippage = (
            None
            if base.slippage is None
            else base.slippage.model_copy(
                update={"k": base.slippage.k * m, "min_ticks": base.slippage.min_ticks * m}
            )
        )
        fx = (
            None
            if base.fx is None
            else base.fx.model_copy(update={"conversion_spread": base.fx.conversion_spread * m})
        )
        return CostModelConfig(
            version=f"{base.version}{scenario.suffix}",
            commissions=commissions,
            spreads={sym: _scale_spread(s, m) for sym, s in base.spreads.items()},
            slippage=slippage,
            latency=_stressed_latency(base.latency, scenario.latency_ms),
            fx=fx,
            taxes=base.taxes,  # statutare, aprobate de operator: nestresate
            quote_spread_multiplier=base.quote_spread_multiplier * m,
        )


def stressed_model(base: CostModelConfig, scenario: StressScenario) -> CompleteCostModel:
    return CompleteCostModel(stressed_config(base, scenario))
