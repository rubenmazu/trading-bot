"""Stresarea ipotezelor de cost și latență pentru pipeline-ul de cercetare (Req 20.3, 20.4, 20.6).

Un `StressSuite` evaluează o strategie la grila de scenarii preînregistrate: multiplicatorii de
cost (×1,0 / ×1,5 / ×2,0) și latențele de stres, construind pentru fiecare o configurație de cost
derivată prin `qts.costs.stress`. Modulul nu rulează el însuși backtestul; primește o funcție de
evaluare (`StressEvaluator`) care, pentru un `StressScenario`, întoarce rezultatul net cumulat al
strategiei sub acel scenariu. Astfel stresul rămâne pur și determinist și poate fi testat fără a
depinde de motorul event-driven.

Scenariul de bază (×1,0, fără latență de stres) este referința. Un scenariu de stres este
`acceptable` dacă rezultatul lui net satisface `Promotion_Criteria.min_net_result_eur`. Dacă vreun
scenariu de stres preînregistrat nu îndeplinește criteriile, suita raportează `passed = False`, iar
apelantul respinge strategia (Req 20.6).

Grila de scenarii este derivată din `PreRegistration.stress`: multiplicatorii de cost și
latențele de stres sunt luate exact de acolo, deci stresul evaluat corespunde celui preînregistrat
(Req 20.3, 20.4). Latențele de stres sunt exprimate ca milisecunde absolute în pre-înregistrare și
sunt deduse din multiplicatorii de latență × latența de bază furnizată de apelant.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from qts.core.models import Dec, Frozen
from qts.costs.stress import StressScenario

from .preregistration import PreRegistration

__all__ = [
    "StressEvaluator",
    "StressOutcome",
    "StressResult",
    "StressSuite",
    "scenarios_from_preregistration",
]

StressEvaluator = Callable[[StressScenario], Decimal]
"""Dată o configurație de stres, întoarce rezultatul net cumulat (EUR) al strategiei."""


class StressOutcome(Frozen):
    """Rezultatul net al strategiei sub un singur scenariu de stres."""

    cost_multiplier: Dec
    latency_ms: int | None
    net_result_eur: Dec
    acceptable: bool
    is_baseline: bool


class StressResult(Frozen):
    """Agregatul tuturor scenariilor de stres evaluate (Req 20.3, 20.4, 20.6)."""

    outcomes: tuple[StressOutcome, ...]

    @property
    def passed(self) -> bool:
        """Adevărat dacă fiecare scenariu de stres este acceptabil (Req 20.6)."""
        return all(o.acceptable for o in self.outcomes)

    @property
    def baseline(self) -> StressOutcome:
        for outcome in self.outcomes:
            if outcome.is_baseline:
                return outcome
        raise ValueError("suita de stres nu conține scenariul de bază")

    def failing(self) -> tuple[StressOutcome, ...]:
        """Scenariile de stres care nu au trecut (fără cel de bază, raportat separat)."""
        return tuple(o for o in self.outcomes if not o.acceptable)


def scenarios_from_preregistration(
    pre: PreRegistration, base_latency_ms: int
) -> tuple[StressScenario, ...]:
    """Derivă grila de scenarii din pre-înregistrare (Req 20.3, 20.4).

    Multiplicatorii de cost provin din `pre.stress.cost_multipliers`; latențele de stres sunt
    `multiplicator × base_latency_ms` pentru fiecare multiplicator de latență > 1 (multiplicatorul
    1,0 reprezintă latența de bază, modelată ca `latency_ms = None`). Fiecare multiplicator de cost
    este combinat cu latența de bază și cu fiecare latență de stres, în ordine deterministă.
    """
    if base_latency_ms < 0:
        raise ValueError("base_latency_ms nu poate fi negativ")
    cost_mults = tuple(dict.fromkeys(pre.stress.cost_multipliers))
    stress_latencies: list[int | None] = [None]
    for mult in pre.stress.latency_multipliers:
        if mult == 1:
            continue
        latency = int((Decimal(base_latency_ms) * mult).to_integral_value())
        if latency not in stress_latencies:
            stress_latencies.append(latency)
    return tuple(
        StressScenario(cost_multiplier=cost, latency_ms=lat)
        for cost in cost_mults
        for lat in stress_latencies
    )


class StressSuite:
    """Evaluează o strategie peste grila de scenarii de stres preînregistrată."""

    def __init__(self, pre: PreRegistration, *, base_latency_ms: int) -> None:
        self._pre = pre
        self._scenarios = scenarios_from_preregistration(pre, base_latency_ms)

    @property
    def scenarios(self) -> tuple[StressScenario, ...]:
        return self._scenarios

    def run(self, evaluate: StressEvaluator) -> StressResult:
        """Rulează fiecare scenariu și clasifică rezultatele față de `Promotion_Criteria`."""
        threshold = self._pre.promotion_criteria.min_net_result_eur
        outcomes = tuple(
            StressOutcome(
                cost_multiplier=scenario.cost_multiplier,
                latency_ms=scenario.latency_ms,
                net_result_eur=net,
                acceptable=net >= threshold,
                is_baseline=scenario.is_baseline,
            )
            for scenario in self._scenarios
            for net in (evaluate(scenario),)
        )
        return StressResult(outcomes=outcomes)
