"""Robustețea parametrilor și testul de eliminare a câștigurilor extreme (Req 19.5, 19.6, 21.6).

Două verificări independente, ambele fail-closed:

1. Vecinătatea parametrilor (Req 19.5, 19.6): pentru parametrii aleși, se construiesc vecinii la
   ±1 pas pe fiecare axă (pasul = `ParameterAxis.neighbor_step`, iar vecinul trebuie să existe în
   `ParameterAxis.values`). Fiecare vecin este evaluat; dacă fracția de vecini acceptabili este sub
   `Promotion_Criteria.min_robust_neighbor_fraction` (implicit 1/2), rezultatul acceptabil apare
   doar într-un punct izolat și strategia este respinsă (Req 19.6). Parametrii aleși sunt incluși în
   evaluare, dar `passed` se bazează pe fracția vecinilor, nu pe punctul central.

2. Eliminarea celor mai mari N câștiguri (Req 21.6): se elimină cele mai mari `drop_top_n`
   rezultate pozitive per tranzacție; dacă rezultatul net rămas este <= 0, avantajul depinde de
   câteva tranzacții norocoase și strategia este respinsă.

Evaluarea unui punct din spațiul parametrilor este furnizată de apelant printr-un
`NeighborEvaluator` pur, deci modulul nu depinde de motorul event-driven și rămâne determinist.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal

from qts.core.models import Dec, Frozen
from qts.core.money import ZERO

from .preregistration import ParameterAxis, PreRegistration

__all__ = [
    "NeighborEvaluator",
    "NeighborOutcome",
    "ParameterPoint",
    "RobustnessResult",
    "drop_top_n_wins",
    "neighbors_of",
    "robustness_check",
]

ParameterPoint = Mapping[str, Decimal]
NeighborEvaluator = Callable[[ParameterPoint], Decimal]
"""Dat un punct din spațiul parametrilor, întoarce rezultatul net cumulat (EUR)."""


class NeighborOutcome(Frozen):
    """Rezultatul evaluării unui punct (centru sau vecin) din spațiul parametrilor."""

    point: tuple[tuple[str, Dec], ...]
    net_result_eur: Dec
    acceptable: bool
    is_center: bool


class RobustnessResult(Frozen):
    """Verdictul de robustețe a vecinătății (Req 19.5, 19.6)."""

    center: NeighborOutcome
    neighbors: tuple[NeighborOutcome, ...]
    acceptable_fraction: Dec
    required_fraction: Dec

    @property
    def passed(self) -> bool:
        """Adevărat dacă destui vecini sunt acceptabili (nu un punct izolat, Req 19.6)."""
        if not self.neighbors:
            return False
        return self.acceptable_fraction >= self.required_fraction


def _point_key(point: ParameterPoint) -> tuple[tuple[str, Dec], ...]:
    return tuple(sorted(point.items()))


def _axis_neighbors(axis: ParameterAxis, value: Decimal) -> list[Decimal]:
    """Valorile ±1 pas față de `value`, prezente în valorile candidate ale axei."""
    if axis.neighbor_step is None:
        return []
    present = set(axis.values)
    found: list[Decimal] = []
    for candidate in (value - axis.neighbor_step, value + axis.neighbor_step):
        for existing in axis.values:
            if existing == candidate and existing not in found:
                found.append(existing)
    # Verificăm apartenența exactă (evită erori de reprezentare între step și valori).
    return [c for c in found if c in present]


def neighbors_of(
    parameter_space: Sequence[ParameterAxis], chosen: ParameterPoint
) -> tuple[dict[str, Decimal], ...]:
    """Toți vecinii la ±1 pas pe fiecare axă (celelalte axe rămân la valoarea aleasă) (Req 19.5)."""
    axes = {axis.name: axis for axis in parameter_space}
    missing = set(chosen) - set(axes)
    if missing:
        raise ValueError(f"parametri necunoscuți în spațiul definit: {sorted(missing)}")
    if set(axes) - set(chosen):
        raise ValueError("punctul ales nu acoperă toate axele spațiului de parametri")
    result: list[dict[str, Decimal]] = []
    for name, value in chosen.items():
        if value not in set(axes[name].values):
            raise ValueError(f"valoarea aleasă {value} nu se află pe axa {name}")
        for neighbor_value in _axis_neighbors(axes[name], value):
            point = dict(chosen)
            point[name] = neighbor_value
            result.append(point)
    return tuple(result)


def robustness_check(
    pre: PreRegistration,
    chosen: ParameterPoint,
    evaluate: NeighborEvaluator,
) -> RobustnessResult:
    """Evaluează centrul și vecinii ±1 pas; respinge dacă acceptabilitatea e izolată (Req 19.6)."""
    threshold = pre.promotion_criteria.min_net_result_eur
    required = pre.promotion_criteria.min_robust_neighbor_fraction
    center_net = evaluate(chosen)
    center = NeighborOutcome(
        point=_point_key(chosen),
        net_result_eur=center_net,
        acceptable=center_net >= threshold,
        is_center=True,
    )
    neighbor_points = neighbors_of(pre.parameter_space, chosen)
    neighbors = tuple(
        NeighborOutcome(
            point=_point_key(point),
            net_result_eur=net,
            acceptable=net >= threshold,
            is_center=False,
        )
        for point in neighbor_points
        for net in (evaluate(point),)
    )
    acceptable = sum(1 for n in neighbors if n.acceptable)
    fraction = Decimal(acceptable) / Decimal(len(neighbors)) if neighbors else ZERO
    return RobustnessResult(
        center=center,
        neighbors=neighbors,
        acceptable_fraction=fraction,
        required_fraction=required,
    )


def drop_top_n_wins(net_results_eur: Sequence[Decimal], drop_top_n: int) -> Decimal:
    """Rezultatul net după eliminarea celor mai mari `drop_top_n` câștiguri pozitive (Req 21.6).

    Se elimină doar rezultate strict pozitive (câștiguri). Dacă există mai puține câștiguri decât
    `drop_top_n`, se elimină toate câștigurile. Pierderile și tranzacțiile nule rămân.
    """
    if drop_top_n < 0:
        raise ValueError("drop_top_n nu poate fi negativ")
    total = sum(net_results_eur, ZERO)
    if drop_top_n == 0:
        return total
    wins = sorted((r for r in net_results_eur if r > 0), reverse=True)
    return total - sum(wins[:drop_top_n], ZERO)
