"""Pre-înregistrarea ipotezelor de cercetare, salvată și hash-uită înaintea evaluării.

Req 19.2, 19.8, 21.1, 21.8.

Înainte de evaluarea finală a unei strategii, cercetătorul fixează — și nu mai poate schimba
fără urmă — ipotezele studiului: metrica principală, `Promotion_Criteria`, spațiul parametrilor,
numărul de variante, metoda de corecție pentru testare multiplă, regimurile de piață,
multiplicatorii de stres pentru costuri și latență și ferestrele walk-forward (lungime, pas,
regula de recalibrare). Toate acestea sunt serializate canonic și hash-uite (`preregistration_id`
= SHA-256 peste conținut). Modificarea oricărui câmp schimbă hash-ul, deci o evaluare efectuată
sub un alt hash este imediat detectabilă.

Fail-closed (Req 19.8, 21.1, 21.8): `require_preregistration` ridică `EvaluationBlockedError` dacă
nu există o pre-înregistrare, dacă hash-ul ei nu se verifică, dacă ferestrele walk-forward nu au
lungime/pas/recalibrare fixate (19.8) sau dacă metoda de corecție pentru testare multiplă nu este
fixată (21.8). O evaluare refuză să ruleze fără o pre-înregistrare validă.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Final

from pydantic import AfterValidator, Field, model_validator

from qts.core.models import Dec, Frozen

__all__ = [
    "PREREGISTRATION_VERSION",
    "CorrectionMethod",
    "EvaluationBlockedError",
    "MarketRegimeSpec",
    "ParameterAxis",
    "PreRegistration",
    "PromotionCriteria",
    "StressMultipliers",
    "WalkForwardSpec",
    "create_preregistration",
    "preregistration_hash",
    "require_preregistration",
]

PREREGISTRATION_VERSION: Final = "preregistration-v1"


class EvaluationBlockedError(Exception):
    """Evaluarea nu poate rula: pre-înregistrare lipsă, invalidă sau incompletă (Req 19.8, 21.1)."""


class CorrectionMethod(StrEnum):
    """Metoda versionată de corecție pentru testarea multiplă (Req 21.2, 21.5, 21.8)."""

    DEFLATED_SHARPE = "deflated_sharpe"
    HOLM_BONFERRONI = "holm_bonferroni"
    DEFLATED_SHARPE_HOLM = "deflated_sharpe_holm"


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sorted_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    # Tupluri sortate (nu set): serializarea canonică și hash-ul rămân deterministe.
    return tuple(sorted(set(values)))


SortedStrs = Annotated[tuple[str, ...], AfterValidator(_sorted_unique)]


# --------------------------------------------------------------------------- componente


class PromotionCriteria(Frozen):
    """Pragurile numerice pe care o strategie trebuie să le atingă pentru a fi promovată.

    `min_net_result_eur` impune implicit cerința ca rezultatul net cumulat OOS să fie > 0
    (Req 21.4); `max_drawdown_eur` și pragurile Sharpe/câștiguri sunt opționale, dar fixate aici
    înaintea evaluării, nu alese ulterior.
    """

    min_net_result_eur: Dec
    min_sharpe: Dec | None = None
    max_drawdown_eur: Dec | None = None
    min_trades: Annotated[int, Field(ge=1)] = 1
    min_regimes: Annotated[int, Field(ge=1)] = 1
    min_robust_neighbor_fraction: Dec = Field(default=Decimal("0.5"))

    @model_validator(mode="after")
    def _check(self) -> PromotionCriteria:
        if self.max_drawdown_eur is not None and self.max_drawdown_eur < 0:
            raise ValueError("max_drawdown_eur nu poate fi negativ")
        if not (0 < self.min_robust_neighbor_fraction <= 1):
            raise ValueError("min_robust_neighbor_fraction trebuie să fie în (0, 1]")
        return self


class ParameterAxis(Frozen):
    """O axă a spațiului de parametri explorat, cu valorile candidate fixate (Req 21.1)."""

    name: str
    values: tuple[Dec, ...]
    neighbor_step: Dec | None = None  # pentru testul de robustețe ±1 pas (Req 19.5)

    @model_validator(mode="after")
    def _check(self) -> ParameterAxis:
        if not self.name.strip():
            raise ValueError("axa de parametri necesită un nume")
        if not self.values:
            raise ValueError(f"axa {self.name} nu are valori candidate")
        if len(self.values) != len(set(self.values)):
            raise ValueError(f"axa {self.name} are valori duplicate")
        if self.neighbor_step is not None and self.neighbor_step <= 0:
            raise ValueError(f"neighbor_step pentru {self.name} trebuie să fie > 0")
        return self


class WalkForwardSpec(Frozen):
    """Ferestrele walk-forward: lungime, pas și recalibrare fixate înainte (Req 19.1, 19.2, 19.8).

    `windows` este numărul de ferestre Out_Of_Sample_Set; cerința de minimum cinci (Req 19.1) este
    impusă aici, înaintea evaluării, nu la rularea walk-forward.
    """

    windows: Annotated[int, Field(ge=5)]
    train_bars: Annotated[int, Field(ge=1)]
    test_bars: Annotated[int, Field(ge=1)]
    step_bars: Annotated[int, Field(ge=1)]
    recalibrate: bool

    @model_validator(mode="after")
    def _check(self) -> WalkForwardSpec:
        # Lungimea (train/test), pasul și regula de recalibrare trebuie să fie fixate (Req 19.2).
        # Pydantic garantează deja că sunt prezente și pozitive; validăm doar coerența.
        if self.step_bars > self.test_bars + self.train_bars:
            raise ValueError("step_bars nu poate depăși lungimea unei ferestre")
        return self


class StressMultipliers(Frozen):
    """Multiplicatorii de stres pentru costuri și latență, fixați în pre-înregistrare (Req 20.3)."""

    cost_multipliers: tuple[Dec, ...] = (Dec(1), Dec("1.5"), Dec(2))
    latency_multipliers: tuple[Dec, ...] = (Dec(1), Dec(2))

    @model_validator(mode="after")
    def _check(self) -> StressMultipliers:
        for name, values in (
            ("cost_multipliers", self.cost_multipliers),
            ("latency_multipliers", self.latency_multipliers),
        ):
            if not values:
                raise ValueError(f"{name} nu poate fi gol")
            if any(v <= 0 for v in values):
                raise ValueError(f"{name} trebuie să conțină valori > 0")
            if Dec(1) not in values:
                raise ValueError(f"{name} trebuie să includă nivelul de bază 1,0")
        return self


class MarketRegimeSpec(Frozen):
    """Un regim de piață definit prin praguri măsurabile, fixate înainte (Req 20.1)."""

    name: str
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> MarketRegimeSpec:
        if not self.name.strip():
            raise ValueError("regimul necesită un nume")
        return self


# --------------------------------------------------------------------------- pre-înregistrare


class PreRegistration(Frozen):
    """Ipotezele de cercetare fixate și hash-uite înaintea evaluării (Req 21.1, 19.8, 21.8).

    `preregistration_id` este SHA-256 peste tot conținutul (toate câmpurile în afară de el însuși),
    calculat prin `preregistration_hash`. `verify` recompută hash-ul și confirmă că nu s-a schimbat
    nimic de la salvare.
    """

    preregistration_id: str
    version: str = PREREGISTRATION_VERSION
    strategy_id: str
    primary_metric: str
    promotion_criteria: PromotionCriteria
    parameter_space: tuple[ParameterAxis, ...]
    variant_count: Annotated[int, Field(ge=1)]
    correction_method: CorrectionMethod
    regimes: tuple[MarketRegimeSpec, ...]
    stress: StressMultipliers
    walk_forward: WalkForwardSpec
    drop_top_n_wins: Annotated[int, Field(ge=0)] = 0  # testul de eliminare a câștigurilor (21.6)

    @model_validator(mode="after")
    def _check(self) -> PreRegistration:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id este obligatoriu")
        if not self.primary_metric.strip():
            raise ValueError("primary_metric este obligatoriu")
        if not self.parameter_space:
            raise ValueError("parameter_space nu poate fi gol")
        names = [axis.name for axis in self.parameter_space]
        if len(names) != len(set(names)):
            raise ValueError("axe de parametri cu nume duplicat")
        if not self.regimes:
            raise ValueError("cel puțin un regim trebuie definit înaintea evaluării (Req 20.1)")
        regime_names = [r.name for r in self.regimes]
        if len(regime_names) != len(set(regime_names)):
            raise ValueError("regimuri cu nume duplicat")
        return self

    def content(self) -> dict[str, Any]:
        """Conținutul hash-uit (tot în afară de `preregistration_id`)."""
        return self.model_dump(mode="json", exclude={"preregistration_id"})

    def verify(self) -> bool:
        """Adevărat dacă `preregistration_id` corespunde conținutului (detectează modificări)."""
        return preregistration_hash(self.content()) == self.preregistration_id


def preregistration_hash(content: dict[str, Any]) -> str:
    """SHA-256 hex peste forma canonică a conținutului pre-înregistrării."""
    return hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


def create_preregistration(
    *,
    strategy_id: str,
    primary_metric: str,
    promotion_criteria: PromotionCriteria,
    parameter_space: tuple[ParameterAxis, ...],
    variant_count: int,
    correction_method: CorrectionMethod,
    regimes: tuple[MarketRegimeSpec, ...],
    stress: StressMultipliers,
    walk_forward: WalkForwardSpec,
    drop_top_n_wins: int = 0,
) -> PreRegistration:
    """Construiește o `PreRegistration` cu `preregistration_id` derivat din conținut (Req 21.1).

    Aceeași pre-înregistrare, construită de două ori, produce același identificator: hash-ul nu
    depinde de niciun element volatil (fără timestamp, fără ordine aleatorie).
    """
    content: dict[str, Any] = {
        "version": PREREGISTRATION_VERSION,
        "strategy_id": strategy_id,
        "primary_metric": primary_metric,
        "promotion_criteria": promotion_criteria.model_dump(mode="json"),
        "parameter_space": [axis.model_dump(mode="json") for axis in parameter_space],
        "variant_count": variant_count,
        "correction_method": correction_method.value,
        "regimes": [r.model_dump(mode="json") for r in regimes],
        "stress": stress.model_dump(mode="json"),
        "walk_forward": walk_forward.model_dump(mode="json"),
        "drop_top_n_wins": drop_top_n_wins,
    }
    pre = PreRegistration(
        preregistration_id=preregistration_hash(content),
        version=PREREGISTRATION_VERSION,
        strategy_id=strategy_id,
        primary_metric=primary_metric,
        promotion_criteria=promotion_criteria,
        parameter_space=parameter_space,
        variant_count=variant_count,
        correction_method=correction_method,
        regimes=regimes,
        stress=stress,
        walk_forward=walk_forward,
        drop_top_n_wins=drop_top_n_wins,
    )
    if not pre.verify():  # serializarea trebuie să fie stabilă (round-trip)
        raise EvaluationBlockedError(
            "conținutul pre-înregistrării nu este serializabil determinist"
        )
    return pre


def require_preregistration(pre: PreRegistration | None) -> PreRegistration:
    """Poartă fail-closed înaintea oricărei evaluări finale (Req 19.8, 21.1, 21.8).

    Refuză evaluarea dacă:
    - nu există pre-înregistrare (Req 21.1);
    - `preregistration_id` nu corespunde conținutului — a fost modificat după salvare;
    - lungimea, pasul sau recalibrarea walk-forward nu sunt fixate (Req 19.8);
    - metoda de corecție pentru testare multiplă nu este fixată (Req 21.8).

    La succes întoarce pre-înregistrarea validată; altfel ridică `EvaluationBlockedError`.
    """
    if pre is None:
        raise EvaluationBlockedError(
            "evaluarea este blocată: nu există o pre-înregistrare (Req 21.1)"
        )
    if not pre.verify():
        raise EvaluationBlockedError(
            "evaluarea este blocată: hash-ul pre-înregistrării nu corespunde conținutului"
        )
    # Req 19.8: ferestrele walk-forward trebuie să aibă lungime, pas și recalibrare fixate.
    # Modelul le impune ca prezente și pozitive; verificăm explicit pentru fail-closed.
    wf = pre.walk_forward
    if wf.windows < 5 or wf.train_bars < 1 or wf.test_bars < 1 or wf.step_bars < 1:
        raise EvaluationBlockedError(
            "evaluarea este blocată: ferestrele walk-forward nu sunt fixate complet (Req 19.8)"
        )
    # Req 21.8: metoda de corecție pentru testare multiplă trebuie preînregistrată.
    if pre.correction_method not in CorrectionMethod:
        raise EvaluationBlockedError(
            "evaluarea este blocată: metoda de corecție nu este preînregistrată (Req 21.8)"
        )
    return pre
