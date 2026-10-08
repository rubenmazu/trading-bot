"""Live_Gate: poartă indivizibilă înaintea oricărei transmisii Live (Req 3.6, 15).

În această versiune poarta este numai închisă:

- `evaluate()` reevaluează de la zero, la fiecare apel, toate condițiile 15.1–15.5 (15.12) și
  întoarce lista celor nesatisfăcute, cu coduri stabile (15.8);
- condiția 15.1 (ieșire aprobată din Initial_Stage) este satisfăcută numai pentru etapele din
  `POST_INITIAL_STAGES`, mulțime goală aici; `ProjectStage` nu poate reprezenta o etapă
  ulterioară, deci poarta nu se poate deschide structural;
- nu există flux de confirmare (15.6, 15.7) și nici comandă care să deschidă poarta;
- condițiile 15.2–15.5 vin prin furnizori injectați, deoarece registrul Open_Decision (17.2),
  Paper_Qualification (17.3), Strategy_Artifact (17.1), reconcilierea (13.1) și Health_Monitor
  (14.1) nu sunt încă implementate. Furnizorii impliciți sunt fail-closed (condiție
  nesatisfăcută, „furnizor indisponibil”), iar orice excepție sau rezultat invalid al unui
  furnizor înseamnă condiție nesatisfăcută.

`LiveGate` satisface `LiveGateLike` din `broker/fail_safe.py` (`is_open() -> bool`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from qts.safety.stage import ProjectStage, StageInfo, read_stage

__all__ = [
    "CONDITION_ORDER",
    "CONDITION_REQUIREMENTS",
    "LIVE_RISK_WARNING",
    "POST_INITIAL_STAGES",
    "ConditionCheck",
    "ConditionProvider",
    "LiveGate",
    "LiveGateCondition",
    "LiveGateEvaluation",
    "StageProvider",
    "UnsatisfiedCondition",
    "unavailable_provider",
]

# Req 15.11: textul afișat la prezentarea activării Live (UI viitor).
LIVE_RISK_WARNING: Final = (
    "AVERTIZARE: tranzacționarea Live poate produce pierderi, inclusiv pierderea capitalului "
    "alocat. Profitul nu este garantat, iar rezultatele din Backtest, Shadow sau Demo nu "
    "garantează rezultate Live."
)

# Etapele în care ieșirea din Initial_Stage este considerată aprobată. Goală în această versiune.
POST_INITIAL_STAGES: Final[frozenset[ProjectStage]] = frozenset()


class LiveGateCondition(StrEnum):
    STAGE_NOT_EXITED = "LG_STAGE_NOT_EXITED"  # 15.1
    BROKER_OPEN_DECISION = "LG_BROKER_OPEN_DECISION"  # 15.2, 3.6
    STRATEGY_ARTIFACT = "LG_STRATEGY_ARTIFACT"  # 15.3
    PAPER_QUALIFICATION = "LG_PAPER_QUALIFICATION"  # 15.3
    RECONCILIATION = "LG_RECONCILIATION"  # 15.4
    HEALTH = "LG_HEALTH"  # 15.4
    CRITICAL_INCIDENTS = "LG_CRITICAL_INCIDENTS"  # 15.4
    RISK_LIMITS = "LG_RISK_LIMITS"  # 15.5
    LIVE_CREDENTIALS = "LG_LIVE_CREDENTIALS"  # 15.5


CONDITION_ORDER: Final[tuple[LiveGateCondition, ...]] = tuple(LiveGateCondition)

CONDITION_REQUIREMENTS: Final[Mapping[LiveGateCondition, str]] = {
    LiveGateCondition.STAGE_NOT_EXITED: "15.1",
    LiveGateCondition.BROKER_OPEN_DECISION: "15.2, 3.6",
    LiveGateCondition.STRATEGY_ARTIFACT: "15.3",
    LiveGateCondition.PAPER_QUALIFICATION: "15.3",
    LiveGateCondition.RECONCILIATION: "15.4",
    LiveGateCondition.HEALTH: "15.4",
    LiveGateCondition.CRITICAL_INCIDENTS: "15.4",
    LiveGateCondition.RISK_LIMITS: "15.5",
    LiveGateCondition.LIVE_CREDENTIALS: "15.5",
}

# Condițiile furnizate din exterior (toate, mai puțin etapa, evaluată intern).
PROVIDED_CONDITIONS: Final[tuple[LiveGateCondition, ...]] = tuple(
    c for c in CONDITION_ORDER if c is not LiveGateCondition.STAGE_NOT_EXITED
)


@dataclass(frozen=True)
class ConditionCheck:
    """Rezultatul unui furnizor: satisfăcut numai dacă `satisfied is True`."""

    satisfied: bool
    detail: str = ""


@dataclass(frozen=True)
class UnsatisfiedCondition:
    code: LiveGateCondition
    requirement: str
    detail: str


@dataclass(frozen=True)
class LiveGateEvaluation:
    open: bool
    unsatisfied: tuple[UnsatisfiedCondition, ...]

    @property
    def codes(self) -> tuple[LiveGateCondition, ...]:
        return tuple(u.code for u in self.unsatisfied)


ConditionProvider = Callable[[], ConditionCheck]
StageProvider = Callable[[], StageInfo]

UNAVAILABLE_DETAIL: Final = "furnizor indisponibil (fail-closed)"


def unavailable_provider() -> ConditionCheck:
    """Furnizorul implicit: condiția este nesatisfăcută."""
    return ConditionCheck(satisfied=False, detail=UNAVAILABLE_DETAIL)


class LiveGate:
    """Poartă numai închisă în Initial_Stage; nu are nicio cale de deschidere."""

    def __init__(
        self,
        *,
        stage: StageProvider,
        providers: Mapping[LiveGateCondition, ConditionProvider] | None = None,
    ) -> None:
        given = dict(providers or {})
        if LiveGateCondition.STAGE_NOT_EXITED in given:
            raise ValueError("condiția de etapă este evaluată intern din stage.lock")
        self._stage = stage
        self._providers: dict[LiveGateCondition, ConditionProvider] = {
            c: given.get(c, unavailable_provider) for c in PROVIDED_CONDITIONS
        }

    @classmethod
    def from_stage_lock(
        cls,
        path: Path,
        providers: Mapping[LiveGateCondition, ConditionProvider] | None = None,
    ) -> LiveGate:
        """Etapa este recitită din `stage.lock` la fiecare evaluare."""
        return cls(stage=lambda: read_stage(path), providers=providers)

    def evaluate(self) -> LiveGateEvaluation:
        """Reevaluează de la zero toate condițiile (15.12); nimic nu este memorat."""
        unsatisfied: list[UnsatisfiedCondition] = []
        stage_detail = self._check_stage()
        if stage_detail is not None:
            unsatisfied.append(self._unsatisfied(LiveGateCondition.STAGE_NOT_EXITED, stage_detail))
        for code in PROVIDED_CONDITIONS:
            detail = self._check_provider(self._providers[code])
            if detail is not None:
                unsatisfied.append(self._unsatisfied(code, detail))
        return LiveGateEvaluation(open=not unsatisfied, unsatisfied=tuple(unsatisfied))

    def is_open(self) -> bool:
        try:
            return self.evaluate().open is True
        except Exception:
            return False

    # ------------------------------------------------------------------ intern

    def _check_stage(self) -> str | None:
        try:
            info = self._stage()
        except Exception as exc:
            return f"etapa nu a putut fi citită: {type(exc).__name__} (fail-closed)"
        if not isinstance(info, StageInfo):
            return "furnizorul de etapă a întors o valoare invalidă (fail-closed)"
        if info.stage not in POST_INITIAL_STAGES:
            return (
                f"proiectul este în etapa {info.stage} (sursă: {info.source}); ieșirea aprobată "
                "din Initial_Stage nu este disponibilă în această versiune"
            )
        return None

    @staticmethod
    def _check_provider(provider: ConditionProvider) -> str | None:
        try:
            result = provider()
        except Exception as exc:
            return f"furnizorul a eșuat: {type(exc).__name__} (fail-closed)"
        if not isinstance(result, ConditionCheck):
            return "furnizorul a întors o valoare invalidă (fail-closed)"
        if result.satisfied is not True:
            return result.detail or "condiție nesatisfăcută"
        return None

    @staticmethod
    def _unsatisfied(code: LiveGateCondition, detail: str) -> UnsatisfiedCondition:
        return UnsatisfiedCondition(code, CONDITION_REQUIREMENTS[code], detail)
