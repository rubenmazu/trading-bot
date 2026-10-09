"""Evaluarea `Paper_Qualification` față de `Promotion_Criteria` preînregistrate (Req 22.4–22.6).

`Paper_Qualification` este calificarea *cumulată* în Shadow și Demo pe care o strategie trebuie să
o obțină înainte de eligibilitatea Live (glosar, Req 22). Acest modul nu rulează el însuși
etapele Shadow/Demo; primește observațiile deja măsurate pentru fiecare etapă (`StageObservations`)
și le evaluează determinist față de criteriile *preînregistrate* (`PromotionCriteria` din
`research.preregistration`, plus regimurile fixate în `PreRegistration`), în același stil ca
`research.report`: produce un verdict pass/fail cu lista criteriilor evaluate și motivele.

Criteriile de acceptare acoperite:

- **Req 22.4** — compară, față de `Promotion_Criteria`, pentru rezultatul agregat Shadow+Demo:
  rezultatul net, execuțiile (numărul de ordine), respingerile, deconectările și reconcilierile.
- **Req 22.5** — fail-closed: dacă rezultatul net cumulat în Shadow **sau** în Demo este ≤ 0 după
  `Complete_Cost_Model`, calificarea este respinsă (nu doar agregatul, ci fiecare etapă în parte).
- **Req 22.6** — fail-closed: dacă există un `Critical_Incident` nerezolvat în oricare etapă,
  calificarea este blocată.

Determinism: fără timestamp, fără ordine aleatorie. Toate valorile monetare sunt `Decimal`
(moneda de raportare EUR), modelele sunt `frozen` și `extra="forbid"`. Același set de observații,
evaluat de două ori, produce exact același verdict și aceeași listă de motive, în aceeași ordine.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Final

from pydantic import Field, model_validator

from qts.core.models import Dec, Frozen
from qts.core.money import REPORTING_CURRENCY, ZERO
from qts.research.preregistration import PreRegistration, PromotionCriteria

__all__ = [
    "PAPER_QUALIFICATION_VERSION",
    "PaperQualificationCriteria",
    "PaperQualificationResult",
    "PaperStage",
    "StageObservations",
    "evaluate_paper_qualification",
]

PAPER_QUALIFICATION_VERSION: Final = "paper-qualification-v1"


class PaperStage(StrEnum):
    """Etapele de date curente care compun `Paper_Qualification` (Req 22)."""

    SHADOW = "shadow"
    DEMO = "demo"


class StageObservations(Frozen):
    """Rezultatele observate într-o etapă (Shadow sau Demo), după `Complete_Cost_Model`.

    Toate câmpurile sunt măsurători, nu praguri. `net_result_eur` este rezultatul net cumulat în
    etapă după modelul complet de costuri (Req 22.5). `regimes_covered` enumeră regimurile de piață
    observate efectiv în etapă (din cele preînregistrate). `unresolved_critical_incidents` numără
    incidentele critice nerezolvate (Req 22.6).
    """

    stage: PaperStage
    net_result_eur: Dec
    orders: Annotated[int, Field(ge=0)]
    rejections: Annotated[int, Field(ge=0)] = 0
    disconnections: Annotated[int, Field(ge=0)] = 0
    failed_reconciliations: Annotated[int, Field(ge=0)] = 0
    regimes_covered: tuple[str, ...] = ()
    unresolved_critical_incidents: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def _check(self) -> StageObservations:
        if len(self.regimes_covered) != len(set(self.regimes_covered)):
            raise ValueError("regimes_covered conține regimuri duplicate")
        return self


class PaperQualificationCriteria(Frozen):
    """Pragurile operaționale `Paper_Qualification`, derivate din pre-înregistrare (Req 22.4).

    `Promotion_Criteria` preînregistrate fixează pragul de rezultat net (`min_net_result_eur`),
    numărul minim de ordine (`min_trades`) și acoperirea minimă de regimuri (`min_regimes`).
    Pragurile operaționale suplimentare — respingeri, deconectări și reconcilieri eșuate maxime —
    sunt tolerante implicit (fără limită) dacă nu sunt fixate, dar pot fi preînregistrate explicit.
    Toate pragurile sunt fixate *înainte* de evaluare, nu alese ulterior.
    """

    min_net_result_eur: Dec
    min_orders: Annotated[int, Field(ge=1)] = 1
    min_regimes: Annotated[int, Field(ge=1)] = 1
    required_regimes: tuple[str, ...] = ()
    max_rejections: Annotated[int, Field(ge=0)] | None = None
    max_disconnections: Annotated[int, Field(ge=0)] | None = None
    max_failed_reconciliations: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def _check(self) -> PaperQualificationCriteria:
        if len(self.required_regimes) != len(set(self.required_regimes)):
            raise ValueError("required_regimes conține regimuri duplicate")
        if self.required_regimes and self.min_regimes > len(self.required_regimes):
            raise ValueError("min_regimes nu poate depăși numărul de regimuri preînregistrate")
        return self

    @classmethod
    def from_preregistration(
        cls,
        pre: PreRegistration,
        *,
        max_rejections: int | None = None,
        max_disconnections: int | None = None,
        max_failed_reconciliations: int | None = None,
    ) -> PaperQualificationCriteria:
        """Construiește criteriile din `PromotionCriteria` și regimurile preînregistrate (Req 22.1).

        Reutilizează `PromotionCriteria` (pragul de rezultat net, numărul minim de ordine,
        acoperirea minimă de regimuri) și preia regimurile fixate în `PreRegistration` ca mulțime
        de regimuri cerute. Pragurile operaționale (respingeri, deconectări, reconcilieri) sunt
        opționale și pot fi date explicit; dacă lipsesc, nu impun o limită superioară.
        """
        criteria: PromotionCriteria = pre.promotion_criteria
        return cls(
            min_net_result_eur=criteria.min_net_result_eur,
            min_orders=criteria.min_trades,
            min_regimes=criteria.min_regimes,
            required_regimes=tuple(sorted(r.name for r in pre.regimes)),
            max_rejections=max_rejections,
            max_disconnections=max_disconnections,
            max_failed_reconciliations=max_failed_reconciliations,
        )


class PaperQualificationResult(Frozen):
    """Verdictul `Paper_Qualification`: pass/fail, criteriile evaluate și motivele (Req 22.4–22.6).

    `passed` este adevărat doar dacă toate criteriile sunt îndeplinite. `reasons` enumeră, în ordine
    deterministă, motivele de respingere (gol când `passed`). `evaluated` enumeră, în ordine, toate
    criteriile verificate, fiecare cu rezultatul său, astfel încât verdictul să fie auditabil.
    `aggregate_net_result_eur` este suma rezultatelor nete Shadow+Demo (Req 22.4).
    """

    version: str = PAPER_QUALIFICATION_VERSION
    reporting_currency: str = REPORTING_CURRENCY
    passed: bool
    aggregate_net_result_eur: Dec
    aggregate_orders: int
    reasons: tuple[str, ...]
    evaluated: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> PaperQualificationResult:
        # O singură monedă de raportare per rulare (nu neapărat EUR global); implicitul rămâne EUR.
        if not self.reporting_currency.strip():
            raise ValueError("moneda de raportare este obligatorie")
        if self.passed and self.reasons:
            raise ValueError("un verdict trecut nu poate avea motive de respingere")
        if not self.passed and not self.reasons:
            raise ValueError("un verdict respins necesită cel puțin un motiv")
        if not self.evaluated:
            raise ValueError("verdictul trebuie să enumere criteriile evaluate")
        return self


def _stage_label(stage: PaperStage) -> str:
    return "Shadow" if stage is PaperStage.SHADOW else "Demo"


def evaluate_paper_qualification(
    criteria: PaperQualificationCriteria,
    observations: tuple[StageObservations, ...],
) -> PaperQualificationResult:
    """Evaluează `Paper_Qualification` cumulat (Shadow+Demo) față de criterii (Req 22.4–22.6).

    Agregă observațiile celor două etape și verifică, determinist:

    - fiecare etapă Shadow și Demo este prezentă exact o dată (calificarea este cumulată peste
      ambele etape; glosar, Req 22);
    - **Req 22.6** — niciun `Critical_Incident` nerezolvat în vreo etapă (fail-closed);
    - **Req 22.5** — rezultatul net cumulat în fiecare etapă (Shadow și Demo) este > 0 după
      `Complete_Cost_Model` (fail-closed, pe etapă, nu doar pe agregat);
    - **Req 22.4** — agregatul Shadow+Demo respectă `Promotion_Criteria`: rezultat net ≥ prag,
      numărul de ordine ≥ prag, acoperirea de regimuri ≥ prag și, dacă sunt fixate, respingerile,
      deconectările și reconcilierile eșuate nu depășesc limitele.

    Verdictul este `passed` doar dacă *toate* criteriile sunt îndeplinite; altfel enumeră motivele
    în ordine deterministă. `evaluated` enumeră fiecare criteriu verificat cu rezultatul său.
    """
    stages = tuple(o.stage for o in observations)
    missing = [s for s in (PaperStage.SHADOW, PaperStage.DEMO) if s not in stages]
    if missing or len(stages) != len(set(stages)):
        names = ", ".join(_stage_label(s) for s in (PaperStage.SHADOW, PaperStage.DEMO))
        raise ValueError(
            f"Paper_Qualification necesită exact o observație pentru fiecare etapă ({names})"
        )

    by_stage = {o.stage: o for o in observations}
    ordered = (by_stage[PaperStage.SHADOW], by_stage[PaperStage.DEMO])

    aggregate_net: Decimal = sum((o.net_result_eur for o in ordered), ZERO)
    aggregate_orders = sum(o.orders for o in ordered)
    aggregate_rejections = sum(o.rejections for o in ordered)
    aggregate_disconnections = sum(o.disconnections for o in ordered)
    aggregate_failed_recon = sum(o.failed_reconciliations for o in ordered)
    regimes_covered: set[str] = set()
    for o in ordered:
        regimes_covered.update(o.regimes_covered)

    reasons: list[str] = []
    evaluated: list[str] = []

    # Req 22.6: incidente critice nerezolvate blochează promovarea (fail-closed, verificat întâi).
    unresolved = sum(o.unresolved_critical_incidents for o in ordered)
    evaluated.append(f"Req 22.6 incidente critice nerezolvate = {unresolved} (prag 0)")
    if unresolved > 0:
        for o in ordered:
            if o.unresolved_critical_incidents > 0:
                reasons.append(
                    f"Req 22.6: {o.unresolved_critical_incidents} Critical_Incident nerezolvate "
                    f"în {_stage_label(o.stage)} blochează promovarea Live"
                )

    # Req 22.5: rezultatul net cumulat în fiecare etapă trebuie să fie > 0 (fail-closed, pe etapă).
    for o in ordered:
        evaluated.append(
            f"Req 22.5 rezultat net {_stage_label(o.stage)} = {o.net_result_eur} EUR (> 0)"
        )
        if o.net_result_eur <= ZERO:
            reasons.append(
                f"Req 22.5: rezultatul net cumulat în {_stage_label(o.stage)} "
                f"({o.net_result_eur} EUR) este ≤ 0 după Complete_Cost_Model"
            )

    # Req 22.4: agregatul Shadow+Demo față de Promotion_Criteria.
    evaluated.append(
        f"Req 22.4 rezultat net agregat = {aggregate_net} EUR "
        f"(prag ≥ {criteria.min_net_result_eur})"
    )
    if aggregate_net < criteria.min_net_result_eur:
        reasons.append(
            f"Req 22.4: rezultatul net agregat ({aggregate_net} EUR) este sub pragul "
            f"Promotion_Criteria ({criteria.min_net_result_eur} EUR)"
        )

    evaluated.append(
        f"Req 22.4 ordine agregate = {aggregate_orders} (prag ≥ {criteria.min_orders})"
    )
    if aggregate_orders < criteria.min_orders:
        reasons.append(
            f"Req 22.4: numărul de ordine ({aggregate_orders}) este sub pragul "
            f"Promotion_Criteria ({criteria.min_orders})"
        )

    covered_required = (
        regimes_covered & set(criteria.required_regimes)
        if criteria.required_regimes
        else regimes_covered
    )
    evaluated.append(
        f"Req 22.4 regimuri acoperite = {len(covered_required)} (prag ≥ {criteria.min_regimes})"
    )
    if len(covered_required) < criteria.min_regimes:
        missing_regimes = (
            sorted(set(criteria.required_regimes) - regimes_covered)
            if criteria.required_regimes
            else []
        )
        detail = f"; regimuri lipsă: {missing_regimes}" if missing_regimes else ""
        reasons.append(
            f"Req 22.4: acoperirea de regimuri ({len(covered_required)}) este sub pragul "
            f"Promotion_Criteria ({criteria.min_regimes}){detail}"
        )

    _check_optional_max(
        evaluated,
        reasons,
        label="respingeri",
        value=aggregate_rejections,
        limit=criteria.max_rejections,
    )
    _check_optional_max(
        evaluated,
        reasons,
        label="deconectări",
        value=aggregate_disconnections,
        limit=criteria.max_disconnections,
    )
    _check_optional_max(
        evaluated,
        reasons,
        label="reconcilieri eșuate",
        value=aggregate_failed_recon,
        limit=criteria.max_failed_reconciliations,
    )

    return PaperQualificationResult(
        passed=not reasons,
        aggregate_net_result_eur=aggregate_net,
        aggregate_orders=aggregate_orders,
        reasons=tuple(reasons),
        evaluated=tuple(evaluated),
    )


def _check_optional_max(
    evaluated: list[str],
    reasons: list[str],
    *,
    label: str,
    value: int,
    limit: int | None,
) -> None:
    """Verifică un prag operațional opțional (Req 22.4); fără limită fixată înseamnă tolerant."""
    if limit is None:
        evaluated.append(f"Req 22.4 {label} = {value} (fără limită preînregistrată)")
        return
    evaluated.append(f"Req 22.4 {label} = {value} (prag ≤ {limit})")
    if value > limit:
        reasons.append(
            f"Req 22.4: {label} ({value}) depășesc limita Promotion_Criteria ({limit})"
        )
