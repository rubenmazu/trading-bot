"""Promovare controlată Backtest → Shadow → Demo a unui `StrategyArtifact` (Req 16).

Promovarea respectă strict ordinea Backtest → Shadow → Demo (`Live` este în afara spec-ului și
rămâne blocat). Regulile impuse aici:

- Ordine strictă (Req 16.1): o etapă poate fi finalizată doar dacă precedenta a fost finalizată cu
  succes; nu se poate sări peste etape și nu se poate merge înapoi.
- Logica rămâne neschimbată (Req 16.2): artefactul promovat este același obiect; `Strategy`,
  parametrii, regulile `Risk_Engine` și modelul complet de costuri sunt păstrate.
- Versiune nouă la modificare (Req 16.3): dacă artefactul se schimbă (alt `artifact_id`),
  controllerul se resetează și validarea reîncepe de la Backtest.
- Înregistrarea finalizării (Req 16.4): la finalizarea unei etape se rețin criteriile, rezultatul,
  identitatea aprobatorului și timpul aprobării.
- Blocarea etapei următoare (Req 16.5): dacă `Promotion_Criteria` preînregistrate nu sunt
  îndeplinite, etapa următoare este blocată.
- Doar adaptoare/mediu/credentiale se schimbă (Req 16.6, 16.7): `detect_disallowed_differences`
  compară configurația de rulare a două etape și blochează promovarea dacă există diferențe în
  afara mulțimii permise (adaptoare, configurație de mediu, referințe de credentiale).
- Stare terminală `Respins` (Req 16.8, 22.3): când promovarea este blocată sau respinsă, starea
  curentă a etapei devine `RESPINS`, terminală — nu mai poate fi finalizată fără reset.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Final

from pydantic import model_validator

from qts.core.clock import ensure_utc
from qts.core.models import Frozen, UtcDatetime
from qts.promotion.artifact import StrategyArtifact

__all__ = [
    "ALLOWED_CHANGE_KEYS",
    "PromotionBlockedError",
    "PromotionController",
    "PromotionError",
    "PromotionOrderError",
    "PromotionRecord",
    "PromotionStage",
    "PromotionState",
    "StageStatus",
    "allowed_environment_keys",
    "detect_disallowed_differences",
]


# La promovare se pot schimba numai aceste chei de configurație de rulare (Req 16.6).
ALLOWED_CHANGE_KEYS: Final[frozenset[str]] = frozenset(
    {"adapters", "environment", "credentials"}
)


class PromotionStage(StrEnum):
    """Etapele de promovare, în ordinea permisă (Req 16.1). `Live` este în afara spec-ului."""

    BACKTEST = "backtest"
    SHADOW = "shadow"
    DEMO = "demo"


# Ordinea strictă a etapelor; promovarea avansează doar de la stânga la dreapta.
_STAGE_ORDER: Final[tuple[PromotionStage, ...]] = (
    PromotionStage.BACKTEST,
    PromotionStage.SHADOW,
    PromotionStage.DEMO,
)


class StageStatus(StrEnum):
    """Starea unei etape în fluxul de promovare."""

    PENDING = "pending"
    COMPLETED = "completed"
    RESPINS = "respins"  # stare terminală (Req 16.8, 22.3)


# --------------------------------------------------------------------------- erori


class PromotionError(Exception):
    """Eroare generică de promovare."""


class PromotionOrderError(PromotionError):
    """Promovare în afara ordinii Backtest → Shadow → Demo (Req 16.1)."""


class PromotionBlockedError(PromotionError):
    """Promovare blocată: criterii neîndeplinite sau diferențe interzise (Req 16.5, 16.7)."""


# --------------------------------------------------------------------------- modele


class PromotionRecord(Frozen):
    """Dovada finalizării sau respingerii unei etape (Req 16.4, 16.8).

    Rețineți: criteriile evaluate, rezultatul (acceptat/respins), aprobatorul și timpul aprobării.
    Pentru o etapă respinsă, `criteria_met` este `False` și, opțional, `reason` explică blocajul.
    """

    stage: PromotionStage
    status: StageStatus
    artifact_id: str
    criteria_met: bool
    approver: str
    approved_at: UtcDatetime
    reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> PromotionRecord:
        if not self.approver.strip():
            raise ValueError("identitatea aprobatorului este obligatorie (Req 16.4)")
        if self.status is StageStatus.PENDING:
            raise ValueError("o înregistrare se creează doar la finalizare sau respingere")
        if self.status is StageStatus.COMPLETED and not self.criteria_met:
            raise ValueError("o etapă finalizată trebuie să aibă criteriile îndeplinite")
        if self.status is StageStatus.RESPINS and self.criteria_met:
            raise ValueError("o etapă respinsă nu poate avea criteriile îndeplinite")
        return self


class PromotionState(Frozen):
    """Starea imuabilă a promovării: artefactul și istoricul etapelor finalizate/respinse."""

    artifact_id: str
    records: tuple[PromotionRecord, ...] = ()

    def completed_stages(self) -> tuple[PromotionStage, ...]:
        return tuple(
            r.stage for r in self.records if r.status is StageStatus.COMPLETED
        )

    def is_rejected(self) -> bool:
        """Adevărat dacă ultima etapă atinsă este în starea terminală `Respins`."""
        return bool(self.records) and self.records[-1].status is StageStatus.RESPINS


# --------------------------------------------------------------------------- detector diferențe


def allowed_environment_keys() -> frozenset[str]:
    """Cheile de configurație care au voie să difere între etape la promovare (Req 16.6)."""
    return ALLOWED_CHANGE_KEYS


def detect_disallowed_differences(
    source_config: Mapping[str, Any],
    target_config: Mapping[str, Any],
    *,
    allowed_keys: frozenset[str] = ALLOWED_CHANGE_KEYS,
) -> tuple[str, ...]:
    """Întoarce cheile care diferă între două configurații de rulare în afara mulțimii permise.

    La promovare, numai `adapters`, `environment` și `credentials` au voie să se schimbe
    (Req 16.6). Orice altă cheie care apare doar într-una dintre configurații sau are valori
    diferite este o diferență interzisă (Req 16.7) și este raportată aici, sortată determinist.
    O listă nevidă înseamnă că promovarea trebuie blocată până la eliminarea diferențelor.
    """
    disallowed: set[str] = set()
    for key in set(source_config) | set(target_config):
        if key in allowed_keys:
            continue
        if source_config.get(key) != target_config.get(key):
            disallowed.add(key)
    return tuple(sorted(disallowed))


# --------------------------------------------------------------------------- controller


class PromotionController:
    """Mașina de promovare strict ordonată pentru un singur artefact (Req 16).

    Controllerul este legat de un `StrategyArtifact`. Dacă i se prezintă un artefact cu alt
    `artifact_id` (modificare după aprobare), se resetează și validarea reîncepe de la Backtest
    (Req 16.3). Stările se schimbă doar prin `complete_stage` (succes) sau `reject_stage`
    (blocare/respingere); `reject_stage` duce etapa în starea terminală `Respins` (Req 16.8).
    """

    def __init__(self, artifact: StrategyArtifact) -> None:
        if not artifact.verify():
            raise PromotionError(
                "artefactul nu se verifică: artifact_id nu corespunde conținutului"
            )
        self._artifact = artifact
        self._records: list[PromotionRecord] = []

    @property
    def artifact(self) -> StrategyArtifact:
        return self._artifact

    def state(self) -> PromotionState:
        return PromotionState(artifact_id=self._artifact.artifact_id, records=tuple(self._records))

    # ----------------------------------------------------------------- modificarea artefactului

    def set_artifact(self, artifact: StrategyArtifact) -> bool:
        """Înlocuiește artefactul; dacă `artifact_id` diferă, resetează validarea (Req 16.3).

        Întoarce `True` dacă a avut loc un reset (artefact nou), `False` dacă artefactul este
        identic (niciun efect). Un artefact nou șterge tot istoricul de etape — validarea
        reîncepe de la Backtest.
        """
        if not artifact.verify():
            raise PromotionError(
                "artefactul nu se verifică: artifact_id nu corespunde conținutului"
            )
        if artifact.artifact_id == self._artifact.artifact_id:
            return False
        self._artifact = artifact
        self._records = []
        return True

    # ----------------------------------------------------------------- interogări

    def next_stage(self) -> PromotionStage | None:
        """Următoarea etapă finalizabilă, sau `None` dacă fluxul e încheiat/respins."""
        if self.state().is_rejected():
            return None
        completed = self.state().completed_stages()
        for stage in _STAGE_ORDER:
            if stage not in completed:
                return stage
        return None

    def can_promote_to(self, stage: PromotionStage) -> bool:
        """Adevărat dacă `stage` este exact următoarea etapă validă (ordine strictă, Req 16.1)."""
        return self.next_stage() is stage

    # ----------------------------------------------------------------- tranziții

    def complete_stage(
        self,
        stage: PromotionStage,
        *,
        criteria_met: bool,
        approver: str,
        approved_at: datetime,
        source_config: Mapping[str, Any] | None = None,
        target_config: Mapping[str, Any] | None = None,
    ) -> PromotionRecord:
        """Finalizează o etapă, cu validarea ordinii, criteriilor și diferențelor (Req 16.1–16.7).

        Pași, în ordine:
        1. Ordine strictă: `stage` trebuie să fie exact următoarea etapă (Req 16.1). Dacă fluxul
           este deja respins, promovarea e refuzată.
        2. Diferențe de configurație: dacă se dau `source_config` și `target_config`, se verifică
           să nu existe schimbări în afara adaptoarelor/mediului/credentialelor (Req 16.6, 16.7).
           La o diferență interzisă etapa este respinsă (`Respins`) și se ridică
           `PromotionBlockedError`.
        3. Criterii: dacă `criteria_met` este `False`, etapa este respinsă (`Respins`), blocând
           etapa următoare (Req 16.5, 16.8).
        4. Succes: se înregistrează finalizarea cu criterii, rezultat, aprobator și timp (Req 16.4).
        """
        if self.state().is_rejected():
            raise PromotionOrderError(
                "promovarea este în starea terminală Respins; reluați validarea cu un artefact nou"
            )
        expected = self.next_stage()
        if stage is not expected:
            raise PromotionOrderError(
                f"promovare în afara ordinii: etapa așteptată este {expected}, nu {stage}"
            )

        # Req 16.6, 16.7: la promovare doar adaptoarele, mediul și credentialele pot diferi.
        if source_config is not None and target_config is not None:
            disallowed = detect_disallowed_differences(source_config, target_config)
            if disallowed:
                reason = (
                    "promovare blocată: diferențe în afara adaptoarelor/mediului/credentialelor: "
                    + ", ".join(disallowed)
                )
                self._reject(stage, approver=approver, approved_at=approved_at, reason=reason)
                raise PromotionBlockedError(reason)

        # Req 16.5: criteriile de promovare preînregistrate trebuie îndeplinite.
        if not criteria_met:
            reason = "promovare blocată: Promotion_Criteria preînregistrate neîndeplinite"
            self._reject(stage, approver=approver, approved_at=approved_at, reason=reason)
            raise PromotionBlockedError(reason)

        record = PromotionRecord(
            stage=stage,
            status=StageStatus.COMPLETED,
            artifact_id=self._artifact.artifact_id,
            criteria_met=True,
            approver=approver,
            approved_at=ensure_utc(approved_at),
        )
        self._records.append(record)
        return record

    def reject_stage(
        self,
        stage: PromotionStage,
        *,
        approver: str,
        approved_at: datetime,
        reason: str,
    ) -> PromotionRecord:
        """Respinge explicit o etapă, aducând-o în starea terminală `Respins` (Req 16.8, 22.3)."""
        if self.state().is_rejected():
            raise PromotionOrderError("promovarea este deja în starea terminală Respins")
        expected = self.next_stage()
        if stage is not expected:
            raise PromotionOrderError(
                f"respingere în afara ordinii: etapa așteptată este {expected}, nu {stage}"
            )
        return self._reject(stage, approver=approver, approved_at=approved_at, reason=reason)

    def _reject(
        self,
        stage: PromotionStage,
        *,
        approver: str,
        approved_at: datetime,
        reason: str,
    ) -> PromotionRecord:
        record = PromotionRecord(
            stage=stage,
            status=StageStatus.RESPINS,
            artifact_id=self._artifact.artifact_id,
            criteria_met=False,
            approver=approver,
            approved_at=ensure_utc(approved_at),
            reason=reason,
        )
        self._records.append(record)
        return record
