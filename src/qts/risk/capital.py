"""Reguli pentru modificarea capitalului configurat (Req 28.1–28.7).

Capitalul suplimentar depinde de dovezi aprobate: nicio alimentare automată a contului și nicio
creștere automată a limitelor (28.6). O modificare de capital este acceptată numai dacă
`Capital_Change_Authorization` este adevărat — predicat unic adevărat numai când aprobarea
explicită este validă ȘI noul `Configuration_Snapshot` este creat atomic (vezi glosarul; 28.4).

Reguli impuse aici, în ordine (fail-closed: orice regulă neîndeplinită respinge modificarea):

1. Capitalul de referință rămâne 100 EUR și pierderea totală maximă rămâne 10 EUR (28.1, 28.2);
   dacă limita totală propusă depășește 10 EUR, configurația este respinsă (28.7).
2. Capitalul suplimentar (orice capital peste cel curent) cere `Paper_Qualification` validă și
   `Promotion_Criteria` aprobate (28.3); în plus, creșterile nu pot fi automate (28.6): fiecare
   propunere trebuie să poarte o aprobare explicită.
3. Modificarea trebuie să includă limite `Risk_Engine` reevaluate (28.5): fără o nouă
   configurație de risc atașată, modificarea este respinsă.
4. `Capital_Change_Authorization` este adevărat numai când aprobarea explicită este validă ȘI un
   `Configuration_Snapshot` nou și verificabil însoțește modificarea, creat atomic (28.4).

Modulul nu modifică nimic de la sine; evaluează o `CapitalChangeProposal` și întoarce o
`CapitalChangeDecision`. Aplicarea efectivă (dezactivarea `Kill_Switch(CAPITAL_CONFIG)`, 13.6/13.11)
este responsabilitatea apelantului, numai când `decision.authorized` este adevărat.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Final

from pydantic import model_validator

from qts.config.schema import (
    MAX_TOTAL_LOSS_EUR,
    REFERENCE_CAPITAL_EUR,
    RiskConfig,
)
from qts.config.snapshot import ConfigurationSnapshot
from qts.core.models import Dec, Frozen

__all__ = [
    "CAPITAL_RULES_VERSION",
    "MAX_TOTAL_LOSS_EUR",
    "REFERENCE_CAPITAL_EUR",
    "CapitalChangeApproval",
    "CapitalChangeDecision",
    "CapitalChangeProposal",
    "CapitalRejectReason",
    "evaluate_capital_change",
    "is_capital_change_authorized",
]

CAPITAL_RULES_VERSION: Final = "1"


class CapitalRejectReason(StrEnum):
    """Motive stabile de respingere a unei modificări de capital (audit și teste depind de ele)."""

    TOTAL_LOSS_EXCEEDS_CAP = "CAPITAL_TOTAL_LOSS_EXCEEDS_CAP"  # 28.2, 28.7
    REFERENCE_CAPITAL_CHANGED = "CAPITAL_REFERENCE_CHANGED"  # 28.1
    AUTOMATIC_INCREASE = "CAPITAL_AUTOMATIC_INCREASE"  # 28.6
    EVIDENCE_MISSING = "CAPITAL_EVIDENCE_MISSING"  # 28.3
    RISK_LIMITS_NOT_REEVALUATED = "CAPITAL_RISK_LIMITS_NOT_REEVALUATED"  # 28.5
    APPROVAL_INVALID = "CAPITAL_APPROVAL_INVALID"  # 28.4
    SNAPSHOT_NOT_ATOMIC = "CAPITAL_SNAPSHOT_NOT_ATOMIC"  # 28.4


class CapitalChangeApproval(Frozen):
    """Aprobarea explicită a proprietarului capitalului pentru o modificare (28.3, 28.4, 28.6).

    O creștere nu poate fi automată: `operator_initiated` trebuie să fie adevărat și
    `approval_ref` nevid. Pentru capital suplimentar, `paper_qualification_valid` și
    `promotion_criteria_approved` dovedesc rezultatele convingătoare (28.3).
    """

    approval_ref: str
    approver: str
    operator_initiated: bool
    paper_qualification_valid: bool = False
    promotion_criteria_approved: bool = False

    @model_validator(mode="after")
    def _check(self) -> CapitalChangeApproval:
        if not self.approval_ref.strip():
            raise ValueError("o aprobare necesită approval_ref nevid (28.4)")
        if not self.approver.strip():
            raise ValueError("o aprobare necesită identitatea aprobatorului (28.4)")
        return self

    @property
    def evidence_complete(self) -> bool:
        """Dovezile pentru capital suplimentar: Paper_Qualification + Promotion_Criteria (28.3)."""
        return self.paper_qualification_valid and self.promotion_criteria_approved

    @property
    def is_explicit(self) -> bool:
        """Aprobare explicită, inițiată de operator — niciodată automată (28.6)."""
        return self.operator_initiated


class CapitalChangeProposal(Frozen):
    """O propunere de modificare a capitalului configurat (28.3–28.7).

    `current_capital_eur` este capitalul configurat în prezent; `proposed_capital_eur` cel nou.
    `reevaluated_risk` este configurația `Risk_Engine` reevaluată care trebuie să însoțească
    modificarea (28.5). `new_snapshot` este `Configuration_Snapshot` nou creat atomic pentru noua
    configurație (28.4); absența lui înseamnă că modificarea nu este atomică.
    """

    current_capital_eur: Dec
    proposed_capital_eur: Dec
    approval: CapitalChangeApproval
    reevaluated_risk: RiskConfig | None = None
    new_snapshot: ConfigurationSnapshot | None = None

    @property
    def is_increase(self) -> bool:
        """Adevărat când capitalul propus îl depășește pe cel curent (capital suplimentar, 28.3)."""
        return self.proposed_capital_eur > self.current_capital_eur


class CapitalChangeDecision(Frozen):
    """Rezultatul evaluării: autorizat sau respins, cu toate motivele (fail-closed)."""

    authorized: bool
    reasons: tuple[CapitalRejectReason, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> CapitalChangeDecision:
        if self.authorized and self.reasons:
            raise ValueError("o modificare autorizată nu poate avea motive de respingere")
        if not self.authorized and not self.reasons:
            raise ValueError("o modificare respinsă trebuie să aibă cel puțin un motiv")
        return self


def _total_loss_limit(risk: RiskConfig | None) -> Decimal | None:
    return risk.total_loss_limit_eur if risk is not None else None


def _snapshot_is_atomic(
    snapshot: ConfigurationSnapshot | None, risk: RiskConfig | None
) -> bool:
    """Adevărat când snapshot-ul există, se verifică și reflectă configurația de risc (28.4)."""
    if snapshot is None or not snapshot.verify():
        return False
    return risk is None or snapshot.config.get("risk") == risk.model_dump(mode="json")


def evaluate_capital_change(proposal: CapitalChangeProposal) -> CapitalChangeDecision:
    """Evaluează o modificare de capital față de Req 28; întoarce toate motivele de respingere.

    Autorizează modificarea numai când `Capital_Change_Authorization` este adevărat: aprobarea
    explicită este validă ȘI un `Configuration_Snapshot` nou și verificabil însoțește modificarea
    (28.4). Toate celelalte reguli (28.1–28.3, 28.5–28.7) trebuie, de asemenea, îndeplinite.
    """
    reasons: list[CapitalRejectReason] = []
    risk = proposal.reevaluated_risk

    # 28.5: modificarea trebuie să includă limite Risk_Engine reevaluate.
    if risk is None:
        reasons.append(CapitalRejectReason.RISK_LIMITS_NOT_REEVALUATED)

    # 28.1: capitalul de referință rămâne 100 EUR în domeniul acestui spec.
    if risk is not None and risk.reference_capital_eur != REFERENCE_CAPITAL_EUR:
        reasons.append(CapitalRejectReason.REFERENCE_CAPITAL_CHANGED)

    # 28.2, 28.7: limita totală propusă nu poate depăși 10 EUR.
    limit = _total_loss_limit(risk)
    if limit is not None and limit > MAX_TOTAL_LOSS_EUR:
        reasons.append(CapitalRejectReason.TOTAL_LOSS_EXCEEDS_CAP)

    # 28.6: nicio creștere automată — fiecare propunere poartă o aprobare explicită de operator.
    if not proposal.approval.is_explicit:
        reasons.append(CapitalRejectReason.AUTOMATIC_INCREASE)

    # 28.3: capitalul suplimentar cere Paper_Qualification validă și Promotion_Criteria aprobate.
    if proposal.is_increase and not proposal.approval.evidence_complete:
        reasons.append(CapitalRejectReason.EVIDENCE_MISSING)

    # 28.4: Configuration_Snapshot nou, creat atomic și verificabil; conținutul lui trebuie să
    # corespundă chiar configurației de risc reevaluate (altfel modificarea nu este atomică).
    if not _snapshot_is_atomic(proposal.new_snapshot, risk):
        reasons.append(CapitalRejectReason.SNAPSHOT_NOT_ATOMIC)

    if reasons:
        # Ordine determinist stabilă pentru audit și teste.
        ordered = tuple(r for r in CapitalRejectReason if r in set(reasons))
        return CapitalChangeDecision(authorized=False, reasons=ordered)
    return CapitalChangeDecision(authorized=True)


def is_capital_change_authorized(proposal: CapitalChangeProposal) -> bool:
    """`Capital_Change_Authorization`: adevărat numai când modificarea este pe deplin autorizată.

    Predicat unic adevărat când aprobarea explicită este validă ȘI noul `Configuration_Snapshot`
    este creat atomic, cu toate celelalte reguli din Req 28 îndeplinite (fail-closed).
    """
    return evaluate_capital_change(proposal).authorized
