"""Reguli pentru modificarea capitalului configurat (Req 28.1–28.7).

Acoperă: `Capital_Change_Authorization` adevărat numai cu aprobare explicită validă + snapshot nou
atomic (28.4); respingerea creșterilor automate (28.6); dovezile necesare pentru capital
suplimentar (28.3); limitele Risk_Engine reevaluate obligatorii (28.5); limita totală ≤ 10 EUR
(28.2, 28.7) și capitalul de referință 100 EUR (28.1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from qts.config.loader import parse_config
from qts.config.schema import AppConfig, RiskConfig
from qts.config.snapshot import CodeVersion, ConfigurationSnapshot, create_snapshot
from qts.risk.capital import (
    CapitalChangeApproval,
    CapitalChangeProposal,
    CapitalRejectReason,
    evaluate_capital_change,
    is_capital_change_authorized,
)
from qts.safety.stage import ProjectStage, StageInfo
from tests.helpers import config_dict

STAGE = StageInfo(ProjectStage.INITIAL, "stage.lock")
CODE = CodeVersion("0" * 40, False, "f" * 64)
T0 = datetime(2026, 1, 1, tzinfo=UTC)
REFERENCE = Decimal("100")


def _cfg(**overrides: object) -> AppConfig:
    return parse_config(config_dict(**overrides))


def _snapshot(config: AppConfig) -> ConfigurationSnapshot:
    return create_snapshot(config, STAGE, CODE, T0, ["ds-a"], "art-1")


def _approval(**overrides: object) -> CapitalChangeApproval:
    base: dict[str, object] = {
        "approval_ref": "cap/2026/approval-1",
        "approver": "owner@capital",
        "operator_initiated": True,
        "paper_qualification_valid": True,
        "promotion_criteria_approved": True,
    }
    base.update(overrides)
    return CapitalChangeApproval(**base)


def _proposal(**overrides: object) -> CapitalChangeProposal:
    config = _cfg()
    base: dict[str, object] = {
        "current_capital_eur": REFERENCE,
        "proposed_capital_eur": REFERENCE,
        "approval": _approval(),
        "reevaluated_risk": config.risk,
        "new_snapshot": _snapshot(config),
    }
    base.update(overrides)
    return CapitalChangeProposal(**base)


def _unchecked_proposal(*, reevaluated_risk: RiskConfig) -> CapitalChangeProposal:
    """Construiește o propunere ocolind validarea nested a `RiskConfig`.

    `RiskConfig` impune deja 28.1/28.2/28.7 la încărcare, deci o valoare invalidă nu poate ajunge
    prin construcție normală. Aceste teste verifică apărarea redundantă din regulile de capital,
    deci trebuie să injecteze o configurație de risc în afara limitelor.
    """
    config = _cfg()
    return CapitalChangeProposal.model_construct(
        current_capital_eur=REFERENCE,
        proposed_capital_eur=REFERENCE,
        approval=_approval(),
        reevaluated_risk=reevaluated_risk,
        new_snapshot=_snapshot(config),
    )


# --------------------------------------------------------------------------- autorizare validă


def test_authorized_only_with_explicit_approval_and_atomic_snapshot() -> None:
    """28.4: Capital_Change_Authorization adevărat cu aprobare validă + snapshot nou atomic."""
    proposal = _proposal()
    decision = evaluate_capital_change(proposal)
    assert decision.authorized
    assert decision.reasons == ()
    assert is_capital_change_authorized(proposal)


# --------------------------------------------------------------------------- snapshot atomic (28.4)


def test_rejects_when_no_new_snapshot() -> None:
    decision = evaluate_capital_change(_proposal(new_snapshot=None))
    assert not decision.authorized
    assert CapitalRejectReason.SNAPSHOT_NOT_ATOMIC in decision.reasons
    assert not is_capital_change_authorized(_proposal(new_snapshot=None))


def test_rejects_when_snapshot_does_not_match_reevaluated_risk() -> None:
    """Snapshot-ul atomic trebuie să corespundă chiar configurației de risc reevaluate (28.4)."""
    other = _cfg(run={"seed": 7})  # alt conținut => alt snapshot_id, risc identic
    mismatched = create_snapshot(other, STAGE, CODE, T0, ["ds-a"], "art-1")
    # Snapshot valabil în sine, dar config.risk e identic; forțăm o nepotrivire reală de risc:
    changed_risk = _cfg().risk.model_copy(update={"max_open_positions": 2})
    decision = evaluate_capital_change(
        _proposal(reevaluated_risk=changed_risk, new_snapshot=mismatched)
    )
    assert not decision.authorized
    assert CapitalRejectReason.SNAPSHOT_NOT_ATOMIC in decision.reasons


def test_rejects_when_snapshot_tampered() -> None:
    config = _cfg()
    good = _snapshot(config)
    tampered = good.model_copy(update={"seed": good.seed + 1})  # snapshot_id nu mai corespunde
    assert not tampered.verify()
    decision = evaluate_capital_change(_proposal(new_snapshot=tampered))
    assert not decision.authorized
    assert CapitalRejectReason.SNAPSHOT_NOT_ATOMIC in decision.reasons


# --------------------------------------------------------------------------- fără creșteri automate


def test_rejects_automatic_increase() -> None:
    """28.6: o modificare care nu este inițiată explicit de operator este respinsă."""
    decision = evaluate_capital_change(
        _proposal(approval=_approval(operator_initiated=False))
    )
    assert not decision.authorized
    assert CapitalRejectReason.AUTOMATIC_INCREASE in decision.reasons


# --------------------------------------------------------------- dovezi capital suplimentar (28.3)


def test_increase_requires_paper_qualification_and_promotion_criteria() -> None:
    """28.3: capital suplimentar fără Paper_Qualification/Promotion_Criteria este respins."""
    decision = evaluate_capital_change(
        _proposal(
            proposed_capital_eur=Decimal("120"),
            approval=_approval(paper_qualification_valid=False),
        )
    )
    assert not decision.authorized
    assert CapitalRejectReason.EVIDENCE_MISSING in decision.reasons


def test_non_increase_does_not_require_evidence() -> None:
    """O modificare fără creștere (capital egal) nu cere dovezi suplimentare (28.3)."""
    decision = evaluate_capital_change(
        _proposal(approval=_approval(paper_qualification_valid=False,
                                     promotion_criteria_approved=False))
    )
    assert decision.authorized


# ------------------------------------------------------------------- limite Risk_Engine (28.5)


def test_rejects_when_risk_limits_not_reevaluated() -> None:
    decision = evaluate_capital_change(_proposal(reevaluated_risk=None, new_snapshot=None))
    assert not decision.authorized
    assert CapitalRejectReason.RISK_LIMITS_NOT_REEVALUATED in decision.reasons


# --------------------------------------------------------------------------- limita totală ≤ 10 EUR


def test_rejects_total_loss_above_cap() -> None:
    """28.2, 28.7: limita totală propusă > 10 EUR respinge configurația.

    `RiskConfig` impune deja `total_loss_limit_eur == 10`; construim invalidarea ocolind
    validatorul pentru a verifica regula de capital independent.
    """
    over = RiskConfig.model_construct(
        reference_capital_eur=REFERENCE,
        risk_per_trade_target_eur=Decimal("0.25"),
        risk_per_trade_max_eur=Decimal("0.50"),
        daily_loss_limit_eur=Decimal("2"),
        total_loss_limit_eur=Decimal("11"),
        max_open_positions=3,
    )
    decision = evaluate_capital_change(_unchecked_proposal(reevaluated_risk=over))
    assert not decision.authorized
    assert CapitalRejectReason.TOTAL_LOSS_EXCEEDS_CAP in decision.reasons


def test_total_loss_exactly_cap_is_allowed() -> None:
    """Limita totală fix 10 EUR este acceptată (28.2)."""
    decision = evaluate_capital_change(_proposal())
    assert decision.authorized


# ----------------------------------------------------------------- capital de referință (28.1)


def test_rejects_reference_capital_change() -> None:
    changed = RiskConfig.model_construct(
        reference_capital_eur=Decimal("200"),
        risk_per_trade_target_eur=Decimal("0.25"),
        risk_per_trade_max_eur=Decimal("0.50"),
        daily_loss_limit_eur=Decimal("2"),
        total_loss_limit_eur=Decimal("10"),
        max_open_positions=3,
    )
    decision = evaluate_capital_change(_unchecked_proposal(reevaluated_risk=changed))
    assert not decision.authorized
    assert CapitalRejectReason.REFERENCE_CAPITAL_CHANGED in decision.reasons


# --------------------------------------------------------------------------- motive agregate


def test_multiple_violations_are_all_reported_in_stable_order() -> None:
    """Fail-closed: toate regulile încălcate sunt raportate, în ordine stabilă."""
    decision = evaluate_capital_change(
        CapitalChangeProposal(
            current_capital_eur=REFERENCE,
            proposed_capital_eur=Decimal("150"),
            approval=_approval(
                operator_initiated=False,
                paper_qualification_valid=False,
                promotion_criteria_approved=False,
            ),
            reevaluated_risk=None,
            new_snapshot=None,
        )
    )
    assert not decision.authorized
    assert CapitalRejectReason.RISK_LIMITS_NOT_REEVALUATED in decision.reasons
    assert CapitalRejectReason.AUTOMATIC_INCREASE in decision.reasons
    assert CapitalRejectReason.EVIDENCE_MISSING in decision.reasons
    assert CapitalRejectReason.SNAPSHOT_NOT_ATOMIC in decision.reasons
    # Ordinea urmează definiția enum-ului (determinism).
    order = list(CapitalRejectReason)
    idx = [order.index(r) for r in decision.reasons]
    assert idx == sorted(idx)


# --------------------------------------------------------------------------- validări model


def test_approval_requires_ref_and_approver() -> None:
    with pytest.raises(ValueError):
        CapitalChangeApproval(
            approval_ref="  ", approver="owner", operator_initiated=True
        )
    with pytest.raises(ValueError):
        CapitalChangeApproval(
            approval_ref="ref", approver="  ", operator_initiated=True
        )


def test_decision_consistency_is_enforced() -> None:
    from qts.risk.capital import CapitalChangeDecision

    with pytest.raises(ValueError):
        CapitalChangeDecision(authorized=True, reasons=(CapitalRejectReason.APPROVAL_INVALID,))
    with pytest.raises(ValueError):
        CapitalChangeDecision(authorized=False, reasons=())
