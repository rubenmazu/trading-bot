"""Teste pentru `safety/decisions.py` (Req 30.1–30.9, 22.1, 22.2, 22.8, 3.5, 3.6).

Acoperă: înregistrarea deciziilor, starea deschis vs rezolvat, înregistrarea aprobării, blocarea
etapelor dependente cât timp decizia e deschisă și deblocarea automată la aprobare, prezența celor
patru decizii seed și persistența stării peste reconectarea bazei.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from qts.persistence.db import open_db
from qts.safety.decisions import (
    SEED_DECISIONS,
    Approval,
    DecisionAlreadyApprovedError,
    DecisionDomain,
    DecisionRegistry,
    DecisionSeed,
    DecisionStatus,
    DependentStage,
    OpenDecision,
    RegistryError,
    UnknownDecisionError,
    decision_hash,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
DEADLINE = datetime(2026, 12, 31, tzinfo=UTC)


def _registry() -> DecisionRegistry:
    return DecisionRegistry(open_db(":memory:"))


def _custom_seed(
    topic: str = "custom", stage: DependentStage = DependentStage.LIVE_ELIGIBILITY
) -> DecisionSeed:
    return DecisionSeed(
        topic=topic,
        title="Decizie de test",
        options=("optiunea A", "optiunea B"),
        criteria=("criteriu masurabil 1", "criteriu masurabil 2"),
        responsible="responsabil",
        deadline=DEADLINE,
        affected_domains=(DecisionDomain.SAFETY,),
        blocked_stage=stage,
    )


# --------------------------------------------------------------------------- deciziile seed


def test_four_seed_decisions_present() -> None:
    # Req 30.1 / 3.5: broker, univers, sursă de date și durata minimă Demo.
    reg = _registry()
    topics = {d.topic for d in reg.list_all()}
    assert topics == {"broker", "universe", "data_source", "demo_duration"}
    assert len(SEED_DECISIONS) == 4


def test_seed_decisions_are_open_and_block_stages() -> None:
    reg = _registry()
    # Toate seed-urile încep deschise (Req 30.2).
    assert {d.topic for d in reg.open_decisions()} == {
        "broker",
        "universe",
        "data_source",
        "demo_duration",
    }
    # Req 3.6: brokerul blochează eligibilitatea Live.
    assert reg.is_live_eligibility_blocked() is True
    # Req 30.6: sursa de date / universul blochează validarea Strategy.
    assert reg.is_stage_blocked(DependentStage.STRATEGY_VALIDATION) is True
    # Req 22.2: durata Demo blochează finalizarea Paper_Qualification.
    assert reg.is_stage_blocked(DependentStage.PAPER_QUALIFICATION) is True


def test_broker_options_are_preliminary_ibkr_and_alpaca() -> None:
    # Req 30.3: IBKR și Alpaca rămân opțiuni preliminare.
    reg = _registry()
    broker = reg.get("broker")
    assert broker is not None
    assert set(broker.options) == {"ibkr", "alpaca"}
    # Req 3.5: criteriile de evaluare a brokerului includ accesul pentru rezidenți din România.
    assert any("România" in c for c in broker.criteria)


def test_seeding_is_idempotent() -> None:
    conn = open_db(":memory:")
    DecisionRegistry(conn)
    before = len(DecisionRegistry(conn).list_all())
    DecisionRegistry(conn).seed_decisions()
    after = len(DecisionRegistry(conn).list_all())
    assert before == after == 4


# --------------------------------------------------------------------------- înregistrare


def test_register_builds_hashed_decision() -> None:
    reg = DecisionRegistry(open_db(":memory:"), seed=False)
    decision = reg.register(_custom_seed(), now=NOW)
    assert decision.verify() is True
    assert decision.decision_id == decision_hash(
        topic=decision.topic,
        title=decision.title,
        options=decision.options,
        criteria=decision.criteria,
        responsible=decision.responsible,
        deadline=decision.deadline,
        affected_domains=decision.affected_domains,
        blocked_stage=decision.blocked_stage,
    )


def test_decision_hash_is_order_independent() -> None:
    a = decision_hash(
        topic="t",
        title="T",
        options=("a", "b"),
        criteria=("c1", "c2"),
        responsible="r",
        deadline=DEADLINE,
        affected_domains=(DecisionDomain.SAFETY, DecisionDomain.COSTS),
        blocked_stage=DependentStage.LIVE_ELIGIBILITY,
    )
    b = decision_hash(
        topic="t",
        title="T",
        options=("b", "a"),
        criteria=("c2", "c1"),
        responsible="r",
        deadline=DEADLINE,
        affected_domains=(DecisionDomain.COSTS, DecisionDomain.SAFETY),
        blocked_stage=DependentStage.LIVE_ELIGIBILITY,
    )
    assert a == b


def test_register_rejects_conflicting_content_for_same_topic() -> None:
    reg = DecisionRegistry(open_db(":memory:"), seed=False)
    reg.register(_custom_seed(), now=NOW)
    conflicting = DecisionSeed(
        topic="custom",
        title="Titlu diferit",
        options=("optiunea A", "optiunea B"),
        criteria=("criteriu masurabil 1", "criteriu masurabil 2"),
        responsible="responsabil",
        deadline=DEADLINE,
        affected_domains=(DecisionDomain.SAFETY,),
        blocked_stage=DependentStage.LIVE_ELIGIBILITY,
    )
    with pytest.raises(RegistryError):
        reg.register(conflicting, now=NOW)


def test_decision_requires_two_options() -> None:
    with pytest.raises(ValueError):
        OpenDecision(
            decision_id="x",
            topic="t",
            title="T",
            options=("only",),
            criteria=("c",),
            responsible="r",
            deadline=DEADLINE,
            affected_domains=(DecisionDomain.SAFETY,),
            blocked_stage=DependentStage.LIVE_ELIGIBILITY,
        )


# --------------------------------------------------------------------------- stare și aprobare


def test_open_vs_resolved_state() -> None:
    reg = _registry()
    assert reg.status("broker") is DecisionStatus.OPEN
    assert reg.is_approved("broker") is False
    reg.approve(
        "broker",
        chosen_option="ibkr",
        evidence=("raport de evaluare broker",),
        consequences="Live folosește IBKR",
        approver="responsabil de proiect",
        now=NOW,
    )
    assert reg.status("broker") is DecisionStatus.APPROVED
    assert reg.is_approved("broker") is True


def test_approval_records_choice_evidence_and_approver() -> None:
    # Req 30.4, 22.2, 22.8.
    reg = _registry()
    approval = reg.approve(
        "demo_duration",
        chosen_option="minimum 20 de zile de tranzacționare",
        evidence=("analiză statistică a duratei", "jurnal de ordine Demo"),
        consequences="Paper_Qualification necesită 20 de zile",
        approver="responsabil de proiect",
        now=NOW,
    )
    assert isinstance(approval, Approval)
    assert approval.chosen_option == "minimum 20 de zile de tranzacționare"
    assert approval.approver == "responsabil de proiect"
    assert len(approval.evidence) == 2
    stored = reg.get_approval("demo_duration")
    assert stored == approval


def test_approve_rejects_unknown_option() -> None:
    reg = _registry()
    with pytest.raises(RegistryError):
        reg.approve(
            "broker",
            chosen_option="necunoscut",
            evidence=("x",),
            consequences="y",
            approver="z",
            now=NOW,
        )


def test_approve_twice_is_refused() -> None:
    reg = _registry()
    reg.approve(
        "broker",
        chosen_option="ibkr",
        evidence=("raport",),
        consequences="Live folosește IBKR",
        approver="responsabil",
        now=NOW,
    )
    with pytest.raises(DecisionAlreadyApprovedError):
        reg.approve(
            "broker",
            chosen_option="alpaca",
            evidence=("alt raport",),
            consequences="altceva",
            approver="responsabil",
            now=NOW,
        )


def test_approve_unknown_topic_raises() -> None:
    reg = _registry()
    with pytest.raises(UnknownDecisionError):
        reg.approve(
            "inexistent",
            chosen_option="x",
            evidence=("y",),
            consequences="z",
            approver="w",
            now=NOW,
        )


# --------------------------------------------------------------------------- blocarea etapelor


def test_dependent_stage_blocked_while_open_and_unblocked_after_approval() -> None:
    # Req 30.5 (blocare) și 30.9 (deblocare automată la rezolvare).
    reg = _registry()
    assert reg.is_live_eligibility_blocked() is True
    assert "broker" in {d.topic for d in reg.blocking_decisions(DependentStage.LIVE_ELIGIBILITY)}
    reg.approve(
        "broker",
        chosen_option="ibkr",
        evidence=("raport",),
        consequences="Live folosește IBKR",
        approver="responsabil",
        now=NOW,
    )
    # După aprobare, condiția de eligibilitate Live se deblochează automat.
    assert reg.is_live_eligibility_blocked() is False
    assert reg.blocking_decisions(DependentStage.LIVE_ELIGIBILITY) == ()


def test_paper_qualification_blocked_until_demo_duration_approved() -> None:
    # Req 22.2: durata Demo deschisă blochează finalizarea Paper_Qualification.
    reg = _registry()
    assert reg.is_stage_blocked(DependentStage.PAPER_QUALIFICATION) is True
    reg.approve(
        "demo_duration",
        chosen_option="minimum 40 de zile de tranzacționare",
        evidence=("justificare",),
        consequences="40 de zile",
        approver="responsabil",
        now=NOW,
    )
    assert reg.is_stage_blocked(DependentStage.PAPER_QUALIFICATION) is False


# --------------------------------------------------------------------------- persistență


def test_state_survives_reconnect(tmp_path: Path) -> None:
    db = tmp_path / "decisions.db"
    reg = DecisionRegistry(open_db(db))
    reg.approve(
        "broker",
        chosen_option="alpaca",
        evidence=("raport de evaluare",),
        consequences="Live folosește Alpaca",
        approver="responsabil",
        now=NOW,
    )
    reg.connection.close()

    reopened = DecisionRegistry(open_db(db))
    assert reopened.is_approved("broker") is True
    assert reopened.is_live_eligibility_blocked() is False
    approval = reopened.get_approval("broker")
    assert approval is not None
    assert approval.chosen_option == "alpaca"
    # Deciziile neaprobate rămân deschise după reconectare.
    assert reopened.status("universe") is DecisionStatus.OPEN
