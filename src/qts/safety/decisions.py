"""Registrul Open_Decision și aprobările, cu blocarea etapelor dependente (Req 30, 22, 3).

O `Open_Decision` este o decizie nerezolvată care are opțiuni, criterii măsurabile, un responsabil,
un termen și o stare documentată (Req 30.2). Fiecare decizie afectează cel puțin un domeniu dintre
siguranță, costuri, reproductibilitate sau eligibilitate Live și blochează o etapă dependentă până
la aprobare (Req 30.5). `decision_id` = SHA-256 peste conținutul canonic al deciziei (în stilul
`research/preregistration.py` și `research/partition.py`), deci identitatea deciziei este stabilă,
iar orice modificare a conținutului ar schimba hash-ul.

Starea este binară și derivată, nu stocată redundant: o decizie este „aprobată” exact când există
o aprobare pentru ea; altfel este „deschisă” (Req 30.2 — starea documentată). O aprobare
înregistrează alegerea, criteriile, dovezile, consecințele, aprobatorul și momentul
(Req 30.4, 22.2, 22.8).

Blocarea este generică (Req 30.5): `is_stage_blocked(stage)` întoarce adevărat cât timp există cel
puțin o decizie deschisă care blochează acea etapă. La aprobarea deciziei, etapa se deblochează
automat (Req 30.9) — nu există stare separată de deblocat, blocarea se recalculează din registru la
fiecare interogare, în același spirit fail-closed ca `safety/live_gate.py`.

Registrul este persistent (SQLite, tabelele `decisions` + `approvals` din migrația 5) și imuabil
după scriere: după reconectarea bazei, o decizie aprobată rămâne aprobată. Scrierile folosesc
tranzacții `BEGIN IMMEDIATE`, în stilul `qts.persistence`.

La construcție, registrul populează (idempotent) cele patru decizii concrete cerute de Req 30.1 și
3.5: brokerul, universul inițial exact, sursa de date și durata minimă Demo.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from pydantic import model_validator

from qts.core.clock import ensure_utc
from qts.core.models import Frozen, UtcDatetime

__all__ = [
    "DECISION_VERSION",
    "SEED_DECISIONS",
    "Approval",
    "DecisionAlreadyApprovedError",
    "DecisionDomain",
    "DecisionRegistry",
    "DecisionSeed",
    "DecisionStatus",
    "DependentStage",
    "OpenDecision",
    "RegistryError",
    "UnknownDecisionError",
    "decision_hash",
]

DECISION_VERSION: Final = "open-decision-v1"


# --------------------------------------------------------------------------- enumerări


class DecisionDomain(StrEnum):
    """Domeniile pe care le poate afecta o Open_Decision (Req 30.5)."""

    SAFETY = "safety"
    COSTS = "costs"
    REPRODUCIBILITY = "reproducibility"
    LIVE_ELIGIBILITY = "live_eligibility"


class DependentStage(StrEnum):
    """Etapa dependentă pe care o decizie deschisă o blochează (Req 30.5, 30.6, 22.2, 3.6)."""

    LIVE_ELIGIBILITY = "live_eligibility"  # Req 3.6, 30.5
    STRATEGY_VALIDATION = "strategy_validation"  # Req 30.6
    PAPER_QUALIFICATION = "paper_qualification"  # Req 22.2


class DecisionStatus(StrEnum):
    """Starea documentată a unei Open_Decision (Req 30.2)."""

    OPEN = "open"
    APPROVED = "approved"


# --------------------------------------------------------------------------- erori


class RegistryError(Exception):
    """Eroare generică a registrului de decizii."""


class UnknownDecisionError(RegistryError):
    """Operație asupra unei decizii care nu există în registru."""


class DecisionAlreadyApprovedError(RegistryError):
    """A doua aprobare a aceleiași decizii este refuzată (aprobările sunt imuabile, Req 30.4)."""


# --------------------------------------------------------------------------- modele


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sorted_unique_str(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({v for v in values}))


class OpenDecision(Frozen):
    """O decizie nerezolvată cu opțiuni, criterii, responsabil, termen și stare (Req 30.2).

    `decision_id` este derivat din conținut prin `decision_hash`; `verify` recompută hash-ul și
    confirmă că nimic nu s-a modificat de la înregistrare. `topic` este cheia stabilă prin care se
    referă o decizie concretă (ex. „broker”, „data_source”).
    """

    decision_id: str
    version: str = DECISION_VERSION
    topic: str
    title: str
    options: tuple[str, ...]
    criteria: tuple[str, ...]
    responsible: str
    deadline: UtcDatetime
    affected_domains: tuple[DecisionDomain, ...]
    blocked_stage: DependentStage

    @model_validator(mode="after")
    def _check(self) -> OpenDecision:
        if not self.topic.strip():
            raise ValueError("decizia necesită un topic")
        if not self.title.strip():
            raise ValueError("decizia necesită un titlu")
        if not self.responsible.strip():
            raise ValueError("decizia necesită un responsabil")
        # Req 30.2: opțiunile și criteriile măsurabile trebuie documentate.
        if len(self.options) < 2:
            raise ValueError("o Open_Decision necesită cel puțin două opțiuni (Req 30.2)")
        if len(self.options) != len(set(self.options)):
            raise ValueError("opțiuni duplicate")
        if not self.criteria:
            raise ValueError("o Open_Decision necesită criterii măsurabile (Req 30.2)")
        if len(self.criteria) != len(set(self.criteria)):
            raise ValueError("criterii duplicate")
        # Req 30.5: decizia afectează cel puțin un domeniu dependent.
        if not self.affected_domains:
            raise ValueError("decizia trebuie să afecteze cel puțin un domeniu (Req 30.5)")
        if len(self.affected_domains) != len(set(self.affected_domains)):
            raise ValueError("domenii duplicate")
        expected = decision_hash(
            topic=self.topic,
            title=self.title,
            options=self.options,
            criteria=self.criteria,
            responsible=self.responsible,
            deadline=self.deadline,
            affected_domains=self.affected_domains,
            blocked_stage=self.blocked_stage,
        )
        if self.decision_id != expected:
            raise ValueError("decision_id nu corespunde conținutului deciziei")
        return self

    def verify(self) -> bool:
        """Adevărat dacă `decision_id` corespunde conținutului (detectează modificări)."""
        return self.decision_id == decision_hash(
            topic=self.topic,
            title=self.title,
            options=self.options,
            criteria=self.criteria,
            responsible=self.responsible,
            deadline=self.deadline,
            affected_domains=self.affected_domains,
            blocked_stage=self.blocked_stage,
        )


class Approval(Frozen):
    """Aprobarea unei decizii: alegere, criterii, dovezi, consecințe, aprobator (Req 30.4, 22.2)."""

    decision_id: str
    chosen_option: str
    criteria: tuple[str, ...]
    evidence: tuple[str, ...]
    consequences: str
    approver: str
    approved_at: UtcDatetime

    @model_validator(mode="after")
    def _check(self) -> Approval:
        if not self.chosen_option.strip():
            raise ValueError("aprobarea necesită opțiunea aleasă (Req 30.4)")
        if not self.evidence:
            raise ValueError("aprobarea necesită dovezi (Req 30.4, 22.2)")
        if not self.consequences.strip():
            raise ValueError("aprobarea necesită consecințe documentate (Req 30.4)")
        if not self.approver.strip():
            raise ValueError("aprobarea necesită un aprobator (Req 30.4, 22.8)")
        return self


class DecisionSeed(Frozen):
    """Descrierea unei decizii concrete de populat la pornire (Req 30.1, 3.5)."""

    topic: str
    title: str
    options: tuple[str, ...]
    criteria: tuple[str, ...]
    responsible: str
    deadline: UtcDatetime
    affected_domains: tuple[DecisionDomain, ...]
    blocked_stage: DependentStage


def decision_hash(
    *,
    topic: str,
    title: str,
    options: Sequence[str],
    criteria: Sequence[str],
    responsible: str,
    deadline: datetime,
    affected_domains: Sequence[DecisionDomain],
    blocked_stage: DependentStage,
) -> str:
    """SHA-256 hex peste conținutul canonic al unei Open_Decision.

    Opțiunile, criteriile și domeniile sunt sortate pentru ca hash-ul să nu depindă de ordine,
    la fel ca `research/partition.partition_hash`.
    """
    content = {
        "version": DECISION_VERSION,
        "topic": topic,
        "title": title,
        "options": sorted(options),
        "criteria": sorted(criteria),
        "responsible": responsible,
        "deadline": ensure_utc(deadline).isoformat(),
        "affected_domains": sorted(DecisionDomain(d).value for d in affected_domains),
        "blocked_stage": DependentStage(blocked_stage).value,
    }
    return hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- deciziile concrete

# Termen comun pentru deciziile deschise ale Initial_Stage. Fix (fără volatilitate), ca hash-urile
# deciziilor seed să fie deterministe între rulări.
_SEED_DEADLINE: Final = datetime(2026, 12, 31, tzinfo=UTC)

# Cele patru decizii concrete cerute de Req 30.1 și 3.5: broker, univers, sursă date, durată Demo.
SEED_DECISIONS: Final[tuple[DecisionSeed, ...]] = (
    DecisionSeed(
        topic="broker",
        title="Selecția brokerului pentru Demo și Live",
        # Req 30.3: IBKR și Alpaca rămân doar opțiuni preliminare până la verificarea criteriilor.
        options=("ibkr", "alpaca"),
        criteria=(
            "acces pentru rezidenți din România",  # Req 3.5
            "disponibilitate API documentată",
            "mediu Demo (paper) disponibil",
            "costuri și comisioane publicate",
            "suport pentru fracțiuni",
            "active eligibile acoperite",
            "fiabilitate documentată",
        ),
        responsible="responsabil de proiect",
        deadline=_SEED_DEADLINE,
        # Req 3.6: cât timp decizia brokerului este deschisă, eligibilitatea Live este blocată.
        affected_domains=(
            DecisionDomain.SAFETY,
            DecisionDomain.COSTS,
            DecisionDomain.LIVE_ELIGIBILITY,
        ),
        blocked_stage=DependentStage.LIVE_ELIGIBILITY,
    ),
    DecisionSeed(
        topic="universe",
        title="Universul inițial exact de instrumente",
        options=(
            "ETF-uri UCITS lichide (listă restrânsă)",
            "acțiuni mari-cap din zona euro",
        ),
        criteria=(
            "lichiditate minimă validată",
            "orar de tranzacționare compatibil",
            "eligibilitate pentru rezidenți din România",
        ),
        responsible="responsabil de proiect",
        deadline=_SEED_DEADLINE,
        affected_domains=(DecisionDomain.REPRODUCIBILITY, DecisionDomain.COSTS),
        blocked_stage=DependentStage.STRATEGY_VALIDATION,
    ),
    DecisionSeed(
        topic="data_source",
        title="Sursa de date de piață",
        options=(
            "feed-ul brokerului ales",
            "furnizor de date terț dedicat",
        ),
        criteria=(
            "istoric suficient pentru walk-forward",
            "prospețime și latență documentate",
            "calitate și completitudine verificate",
        ),
        responsible="responsabil de proiect",
        deadline=_SEED_DEADLINE,
        # Req 30.6: sursa de date deschisă blochează validarea finală a Strategy.
        affected_domains=(DecisionDomain.REPRODUCIBILITY, DecisionDomain.SAFETY),
        blocked_stage=DependentStage.STRATEGY_VALIDATION,
    ),
    DecisionSeed(
        topic="demo_duration",
        title="Durata minimă a calificării Demo",
        options=(
            "minimum 20 de zile de tranzacționare",
            "minimum 40 de zile de tranzacționare",
        ),
        criteria=(
            "număr minim de ordine executate",  # Req 22.1
            "acoperirea regimurilor de piață",
            "justificare statistică a duratei",  # Req 22.8
        ),
        responsible="responsabil de proiect",
        deadline=_SEED_DEADLINE,
        # Req 22.2: durata Demo deschisă blochează finalizarea Paper_Qualification.
        affected_domains=(DecisionDomain.SAFETY, DecisionDomain.LIVE_ELIGIBILITY),
        blocked_stage=DependentStage.PAPER_QUALIFICATION,
    ),
)


# --------------------------------------------------------------------------- registru persistent


class DecisionRegistry:
    """Registru persistent de Open_Decision cu aprobări imuabile și blocare de etape (Req 30).

    Scrie în tabelele `decisions` (append-only, conținutul definitoriu al deciziei) și `approvals`
    (append-only, aprobarea unică a fiecărei decizii). Starea deciziei se derivă din existența unei
    aprobări. La construcție populează deciziile seed (Req 30.1, 3.5), idempotent.
    """

    def __init__(self, conn: sqlite3.Connection, *, seed: bool = True) -> None:
        self._conn = conn
        if seed:
            self.seed_decisions()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    # -- populare ------------------------------------------------------------

    def seed_decisions(self, *, now: datetime | None = None) -> None:
        """Înregistrează (idempotent) cele patru decizii concrete cerute (Req 30.1, 3.5)."""
        for seed in SEED_DECISIONS:
            self.register(seed, now=now)

    # -- înregistrare --------------------------------------------------------

    def register(self, seed: DecisionSeed, *, now: datetime | None = None) -> OpenDecision:
        """Înregistrează o Open_Decision (Req 30.2); idempotentă dacă hash-ul coincide."""
        decision = self._build(seed)
        created_at = ensure_utc(now) if now is not None else _utcnow()
        existing = self._conn.execute(
            "SELECT decision_id, content_hash FROM decisions WHERE topic = ?", (seed.topic,)
        ).fetchone()
        if existing is not None:
            if existing["content_hash"] != decision.decision_id:
                raise RegistryError(
                    f"topicul {seed.topic!r} există deja cu alt conținut (deciziile sunt imuabile)"
                )
            return decision
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO decisions (decision_id, content_hash, topic, title, options, "
                "criteria, responsible, deadline, affected_domains, blocked_stage, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.decision_id,
                    decision.decision_id,
                    decision.topic,
                    decision.title,
                    _canonical(list(decision.options)),
                    _canonical(list(decision.criteria)),
                    decision.responsible,
                    decision.deadline.isoformat(),
                    _canonical([d.value for d in decision.affected_domains]),
                    decision.blocked_stage.value,
                    created_at.isoformat(),
                ),
            )
        return decision

    # -- aprobare ------------------------------------------------------------

    def approve(
        self,
        topic: str,
        *,
        chosen_option: str,
        evidence: Sequence[str],
        consequences: str,
        approver: str,
        criteria: Sequence[str] | None = None,
        now: datetime | None = None,
    ) -> Approval:
        """Aprobă o decizie, înregistrând alegerea, dovezile, consecințele și aprobatorul.

        Req 30.4, 22.2, 22.8. Opțiunea aleasă trebuie să fie una dintre opțiunile deciziei. O a doua
        aprobare a aceleiași decizii este refuzată (aprobările sunt imuabile).
        """
        decision = self._require(topic)
        if chosen_option not in decision.options:
            raise RegistryError(
                f"opțiunea {chosen_option!r} nu face parte din decizia {topic!r}"
            )
        approved_at = ensure_utc(now) if now is not None else _utcnow()
        chosen_criteria = tuple(criteria) if criteria is not None else decision.criteria
        approval = Approval(
            decision_id=decision.decision_id,
            chosen_option=chosen_option,
            criteria=chosen_criteria,
            evidence=tuple(evidence),
            consequences=consequences,
            approver=approver,
            approved_at=approved_at,
        )
        with self._tx() as conn:
            exists = conn.execute(
                "SELECT 1 FROM approvals WHERE decision_id = ?", (decision.decision_id,)
            ).fetchone()
            if exists is not None:
                raise DecisionAlreadyApprovedError(
                    f"decizia {topic!r} este deja aprobată (aprobările sunt imuabile)"
                )
            conn.execute(
                "INSERT INTO approvals (decision_id, chosen_option, criteria, evidence, "
                "consequences, approver, approved_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    approval.decision_id,
                    approval.chosen_option,
                    _canonical(list(approval.criteria)),
                    _canonical(list(approval.evidence)),
                    approval.consequences,
                    approval.approver,
                    approval.approved_at.isoformat(),
                ),
            )
        return approval

    # -- interogare ----------------------------------------------------------

    def get(self, topic: str) -> OpenDecision | None:
        """Întoarce decizia după topic sau `None` dacă nu există."""
        row = self._conn.execute(
            "SELECT * FROM decisions WHERE topic = ?", (topic,)
        ).fetchone()
        return _row_to_decision(row) if row is not None else None

    def get_approval(self, topic: str) -> Approval | None:
        """Întoarce aprobarea deciziei sau `None` dacă nu este încă aprobată."""
        decision = self.get(topic)
        if decision is None:
            return None
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE decision_id = ?", (decision.decision_id,)
        ).fetchone()
        return _row_to_approval(row) if row is not None else None

    def status(self, topic: str) -> DecisionStatus:
        """Starea unei decizii: aprobată dacă are o aprobare, altfel deschisă (Req 30.2)."""
        self._require(topic)
        if self.get_approval(topic) is not None:
            return DecisionStatus.APPROVED
        return DecisionStatus.OPEN

    def is_approved(self, topic: str) -> bool:
        """Adevărat dacă decizia a fost aprobată."""
        return self.status(topic) is DecisionStatus.APPROVED

    def list_all(self) -> tuple[OpenDecision, ...]:
        """Toate deciziile înregistrate, ordonate după topic."""
        rows = self._conn.execute("SELECT * FROM decisions ORDER BY topic").fetchall()
        return tuple(_row_to_decision(r) for r in rows)

    def open_decisions(self) -> tuple[OpenDecision, ...]:
        """Deciziile încă nerezolvate (fără aprobare), ordonate după topic (Req 30.2)."""
        return tuple(d for d in self.list_all() if self.get_approval(d.topic) is None)

    # -- blocarea etapelor dependente ---------------------------------------

    def is_stage_blocked(self, stage: DependentStage) -> bool:
        """Adevărat cât timp o decizie deschisă blochează etapa `stage` (Req 30.5, 22.2, 3.6).

        Blocarea se recalculează din registru la fiecare apel: la aprobarea deciziei, etapa se
        deblochează automat (Req 30.9), fără stare separată de modificat.
        """
        return any(d.blocked_stage is stage for d in self.open_decisions())

    def blocking_decisions(self, stage: DependentStage) -> tuple[OpenDecision, ...]:
        """Deciziile deschise care blochează o anumită etapă dependentă."""
        return tuple(d for d in self.open_decisions() if d.blocked_stage is stage)

    def is_live_eligibility_blocked(self) -> bool:
        """Adevărat cât timp o decizie deschisă blochează eligibilitatea Live (Req 3.6, 30.5)."""
        return self.is_stage_blocked(DependentStage.LIVE_ELIGIBILITY)

    # -- intern --------------------------------------------------------------

    def _build(self, seed: DecisionSeed) -> OpenDecision:
        digest = decision_hash(
            topic=seed.topic,
            title=seed.title,
            options=seed.options,
            criteria=seed.criteria,
            responsible=seed.responsible,
            deadline=seed.deadline,
            affected_domains=seed.affected_domains,
            blocked_stage=seed.blocked_stage,
        )
        return OpenDecision(
            decision_id=digest,
            topic=seed.topic,
            title=seed.title,
            options=seed.options,
            criteria=seed.criteria,
            responsible=seed.responsible,
            deadline=seed.deadline,
            affected_domains=seed.affected_domains,
            blocked_stage=seed.blocked_stage,
        )

    def _require(self, topic: str) -> OpenDecision:
        decision = self.get(topic)
        if decision is None:
            raise UnknownDecisionError(f"decizia {topic!r} nu există în registru")
        return decision

    def _tx(self) -> _Transaction:
        return _Transaction(self._conn)


class _Transaction:
    """Context de tranzacție `BEGIN IMMEDIATE`, în stilul `qts.persistence`."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._owns = False

    def __enter__(self) -> sqlite3.Connection:
        if not self._conn.in_transaction:
            self._conn.execute("BEGIN IMMEDIATE")
            self._owns = True
        return self._conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._owns:
            return
        if exc_type is None:
            self._conn.execute("COMMIT")
        else:
            self._conn.execute("ROLLBACK")


def _row_to_decision(row: sqlite3.Row) -> OpenDecision:
    return OpenDecision(
        decision_id=row["decision_id"],
        topic=row["topic"],
        title=row["title"],
        options=tuple(json.loads(row["options"])),
        criteria=tuple(json.loads(row["criteria"])),
        responsible=row["responsible"],
        deadline=datetime.fromisoformat(row["deadline"]),
        affected_domains=tuple(DecisionDomain(d) for d in json.loads(row["affected_domains"])),
        blocked_stage=DependentStage(row["blocked_stage"]),
    )


def _row_to_approval(row: sqlite3.Row) -> Approval:
    return Approval(
        decision_id=row["decision_id"],
        chosen_option=row["chosen_option"],
        criteria=tuple(json.loads(row["criteria"])),
        evidence=tuple(json.loads(row["evidence"])),
        consequences=row["consequences"],
        approver=row["approver"],
        approved_at=datetime.fromisoformat(row["approved_at"]),
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)
