"""Recuperare deterministă din jurnal (Req 26.1–26.5, 26.8, 26.9).

Principiu (design: *Event sourcing*): starea financiară nu se recalculează re-rulând motorul.
Jurnalul append-only cu lanț hash este simultan `Recovery_Point` și bază de audit; fiecare
înregistrare `oms.*` codifică deja *rezultatul* unei operații (starea ordinului după tranziție,
costurile execuției aplicate). Recuperarea *reconstruiește proiecțiile* din aceste rezultate,
fără să treacă din nou prin FSM, risc sau broker. Astfel reluarea este idempotentă: o execuție
deja aplicată (același `broker_exec_id`) nu se aplică a doua oară (Req 26.8).

Ce se citește din jurnal vs. ce se recalculează
    - `IdempotencyRegistry`      — citit: cheile de ordin din `oms.ORDER_CREATED`,
      `broker_exec_id` aplicate din `oms.EXEC_APPLIED` și cele reflectate de `oms.RECONCILED`.
    - stările ordinelor          — citite: ultima stare din `payload["order"]` al fiecărui
      `oms.*` care poartă ordinul (`ORDER_CREATED`, `TRANSITION`, `EXEC_APPLIED`, `RECONCILED`,
      `CANCEL_PENDING_RESTORED`). Nu se re-evaluează FSM-ul.
    - portofoliul                — *recalculat*: fiecare `oms.EXEC_APPLIED` care poartă un fill
      (`fill_costs` prezent) este rejucat prin `portfolio.apply_fill`. Numai execuțiile aplicate
      produc fill-uri, deci deduplicarea după `broker_exec_id` este deja implicită în jurnal.
    - `Kill_Switch`              — citit: reconstruit de `KillSwitch` din `kill_switch.*`
      (vezi `safety/kill_switch.py`); recuperarea doar raportează dacă `GLOBAL`/`CAPITAL_CONFIG`
      este activ, nu reconstruiește obiectul.

Blocarea instrumentelor (Req 26.5, 26.9)
    Un ordin lăsat într-o stare neterminală la momentul căderii — mai ales `SUBMITTED` sau
    `UNKNOWN` (cerere trimisă fără confirmare) — ține instrumentul său blocat până când
    reconcilierea confirmă rezultatul brokerului. `RecoveredState.frozen_instruments` enumeră
    aceste instrumente; `resume_seq` este `seq`-ul primului astfel de eveniment neconfirmat
    (Req 26.1, 26.3).

Recovery_Point invalid (Req 26.4)
    `validate_integrity` este verificare pură în memorie (lanț hash + eventual checkpoint), deci
    rulează cu mult sub 1 s. La verdict negativ, apelantul activează `Kill_Switch(GLOBAL)`;
    `global_kill_on_invalid_recovery` oferă calea rapidă.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Final

from qts.core.clock import Clock, ensure_utc
from qts.core.models import CostBreakdown, OrderState
from qts.oms.idempotency import AppliedExec, IdempotencyRegistry
from qts.oms.manager import RecordType
from qts.persistence.audit import ChainVerification, verify_journal
from qts.persistence.journal import Journal
from qts.portfolio.portfolio import PortfolioFill, PortfolioState, apply_fill
from qts.risk.context import KillSwitchScope
from qts.safety.kill_switch import KillSwitch

__all__ = [
    "RECOVERY_VERSION",
    "Checkpoint",
    "RecoveredState",
    "RecoveryVerdict",
    "global_kill_on_invalid_recovery",
    "latest_checkpoint",
    "projections_hash",
    "recover",
    "validate_integrity",
    "write_checkpoint",
]

RECOVERY_VERSION: Final = "1"

# Prefixul tipurilor OMS în jurnal (vezi `oms.manager.JournalOmsSink`).
_OMS_PREFIX: Final = "oms."

# Stările neterminale care necesită confirmarea brokerului înainte de ordine noi (Req 26.5, 26.9).
# `SUBMITTED` și `UNKNOWN` sunt rezultatul tipic al unei căderi între jurnal și broker.
PENDING_STATES: Final[frozenset[OrderState]] = frozenset(
    {
        OrderState.SUBMITTED,
        OrderState.UNKNOWN,
        OrderState.ACKNOWLEDGED,
        OrderState.PARTIALLY_FILLED,
        OrderState.CANCEL_PENDING,
    }
)

# --------------------------------------------------------------------------- checkpoint


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Recovery_Point persistat: până unde a fost validat jurnalul și hash-ul proiecțiilor."""

    journal_seq: int
    journal_hash: str
    projections_hash: str
    ts: datetime


def _dec(value: Any) -> Decimal:
    return Decimal(str(value))


def projections_hash(
    *,
    registry: IdempotencyRegistry,
    order_states: dict[str, OrderState],
    portfolio: PortfolioState,
) -> str:
    """Hash determinist al proiecțiilor reconstruite (ordine, idempotență, portofoliu).

    Forma este canonică (chei sortate, fără float), deci aceeași stare dă același hash după
    repornire. Hash-ul este stocat în checkpoint și reverificat la pornire (Req 26.2).
    """
    body = {
        "order_states": {k: order_states[k].value for k in sorted(order_states)},
        "applied_execs": {
            k: [a.client_order_id, a.record_seq, a.fingerprint]
            for k, a in sorted(registry.applied_execs.items())
        },
        "portfolio": json.loads(portfolio.model_dump_json()),
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_checkpoint(
    journal: Journal,
    *,
    proj_hash: str,
    ts: datetime,
) -> Checkpoint:
    """Scrie un Recovery_Point la head-ul curent al jurnalului (append-only)."""
    seq, digest = journal.head
    ts_iso = ensure_utc(ts).isoformat()
    with journal.transaction() as conn:
        conn.execute(
            "INSERT INTO checkpoints (journal_seq, journal_hash, projections_hash, ts) "
            "VALUES (?, ?, ?, ?)",
            (seq, digest, proj_hash, ts_iso),
        )
    return Checkpoint(
        journal_seq=seq, journal_hash=digest, projections_hash=proj_hash, ts=ensure_utc(ts)
    )


def latest_checkpoint(conn: sqlite3.Connection) -> Checkpoint | None:
    """Cel mai recent Recovery_Point (după `journal_seq`, apoi `id`), sau None."""
    row = conn.execute(
        "SELECT journal_seq, journal_hash, projections_hash, ts "
        "FROM checkpoints ORDER BY journal_seq DESC, id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return Checkpoint(
        journal_seq=row["journal_seq"],
        journal_hash=row["journal_hash"],
        projections_hash=row["projections_hash"],
        ts=datetime.fromisoformat(row["ts"]),
    )


# --------------------------------------------------------------------------- integritate


@dataclass(frozen=True, slots=True)
class RecoveryVerdict:
    """Rezultatul validării integrității Recovery_Point (Req 26.2, 26.4)."""

    ok: bool
    reason: str | None = None
    first_bad_seq: int | None = None
    records_checked: int = 0
    head_hash: str | None = None

    @classmethod
    def passed(cls, chain: ChainVerification) -> RecoveryVerdict:
        return cls(True, None, None, chain.records_checked, chain.head_hash)

    @classmethod
    def failed(
        cls,
        reason: str,
        *,
        first_bad_seq: int | None = None,
        chain: ChainVerification | None = None,
    ) -> RecoveryVerdict:
        return cls(
            False,
            reason,
            first_bad_seq
            if first_bad_seq is not None
            else (chain.first_bad_seq if chain else None),
            chain.records_checked if chain else 0,
            chain.head_hash if chain else None,
        )


def validate_integrity(journal: Journal, checkpoint: Checkpoint | None = None) -> RecoveryVerdict:
    """Validează integritatea Recovery_Point înainte de evenimente noi (Req 26.2).

    Pași, toți în memorie (calea rapidă pentru cerința ≤ 1 s, Req 26.4):
    1. verifică lanțul hash complet (`verify_journal`);
    2. dacă există un checkpoint, confirmă că jurnalul atinge `journal_seq` și că hash-ul
       înregistrării de la acel `seq` corespunde `checkpoint.journal_hash` (detectează
       trunchierea cozii sau rescrierea față de punctul validat);
    3. confirmă că proiecțiile reconstruite până la `journal_seq` au același `projections_hash`.

    La orice eșec întoarce un verdict negativ cu motiv și, când e cunoscut, `first_bad_seq`.
    """
    chain = verify_journal(journal)
    if not chain.ok:
        return RecoveryVerdict.failed("lanț hash rupt", chain=chain)
    if checkpoint is None:
        return RecoveryVerdict.passed(chain)

    if chain.records_checked < checkpoint.journal_seq:
        return RecoveryVerdict.failed(
            "jurnal trunchiat sub checkpoint",
            first_bad_seq=chain.records_checked + 1,
            chain=chain,
        )
    if checkpoint.journal_seq > 0:
        row = journal.connection.execute(
            "SELECT hash FROM journal WHERE seq = ?", (checkpoint.journal_seq,)
        ).fetchone()
        if row is None or row["hash"] != checkpoint.journal_hash:
            return RecoveryVerdict.failed(
                "head-ul checkpoint-ului nu corespunde jurnalului",
                first_bad_seq=checkpoint.journal_seq,
                chain=chain,
            )

    rebuilt = _rebuild(journal, up_to_seq=checkpoint.journal_seq)
    actual = projections_hash(
        registry=rebuilt.registry,
        order_states=rebuilt.order_states,
        portfolio=rebuilt.portfolio,
    )
    if actual != checkpoint.projections_hash:
        return RecoveryVerdict.failed(
            "hash-ul proiecțiilor nu corespunde checkpoint-ului",
            first_bad_seq=checkpoint.journal_seq,
            chain=chain,
        )
    return RecoveryVerdict.passed(chain)


def global_kill_on_invalid_recovery(
    kill_switch: KillSwitch,
    verdict: RecoveryVerdict,
    *,
    operator: str = "recovery",
    reason_code: str = "RECOVERY_POINT_INVALID",
) -> bool:
    """Activează `Kill_Switch(GLOBAL)` dacă Recovery_Point este invalid (Req 26.4).

    Întoarce `True` dacă a activat kill switch-ul (verdict negativ). Blocarea este sincronă
    în memorie (vezi `safety/kill_switch.py`), deci efectul este sub 1 s.
    """
    if verdict.ok:
        return False
    kill_switch.activate_automatic(
        KillSwitchScope.GLOBAL,
        component="recovery",
        reason_code=reason_code,
        detail=verdict.reason or "Recovery_Point invalid",
    )
    return True


# --------------------------------------------------------------------------- reconstrucție


@dataclass(frozen=True, slots=True)
class RecoveredState:
    """Rezultatul recuperării: proiecții reconstruite și punctul de reluare (Req 26.1–26.5)."""

    head: tuple[int, str]
    registry: IdempotencyRegistry
    order_states: dict[str, OrderState]
    order_instruments: dict[str, str]
    portfolio: PortfolioState
    pending_orders: tuple[str, ...]
    frozen_instruments: frozenset[str]
    resume_seq: int | None
    verdict: RecoveryVerdict

    @property
    def recovery_complete(self) -> bool:
        """Recuperarea e completă dacă nu rămân ordine în așteptare (Req 26.5)."""
        return self.verdict.ok and not self.pending_orders


@dataclass
class _Rebuild:
    registry: IdempotencyRegistry = field(default_factory=IdempotencyRegistry)
    order_states: dict[str, OrderState] = field(default_factory=dict)
    order_instruments: dict[str, str] = field(default_factory=dict)
    portfolio: PortfolioState = field(default_factory=lambda: PortfolioState.initial(Decimal(0)))
    # Pentru fiecare ordin, seq-ul la care a intrat *prima dată* în SUBMITTED/UNKNOWN. Punctul
    # de reluare (Req 26.3) este minimul peste ordinele rămase neconfirmate la final.
    first_submit_seq: dict[str, int] = field(default_factory=dict)


def _costs_from(payload: dict[str, Any]) -> CostBreakdown:
    raw = payload.get("fill_costs")
    if not raw:
        return CostBreakdown()
    return CostBreakdown.model_validate(raw)


def _rebuild(
    journal: Journal, *, up_to_seq: int | None = None, initial_cash: Decimal = Decimal(0)
) -> _Rebuild:
    """Reconstruiește proiecțiile din înregistrările `oms.*` până la `up_to_seq` inclusiv."""
    rebuild = _Rebuild(portfolio=PortfolioState.initial(initial_cash))
    for record in journal.read():
        if up_to_seq is not None and record.seq > up_to_seq:
            break
        if not record.type.startswith(_OMS_PREFIX):
            continue
        oms_type = record.type[len(_OMS_PREFIX) :]
        _apply_oms_record(
            rebuild, oms_type, record.correlation_id, record.payload, record.seq, record.ts
        )
    return rebuild


def _apply_oms_record(
    rebuild: _Rebuild,
    oms_type: str,
    coid: str,
    payload: dict[str, Any],
    seq: int,
    ts: datetime,
) -> None:
    order = payload.get("order")

    if oms_type == RecordType.ORDER_CREATED.value and order is not None:
        rebuild.registry.register_order(order["client_order_id"])

    # Starea ordinului: înregistrările cu `order` (ORDER_CREATED, EXEC_APPLIED, RECONCILED) o
    # poartă complet; TRANSITION și CANCEL_PENDING_RESTORED poartă numai `to_state` pentru
    # ordinul din `correlation_id`. Instrumentul se cunoaște deja din ORDER_CREATED.
    new_state: OrderState | None = None
    if order is not None:
        rebuild.order_states[order["client_order_id"]] = OrderState(order["state"])
        rebuild.order_instruments[order["client_order_id"]] = order["instrument"]
        new_state = OrderState(order["state"])
    elif "to_state" in payload and coid:
        new_state = OrderState(payload["to_state"])
        rebuild.order_states[coid] = new_state

    if oms_type == RecordType.EXEC_APPLIED.value and order is not None:
        broker_exec_id = payload["broker_exec_id"]
        if rebuild.registry.applied(broker_exec_id) is None:
            rebuild.registry.mark_applied(
                AppliedExec(broker_exec_id, order["client_order_id"], seq, None)
            )
            _replay_fill_at(rebuild, order, payload, ts)

    if oms_type == RecordType.RECONCILED.value:
        for exec_id in payload.get("reflected_exec_ids", ()):
            if rebuild.registry.applied(exec_id) is None:
                rebuild.registry.mark_applied(AppliedExec(exec_id, coid, seq, None))

    # Reține seq-ul primei intrări în SUBMITTED/UNKNOWN pe ordin; resume_seq se alege la final
    # numai dintre ordinele rămase neconfirmate (un ordin ajuns terminal nu mai e neconfirmat).
    if (
        new_state in (OrderState.SUBMITTED, OrderState.UNKNOWN)
        and coid
        and coid not in rebuild.first_submit_seq
    ):
        rebuild.first_submit_seq[coid] = seq


def _replay_fill_at(
    rebuild: _Rebuild, order: dict[str, Any], payload: dict[str, Any], ts: datetime
) -> None:
    qty = payload.get("exec_qty")
    price = payload.get("exec_price")
    if qty is None or price is None:
        return
    fill = PortfolioFill(
        fill_id=payload["broker_exec_id"],
        instrument=order["instrument"],
        side=order["side"],
        qty=_dec(qty),
        price=_dec(price),
        ts=ensure_utc(ts),
        costs=_costs_from(payload),
    )
    rebuild.portfolio = apply_fill(rebuild.portfolio, fill)


def recover(
    journal: Journal,
    *,
    checkpoint: Checkpoint | None = None,
    initial_cash: Decimal = Decimal(0),
    clock: Clock | None = None,
) -> RecoveredState:
    """Reconstruiește starea din jurnal și întoarce punctul de reluare (Req 26.1–26.5, 26.8).

    `initial_cash` este numerarul inițial al portofoliului rulării (de obicei
    `risk.reference_capital_eur`), peste care se rejoacă fill-urile aplicate.
    `clock` este opțional și nu este necesar pentru reconstrucție; e acceptat pentru simetrie
    cu restul componentelor.
    """
    verdict = validate_integrity(journal, checkpoint)
    rebuild = _rebuild(journal, initial_cash=initial_cash)

    pending = tuple(
        coid for coid, state in sorted(rebuild.order_states.items()) if state in PENDING_STATES
    )
    frozen = frozenset(
        rebuild.order_instruments[coid] for coid in pending if coid in rebuild.order_instruments
    )
    # Reluare de la primul eveniment neconfirmat (Req 26.1, 26.3): cel mai mic seq la care un
    # ordin încă în așteptare a intrat în SUBMITTED/UNKNOWN. Fără ordine în așteptare: None.
    resume_candidates = [
        rebuild.first_submit_seq[coid] for coid in pending if coid in rebuild.first_submit_seq
    ]
    resume_seq = min(resume_candidates) if resume_candidates else None
    return RecoveredState(
        head=journal.head,
        registry=rebuild.registry,
        order_states=dict(rebuild.order_states),
        order_instruments=dict(rebuild.order_instruments),
        portfolio=rebuild.portfolio,
        pending_orders=pending,
        frozen_instruments=frozen,
        resume_seq=resume_seq,
        verdict=verdict,
    )
