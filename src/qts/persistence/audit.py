"""Verificarea integrității, exportul și reconstrucția deciziilor (Req 24.4, 24.5, 24.7)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from qts.core.models import AuditRecord
from qts.persistence.journal import GENESIS_HASH, Journal, hash_fields, record_hash

EXPORT_SCHEMA_VERSION: Final = "1"

# Ordinea etapelor în lanțul unei decizii (Req 24.5).
DECISION_CHAIN_TYPES: Final = (
    "market_event",
    "signal",
    "order_intent",
    "risk_decision",
    "broker_request",
    "execution_event",
)


@dataclass(frozen=True)
class ChainVerification:
    ok: bool
    records_checked: int
    head_hash: str
    first_bad_seq: int | None = None
    reason: str | None = None


def verify_chain(records: Iterable[AuditRecord]) -> ChainVerification:
    """Verifică secvența continuă de la 1, legătura `prev_hash` și hash-ul recalculat."""
    expected_seq = 1
    prev = GENESIS_HASH
    count = 0
    for rec in records:
        count += 1
        if rec.seq != expected_seq:
            return ChainVerification(
                False, count, prev, rec.seq, f"secvență întreruptă: așteptat {expected_seq}"
            )
        if rec.prev_hash != prev:
            return ChainVerification(False, count, prev, rec.seq, "prev_hash nu corespunde")
        if record_hash(prev, hash_fields(rec)) != rec.hash:
            return ChainVerification(False, count, prev, rec.seq, "hash recalculat diferit")
        prev = rec.hash
        expected_seq += 1
    return ChainVerification(True, count, prev)


def verify_journal(
    journal: Journal, expected_head: tuple[int, str] | None = None
) -> ChainVerification:
    """Verifică jurnalul; opțional îl compară cu un head cunoscut (detectează trunchierea)."""
    result = verify_chain(journal.read())
    if result.ok and expected_head is not None:
        seq, digest = expected_head
        if result.records_checked < seq:
            return ChainVerification(
                False,
                result.records_checked,
                result.head_hash,
                result.records_checked + 1,
                "jurnal trunchiat față de head-ul cunoscut",
            )
        records = list(journal.read(from_seq=seq))
        if not records or records[0].hash != digest:
            return ChainVerification(
                False, result.records_checked, result.head_hash, seq, "head-ul cunoscut diferă"
            )
    return result


HASH_ALGORITHM: Final = "sha256(prev_hash || canonical_json(fields))"
_HEADER_KEYS: Final = frozenset(
    {
        "export_schema_version",
        "db_schema_version",
        "record_count",
        "head_hash",
        "genesis_hash",
        "hash_algorithm",
    }
)


def _db_schema_version(journal: Journal) -> int:
    return int(journal.connection.execute("PRAGMA user_version").fetchone()[0])


def export_journal(journal: Journal, path: Path) -> ChainVerification:
    """Exportă JSONL (Req 24.7): un antet cu versiunile schemei și dovada integrității
    (număr de înregistrări, hash-ul head-ului, algoritmul), apoi înregistrările în ordinea `seq`.

    Exportul este refuzat dacă jurnalul nu trece verificarea lanțului.
    """
    records = list(journal.read())
    verification = verify_chain(records)
    if not verification.ok:
        raise RuntimeError(f"export refuzat: jurnal corupt la seq {verification.first_bad_seq}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        header = {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "db_schema_version": _db_schema_version(journal),
            "record_count": verification.records_checked,
            "head_hash": verification.head_hash,
            "genesis_hash": GENESIS_HASH,
            "hash_algorithm": HASH_ALGORITHM,
        }
        fh.write(json.dumps(header, sort_keys=True) + "\n")
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")
    return verification


def _bad_export(
    reason: str, checked: int = 0, first_bad_seq: int | None = None
) -> ChainVerification:
    return ChainVerification(False, checked, GENESIS_HASH, first_bad_seq, reason)


def verify_export(path: Path) -> ChainVerification:
    """Verifică un export independent de baza de date: antet, lanț și dovada din antet."""
    with path.open(encoding="utf-8") as fh:
        try:
            header = json.loads(fh.readline())
        except json.JSONDecodeError:
            return _bad_export("antet ilizibil")
        if not isinstance(header, dict) or not header.keys() >= _HEADER_KEYS:
            return _bad_export("antet incomplet")
        if header["export_schema_version"] != EXPORT_SCHEMA_VERSION:
            return _bad_export(f"versiune export necunoscută: {header['export_schema_version']}")
        if header["genesis_hash"] != GENESIS_HASH or header["hash_algorithm"] != HASH_ALGORITHM:
            return _bad_export("algoritm sau genesis necunoscut")
        records: list[AuditRecord] = []
        for line in fh:
            if not line.strip():
                continue
            try:
                records.append(AuditRecord.model_validate_json(line))
            except ValueError:
                return _bad_export("înregistrare ilizibilă", len(records), len(records) + 1)
    result = verify_chain(records)
    if result.ok and (
        result.records_checked != header["record_count"] or result.head_hash != header["head_hash"]
    ):
        return ChainVerification(
            False, result.records_checked, result.head_hash, None, "antetul nu corespunde"
        )
    return result


# Etapele obligatorii; `broker_request` este opțional (nu apare în simulare/backtest).
REQUIRED_DECISION_TYPES: Final = (
    "market_event",
    "signal",
    "order_intent",
    "risk_decision",
    "execution_event",
)
# După o respingere de risc nu urmează cerere la broker și nici Execution_Event.
_POST_RISK_TYPES: Final = frozenset({"broker_request", "execution_event"})


@dataclass(frozen=True)
class DecisionChain:
    correlation_id: str
    records: tuple[AuditRecord, ...]
    missing: tuple[str, ...]
    risk_rejected: bool

    @property
    def complete(self) -> bool:
        return not self.missing


def _is_rejection(outcome: str) -> bool:
    return "reject" in outcome.lower()


def reconstruct_decision(journal: Journal, correlation_id: str) -> list[AuditRecord]:
    """Lanțul Market_Event–Signal–Order_Intent–risc–Execution_Event pentru o decizie."""
    order = {t: i for i, t in enumerate(DECISION_CHAIN_TYPES)}
    records = [r for r in journal.by_correlation(correlation_id) if r.type in order]
    return sorted(records, key=lambda r: (order[r.type], r.seq))


def reconstruct_decision_chain(journal: Journal, correlation_id: str) -> DecisionChain:
    """Reconstrucția completă (Req 24.5), cu etapele lipsă raportate explicit.

    Dacă decizia de risc este o respingere, etapele de după risc nu sunt cerute.
    """
    records = reconstruct_decision(journal, correlation_id)
    present = {r.type for r in records}
    risk_rejected = any(r.type == "risk_decision" and _is_rejection(r.outcome) for r in records)
    required = [t for t in REQUIRED_DECISION_TYPES if not (risk_rejected and t in _POST_RISK_TYPES)]
    missing = tuple(t for t in required if t not in present)
    return DecisionChain(correlation_id, tuple(records), missing, risk_rejected)
