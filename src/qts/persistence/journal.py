"""Jurnal append-only cu lanț SHA-256 (Req 24.1–24.4, 26.1).

Fiecare înregistrare conține `prev_hash`, iar `hash = SHA256(prev_hash ‖ conținut_canonic)`.
Modificarea, ștergerea sau reordonarea oricărei înregistrări rupe lanțul, iar ruptura este
detectată de `qts.persistence.audit.verify_chain`.

Jurnalul refuză orice payload care conține o valoare secretă cunoscută (Req 23.7).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Final

from qts.core.clock import ensure_utc
from qts.core.models import AuditRecord
from qts.secrets.store import REDACTOR, Redactor

GENESIS_HASH: Final = "0" * 64


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def record_hash(prev_hash: str, fields: dict[str, Any]) -> str:
    """Hash-ul unei înregistrări, calculat din toate câmpurile mai puțin `hash`."""
    body = _canonical({k: v for k, v in fields.items() if k != "hash"})
    return hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()


def _row_to_record(row: sqlite3.Row) -> AuditRecord:
    return AuditRecord(
        seq=row["seq"],
        ts=datetime.fromisoformat(row["ts"]),
        type=row["type"],
        correlation_id=row["correlation_id"],
        component=row["component"],
        component_version=row["component_version"],
        actor=row["actor"],
        outcome=row["outcome"],
        payload=json.loads(row["payload"]),
        prev_hash=row["prev_hash"],
        hash=row["hash"],
    )


def hash_fields(record: AuditRecord) -> dict[str, Any]:
    """Câmpurile, în forma exactă folosită la calculul hash-ului."""
    return {
        "seq": record.seq,
        "ts": record.ts.isoformat(),
        "type": record.type,
        "correlation_id": record.correlation_id,
        "component": record.component,
        "component_version": record.component_version,
        "actor": record.actor,
        "outcome": record.outcome,
        "payload": record.payload,
        "prev_hash": record.prev_hash,
    }


class Journal:
    def __init__(self, conn: sqlite3.Connection, redactor: Redactor = REDACTOR) -> None:
        self._conn = conn
        self._redactor = redactor
        row = conn.execute("SELECT seq, hash FROM journal ORDER BY seq DESC LIMIT 1").fetchone()
        self._last_seq: int = row["seq"] if row else 0
        self._last_hash: str = row["hash"] if row else GENESIS_HASH

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @property
    def head(self) -> tuple[int, str]:
        return self._last_seq, self._last_hash

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Tranzacție atomică pentru jurnal și proiecții. Suportă imbricarea (fără efect)."""
        if self._conn.in_transaction:
            yield self._conn
            return
        saved = (self._last_seq, self._last_hash)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            self._last_seq, self._last_hash = saved
            raise
        else:
            self._conn.execute("COMMIT")

    def append(
        self,
        *,
        ts: datetime,
        type: str,
        correlation_id: str,
        component: str,
        component_version: str,
        actor: str,
        outcome: str,
        payload: dict[str, Any],
    ) -> AuditRecord:
        where = f"jurnal ({type})"
        # Req 24.3: fiecare Audit_Record are tip, corelare, versiune, actor și rezultat.
        required = {
            "type": type,
            "correlation_id": correlation_id,
            "component": component,
            "component_version": component_version,
            "actor": actor,
            "outcome": outcome,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"{where}: câmpuri obligatorii goale: {', '.join(missing)}")
        self._redactor.assert_clean([payload, correlation_id, actor, outcome], where)
        payload_json = _canonical(payload)
        self._redactor.assert_clean(payload_json, where)
        normalized_payload: dict[str, Any] = json.loads(payload_json)
        seq = self._last_seq + 1
        fields: dict[str, Any] = {
            "seq": seq,
            "ts": ensure_utc(ts).isoformat(),
            "type": type,
            "correlation_id": correlation_id,
            "component": component,
            "component_version": component_version,
            "actor": actor,
            "outcome": outcome,
            "payload": normalized_payload,
            "prev_hash": self._last_hash,
        }
        digest = record_hash(self._last_hash, fields)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO journal (seq, ts, type, correlation_id, component, "
                "component_version, actor, outcome, payload, prev_hash, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    seq,
                    fields["ts"],
                    type,
                    correlation_id,
                    component,
                    component_version,
                    actor,
                    outcome,
                    payload_json,
                    fields["prev_hash"],
                    digest,
                ),
            )
            self._last_seq, self._last_hash = seq, digest
        return AuditRecord(hash=digest, **{**fields, "ts": ensure_utc(ts)})

    def read(self, from_seq: int = 1) -> Iterator[AuditRecord]:
        cursor = self._conn.execute(
            "SELECT * FROM journal WHERE seq >= ? ORDER BY seq", (from_seq,)
        )
        for row in cursor:
            yield _row_to_record(row)

    def by_correlation(self, correlation_id: str) -> list[AuditRecord]:
        rows = self._conn.execute(
            "SELECT * FROM journal WHERE correlation_id = ? ORDER BY seq", (correlation_id,)
        ).fetchall()
        return [_row_to_record(r) for r in rows]
