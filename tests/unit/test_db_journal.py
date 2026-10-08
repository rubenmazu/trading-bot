"""Teste unitare pentru `persistence/db.py` și `persistence/journal.py`.

Req 24.1, 24.3, 24.4, 26.1.
"""

import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from qts.persistence.db import MIGRATIONS, migrate, open_db
from qts.persistence.journal import GENESIS_HASH, Journal, hash_fields, record_hash
from qts.secrets.store import Redactor

T0 = datetime(2026, 1, 1, 9, tzinfo=UTC)


def _kwargs(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ts": T0,
        "type": "signal",
        "correlation_id": "c1",
        "component": "strategy",
        "component_version": "1.0",
        "actor": "system",
        "outcome": "ok",
        "payload": {"value": "1.5"},
    }
    base.update(over)
    return base


def test_pragmas_and_schema_version(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "db" / "x.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    assert migrate(conn) == len(MIGRATIONS)  # idempotent
    conn.close()


def test_newer_schema_is_rejected() -> None:
    conn = open_db(":memory:")
    conn.execute(f"PRAGMA user_version={len(MIGRATIONS) + 1}")
    with pytest.raises(RuntimeError, match="mai nouă"):
        migrate(conn)


def test_seq_monotonic_and_hash_chain() -> None:
    j = Journal(open_db(":memory:"), Redactor())
    assert j.head == (0, GENESIS_HASH)
    records = [
        j.append(**_kwargs(ts=T0 + timedelta(seconds=i), payload={"n": i})) for i in range(4)
    ]
    assert [r.seq for r in records] == [1, 2, 3, 4]
    prev = GENESIS_HASH
    for r in records:
        assert r.prev_hash == prev
        assert r.hash == record_hash(prev, hash_fields(r))
        prev = r.hash
    assert j.head == (4, prev)
    assert list(j.read()) == records
    assert list(j.read(from_seq=3)) == records[2:]


def test_record_contains_required_fields_in_utc() -> None:
    j = Journal(open_db(":memory:"), Redactor())
    local = datetime(2026, 1, 1, 11, tzinfo=timezone(timedelta(hours=2)))
    rec = j.append(**_kwargs(ts=local, correlation_id="d7"))
    assert rec.ts == T0 and rec.ts.tzinfo is not None
    stored = j.by_correlation("d7")
    assert stored == [rec]
    assert (rec.type, rec.component_version, rec.actor, rec.outcome) == (
        "signal",
        "1.0",
        "system",
        "ok",
    )


@pytest.mark.parametrize(
    "field", ["type", "correlation_id", "component", "component_version", "actor", "outcome"]
)
def test_empty_required_field_rejected(field: str) -> None:
    j = Journal(open_db(":memory:"), Redactor())
    with pytest.raises(ValueError, match=field):
        j.append(**_kwargs(**{field: " "}))
    assert j.head == (0, GENESIS_HASH)


def test_append_only_triggers() -> None:
    conn = open_db(":memory:")
    j = Journal(conn, Redactor())
    j.append(**_kwargs())
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE journal SET payload='{}'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM journal")


def test_transaction_groups_writes_atomically() -> None:
    conn = open_db(":memory:")
    j = Journal(conn, Redactor())
    with j.transaction():
        j.append(**_kwargs(payload={"n": 1}))
        j.append(**_kwargs(payload={"n": 2}))
    assert j.head[0] == 2
    with pytest.raises(RuntimeError), j.transaction():
        j.append(**_kwargs(payload={"n": 3}))
        raise RuntimeError("abort")
    assert conn.execute("SELECT COUNT(*) FROM journal").fetchone()[0] == 2
    assert j.head[0] == 2
