import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from qts.persistence.audit import (
    export_journal,
    reconstruct_decision,
    verify_export,
    verify_journal,
)
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.secrets.store import Redactor, SecretLeakError

T0 = datetime(2026, 1, 1, 9, tzinfo=UTC)


def _append(j: Journal, type_: str, corr: str = "c1", **payload: object) -> None:
    j.append(
        ts=T0,
        type=type_,
        correlation_id=corr,
        component="test",
        component_version="1",
        actor="system",
        outcome="ok",
        payload=dict(payload),
    )


def test_wal_and_append_only(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "x.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    j = Journal(conn, Redactor())
    _append(j, "signal", value="1.5")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE journal SET outcome='x'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM journal")


def test_chain_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    j = Journal(open_db(path), Redactor())
    _append(j, "a")
    _append(j, "b")
    head = j.head
    j.connection.close()
    j2 = Journal(open_db(path), Redactor())
    assert j2.head == head
    _append(j2, "c")
    assert verify_journal(j2).ok
    assert verify_journal(j2, expected_head=head).ok


def test_rollback_keeps_head_consistent() -> None:
    j = Journal(open_db(":memory:"), Redactor())
    _append(j, "a")
    head = j.head
    with pytest.raises(RuntimeError), j.transaction():
        _append(j, "b")
        raise RuntimeError("abort")
    assert j.head == head
    _append(j, "c")
    assert verify_journal(j).ok


def test_secret_in_payload_is_blocked() -> None:
    redactor = Redactor()
    redactor.register("super-secret-token")
    j = Journal(open_db(":memory:"), redactor)
    with pytest.raises(SecretLeakError):
        _append(j, "a", note="x super-secret-token y")
    assert j.head[0] == 0


def test_export_and_verify(tmp_path: Path) -> None:
    j = Journal(open_db(":memory:"), Redactor())
    for i in range(5):
        _append(j, "signal", n=i)
    out = tmp_path / "audit.jsonl"
    assert export_journal(j, out).records_checked == 5
    assert verify_export(out).ok
    lines = out.read_text(encoding="utf-8").splitlines()
    lines[3] = lines[3].replace('"n":2', '"n":9')
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not verify_export(out).ok


def test_reconstruct_decision_orders_chain() -> None:
    j = Journal(open_db(":memory:"), Redactor())
    _append(j, "execution_event", corr="d1")
    _append(j, "signal", corr="d1")
    _append(j, "signal", corr="other")
    _append(j, "risk_decision", corr="d1")
    _append(j, "market_event", corr="d1")
    chain = [r.type for r in reconstruct_decision(j, "d1")]
    assert chain == ["market_event", "signal", "risk_decision", "execution_event"]


def test_truncation_detected_with_known_head() -> None:
    conn = open_db(":memory:")
    j = Journal(conn, Redactor())
    for i in range(3):
        _append(j, "a", n=i)
    head = j.head
    conn.execute("DROP TRIGGER journal_no_delete")
    conn.execute("DELETE FROM journal WHERE seq = 3")
    assert verify_journal(Journal(conn, Redactor())).ok  # fără head extern nu se poate vedea
    assert not verify_journal(Journal(conn, Redactor()), expected_head=head).ok
