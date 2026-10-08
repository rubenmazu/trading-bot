"""Teste pentru `qts.persistence.audit` (Req 24.4, 24.5, 24.7)."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from qts.persistence.audit import (
    EXPORT_SCHEMA_VERSION,
    export_journal,
    reconstruct_decision_chain,
    verify_chain,
    verify_export,
)
from qts.persistence.db import MIGRATIONS, open_db
from qts.persistence.journal import GENESIS_HASH, Journal
from qts.secrets.store import Redactor

T0 = datetime(2026, 1, 1, 9, tzinfo=UTC)


def _append(
    j: Journal, type_: str, corr: str = "c1", outcome: str = "ok", **payload: object
) -> None:
    j.append(
        ts=T0,
        type=type_,
        correlation_id=corr,
        component="test",
        component_version="1",
        actor="system",
        outcome=outcome,
        payload=dict(payload),
    )


def _journal(n: int) -> tuple[sqlite3.Connection, Journal]:
    conn = open_db(":memory:")
    j = Journal(conn, Redactor())
    for i in range(n):
        _append(j, "signal", n=i)
    return conn, j


def test_verify_chain_empty_and_valid() -> None:
    empty = verify_chain([])
    assert empty.ok and empty.records_checked == 0 and empty.head_hash == GENESIS_HASH
    _conn, j = _journal(4)
    result = verify_chain(j.read())
    assert result.ok and result.records_checked == 4 and result.head_hash == j.head[1]


def test_verify_chain_reports_first_broken_seq() -> None:
    conn, j = _journal(5)
    conn.execute("DROP TRIGGER journal_no_update")
    conn.execute("UPDATE journal SET outcome='tampered' WHERE seq=3")
    result = verify_chain(j.read())
    assert not result.ok and result.first_bad_seq == 3


def test_verify_chain_reports_deletion_gap() -> None:
    conn, j = _journal(5)
    conn.execute("DROP TRIGGER journal_no_delete")
    conn.execute("DELETE FROM journal WHERE seq=2")
    result = verify_chain(j.read())
    assert not result.ok and result.first_bad_seq == 3


def test_export_header_has_schema_version_and_proof(tmp_path: Path) -> None:
    _conn, j = _journal(3)
    out = tmp_path / "exp" / "audit.jsonl"
    export_journal(j, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["export_schema_version"] == EXPORT_SCHEMA_VERSION
    assert header["db_schema_version"] == len(MIGRATIONS)
    assert header["record_count"] == 3
    assert header["head_hash"] == j.head[1]
    assert len(lines) == 4
    assert verify_export(out).ok


def test_export_of_empty_journal_verifies(tmp_path: Path) -> None:
    _conn, j = _journal(0)
    out = tmp_path / "audit.jsonl"
    export_journal(j, out)
    assert verify_export(out).ok


def test_export_refused_for_corrupt_journal(tmp_path: Path) -> None:
    conn, j = _journal(3)
    conn.execute("DROP TRIGGER journal_no_update")
    conn.execute("UPDATE journal SET actor='x' WHERE seq=2")
    with pytest.raises(RuntimeError, match="seq 2"):
        export_journal(j, tmp_path / "audit.jsonl")


def test_verify_export_detects_dropped_tail_and_bad_header(tmp_path: Path) -> None:
    _conn, j = _journal(3)
    out = tmp_path / "audit.jsonl"
    export_journal(j, out)
    lines = out.read_text(encoding="utf-8").splitlines()

    out.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # ultima înregistrare lipsă
    assert not verify_export(out).ok

    header = json.loads(lines[0])
    header["export_schema_version"] = "999"
    out.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8")
    assert not verify_export(out).ok

    out.write_text("\n".join([lines[0], "{not json", *lines[2:]]) + "\n", encoding="utf-8")
    assert not verify_export(out).ok


def test_reconstruct_complete_chain() -> None:
    j = Journal(open_db(":memory:"), Redactor())
    for t in ("execution_event", "order_intent", "market_event", "risk_decision", "signal"):
        _append(j, t, corr="d1")
    _append(j, "signal", corr="d2")
    _append(j, "config_change", corr="d1")  # tip din afara lanțului: ignorat
    chain = reconstruct_decision_chain(j, "d1")
    assert chain.complete and not chain.risk_rejected
    assert [r.type for r in chain.records] == [
        "market_event",
        "signal",
        "order_intent",
        "risk_decision",
        "execution_event",
    ]


def test_reconstruct_reports_missing_stages() -> None:
    j = Journal(open_db(":memory:"), Redactor())
    _append(j, "market_event", corr="d1")
    _append(j, "signal", corr="d1")
    chain = reconstruct_decision_chain(j, "d1")
    assert not chain.complete
    assert chain.missing == ("order_intent", "risk_decision", "execution_event")
    assert reconstruct_decision_chain(j, "absent").missing[0] == "market_event"


def test_reconstruct_risk_rejection_needs_no_execution() -> None:
    j = Journal(open_db(":memory:"), Redactor())
    for t in ("market_event", "signal", "order_intent"):
        _append(j, t, corr="d1")
    _append(j, "risk_decision", corr="d1", outcome="REJECTED_RISK")
    chain = reconstruct_decision_chain(j, "d1")
    assert chain.risk_rejected and chain.complete
