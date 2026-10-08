"""P12: orice modificare, ștergere sau reordonare în jurnal este detectată (Req 24.4).

Jurnalele sunt generate arbitrar (lungimi, payload-uri, timpi), apoi li se aplică o mutație
arbitrară: modificarea oricărui câmp inclus în hash (inclusiv `seq`, `prev_hash`, `hash`),
ștergerea oricărei înregistrări (inclusiv a ultimei: trunchiere), reordonarea și inserarea.
Atacatorul poate și să recalculeze hash-ul înregistrării modificate sau al întregului lanț
(„resigilare”); atunci detecția se face față de head-ul cunoscut (`expected_head`) și față de
dovada din antetul exportului. Mutația este verificată atât în baza de date (`verify_journal`),
cât și în fișierul de export (`verify_export`). Jurnalele nemodificate trebuie să treacă.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from qts.core.models import AuditRecord
from qts.persistence.audit import export_journal, verify_export, verify_journal
from qts.persistence.db import open_db
from qts.persistence.journal import GENESIS_HASH, Journal, hash_fields, record_hash
from qts.secrets.store import Redactor

T0 = datetime(2026, 1, 1, tzinfo=UTC)

CONTENT_FIELDS = (
    "ts",
    "type",
    "correlation_id",
    "component",
    "component_version",
    "actor",
    "outcome",
    "payload",
)
CHAIN_FIELDS = ("seq", "prev_hash", "hash")

json_scalars = st.one_of(
    st.none(), st.booleans(), st.integers(-(10**12), 10**12), st.text(max_size=10)
)
payloads = st.dictionaries(
    st.text(min_size=1, max_size=6),
    st.one_of(json_scalars, st.lists(json_scalars, max_size=3)),
    max_size=4,
)
names = st.text(min_size=1, max_size=12).filter(lambda s: s.strip() != "")
hex64 = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)


@st.composite
def entries(draw: st.DrawFn) -> dict[str, Any]:
    return {
        "type": draw(st.sampled_from(["market_event", "signal", "order_intent", "evt"])),
        "actor": draw(st.sampled_from(["system", "operator"])),
        "outcome": draw(names),
        "payload": draw(payloads),
        "gap_us": draw(st.integers(0, 10**9)),
    }


journals = st.lists(entries(), min_size=1, max_size=12)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _build(items: list[dict[str, Any]]) -> Journal:
    journal = Journal(open_db(":memory:"), Redactor())
    ts = T0
    for i, item in enumerate(items):
        ts += timedelta(microseconds=item["gap_us"])
        journal.append(
            ts=ts,
            type=item["type"],
            correlation_id=f"c{i}",
            component="t",
            component_version="1",
            actor=item["actor"],
            outcome=item["outcome"],
            payload=item["payload"],
        )
    return journal


def _store(records: list[AuditRecord]) -> Journal | None:
    """Scrie înregistrările (mutate) direct într-o bază nouă, ocolind protecțiile append-only.

    Întoarce None dacă mutația nu poate exista în tabel (de ex. `seq`/`hash` duplicat).
    """
    conn = open_db(":memory:")
    conn.execute("DROP TRIGGER journal_no_update")
    conn.execute("DROP TRIGGER journal_no_delete")
    try:
        for r in records:
            conn.execute(
                "INSERT INTO journal (seq, ts, type, correlation_id, component, "
                "component_version, actor, outcome, payload, prev_hash, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r.seq,
                    r.ts.isoformat(),
                    r.type,
                    r.correlation_id,
                    r.component,
                    r.component_version,
                    r.actor,
                    r.outcome,
                    _canonical(r.payload),
                    r.prev_hash,
                    r.hash,
                ),
            )
    except sqlite3.IntegrityError:
        return None
    return Journal(conn, Redactor())


def _rehash(rec: AuditRecord) -> AuditRecord:
    return rec.model_copy(update={"hash": record_hash(rec.prev_hash, hash_fields(rec))})


def _reseal(records: list[AuditRecord]) -> list[AuditRecord]:
    """Atacatorul renumerotează și recalculează întregul lanț."""
    out: list[AuditRecord] = []
    prev = GENESIS_HASH
    for i, rec in enumerate(records, start=1):
        sealed = _rehash(rec.model_copy(update={"seq": i, "prev_hash": prev}))
        out.append(sealed)
        prev = sealed.hash
    return out


def _fingerprint(records: list[AuditRecord]) -> list[tuple[str, str]]:
    return [(_canonical(hash_fields(r)), r.hash) for r in records]


def _assert_detected(
    original: Journal, mutated: list[AuditRecord], *, db_representable: bool = True
) -> None:
    records = list(original.read())
    assert _fingerprint(mutated) != _fingerprint(records), "mutația generată este identitate"
    head = original.head

    if db_representable:
        tampered = _store(mutated)
        if tampered is not None:
            assert not verify_journal(tampered, expected_head=head).ok

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "export.jsonl"
        export_journal(original, path)
        header = path.read_text(encoding="utf-8").splitlines()[0]
        lines = [header, *(r.model_dump_json() for r in mutated)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        assert not verify_export(path).ok


def _new_value(data: st.DataObject, rec: AuditRecord, field: str) -> Any:
    old = getattr(rec, field)
    if field == "ts":
        delta = data.draw(st.integers(-(10**9), 10**9).filter(lambda d: d != 0))
        return old + timedelta(microseconds=delta)
    if field == "payload":
        return data.draw(payloads.filter(lambda p: _canonical(p) != _canonical(old)))
    if field == "seq":
        return data.draw(st.integers(-3, 20).filter(lambda s: s != old))
    if field in ("prev_hash", "hash"):
        return data.draw(st.one_of(hex64, st.text(max_size=70)).filter(lambda s: s != old))
    return data.draw(st.text(max_size=20).filter(lambda s: s != old))


@given(journals)
def test_property_12_unmodified_journal_verifies(items: list[dict[str, Any]]) -> None:
    journal = _build(items)
    head = journal.head
    assert verify_journal(journal).ok
    assert verify_journal(journal, expected_head=head).ok
    copy = _store(list(journal.read()))
    assert copy is not None
    assert verify_journal(copy, expected_head=head).ok
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "export.jsonl"
        export_journal(journal, path)
        result = verify_export(path)
        assert result.ok
        assert (result.records_checked, result.head_hash) == head


@given(journals, st.data())
def test_property_12_modification_detected(
    items: list[dict[str, Any]], data: st.DataObject
) -> None:
    journal = _build(items)
    records = list(journal.read())
    idx = data.draw(st.integers(0, len(records) - 1), label="idx")
    field = data.draw(st.sampled_from(CONTENT_FIELDS + CHAIN_FIELDS), label="field")
    value = _new_value(data, records[idx], field)
    # Câmpurile de lanț ar fi refăcute de resigilare, deci se modifică doar „brut”.
    modes = ["raw", "rehash_record", "reseal"] if field in CONTENT_FIELDS else ["raw"]
    mode = data.draw(st.sampled_from(modes), label="mode")

    changed = records[idx].model_copy(update={field: value})
    if mode == "rehash_record":
        changed = _rehash(changed)
    mutated = [*records[:idx], changed, *records[idx + 1 :]]
    if mode == "reseal":
        mutated = _reseal(mutated)
    _assert_detected(journal, mutated)


@given(journals, st.data())
def test_property_12_deletion_detected(items: list[dict[str, Any]], data: st.DataObject) -> None:
    journal = _build(items)
    records = list(journal.read())
    idx = data.draw(st.integers(0, len(records) - 1), label="idx")  # inclusiv ultima: trunchiere
    mutated = [*records[:idx], *records[idx + 1 :]]
    if data.draw(st.booleans(), label="reseal"):
        mutated = _reseal(mutated)
    _assert_detected(journal, mutated)


@given(st.lists(entries(), min_size=2, max_size=12), st.data())
def test_property_12_reordering_detected(items: list[dict[str, Any]], data: st.DataObject) -> None:
    journal = _build(items)
    records = list(journal.read())
    n = len(records)
    perm = data.draw(
        st.permutations(range(n)).filter(lambda p: list(p) != list(range(n))), label="perm"
    )
    moved = [records[i] for i in perm]
    mode = data.draw(st.sampled_from(["renumber", "reseal", "lines_only"]), label="mode")
    if mode == "lines_only":
        # Doar ordinea liniilor din export; în bază ordinea este dată de `seq`.
        _assert_detected(journal, moved, db_representable=False)
        return
    mutated = [r.model_copy(update={"seq": i}) for i, r in enumerate(moved, start=1)]
    if mode == "reseal":
        mutated = _reseal(mutated)
    _assert_detected(journal, mutated)


@given(journals, entries(), st.data())
def test_property_12_insertion_detected(
    items: list[dict[str, Any]], extra: dict[str, Any], data: st.DataObject
) -> None:
    journal = _build(items)
    records = list(journal.read())
    # Inserarea după ultima înregistrare este o adăugare legitimă, deci poziții < n.
    pos = data.draw(st.integers(0, len(records) - 1), label="pos")
    template = records[pos]
    forged = _rehash(
        template.model_copy(
            update={
                "seq": pos + 1,
                "prev_hash": records[pos - 1].hash if pos > 0 else GENESIS_HASH,
                "correlation_id": "forged",
                "outcome": extra["outcome"],
                "payload": extra["payload"],
            }
        )
    )
    shifted = [r.model_copy(update={"seq": r.seq + 1}) for r in records[pos:]]
    mutated = [*records[:pos], forged, *shifted]
    if data.draw(st.booleans(), label="reseal"):
        mutated = _reseal(mutated)
    _assert_detected(journal, mutated)
