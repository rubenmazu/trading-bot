"""Teste pentru `research/partition.py` (Req 18.1–18.6).

Acoperă: separarea cronologică Development/OOS, rezervarea + prima consumare, refuzul celei de-a
doua consumări (consum unic), calea de reclasificare și persistența stării peste reconectare.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from qts.persistence.db import open_db
from qts.research.partition import (
    DataPartitioner,
    OosAlreadyConsumedError,
    OosNotReservedError,
    OosRegistry,
    OosStatus,
    Partition,
    PartitionRole,
    RegistryError,
    partition_hash,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 1, tzinfo=UTC)
INSTRUMENTS = ("AAA", "BBB")


def _split() -> tuple[Partition, Partition]:
    split = DataPartitioner(oos_fraction=0.25).split(
        start_ts=START, end_ts=END, instruments=INSTRUMENTS, label="study"
    )
    return split.development, split.out_of_sample


# --------------------------------------------------------------------------- partiționare


def test_split_is_chronological() -> None:
    # Req 18.1/18.2: Development precede complet Out_Of_Sample, fără suprapunere.
    dev, oos = _split()
    assert dev.role is PartitionRole.DEVELOPMENT
    assert oos.role is PartitionRole.OUT_OF_SAMPLE
    assert dev.start_ts == START
    assert oos.end_ts == END
    assert dev.end_ts == oos.start_ts
    assert dev.start_ts < dev.end_ts <= oos.start_ts < oos.end_ts


def test_split_fraction_controls_cut_point() -> None:
    dev, _ = _split()
    total = (END - START).total_seconds()
    dev_len = (dev.end_ts - dev.start_ts).total_seconds()
    assert dev_len == pytest.approx(total * 0.75)


def test_partition_retains_bounds_identifier_and_role() -> None:
    # Req 18.5: fiecare partiție păstrează limitele temporale, identificatorul și rolul.
    dev, _ = _split()
    assert dev.dataset_id == dev.partition_hash
    assert dev.dataset_id == partition_hash(
        role=dev.role,
        label=dev.label,
        start_ts=dev.start_ts,
        end_ts=dev.end_ts,
        instruments=dev.instruments,
    )


def test_split_is_deterministic() -> None:
    a = DataPartitioner(oos_fraction=0.3).split(
        start_ts=START, end_ts=END, instruments=INSTRUMENTS
    )
    b = DataPartitioner(oos_fraction=0.3).split(
        start_ts=START, end_ts=END, instruments=INSTRUMENTS
    )
    assert a == b


def test_invalid_fraction_rejected() -> None:
    with pytest.raises(ValueError):
        DataPartitioner(oos_fraction=0.0)
    with pytest.raises(ValueError):
        DataPartitioner(oos_fraction=1.0)


def test_split_at_requires_cut_inside_interval() -> None:
    p = DataPartitioner()
    with pytest.raises(ValueError):
        p.split_at(start_ts=START, cut_ts=START, end_ts=END, instruments=INSTRUMENTS)
    with pytest.raises(ValueError):
        p.split_at(start_ts=START, cut_ts=END, end_ts=END, instruments=INSTRUMENTS)


def test_tampered_partition_hash_rejected() -> None:
    dev, _ = _split()
    # Construirea unei partiții cu un câmp schimbat dar hash-ul vechi este respinsă.
    with pytest.raises(ValueError, match="nu corespund"):
        Partition(
            dataset_id=dev.dataset_id,
            partition_hash=dev.partition_hash,
            role=dev.role,
            label="altceva",
            start_ts=dev.start_ts,
            end_ts=dev.end_ts,
            instruments=dev.instruments,
        )


# --------------------------------------------------------------------------- registru


def _registry() -> OosRegistry:
    return OosRegistry(open_db(":memory:"))


def test_reserve_then_first_consume_succeeds() -> None:
    reg = _registry()
    _, oos = _split()
    rec = reg.reserve(oos, decision_id="D1", preregistration_hash="pre1")
    assert rec.status is OosStatus.RESERVED
    assert reg.is_available(oos.dataset_id)

    consumed = reg.consume(oos.dataset_id, evaluation_hash="eval1")
    assert consumed.status is OosStatus.CONSUMED
    assert consumed.evaluation_hash == "eval1"
    assert consumed.consumed_at is not None
    assert not reg.is_available(oos.dataset_id)


def test_second_consume_is_refused() -> None:
    # Req 18.3, 18.6: un OOS poate fi evaluat o singură dată.
    reg = _registry()
    _, oos = _split()
    reg.reserve(oos, decision_id="D1", preregistration_hash="pre1")
    reg.consume(oos.dataset_id, evaluation_hash="eval1")
    with pytest.raises(OosAlreadyConsumedError):
        reg.consume(oos.dataset_id, evaluation_hash="eval2")
    # starea rămâne consumată, nu se corupe
    assert reg.status(oos.dataset_id) is OosStatus.CONSUMED


def test_consume_unreserved_is_refused() -> None:
    reg = _registry()
    _, oos = _split()
    with pytest.raises(OosNotReservedError):
        reg.consume(oos.dataset_id, evaluation_hash="eval1")


def test_reclassification_blocks_further_consumption() -> None:
    # Req 18.4: dacă OOS influențează modificarea strategiei, este reclasificat ca Development.
    reg = _registry()
    _, oos = _split()
    reg.reserve(oos, decision_id="D1", preregistration_hash="pre1")
    rec = reg.reclassify(oos.dataset_id, reason="a influențat alegerea pragului")
    assert rec.status is OosStatus.RECLASSIFIED
    assert rec.reclassified_reason == "a influențat alegerea pragului"
    assert not reg.is_available(oos.dataset_id)
    with pytest.raises(OosNotReservedError):
        reg.consume(oos.dataset_id, evaluation_hash="eval1")


def test_new_oos_reserved_after_reclassification() -> None:
    # Req 18.4: după reclasificare se rezervă un nou OOS ulterior (interval diferit).
    reg = _registry()
    _, oos = _split()
    reg.reserve(oos, decision_id="D1", preregistration_hash="pre1")
    reg.reclassify(oos.dataset_id, reason="contaminat")

    later = DataPartitioner(oos_fraction=0.25).split_at(
        start_ts=START,
        cut_ts=datetime(2025, 6, 1, tzinfo=UTC),
        end_ts=datetime(2026, 1, 1, tzinfo=UTC),
        instruments=INSTRUMENTS,
        label="study-2",
    ).out_of_sample
    assert later.dataset_id != oos.dataset_id
    new_rec = reg.reserve(later, decision_id="D1", preregistration_hash="pre1")
    assert new_rec.status is OosStatus.RESERVED
    assert reg.consume(later.dataset_id, evaluation_hash="eval1").status is OosStatus.CONSUMED


def test_double_reserve_is_refused() -> None:
    reg = _registry()
    _, oos = _split()
    reg.reserve(oos, decision_id="D1", preregistration_hash="pre1")
    with pytest.raises(RegistryError):
        reg.reserve(oos, decision_id="D2", preregistration_hash="pre2")


def test_reserve_requires_oos_role() -> None:
    reg = _registry()
    dev, _ = _split()
    with pytest.raises(RegistryError):
        reg.reserve(dev, decision_id="D1", preregistration_hash="pre1")


def test_consumed_state_survives_reconnect(tmp_path: Path) -> None:
    # Persistența registrului: după reconectare, un OOS consumat rămâne consumat (Req 18.6).
    path = tmp_path / "research.db"
    _, oos = _split()

    reg = OosRegistry(open_db(path))
    reg.reserve(oos, decision_id="D1", preregistration_hash="pre1")
    reg.consume(oos.dataset_id, evaluation_hash="eval1")
    reg.connection.close()

    reg2 = OosRegistry(open_db(path))
    assert reg2.status(oos.dataset_id) is OosStatus.CONSUMED
    with pytest.raises(OosAlreadyConsumedError):
        reg2.consume(oos.dataset_id, evaluation_hash="eval2")
    reg2.connection.close()


def test_datasets_table_is_append_only(tmp_path: Path) -> None:
    import sqlite3

    reg = OosRegistry(open_db(tmp_path / "r.db"))
    _, oos = _split()
    reg.reserve(oos, decision_id="D1", preregistration_hash="pre1")
    with pytest.raises(sqlite3.DatabaseError, match="imuabil"):
        reg.connection.execute("UPDATE datasets SET label='x'")
    with pytest.raises(sqlite3.DatabaseError, match="imuabil"):
        reg.connection.execute("DELETE FROM datasets")
    reg.connection.close()
