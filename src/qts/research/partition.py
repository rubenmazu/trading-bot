"""Partiționarea cronologică a datelor și registrul OOS cu consum unic (Req 18.1–18.6).

Separarea Development_Set / Out_Of_Sample_Set este cronologică: tot ce intră în `Development_Set`
precede temporal `Out_Of_Sample_Set`, astfel încât selecția și calibrarea nu pot „privi” datele de
test (Req 18.1, 18.2). Fiecare partiție păstrează limitele temporale, identificatorul și rolul
(Req 18.5), iar `partition_hash` = SHA-256 peste conținut leagă stabil o partiție de datele ei.

Registrul OOS (`OosRegistry`) este persistent (SQLite, tabelele `datasets` + `oos_registry` din
migrația 4). Fluxul OOS este:

- `reserve`: rezervă un `Out_Of_Sample_Set` pentru o decizie și o pre-înregistrare (status
  „reserved”).
- `consume`: marchează OOS ca „consumed” la prima evaluare și întoarce înregistrarea; o a doua
  consumare a aceluiași OOS este refuzată cu `OosAlreadyConsumedError` (Req 18.3, 18.6).
- `reclassify`: dacă un OOS a influențat modificarea strategiei, datele consultate sunt
  reclasificate ca Development_Set (status „reclassified”), iar un nou OOS trebuie rezervat ulterior
  (Req 18.4). Un OOS reclasificat nu mai poate fi consumat.

Starea este durabilă: după reconectarea bazei, un OOS consumat rămâne consumat.
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
    "PARTITION_VERSION",
    "DataPartitioner",
    "OosAlreadyConsumedError",
    "OosNotReservedError",
    "OosRecord",
    "OosRegistry",
    "OosStatus",
    "Partition",
    "PartitionRole",
    "PartitionSplit",
    "RegistryError",
    "partition_hash",
]

PARTITION_VERSION: Final = "partition-v1"


class PartitionRole(StrEnum):
    """Rolul unei partiții în pipeline-ul de cercetare (Req 18.5)."""

    DEVELOPMENT = "development"
    OUT_OF_SAMPLE = "out_of_sample"


class OosStatus(StrEnum):
    """Starea unui `Out_Of_Sample_Set` în registru (Req 18.3, 18.4, 18.6)."""

    RESERVED = "reserved"
    CONSUMED = "consumed"
    RECLASSIFIED = "reclassified"


# --------------------------------------------------------------------------- erori


class RegistryError(Exception):
    """Eroare generică a registrului OOS."""


class OosAlreadyConsumedError(RegistryError):
    """A doua evaluare a unui `Out_Of_Sample_Set` deja consumat este refuzată (Req 18.3, 18.6)."""


class OosNotReservedError(RegistryError):
    """Operație asupra unui OOS care nu a fost rezervat (sau a fost reclasificat)."""


# --------------------------------------------------------------------------- modele


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class Partition(Frozen):
    """O partiție cronologică de date cu limitele, identificatorul și rolul ei (Req 18.5).

    `dataset_id` este derivat din `partition_hash` (SHA-256 peste conținut), deci partiții identice
    primesc același identificator, iar orice modificare a limitelor sau instrumentelor îl schimbă.
    """

    dataset_id: str
    partition_hash: str
    role: PartitionRole
    label: str
    start_ts: UtcDatetime
    end_ts: UtcDatetime
    instruments: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Partition:
        if not self.label.strip():
            raise ValueError("partiția necesită un label")
        if self.end_ts <= self.start_ts:
            raise ValueError("end_ts trebuie să fie strict după start_ts")
        if not self.instruments:
            raise ValueError("partiția necesită cel puțin un instrument")
        if len(self.instruments) != len(set(self.instruments)):
            raise ValueError("instrumente duplicate în partiție")
        expected = partition_hash(
            role=self.role,
            label=self.label,
            start_ts=self.start_ts,
            end_ts=self.end_ts,
            instruments=self.instruments,
        )
        if self.partition_hash != expected or self.dataset_id != expected:
            raise ValueError("partition_hash/dataset_id nu corespund conținutului partiției")
        return self


class PartitionSplit(Frozen):
    """Rezultatul separării cronologice: Development_Set urmat de Out_Of_Sample_Set (Req 18.1)."""

    development: Partition
    out_of_sample: Partition

    @model_validator(mode="after")
    def _check(self) -> PartitionSplit:
        if self.development.role is not PartitionRole.DEVELOPMENT:
            raise ValueError("prima partiție trebuie să fie Development_Set")
        if self.out_of_sample.role is not PartitionRole.OUT_OF_SAMPLE:
            raise ValueError("a doua partiție trebuie să fie Out_Of_Sample_Set")
        # Req 18.1/18.2: separarea este cronologică — dezvoltarea precede complet OOS.
        if self.out_of_sample.start_ts < self.development.end_ts:
            raise ValueError("Out_Of_Sample_Set trebuie să urmeze cronologic Development_Set")
        return self


class OosRecord(Frozen):
    """O linie din registrul OOS: rezervare, consum și eventuală reclasificare (Req 18.3, 18.4)."""

    dataset_id: str
    decision_id: str
    preregistration_hash: str
    status: OosStatus
    reserved_at: UtcDatetime
    consumed_at: UtcDatetime | None = None
    evaluation_hash: str | None = None
    reclassified_at: UtcDatetime | None = None
    reclassified_reason: str | None = None


def partition_hash(
    *,
    role: PartitionRole,
    label: str,
    start_ts: datetime,
    end_ts: datetime,
    instruments: Sequence[str],
) -> str:
    """SHA-256 hex peste conținutul canonic al unei partiții (rol, label, limite, instrumente)."""
    content = {
        "version": PARTITION_VERSION,
        "role": PartitionRole(role).value,
        "label": label,
        "start_ts": ensure_utc(start_ts).isoformat(),
        "end_ts": ensure_utc(end_ts).isoformat(),
        "instruments": sorted(instruments),
    }
    return hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- partiționare


class DataPartitioner:
    """Împarte cronologic un interval de date în Development_Set și Out_Of_Sample_Set (Req 18.1).

    `oos_fraction` este fracțiunea finală de timp rezervată Out_Of_Sample_Set. Punctul de tăiere
    este plasat cronologic, astfel încât tot Development_Set precede OOS (fără suprapunere), ceea ce
    menține separarea cerută de Req 18.1/18.2.
    """

    def __init__(self, *, oos_fraction: float = 0.3) -> None:
        if not (0 < oos_fraction < 1):
            raise ValueError("oos_fraction trebuie să fie în (0, 1)")
        self._oos_fraction = oos_fraction

    def split(
        self,
        *,
        start_ts: datetime,
        end_ts: datetime,
        instruments: Sequence[str],
        label: str = "default",
    ) -> PartitionSplit:
        """Separare cronologică la fracțiunea `oos_fraction` din durata totală."""
        start = ensure_utc(start_ts)
        end = ensure_utc(end_ts)
        if end <= start:
            raise ValueError("end_ts trebuie să fie strict după start_ts")
        cut = ensure_utc(start + (end - start) * (1 - self._oos_fraction))
        if cut <= start or cut >= end:
            raise ValueError("intervalul este prea scurt pentru fracțiunea OOS cerută")
        return self.split_at(
            start_ts=start, cut_ts=cut, end_ts=end, instruments=instruments, label=label
        )

    def split_at(
        self,
        *,
        start_ts: datetime,
        cut_ts: datetime,
        end_ts: datetime,
        instruments: Sequence[str],
        label: str = "default",
    ) -> PartitionSplit:
        """Separare la un punct de tăiere explicit (`cut_ts`), util pentru walk-forward."""
        start = ensure_utc(start_ts)
        cut = ensure_utc(cut_ts)
        end = ensure_utc(end_ts)
        if not (start < cut < end):
            raise ValueError("cut_ts trebuie să fie strict între start_ts și end_ts")
        instruments = tuple(instruments)
        development = _build_partition(
            role=PartitionRole.DEVELOPMENT,
            label=f"{label}:dev",
            start_ts=start,
            end_ts=cut,
            instruments=instruments,
        )
        out_of_sample = _build_partition(
            role=PartitionRole.OUT_OF_SAMPLE,
            label=f"{label}:oos",
            start_ts=cut,
            end_ts=end,
            instruments=instruments,
        )
        return PartitionSplit(development=development, out_of_sample=out_of_sample)


def _build_partition(
    *,
    role: PartitionRole,
    label: str,
    start_ts: datetime,
    end_ts: datetime,
    instruments: tuple[str, ...],
) -> Partition:
    digest = partition_hash(
        role=role, label=label, start_ts=start_ts, end_ts=end_ts, instruments=instruments
    )
    return Partition(
        dataset_id=digest,
        partition_hash=digest,
        role=role,
        label=label,
        start_ts=start_ts,
        end_ts=end_ts,
        instruments=instruments,
    )


# --------------------------------------------------------------------------- registru persistent


class OosRegistry:
    """Registru persistent al Out_Of_Sample_Set cu consum unic și reclasificare (Req 18.3–18.6).

    Scrie în tabelele `datasets` (append-only, limitele/identificatorul/rolul partiției — Req 18.5)
    și `oos_registry` (starea de rezervare/consum). Toate scrierile folosesc tranzacții explicite
    `BEGIN IMMEDIATE`, în stilul `qts.persistence`.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    # -- scrierea partițiilor ------------------------------------------------

    def register_partition(self, partition: Partition, *, now: datetime | None = None) -> None:
        """Înregistrează o partiție în `datasets` (idempotent dacă hash-ul coincide)."""
        created_at = ensure_utc(now) if now is not None else _utcnow()
        existing = self._conn.execute(
            "SELECT partition_hash FROM datasets WHERE dataset_id = ?", (partition.dataset_id,)
        ).fetchone()
        if existing is not None:
            if existing["partition_hash"] != partition.partition_hash:
                raise RegistryError(
                    f"dataset_id {partition.dataset_id} există cu alt partition_hash"
                )
            return
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO datasets (dataset_id, partition_hash, role, label, start_ts, "
                "end_ts, instruments, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    partition.dataset_id,
                    partition.partition_hash,
                    partition.role.value,
                    partition.label,
                    partition.start_ts.isoformat(),
                    partition.end_ts.isoformat(),
                    _canonical(list(partition.instruments)),
                    created_at.isoformat(),
                ),
            )

    # -- ciclul de viață OOS -------------------------------------------------

    def reserve(
        self,
        oos: Partition,
        *,
        decision_id: str,
        preregistration_hash: str,
        now: datetime | None = None,
    ) -> OosRecord:
        """Rezervă un Out_Of_Sample_Set pentru o decizie (status „reserved”)."""
        if oos.role is not PartitionRole.OUT_OF_SAMPLE:
            raise RegistryError("doar un Out_Of_Sample_Set poate fi rezervat")
        if not decision_id.strip():
            raise RegistryError("decision_id este obligatoriu")
        if not preregistration_hash.strip():
            raise RegistryError("preregistration_hash este obligatoriu")
        reserved_at = ensure_utc(now) if now is not None else _utcnow()
        self.register_partition(oos, now=reserved_at)
        if self._fetch(oos.dataset_id) is not None:
            raise RegistryError(f"OOS {oos.dataset_id} este deja rezervat")
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO oos_registry (dataset_id, decision_id, preregistration_hash, "
                "status, reserved_at) VALUES (?, ?, ?, ?, ?)",
                (
                    oos.dataset_id,
                    decision_id,
                    preregistration_hash,
                    OosStatus.RESERVED.value,
                    reserved_at.isoformat(),
                ),
            )
        return self._require(oos.dataset_id)

    def consume(
        self,
        dataset_id: str,
        *,
        evaluation_hash: str,
        now: datetime | None = None,
    ) -> OosRecord:
        """Prima evaluare: marchează OOS „consumed”. A doua oară refuză (Req 18.3, 18.6)."""
        if not evaluation_hash.strip():
            raise RegistryError("evaluation_hash este obligatoriu")
        consumed_at = ensure_utc(now) if now is not None else _utcnow()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT status FROM oos_registry WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            if row is None:
                raise OosNotReservedError(f"OOS {dataset_id} nu este rezervat")
            status = OosStatus(row["status"])
            if status is OosStatus.CONSUMED:
                raise OosAlreadyConsumedError(
                    f"OOS {dataset_id} a fost deja consumat o dată (Req 18.3, 18.6)"
                )
            if status is OosStatus.RECLASSIFIED:
                raise OosNotReservedError(
                    f"OOS {dataset_id} a fost reclasificat; rezervă un nou OOS (Req 18.4)"
                )
            conn.execute(
                "UPDATE oos_registry SET status = ?, consumed_at = ?, evaluation_hash = ? "
                "WHERE dataset_id = ?",
                (OosStatus.CONSUMED.value, consumed_at.isoformat(), evaluation_hash, dataset_id),
            )
        return self._require(dataset_id)

    def reclassify(
        self,
        dataset_id: str,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> OosRecord:
        """Reclasifică un OOS consultat ca Development_Set, oprind consumul viitor (Req 18.4)."""
        if not reason.strip():
            raise RegistryError("reclasificarea necesită un motiv")
        reclassified_at = ensure_utc(now) if now is not None else _utcnow()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT status FROM oos_registry WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            if row is None:
                raise OosNotReservedError(f"OOS {dataset_id} nu este rezervat")
            if OosStatus(row["status"]) is OosStatus.RECLASSIFIED:
                raise RegistryError(f"OOS {dataset_id} este deja reclasificat")
            conn.execute(
                "UPDATE oos_registry SET status = ?, reclassified_at = ?, "
                "reclassified_reason = ? WHERE dataset_id = ?",
                (OosStatus.RECLASSIFIED.value, reclassified_at.isoformat(), reason, dataset_id),
            )
        return self._require(dataset_id)

    # -- interogare ----------------------------------------------------------

    def get(self, dataset_id: str) -> OosRecord | None:
        """Întoarce înregistrarea OOS sau `None` dacă nu există."""
        return self._fetch(dataset_id)

    def status(self, dataset_id: str) -> OosStatus:
        """Starea curentă a unui OOS; ridică dacă nu este rezervat."""
        record = self._fetch(dataset_id)
        if record is None:
            raise OosNotReservedError(f"OOS {dataset_id} nu este rezervat")
        return record.status

    def is_available(self, dataset_id: str) -> bool:
        """Adevărat dacă OOS poate fi încă evaluat (rezervat, neconsumat, nereclasificat)."""
        record = self._fetch(dataset_id)
        return record is not None and record.status is OosStatus.RESERVED

    # -- intern --------------------------------------------------------------

    def _fetch(self, dataset_id: str) -> OosRecord | None:
        row = self._conn.execute(
            "SELECT * FROM oos_registry WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def _require(self, dataset_id: str) -> OosRecord:
        record = self._fetch(dataset_id)
        if record is None:  # defensiv: rândul tocmai a fost scris în aceeași tranzacție
            raise RegistryError(f"OOS {dataset_id} a dispărut după scriere")
        return record

    def _tx(self) -> _Transaction:
        return _Transaction(self._conn)


class _Transaction:
    """Context de tranzacție `BEGIN IMMEDIATE`, în stilul `qts.persistence.journal`."""

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


def _row_to_record(row: sqlite3.Row) -> OosRecord:
    return OosRecord(
        dataset_id=row["dataset_id"],
        decision_id=row["decision_id"],
        preregistration_hash=row["preregistration_hash"],
        status=OosStatus(row["status"]),
        reserved_at=datetime.fromisoformat(row["reserved_at"]),
        consumed_at=_opt_dt(row["consumed_at"]),
        evaluation_hash=row["evaluation_hash"],
        reclassified_at=_opt_dt(row["reclassified_at"]),
        reclassified_reason=row["reclassified_reason"],
    )


def _opt_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _utcnow() -> datetime:
    return datetime.now(UTC)
