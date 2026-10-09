"""Reluare istorică din CSV/Parquet cu manifest (Req 5.1, 5.6, 5.8).

`CsvSource` implementează portul `DataAdapter` pentru Backtest:

1. încarcă manifestul și verifică SHA-256; la nepotrivire setul este invalidat
   (`DatasetInvalidError`) înainte de a emite vreun eveniment;
2. citește fișierul cu polars, fără conversie prin float: CSV-ul este citit integral ca text,
   iar coloanele float din Parquet invalidează setul;
3. fiecare rând trece prin `normalize` → `Deduplicator` → `BarValidator`; rândurile respinse
   sunt excluse, iar motivul este reținut în `rejections` și jurnalizat;
4. evenimentele acceptate sunt emise în ordine temporală `(ts_source, instrument)`.

Coloane obligatorii: `instrument, ts_open, ts_close, interval_min, open, high, low, close,
volume`; opțional `seq`. Timpii fără offset sunt interpretați în fusul orar din manifest.
În reluarea istorică `ts_receipt = ts_close`: bara devine disponibilă la închidere.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

from qts.core.models import Bar, MarketEvent
from qts.data.manifest import DatasetInvalidError, DatasetManifest, load_verified
from qts.data.normalize import Deduplicator, DedupOutcome, NormalizationError, normalize
from qts.data.validate import REQUIRED_FIELDS, BarRejectReason, BarValidator, validate_bar

logger = logging.getLogger(__name__)

_TS_FIELDS: Final = ("ts_open", "ts_close")
_UTC_NAMES: Final = frozenset({"UTC", "ETC/UTC", "Z"})

OUTSIDE_MANIFEST_INTERVAL: Final = "BAR_OUTSIDE_MANIFEST_INTERVAL"
DUPLICATE_CONFLICT: Final = "BAR_DUPLICATE_CONFLICT"


@dataclass(frozen=True, slots=True)
class RowRejection:
    """Rând exclus; `row` este indexul rândului de date (0 = primul rând după antet)."""

    row: int
    instrument: str | None
    reason: str
    detail: str


def _resolve_tz(name: str) -> tzinfo:
    if name.upper() in _UTC_NAMES:
        return UTC
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise DatasetInvalidError("DATASET_TIMEZONE_UNKNOWN", name) from exc


def _read_frame(path: Path, manifest: DatasetManifest) -> pl.DataFrame:
    try:
        if manifest.format == "csv":
            frame = pl.read_csv(path, infer_schema=False)  # toate coloanele ca text
        else:
            frame = pl.read_parquet(path)
    except (pl.exceptions.PolarsError, OSError) as exc:
        raise DatasetInvalidError("DATASET_UNREADABLE", str(exc).splitlines()[0]) from exc
    missing = [c for c in REQUIRED_FIELDS if c not in frame.columns]
    if missing:
        raise DatasetInvalidError("DATASET_SCHEMA_INVALID", f"coloane lipsă: {missing}")
    floats = [name for name, dtype in frame.schema.items() if dtype.is_float()]
    if floats:
        raise DatasetInvalidError("DATASET_FLOAT_COLUMN", f"coloane float interzise: {floats}")
    # Timpii Parquet devin text ISO-8601 în polars: evită conversia fusului orar în Python
    # (care ar cere baza tzdata) și păstrează o singură cale de parsare cu CSV.
    stamps: list[pl.Expr] = []
    for name, dtype in frame.schema.items():
        if isinstance(dtype, pl.Datetime):
            if dtype.time_zone is None:
                stamps.append(pl.col(name).dt.strftime("%Y-%m-%dT%H:%M:%S%.f"))
            else:
                utc = pl.col(name).dt.convert_time_zone("UTC")
                stamps.append(utc.dt.strftime("%Y-%m-%dT%H:%M:%S%.f+00:00"))
    return frame.with_columns(stamps) if stamps else frame


class CsvSource:
    """`DataAdapter` pentru reluarea istorică a unui fișier CSV/Parquet cu manifest."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.manifest: DatasetManifest = load_verified(data_path)
        self.source_id: str = self.manifest.source_id
        self.rejections: list[RowRejection] = []

    @property
    def dataset_id(self) -> str:
        return self.manifest.dataset_id

    def stream(self) -> Iterator[MarketEvent]:
        # Eager: verificarea și validarea se fac la apel, deci un set corupt nu emite nimic.
        return iter(self.load())

    def load(self) -> list[MarketEvent]:
        """Reverifică suma de control și întoarce evenimentele acceptate, în ordine temporală."""
        self.manifest = load_verified(self.data_path)
        frame = _read_frame(self.data_path, self.manifest)
        zone = _resolve_tz(self.manifest.timezone) if self._has_naive(frame) else UTC

        self.rejections = []
        dedup = Deduplicator()
        validator = BarValidator()
        accepted: list[MarketEvent] = []
        for idx, row in enumerate(frame.iter_rows(named=True)):
            event = self._to_event(idx, row, zone)
            if event is None:
                continue
            outcome = dedup.offer(event)
            if outcome is DedupOutcome.DUPLICATE:
                continue
            if outcome is DedupOutcome.CONFLICT:
                self._reject(idx, event.instrument, DUPLICATE_CONFLICT, str(event.canonical_key))
                continue
            if not isinstance(event.payload, Bar):  # imposibil: kind="bar" impus mai sus
                raise TypeError("payload-ul unui eveniment bar trebuie să fie Bar")
            verdict = validator.validate(event.payload)
            if verdict.bar is None:
                self._reject(idx, event.instrument, str(verdict.reason), verdict.detail)
                continue
            accepted.append(event)
        accepted.sort(key=lambda e: (e.ts_source, e.instrument))
        logger.info(
            "set încărcat dataset_id=%s accepted=%d rejected=%d",
            self.dataset_id,
            len(accepted),
            len(self.rejections),
        )
        return accepted

    # ------------------------------------------------------------------ intern

    @staticmethod
    def _has_naive(frame: pl.DataFrame) -> bool:
        for col in _TS_FIELDS:
            for value in frame.get_column(col).drop_nulls():
                parsed = value if isinstance(value, datetime) else None
                if parsed is None:
                    try:
                        parsed = datetime.fromisoformat(str(value))
                    except ValueError:
                        continue
                if parsed.tzinfo is None:
                    return True
        return False

    def _reject(self, idx: int, instrument: str | None, reason: str, detail: str) -> None:
        logger.warning(
            "rând exclus dataset_id=%s row=%d instrument=%s reason=%s detail=%s",
            self.dataset_id,
            idx,
            instrument,
            reason,
            detail,
        )
        self.rejections.append(RowRejection(idx, instrument, reason, detail))

    def _to_event(self, idx: int, row: Mapping[str, Any], zone: tzinfo) -> MarketEvent | None:
        fields: dict[str, Any] = {f: row.get(f) for f in REQUIRED_FIELDS}
        instrument = fields["instrument"] if isinstance(fields["instrument"], str) else None
        try:
            for col in _TS_FIELDS:
                value = fields[col]
                if value is None:
                    continue
                ts = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
                fields[col] = ts.replace(tzinfo=zone) if ts.tzinfo is None else ts
            seq_raw = row.get("seq")
            seq = None if seq_raw is None else int(seq_raw)
        except (ValueError, TypeError) as exc:
            self._reject(idx, instrument, str(BarRejectReason.INVALID_FIELD), str(exc))
            return None

        try:
            event = normalize(
                {
                    "source_id": self.source_id,
                    "instrument": instrument,
                    "ts_receipt": fields["ts_close"],
                    "seq": seq,
                    "kind": "bar",
                    "payload": fields,
                }
            )
        except NormalizationError as exc:
            verdict = validate_bar(fields)
            reason = verdict.reason or BarRejectReason.INVALID_FIELD
            self._reject(idx, instrument, str(reason), verdict.detail or str(exc))
            return None

        ts_open: datetime = fields["ts_open"]
        ts_close: datetime = fields["ts_close"]
        if ts_open < self.manifest.start or ts_close > self.manifest.end:
            self._reject(
                idx,
                instrument,
                OUTSIDE_MANIFEST_INTERVAL,
                f"[{ts_open.isoformat()}, {ts_close.isoformat()}] în afara "
                f"[{self.manifest.start.isoformat()}, {self.manifest.end.isoformat()}]",
            )
            return None
        return event
