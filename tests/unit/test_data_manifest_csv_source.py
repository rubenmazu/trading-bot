"""Teste pentru `data/manifest.py` și `data/csv_source.py` (Req 5.1, 5.6, 5.8)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from qts.core.models import Bar
from qts.data.adapter import DataAdapter
from qts.data.csv_source import DUPLICATE_CONFLICT, OUTSIDE_MANIFEST_INTERVAL, CsvSource
from qts.data.manifest import (
    DatasetInvalidError,
    create_manifest,
    file_sha256,
    load_manifest,
    manifest_path_for,
)
from qts.data.validate import BarRejectReason

HEADER = "instrument,ts_open,ts_close,interval_min,open,high,low,close,volume,seq"
START = datetime(2024, 1, 2, tzinfo=UTC)
END = datetime(2024, 1, 3, tzinfo=UTC)


def _row(
    instrument: str,
    hh_mm_open: str,
    hh_mm_close: str,
    ohlcv: str = "100.10,101.00,99.90,100.50,1200",
    seq: str = "",
    offset: str = "+00:00",
) -> str:
    return (
        f"{instrument},2024-01-02T{hh_mm_open}:00{offset},2024-01-02T{hh_mm_close}:00{offset},"
        f"15,{ohlcv},{seq}"
    )


def _dataset(tmp_path: Path, rows: list[str], timezone: str = "UTC", name: str = "d.csv") -> Path:
    path = tmp_path / name
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8", newline="\n")
    create_manifest(
        path,
        source_id="synthetic",
        start=START,
        end=END,
        timezone=timezone,
        calendar_id="XETR",
        corporate_adjustments="none",
    )
    return path


def _bar(source: CsvSource, index: int) -> Bar:
    payload = source.load()[index].payload
    assert isinstance(payload, Bar)
    return payload


# ----------------------------------------------------------------------------- manifest


def test_manifest_records_metadata_and_checksum(tmp_path: Path) -> None:
    path = _dataset(tmp_path, [_row("XYZ", "10:00", "10:15")])
    manifest = load_manifest(manifest_path_for(path))
    assert manifest.source_id == "synthetic"
    assert manifest.format == "csv"
    assert manifest.data_file == "d.csv"
    assert (manifest.start, manifest.end) == (START, END)
    assert (manifest.timezone, manifest.calendar_id) == ("UTC", "XETR")
    assert manifest.corporate_adjustments == "none"
    assert manifest.sha256 == file_sha256(path)
    assert manifest.dataset_id == f"ds_{manifest.sha256[:16]}"
    assert json.loads(manifest_path_for(path).read_text(encoding="utf-8"))["sha256"] == (
        manifest.sha256
    )


def test_dataset_id_changes_with_content(tmp_path: Path) -> None:
    a = CsvSource(_dataset(tmp_path, [_row("XYZ", "10:00", "10:15")], name="a.csv"))
    b = CsvSource(_dataset(tmp_path, [_row("XYZ", "10:15", "10:30")], name="b.csv"))
    c = CsvSource(_dataset(tmp_path, [_row("XYZ", "10:00", "10:15")], name="c.csv"))
    assert a.dataset_id != b.dataset_id
    assert a.dataset_id == c.dataset_id


def test_manifest_rejects_end_before_start(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    path.write_text(HEADER + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="end"):
        create_manifest(
            path,
            source_id="s",
            start=END,
            end=START,
            timezone="UTC",
            calendar_id="XETR",
            corporate_adjustments="none",
        )


# ----------------------------------------------------------------------------- invalidare


def test_checksum_mismatch_invalidates_dataset(tmp_path: Path) -> None:
    path = _dataset(tmp_path, [_row("XYZ", "10:00", "10:15")])
    path.write_text(HEADER + "\n" + _row("XYZ", "10:00", "10:15", seq="9") + "\n")
    with pytest.raises(DatasetInvalidError) as err:
        CsvSource(path)
    assert err.value.reason_code == "DATASET_CHECKSUM_MISMATCH"


def test_tampering_after_open_yields_nothing(tmp_path: Path) -> None:
    path = _dataset(tmp_path, [_row("XYZ", "10:00", "10:15")])
    source = CsvSource(path)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(_row("XYZ", "10:15", "10:30") + "\n")
    emitted: list[object] = []
    with pytest.raises(DatasetInvalidError) as err:
        emitted.extend(source.stream())
    assert err.value.reason_code == "DATASET_CHECKSUM_MISMATCH"
    assert emitted == []


def test_missing_manifest_invalidates_dataset(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    path.write_text(HEADER + "\n", encoding="utf-8")
    with pytest.raises(DatasetInvalidError) as err:
        CsvSource(path)
    assert err.value.reason_code == "DATASET_MANIFEST_MISSING"


def test_missing_required_column_invalidates_dataset(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    path.write_text("instrument,ts_open\nXYZ,2024-01-02T10:00:00+00:00\n", encoding="utf-8")
    create_manifest(
        path,
        source_id="s",
        start=START,
        end=END,
        timezone="UTC",
        calendar_id="XETR",
        corporate_adjustments="none",
    )
    with pytest.raises(DatasetInvalidError) as err:
        CsvSource(path).load()
    assert err.value.reason_code == "DATASET_SCHEMA_INVALID"


# ----------------------------------------------------------------------------- reluare


def test_stream_emits_normalized_events_in_time_order(tmp_path: Path) -> None:
    path = _dataset(
        tmp_path,
        [
            _row("XYZ", "10:15", "10:30", seq="2"),
            _row("ABC", "10:00", "10:15"),
            _row("XYZ", "10:30", "10:45"),
        ],
    )
    source = CsvSource(path)
    assert isinstance(source, DataAdapter)
    events = list(source.stream())
    assert [(e.instrument, e.ts_source.hour, e.ts_source.minute) for e in events] == [
        ("ABC", 10, 15),
        ("XYZ", 10, 30),
        ("XYZ", 10, 45),
    ]
    first = events[0]
    assert first.source_id == "synthetic"
    assert first.ts_receipt == first.ts_source
    assert events[1].seq == 2
    bar = first.payload
    assert isinstance(bar, Bar)
    assert bar.open == Decimal("100.10")
    assert str(bar.open) == "100.10"  # fără trecere prin float
    assert source.rejections == []


def test_invalid_bars_are_excluded_with_reason(tmp_path: Path) -> None:
    path = _dataset(
        tmp_path,
        [
            _row("XYZ", "10:00", "10:15"),
            _row("XYZ", "10:00", "10:15"),  # duplicat identic: o singură reprezentare
            _row("XYZ", "10:00", "10:15", ohlcv="100,101,99,100.7,5"),  # conflict
            _row("XYZ", "10:15", "10:30", ohlcv="100,99,98,100,5"),  # high < open
            _row("XYZ", "10:30", "10:45", ohlcv="100,101,99,,5"),  # close absent
            _row("XYZ", "10:45", "11:00", ohlcv="100,101,99,abc,5"),  # close invalid
            _row("XYZ", "11:00", "11:15"),
        ],
    )
    source = CsvSource(path)
    events = source.load()
    assert [e.ts_source.strftime("%H:%M") for e in events] == ["10:15", "11:15"]
    assert [(r.row, r.reason) for r in source.rejections] == [
        (2, DUPLICATE_CONFLICT),
        (3, BarRejectReason.OHLC_INCONSISTENT.value),
        (4, BarRejectReason.MISSING_FIELD.value),
        (5, BarRejectReason.INVALID_FIELD.value),
    ]


def test_out_of_order_bar_is_rejected(tmp_path: Path) -> None:
    path = _dataset(tmp_path, [_row("XYZ", "10:15", "10:30"), _row("XYZ", "10:00", "10:15")])
    source = CsvSource(path)
    assert len(source.load()) == 1
    assert [r.reason for r in source.rejections] == [BarRejectReason.TIME_NOT_INCREASING.value]


def test_bar_outside_manifest_interval_is_rejected(tmp_path: Path) -> None:
    outside = "XYZ,2024-01-03T00:00:00+00:00,2024-01-03T00:15:00+00:00,15,1,1,1,1,1,"
    source = CsvSource(_dataset(tmp_path, [_row("XYZ", "10:00", "10:15"), outside]))
    assert len(source.load()) == 1
    assert [r.reason for r in source.rejections] == [OUTSIDE_MANIFEST_INTERVAL]


def test_offsets_are_converted_to_utc(tmp_path: Path) -> None:
    source = CsvSource(_dataset(tmp_path, [_row("XYZ", "12:00", "12:15", offset="+02:00")]))
    assert _bar(source, 0).ts_open == datetime(2024, 1, 2, 10, 0, tzinfo=UTC)


def test_naive_timestamps_use_manifest_timezone(tmp_path: Path) -> None:
    source = CsvSource(_dataset(tmp_path, [_row("XYZ", "10:00", "10:15", offset="")]))
    assert _bar(source, 0).ts_open == datetime(2024, 1, 2, 10, 0, tzinfo=UTC)


def test_naive_timestamps_with_unknown_timezone_invalidate(tmp_path: Path) -> None:
    path = _dataset(tmp_path, [_row("XYZ", "10:00", "10:15", offset="")], timezone="Mars/Base")
    with pytest.raises(DatasetInvalidError) as err:
        CsvSource(path).load()
    assert err.value.reason_code == "DATASET_TIMEZONE_UNKNOWN"


# ----------------------------------------------------------------------------- parquet


def _parquet_frame(prices: pl.Series) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": ["XYZ"],
            "ts_open": [datetime(2024, 1, 2, 10, 0, tzinfo=UTC)],
            "ts_close": [datetime(2024, 1, 2, 10, 15, tzinfo=UTC)],
            "interval_min": [15],
            "open": prices,
            "high": ["101.00"],
            "low": ["99.90"],
            "close": ["100.50"],
            "volume": ["1200"],
        }
    )


def _write_parquet(tmp_path: Path, frame: pl.DataFrame) -> Path:
    path = tmp_path / "d.parquet"
    frame.write_parquet(path)
    create_manifest(
        path,
        source_id="synthetic",
        start=START,
        end=END,
        timezone="UTC",
        calendar_id="XETR",
        corporate_adjustments="none",
    )
    return path


def test_parquet_with_text_prices_loads(tmp_path: Path) -> None:
    path = _write_parquet(tmp_path, _parquet_frame(pl.Series(["100.10"])))
    source = CsvSource(path)
    assert source.manifest.format == "parquet"
    bar = _bar(source, 0)
    assert str(bar.open) == "100.10"
    assert bar.ts_close == datetime(2024, 1, 2, 10, 15, tzinfo=UTC)


def test_parquet_float_column_invalidates_dataset(tmp_path: Path) -> None:
    path = _write_parquet(tmp_path, _parquet_frame(pl.Series([float("100.1")])))
    with pytest.raises(DatasetInvalidError) as err:
        CsvSource(path).load()
    assert err.value.reason_code == "DATASET_FLOAT_COLUMN"
