"""Cazuri limită pentru ingestie, cap-coadă prin `CsvSource` (Req 5.4, 5.5, 5.7, 5.8).

Completează testele modulare din `test_data_validate_normalize.py`,
`test_data_manifest_csv_source.py` și `test_data_bars_freshness.py`:
- barele invalide sunt excluse cu cod de motiv, iar cele valide (inclusiv ale altor
  instrumente) continuă să fie emise;
- duplicatele exacte devin un singur eveniment, iar conflictele păstrează prima reprezentare;
- orice modificare a fișierului sau a manifestului invalidează setul fără a emite nimic;
- prospețimea blochează per instrument, pe o secvență de evenimente reluate cu `SimClock`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from qts.core.clock import SimClock
from qts.core.models import Bar, MarketEvent
from qts.data.csv_source import DUPLICATE_CONFLICT, CsvSource
from qts.data.freshness import FreshnessReason, FreshnessTracker
from qts.data.manifest import DatasetInvalidError, create_manifest, manifest_path_for
from qts.data.validate import BarRejectReason

HEADER = "instrument,ts_open,ts_close,interval_min,open,high,low,close,volume,seq"
START = datetime(2024, 1, 2, tzinfo=UTC)
END = datetime(2024, 1, 3, tzinfo=UTC)
DAY = datetime(2024, 1, 2, tzinfo=UTC)
GOOD = "100.10,101.00,99.90,100.50,1200"


def row(
    instrument: str,
    open_hm: str,
    close_hm: str,
    ohlcv: str = GOOD,
    *,
    seq: str = "",
    interval: int = 15,
    offset: str = "+00:00",
) -> str:
    return (
        f"{instrument},2024-01-02T{open_hm}:00{offset},2024-01-02T{close_hm}:00{offset},"
        f"{interval},{ohlcv},{seq}"
    )


def write_dataset(tmp_path: Path, rows: list[str], name: str = "d.csv") -> Path:
    path = tmp_path / name
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8", newline="\n")
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


def at(hm: str) -> datetime:
    hh, mm = hm.split(":")
    return DAY.replace(hour=int(hh), minute=int(mm))


def closes(events: list[MarketEvent]) -> list[tuple[str, str]]:
    return [(e.instrument, e.ts_source.strftime("%H:%M")) for e in events]


# ----------------------------------------------------------------- bare invalide (Req 5.4)


def test_invalid_bars_excluded_while_valid_ones_flow_across_instruments(tmp_path: Path) -> None:
    path = write_dataset(
        tmp_path,
        [
            row("XYZ", "10:00", "10:15"),
            row("ABC", "10:00", "10:15", "0,101,99,100,5"),  # preț zero
            row("XYZ", "10:15", "10:30", "100,101,99,100,-1"),  # volum negativ
            row("ABC", "10:00", "10:30"),  # 30 min cu interval_min=15
            row("", "10:15", "10:30"),  # instrument absent
            # valide după respingeri pe același instrument (respingerile nu avansează timpul);
            # seq distinct: altfel cheia canonică ar fi deja ocupată de bara respinsă
            row("XYZ", "10:15", "10:30", seq="2"),
            row("ABC", "10:15", "10:30", seq="2"),
        ],
    )
    source = CsvSource(path)
    events = source.load()

    assert closes(events) == [("XYZ", "10:15"), ("ABC", "10:30"), ("XYZ", "10:30")]
    assert [(r.row, r.instrument, r.reason) for r in source.rejections] == [
        (1, "ABC", BarRejectReason.NON_POSITIVE_PRICE.value),
        (2, "XYZ", BarRejectReason.NEGATIVE_VOLUME.value),
        (3, "ABC", BarRejectReason.INTERVAL_MISMATCH.value),
        (4, None, BarRejectReason.MISSING_FIELD.value),
    ]
    assert all(r.detail for r in source.rejections)
    # niciun eveniment emis nu încalcă invariantele barei
    for e in events:
        bar = e.payload
        assert isinstance(bar, Bar)
        assert bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high


def test_rejections_are_logged_with_reason_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_dataset(tmp_path, [row("XYZ", "10:00", "10:15", "100,99,98,100,5")])
    with caplog.at_level("WARNING", logger="qts.data.csv_source"):
        assert CsvSource(path).load() == []
    assert BarRejectReason.OHLC_INCONSISTENT.value in caplog.text


def test_reload_resets_rejections(tmp_path: Path) -> None:
    path = write_dataset(
        tmp_path, [row("XYZ", "10:00", "10:15"), row("XYZ", "10:15", "10:30", "1,1,1,1,-1")]
    )
    source = CsvSource(path)
    first = source.load()
    second = source.load()
    assert first == second
    assert len(source.rejections) == 1


# ----------------------------------------------------------------- duplicate (Req 5.7)


def test_exact_duplicate_rows_become_single_event(tmp_path: Path) -> None:
    dup = row("XYZ", "10:00", "10:15", seq="5")
    path = write_dataset(tmp_path, [dup, dup, row("XYZ", "10:15", "10:30"), dup])
    source = CsvSource(path)
    events = source.load()
    assert closes(events) == [("XYZ", "10:15"), ("XYZ", "10:30")]
    assert events[0].seq == 5
    # duplicatele exacte sunt ignorate silențios, nu raportate ca bare invalide
    assert source.rejections == []


def test_same_instant_with_different_offset_is_exact_duplicate(tmp_path: Path) -> None:
    path = write_dataset(
        tmp_path,
        [row("XYZ", "10:00", "10:15"), row("XYZ", "12:00", "12:15", offset="+02:00")],
    )
    source = CsvSource(path)
    assert len(source.load()) == 1
    assert source.rejections == []


def test_conflicting_duplicate_keeps_first_representation(tmp_path: Path) -> None:
    path = write_dataset(
        tmp_path,
        [
            row("XYZ", "10:00", "10:15", "100,101,99,100.50,5", seq="1"),
            row("XYZ", "10:00", "10:15", "100,101,99,100.70,5", seq="1"),
        ],
    )
    source = CsvSource(path)
    (event,) = source.load()
    assert isinstance(event.payload, Bar)
    assert event.payload.close == Decimal("100.50")
    assert [(r.row, r.reason) for r in source.rejections] == [(1, DUPLICATE_CONFLICT)]


def test_same_bar_on_other_source_key_is_not_a_duplicate(tmp_path: Path) -> None:
    # alt instrument → altă cheie canonică: ambele bare sunt emise
    path = write_dataset(tmp_path, [row("XYZ", "10:00", "10:15"), row("ABC", "10:00", "10:15")])
    assert closes(CsvSource(path).load()) == [("ABC", "10:15"), ("XYZ", "10:15")]


def test_same_bar_with_different_seq_is_rejected_by_time_order(tmp_path: Path) -> None:
    # chei canonice diferite (seq), dar aceeași fereastră temporală pe instrument
    path = write_dataset(
        tmp_path, [row("XYZ", "10:00", "10:15", seq="1"), row("XYZ", "10:00", "10:15", seq="2")]
    )
    source = CsvSource(path)
    assert len(source.load()) == 1
    assert [r.reason for r in source.rejections] == [BarRejectReason.TIME_NOT_INCREASING.value]


# ----------------------------------------------------------------- sume de control (Req 5.8)


def _flip_one_byte(path: Path, old: bytes, new: bytes) -> bytes:
    original = path.read_bytes()
    assert len(old) == len(new) == 1 and original.count(old) >= 1
    idx = original.rindex(old)
    path.write_bytes(original[:idx] + new + original[idx + 1 :])
    return original


def _assert_invalid_and_silent(source_factory: object, code: str) -> None:
    emitted: list[MarketEvent] = []
    with pytest.raises(DatasetInvalidError) as err:
        emitted.extend(source_factory())  # type: ignore[operator]
    assert err.value.reason_code == code
    assert emitted == []


def test_single_byte_change_invalidates_dataset(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [row("XYZ", "10:00", "10:15"), row("XYZ", "10:15", "10:30")])
    original = _flip_one_byte(path, b"0", b"1")  # aceeași lungime, conținut diferit
    _assert_invalid_and_silent(lambda: CsvSource(path).stream(), "DATASET_CHECKSUM_MISMATCH")
    # restaurarea octetului face setul din nou valid: verificarea depinde doar de conținut
    path.write_bytes(original)
    assert len(CsvSource(path).load()) == 2


def test_single_byte_change_after_open_emits_nothing(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [row("XYZ", "10:00", "10:15")])
    source = CsvSource(path)
    _flip_one_byte(path, b"5", b"6")
    _assert_invalid_and_silent(source.stream, "DATASET_CHECKSUM_MISMATCH")


def test_data_file_deleted_after_open(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [row("XYZ", "10:00", "10:15")])
    source = CsvSource(path)
    path.unlink()
    _assert_invalid_and_silent(source.stream, "DATASET_FILE_MISSING")


def _edit_manifest(path: Path, **changes: object) -> None:
    mpath = manifest_path_for(path)
    data = json.loads(mpath.read_text(encoding="utf-8"))
    data.update(changes)
    mpath.write_text(json.dumps(data), encoding="utf-8")


def test_manifest_checksum_tampering_invalidates(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [row("XYZ", "10:00", "10:15")])
    _edit_manifest(path, sha256="0" * 64)
    _assert_invalid_and_silent(lambda: CsvSource(path).stream(), "DATASET_CHECKSUM_MISMATCH")


def test_manifest_rehashed_for_tampered_data_is_accepted_with_new_id(tmp_path: Path) -> None:
    # recalcularea manifestului este singura cale legitimă de a accepta conținut nou,
    # iar identificatorul setului se schimbă odată cu conținutul
    path = write_dataset(tmp_path, [row("XYZ", "10:00", "10:15")])
    before = CsvSource(path).dataset_id
    _flip_one_byte(path, b"5", b"6")
    create_manifest(
        path,
        source_id="synthetic",
        start=START,
        end=END,
        timezone="UTC",
        calendar_id="XETR",
        corporate_adjustments="none",
    )
    assert CsvSource(path).dataset_id != before


@pytest.mark.parametrize(
    "changes",
    [
        {"sha256": "not-a-hash"},
        {"data_file": "other.csv"},
        {"data_file": "../d.csv"},
        {"end": "2024-01-01T00:00:00+00:00"},  # end < start
        {"unexpected": "field"},
    ],
)
def test_manifest_field_tampering_invalidates(tmp_path: Path, changes: dict[str, object]) -> None:
    path = write_dataset(tmp_path, [row("XYZ", "10:00", "10:15")])
    _edit_manifest(path, **changes)
    _assert_invalid_and_silent(lambda: CsvSource(path).stream(), "DATASET_MANIFEST_INVALID")


def test_corrupt_manifest_json_invalidates(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [row("XYZ", "10:00", "10:15")])
    manifest_path_for(path).write_text("{not json", encoding="utf-8")
    _assert_invalid_and_silent(lambda: CsvSource(path).stream(), "DATASET_MANIFEST_INVALID")


def test_manifest_deleted_after_open(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [row("XYZ", "10:00", "10:15")])
    source = CsvSource(path)
    manifest_path_for(path).unlink()
    _assert_invalid_and_silent(source.stream, "DATASET_MANIFEST_MISSING")


# ----------------------------------------------------------------- prospețime (Req 5.5)


def test_freshness_blocks_per_instrument_over_replayed_events(tmp_path: Path) -> None:
    path = write_dataset(
        tmp_path,
        [
            row("XYZ", "10:00", "10:15"),
            row("ABC", "10:00", "10:15"),
            row("XYZ", "10:15", "10:30"),
            row("XYZ", "10:30", "10:45"),
            row("ABC", "10:30", "10:45", "1,1,1,1,-1"),  # invalidă: nu împrospătează ABC
        ],
    )
    events = CsvSource(path).load()
    clock = SimClock(at("10:00"))
    tracker = FreshnessTracker(clock, timedelta(minutes=20), {"XYZ": timedelta(minutes=5)})
    universe = ["XYZ", "ABC", "NODATA"]

    assert tracker.blocked_instruments(universe) == universe  # nimic recepționat încă

    blocked_after: list[tuple[str, list[str]]] = []
    for e in events:
        clock.advance_to(e.ts_receipt)
        tracker.record(e)
        blocked_after.append((clock.now().strftime("%H:%M"), tracker.blocked_instruments(universe)))

    assert blocked_after == [
        ("10:15", ["XYZ", "NODATA"]),  # ABC (ordonat primul) recepționat, XYZ încă nu
        ("10:15", ["NODATA"]),
        ("10:30", ["NODATA"]),
        # ABC: ultima recepție validă la 10:15, vârsta 30 min > 20 min
        ("10:45", ["ABC", "NODATA"]),
    ]
    assert tracker.check("ABC").reason is FreshnessReason.STALE
    assert tracker.check("NODATA").reason is FreshnessReason.NEVER_RECEIVED

    # fără evenimente noi, XYZ expiră după pragul propriu de 5 min (limita inclusă)
    clock.advance_to(at("10:50"))
    assert tracker.is_fresh("XYZ")
    clock.advance_to(at("10:50") + timedelta(microseconds=1))
    verdict = tracker.check("XYZ")
    assert verdict.blocks_order_intent and verdict.reason is FreshnessReason.STALE
    assert verdict.last_receipt == at("10:45")
