"""Teste unitare pentru `qts.data.validate` și `qts.data.normalize` (Req 5.3, 5.4, 5.7)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from qts.core.models import Bar, MarketEvent
from qts.data.normalize import (
    Deduplicator,
    DedupOutcome,
    NormalizationError,
    deduplicate,
    normalize,
)
from qts.data.validate import BarRejectReason, BarValidator, filter_bars, validate_bar

T0 = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)


def bar_dict(i: int = 0, **overrides: Any) -> dict[str, Any]:
    ts_open = T0 + timedelta(minutes=15 * i)
    data: dict[str, Any] = {
        "instrument": "XYZ",
        "ts_open": ts_open,
        "ts_close": ts_open + timedelta(minutes=15),
        "interval_min": 15,
        "open": "10.00",
        "high": "10.50",
        "low": "9.80",
        "close": "10.20",
        "volume": "100",
    }
    data.update(overrides)
    return data


def make_bar(i: int = 0, **overrides: Any) -> Bar:
    return Bar.model_validate(bar_dict(i, **overrides))


# --------------------------------------------------------------------------- validate


def test_valid_bar_accepted() -> None:
    verdict = validate_bar(make_bar())
    assert verdict.ok and verdict.bar is not None and verdict.reason is None


def test_raw_mapping_accepted() -> None:
    raw = bar_dict()
    raw["ts_open"] = raw["ts_open"].isoformat()
    raw["ts_close"] = raw["ts_close"].isoformat()
    verdict = validate_bar(raw)
    assert verdict.ok and verdict.bar == make_bar()


def test_boundary_ohlc_equal_prices_accepted() -> None:
    verdict = validate_bar(make_bar(open="10", high="10", low="10", close="10", volume="0"))
    assert verdict.ok


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"high": "10.10"}, BarRejectReason.OHLC_INCONSISTENT),  # high < close
        ({"low": "10.10"}, BarRejectReason.OHLC_INCONSISTENT),  # low > open
        ({"high": "9.00", "low": "11.00"}, BarRejectReason.OHLC_INCONSISTENT),
        ({"volume": "-1"}, BarRejectReason.NEGATIVE_VOLUME),
        ({"low": "0"}, BarRejectReason.NON_POSITIVE_PRICE),
        (
            {"ts_close": T0 + timedelta(minutes=20)},
            BarRejectReason.INTERVAL_MISMATCH,
        ),
    ],
)
def test_impossible_bars_rejected(overrides: dict[str, Any], reason: BarRejectReason) -> None:
    verdict = validate_bar(make_bar(**overrides))
    assert not verdict.ok and verdict.bar is None
    assert verdict.reason is reason and verdict.detail


def test_missing_field_rejected() -> None:
    raw = bar_dict()
    del raw["volume"]
    verdict = validate_bar(raw)
    assert verdict.reason is BarRejectReason.MISSING_FIELD
    assert "volume" in verdict.detail


def test_none_field_rejected_as_missing() -> None:
    verdict = validate_bar(bar_dict(close=None))
    assert verdict.reason is BarRejectReason.MISSING_FIELD


@pytest.mark.parametrize(
    "overrides",
    [
        {"open": "abc"},
        {"open": 10.0},  # float refuzat
        {"ts_open": datetime(2024, 1, 2, 9, 0)},  # noqa: DTZ001 - naive, intenționat
        {"interval_min": 1},
    ],
)
def test_invalid_field_rejected(overrides: dict[str, Any]) -> None:
    verdict = validate_bar(bar_dict(**overrides))
    assert verdict.reason is BarRejectReason.INVALID_FIELD


def test_time_must_strictly_increase_per_instrument() -> None:
    validator = BarValidator()
    assert validator.validate(make_bar(1)).ok
    # aceeași bară, o bară anterioară și o bară suprapusă sunt respinse
    for bad in (make_bar(1), make_bar(0)):
        verdict = validator.validate(bad)
        assert verdict.reason is BarRejectReason.TIME_NOT_INCREASING
    overlapping = make_bar(ts_open=T0 + timedelta(minutes=20), ts_close=T0 + timedelta(minutes=35))
    assert validator.validate(overlapping).reason is BarRejectReason.TIME_NOT_INCREASING
    # starea nu avansează pe bare respinse; o bară următoare este acceptată
    assert validator.last_close("XYZ") == T0 + timedelta(minutes=30)
    assert validator.validate(make_bar(2)).ok
    # alt instrument are propria ordine temporală
    assert validator.validate(make_bar(0, instrument="ABC")).ok


def test_gap_between_bars_is_allowed() -> None:
    validator = BarValidator()
    assert validator.validate(make_bar(0)).ok
    assert validator.validate(make_bar(10)).ok


def test_filter_bars_excludes_invalid_with_reasons() -> None:
    bars: list[Bar | dict[str, Any]] = [
        make_bar(0),
        make_bar(1, volume="-5"),
        make_bar(1),
        make_bar(1),
        bar_dict(2, high=None),
    ]
    accepted, rejected = filter_bars(bars)
    assert accepted == [make_bar(0), make_bar(1)]
    assert [v.reason for v in rejected] == [
        BarRejectReason.NEGATIVE_VOLUME,
        BarRejectReason.TIME_NOT_INCREASING,
        BarRejectReason.MISSING_FIELD,
    ]


def test_rejection_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="qts.data.validate"):
        BarValidator().validate(make_bar(volume="-1"))
    assert BarRejectReason.NEGATIVE_VOLUME.value in caplog.text


# --------------------------------------------------------------------------- normalize


RECEIPT = T0 + timedelta(minutes=15, seconds=2)


def raw_bar_event(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source_id": "csv",
        "kind": "bar",
        "ts_receipt": RECEIPT,
        "seq": 7,
        **bar_dict(),
    }
    data.update(overrides)
    return data


def test_normalize_flat_bar_keeps_envelope_fields() -> None:
    event = normalize(raw_bar_event())
    assert isinstance(event, MarketEvent)
    assert event.payload == make_bar()
    assert (event.source_id, event.instrument, event.seq) == ("csv", "XYZ", 7)
    assert event.ts_source == make_bar().ts_close  # implicit pentru bare
    assert event.ts_receipt == RECEIPT
    assert event.canonical_key == ("csv", "XYZ", make_bar().ts_close, 7)


def test_normalize_nested_and_flat_produce_same_schema() -> None:
    nested = {
        "source_id": "csv",
        "instrument": "XYZ",
        "kind": "bar",
        "ts_receipt": RECEIPT,
        "seq": 7,
        "payload": {k: v for k, v in bar_dict().items() if k != "instrument"},
    }
    assert normalize(nested) == normalize(raw_bar_event())


def test_normalize_quote_and_trade() -> None:
    ts = T0.isoformat()
    quote = normalize(
        {
            "source_id": "s",
            "instrument": "XYZ",
            "kind": "quote",
            "ts_receipt": ts,
            "ts": ts,
            "bid": "9.99",
            "ask": "10.01",
        }
    )
    trade = normalize(
        {
            "source_id": "s",
            "instrument": "XYZ",
            "kind": "trade",
            "ts_receipt": ts,
            "ts": ts,
            "price": "10",
            "size": "3",
        }
    )
    assert quote.kind == "quote" and quote.ts_source == T0 and quote.seq is None
    assert trade.kind == "trade" and trade.ts_source == T0


def test_normalize_explicit_ts_source_kept() -> None:
    event = normalize(raw_bar_event(ts_source=T0 + timedelta(minutes=16)))
    assert event.ts_source == T0 + timedelta(minutes=16)


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "book"},
        {"source_id": None},
        {"ts_receipt": None},
        {"ts_receipt": datetime(2024, 1, 2, 9, 15)},  # noqa: DTZ001 - naive, intenționat
        {"open": 10.0},
        {"open": "abc"},
        {"payload": bar_dict(instrument="OTHER")},  # instrument diferit de anvelopă
    ],
)
def test_normalize_rejects_invalid_input(overrides: dict[str, Any]) -> None:
    with pytest.raises(NormalizationError):
        normalize(raw_bar_event(**overrides))


# --------------------------------------------------------------------------- deduplicare


def test_dedup_keeps_single_canonical_representation() -> None:
    first = normalize(raw_bar_event())
    retransmitted = normalize(raw_bar_event(ts_receipt=RECEIPT + timedelta(seconds=5)))
    conflicting = normalize(raw_bar_event(close="10.30"))
    other_seq = normalize(raw_bar_event(seq=8))

    dedup = Deduplicator()
    assert dedup.offer(first) is DedupOutcome.NEW
    assert dedup.offer(retransmitted) is DedupOutcome.DUPLICATE
    assert dedup.offer(conflicting) is DedupOutcome.CONFLICT
    assert dedup.offer(other_seq) is DedupOutcome.NEW
    assert len(dedup) == 2

    assert deduplicate([first, retransmitted, conflicting, other_seq, first]) == [
        first,
        other_seq,
    ]


def test_dedup_seq_none_is_part_of_key() -> None:
    a = normalize(raw_bar_event(seq=None))
    b = normalize(raw_bar_event(seq=None, source_id="other"))
    assert deduplicate([a, a, b]) == [a, b]
