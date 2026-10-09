"""Normalizare în schema unică `MarketEvent` și deduplicare pe cheia canonică (Req 5.2, 5.3, 5.7).

Toate modurile (Backtest, Shadow, Demo) trec prin aceeași funcție `normalize`, deci strategia
primește aceeași schemă indiferent de sursă. Intrarea brută este un mapping cu:

- `source_id`, `instrument`, `ts_receipt`, `kind` ∈ {bar, quote, trade}, `seq` opțional;
- `ts_source` opțional: implicit `ts_close` pentru bare și `ts` pentru quote/trade;
- câmpurile payload-ului fie sub cheia `payload`, fie direct la nivelul de sus.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import datetime
from decimal import DecimalException
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from qts.core.models import Bar, MarketEvent, Quote, Trade, canonical_json

logger = logging.getLogger(__name__)

_PAYLOAD_TYPES: Final[dict[str, type[BaseModel]]] = {"bar": Bar, "quote": Quote, "trade": Trade}
_DEFAULT_TS_FIELD: Final[dict[str, str]] = {"bar": "ts_close", "quote": "ts", "trade": "ts"}
_ENVELOPE_FIELDS: Final = frozenset(
    {"source_id", "instrument", "ts_source", "ts_receipt", "seq", "kind", "payload"}
)

CanonicalKey = tuple[str, str, datetime, int | None]


class NormalizationError(ValueError):
    """Intrare brută care nu poate fi adusă la schema `MarketEvent`."""

    reason_code = "EVENT_NORMALIZATION_FAILED"


def normalize(raw: Mapping[str, Any]) -> MarketEvent:
    """Transformă o observație brută în `MarketEvent`; ridică `NormalizationError` la eșec."""
    kind = raw.get("kind")
    if kind not in _PAYLOAD_TYPES:
        raise NormalizationError(f"kind necunoscut: {kind!r}")
    instrument = raw.get("instrument")

    nested = raw.get("payload")
    if nested is None:
        fields = {k: v for k, v in raw.items() if k not in _ENVELOPE_FIELDS}
    elif isinstance(nested, Mapping):
        fields = dict(nested)
    else:
        fields = {}
    fields.setdefault("instrument", instrument)

    try:
        payload = (
            nested
            if isinstance(nested, _PAYLOAD_TYPES[kind])
            else _PAYLOAD_TYPES[kind].model_validate(fields)
        )
        ts_source = raw.get("ts_source")
        if ts_source is None:
            ts_source = getattr(payload, _DEFAULT_TS_FIELD[kind])
        return MarketEvent.model_validate(
            {
                "source_id": raw.get("source_id"),
                "instrument": instrument,
                "ts_source": ts_source,
                "ts_receipt": raw.get("ts_receipt"),
                "seq": raw.get("seq"),
                "kind": kind,
                "payload": payload,
            }
        )
    except (ValidationError, ValueError, TypeError, DecimalException) as exc:
        raise NormalizationError(str(exc).splitlines()[0]) from exc


class DedupOutcome(StrEnum):
    NEW = "NEW"
    DUPLICATE = "DUPLICATE"  # aceeași cheie, același conținut
    CONFLICT = "CONFLICT"  # aceeași cheie, conținut diferit: prima reprezentare rămâne


def _content(event: MarketEvent) -> str:
    # `ts_receipt` diferă natural între retransmisii; nu face parte din identitatea observației.
    return canonical_json(event.model_copy(update={"ts_receipt": event.ts_source}))


class Deduplicator:
    """Păstrează o singură reprezentare canonică per `(source_id, instrument, ts_source, seq)`."""

    def __init__(self) -> None:
        self._seen: dict[CanonicalKey, str] = {}

    def __len__(self) -> int:
        return len(self._seen)

    def offer(self, event: MarketEvent) -> DedupOutcome:
        key = event.canonical_key
        content = _content(event)
        known = self._seen.get(key)
        if known is None:
            self._seen[key] = content
            return DedupOutcome.NEW
        if known == content:
            logger.info("eveniment duplicat ignorat key=%s", key)
            return DedupOutcome.DUPLICATE
        logger.warning("conflict pe cheia canonică key=%s; se păstrează prima reprezentare", key)
        return DedupOutcome.CONFLICT


def deduplicate(events: Iterable[MarketEvent]) -> list[MarketEvent]:
    """Întoarce prima apariție a fiecărei chei canonice, în ordinea de intrare."""
    dedup = Deduplicator()
    return [e for e in events if dedup.offer(e) is DedupOutcome.NEW]
