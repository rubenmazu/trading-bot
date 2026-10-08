"""Tabele de comisioane versionate, per broker și (opțional) per bursă (Req 8.1, 8.2).

Un tabel definește un procent din valoarea nominală, limitat între `minimum` și `maximum`,
plus taxe de bursă (procent și sumă fixă). Toate sumele fixe sunt în `currency`.
Pentru un broker pot exista mai multe versiuni; se aplică versiunea cu `valid_from` maxim
care nu depășește momentul ordinului. Un tabel specific bursei are prioritate față de unul
generic (`venue=None`).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, localcontext

from pydantic import model_validator

from qts.core.clock import ensure_utc
from qts.core.models import Dec, Frozen, UtcDatetime
from qts.core.money import ZERO

from .errors import CostModelIncomplete, cost_context


class CommissionTable(Frozen):
    broker: str
    version: str
    valid_from: UtcDatetime
    currency: str
    venue: str | None = None
    percent: Dec  # fracție din nominal, de ex. 0.0005 = 5 bps
    minimum: Dec = ZERO
    maximum: Dec | None = None
    exchange_fee_percent: Dec = ZERO
    exchange_fee_fixed: Dec = ZERO

    @model_validator(mode="after")
    def _check(self) -> CommissionTable:
        values = (self.percent, self.minimum, self.exchange_fee_percent, self.exchange_fee_fixed)
        if any(v < 0 for v in values):
            raise ValueError("valorile tabelului de comisioane nu pot fi negative")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum trebuie să fie >= minimum")
        return self

    def commission(self, notional: Decimal) -> Decimal:
        """Comisionul în `currency` pentru o valoare nominală exprimată tot în `currency`."""
        if notional < 0:
            raise ValueError("notional nu poate fi negativ")
        with localcontext(cost_context()):
            base = max(notional * self.percent, self.minimum)
            if self.maximum is not None:
                base = min(base, self.maximum)
            return base + notional * self.exchange_fee_percent + self.exchange_fee_fixed


class CommissionSchedule(Frozen):
    """Colecție versionată de tabele de comisioane."""

    tables: tuple[CommissionTable, ...]

    @model_validator(mode="after")
    def _unique(self) -> CommissionSchedule:
        keys = [(t.broker, t.venue, t.valid_from) for t in self.tables]
        if len(keys) != len(set(keys)):
            raise ValueError("tabele duplicate pentru același broker, bursă și valid_from")
        return self

    def lookup(self, broker: str, venue: str, ts: datetime) -> CommissionTable:
        """Tabelul aplicabil; ridică `CostModelIncomplete` dacă nu există niciunul."""
        ts = ensure_utc(ts)
        candidates = [t for t in self.tables if t.broker == broker and t.valid_from <= ts]
        for venue_key in (venue, None):
            matching = [t for t in candidates if t.venue == venue_key]
            if matching:
                return max(matching, key=lambda t: t.valid_from)
        raise CostModelIncomplete(
            "commission", f"niciun tabel de comisioane pentru {broker}/{venue} la {ts.isoformat()}"
        )
