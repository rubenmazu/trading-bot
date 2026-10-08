"""Modele de domeniu imuabile, comune tuturor modurilor de operare.

Toate modelele sunt `frozen` și `extra="forbid"`. Timpii sunt UTC, iar banii și cantitățile
sunt `Decimal` construite fără float. `canonical_json` oferă o serializare stabilă, folosită
pentru hash-uri, audit și testele de determinism.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from qts.core.clock import ensure_utc
from qts.core.money import ZERO, FloatNotAllowedError, dec


def _to_decimal(value: Any) -> Decimal:
    if isinstance(value, float):
        raise FloatNotAllowedError("valorile monetare nu pot fi float; folosiți str sau Decimal")
    if isinstance(value, (Decimal, int, str)) and not isinstance(value, bool):
        return dec(value)
    raise ValueError(f"valoare zecimală invalidă: {value!r}")


Dec = Annotated[Decimal, BeforeValidator(_to_decimal)]
UtcDatetime = Annotated[datetime, AfterValidator(ensure_utc)]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=False)


def canonical_json(model: BaseModel) -> str:
    """Serializare canonică: chei sortate, fără spații, Decimal ca text, datetime ISO-8601 UTC."""
    return json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def canonical_bytes(model: BaseModel) -> bytes:
    """Forma canonică în UTF-8, pentru comparații byte-cu-byte (Req 6.5)."""
    return canonical_json(model).encode("utf-8")


def canonical_hash(model: BaseModel) -> str:
    """SHA-256 hex peste forma canonică (Req 6.3, 24.3)."""
    return hashlib.sha256(canonical_bytes(model)).hexdigest()


# --------------------------------------------------------------------------- instrumente


AssetClass = Literal["etf", "stock", "fx", "commodity_etp", "index_etp", "crypto"]


class Instrument(Frozen):
    symbol: str
    venue: str
    asset_class: AssetClass
    currency: str
    tick_size: Dec
    qty_step: Dec
    min_qty: Dec
    min_notional: Dec = ZERO
    fractional: bool = False
    calendar_id: str
    requires_leverage: bool = False
    requires_short: bool = False
    is_derivative: bool = False  # futures, CFD, opțiuni

    @model_validator(mode="after")
    def _check(self) -> Instrument:
        if self.tick_size <= 0 or self.qty_step <= 0:
            raise ValueError("tick_size și qty_step trebuie să fie > 0")
        if self.min_qty < 0 or self.min_notional < 0:
            raise ValueError("min_qty și min_notional nu pot fi negative")
        return self


# --------------------------------------------------------------------------- date de piață


class Bar(Frozen):
    """Bară OHLCV. Validarea semantică (invariante OHLC) se face în `qts.data.validate`."""

    instrument: str
    ts_open: UtcDatetime
    ts_close: UtcDatetime
    interval_min: Annotated[int, Field(ge=5, le=60)]  # Req 4.5
    open: Dec
    high: Dec
    low: Dec
    close: Dec
    volume: Dec


class Quote(Frozen):
    instrument: str
    ts: UtcDatetime
    bid: Dec
    ask: Dec


class Trade(Frozen):
    instrument: str
    ts: UtcDatetime
    price: Dec
    size: Dec


class MarketEvent(Frozen):
    source_id: str
    instrument: str
    ts_source: UtcDatetime
    ts_receipt: UtcDatetime
    seq: int | None = None
    kind: Literal["bar", "quote", "trade"]
    payload: Bar | Quote | Trade

    @model_validator(mode="after")
    def _kind_matches(self) -> MarketEvent:
        expected = {"bar": Bar, "quote": Quote, "trade": Trade}[self.kind]
        if not isinstance(self.payload, expected):
            raise ValueError(f"kind={self.kind} nu corespunde payload-ului")
        if self.payload.instrument != self.instrument:
            raise ValueError("instrumentul payload-ului diferă de instrumentul evenimentului")
        return self

    @property
    def canonical_key(self) -> tuple[str, str, datetime, int | None]:
        """Cheia de deduplicare (Req 5.7)."""
        return (self.source_id, self.instrument, self.ts_source, self.seq)


# --------------------------------------------------------------------------- semnale și ordine


SignalAction = Literal["ENTER_LONG", "EXIT", "NONE"]
Side = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]


class Signal(Frozen):
    signal_id: str
    strategy_id: str
    strategy_version: str
    instrument: str
    ts: UtcDatetime
    action: SignalAction
    stop_price: Dec | None = None
    reason_code: str
    inputs: dict[str, Dec]
    rules_evaluated: list[str]
    config_snapshot_id: str
    data_ids: list[str]

    @model_validator(mode="after")
    def _entry_has_stop(self) -> Signal:
        if self.action == "ENTER_LONG" and self.stop_price is None:
            raise ValueError("ENTER_LONG necesită stop_price")
        return self


class OrderIntent(Frozen):
    intent_id: str
    signal_id: str
    instrument: str
    side: Side
    ref_price: Dec
    stop_price: Dec | None = None
    order_type: OrderType = "MARKET"
    limit_price: Dec | None = None
    requested_qty: Dec | None = None  # pentru ieșiri: cantitatea deținută

    @model_validator(mode="after")
    def _check(self) -> OrderIntent:
        if self.ref_price <= 0:
            raise ValueError("ref_price trebuie să fie > 0")
        if self.order_type == "LIMIT" and self.limit_price is None:
            raise ValueError("ordinul LIMIT necesită limit_price")
        return self


class CostBreakdown(Frozen):
    """Costurile pe categorii, în moneda de raportare (Req 8.1, 8.4)."""

    spread: Dec = ZERO
    commission: Dec = ZERO
    slippage: Dec = ZERO
    latency: Dec = ZERO
    fx_conversion: Dec = ZERO
    taxes: Dec = ZERO

    @property
    def total(self) -> Decimal:
        return (
            self.spread
            + self.commission
            + self.slippage
            + self.latency
            + self.fx_conversion
            + self.taxes
        )

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            spread=self.spread + other.spread,
            commission=self.commission + other.commission,
            slippage=self.slippage + other.slippage,
            latency=self.latency + other.latency,
            fx_conversion=self.fx_conversion + other.fx_conversion,
            taxes=self.taxes + other.taxes,
        )


class OrderState(StrEnum):
    CREATED = "CREATED"
    APPROVED = "APPROVED"
    REJECTED_RISK = "REJECTED_RISK"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REJECTED_BROKER = "REJECTED_BROKER"
    UNKNOWN = "UNKNOWN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class Order(Frozen):
    client_order_id: str
    intent_id: str
    instrument: str
    side: Side
    qty: Dec
    filled_qty: Dec = ZERO
    avg_fill_price: Dec | None = None
    costs: CostBreakdown = CostBreakdown()
    state: OrderState = OrderState.CREATED
    broker_order_id: str | None = None
    version: int = 0

    @property
    def remaining_qty(self) -> Decimal:
        return self.qty - self.filled_qty

    @model_validator(mode="after")
    def _quantity_invariant(self) -> Order:
        if self.qty <= 0:
            raise ValueError("qty trebuie să fie > 0")
        if not ZERO <= self.filled_qty <= self.qty:
            raise ValueError("filled_qty trebuie să fie în [0, qty]")
        return self


class ExecKind(StrEnum):
    ACK = "ACK"
    REJECT = "REJECT"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCELLED = "CANCELLED"
    CANCEL_REJECTED = "CANCEL_REJECTED"
    EXPIRED = "EXPIRED"


class ExecutionEvent(Frozen):
    broker_exec_id: str
    client_order_id: str
    kind: ExecKind
    qty: Dec | None = None
    price: Dec | None = None
    commission: Dec | None = None
    broker_order_id: str | None = None
    reason: str | None = None
    ts_broker: UtcDatetime
    ts_receipt: UtcDatetime
    seq: int | None = None

    @model_validator(mode="after")
    def _fill_fields(self) -> ExecutionEvent:
        is_fill = self.kind in (ExecKind.PARTIAL_FILL, ExecKind.FILL)
        if is_fill and (self.qty is None or self.qty <= 0 or self.price is None or self.price <= 0):
            raise ValueError("execuția necesită qty > 0 și price > 0")
        return self


# --------------------------------------------------------------------------- audit


class AuditRecord(Frozen):
    seq: int
    ts: UtcDatetime
    type: str
    correlation_id: str
    component: str
    component_version: str
    actor: str
    outcome: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str
