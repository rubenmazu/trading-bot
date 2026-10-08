"""Portul `BrokerAdapter`: protocol, capabilități, cereri, confirmări și snapshot (Req 3.1–3.4).

Toate adaptoarele (sim, fake, demo) primesc ordine aprobate ca `OrderRequest` și emit numai
modele normalizate `ExecutionEvent` (`qts.core.models`), indiferent de formatul brokerului.

Capabilități (3.4)
    `BrokerCapabilities` descrie ce acceptă brokerul. `check_request()` verifică o cerere față
    de capabilități și instrument și întoarce primul motiv de respingere, cu un `ReasonCode`
    stabil. Codurile `CAPABILITY_UNSUPPORTED_*` înseamnă că brokerul nu oferă capabilitatea
    cerută; celelalte coduri descriu cereri invalide sau stări incompatibile.

Idempotență
    `submit()` este idempotent pe `client_order_id`: aceeași cerere întoarce aceeași confirmare
    fără un ordin nou; același identificator cu alt conținut este respins cu
    `CLIENT_ORDER_ID_CONFLICT`.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from qts.core.models import (
    AssetClass,
    Dec,
    ExecutionEvent,
    Frozen,
    Instrument,
    Order,
    OrderIntent,
    OrderState,
    OrderType,
    Side,
    UtcDatetime,
)
from qts.core.money import REPORTING_CURRENCY, ZERO

__all__ = [
    "BrokerAdapter",
    "BrokerCapabilities",
    "BrokerOrderStatus",
    "BrokerSnapshot",
    "CancelAck",
    "Environment",
    "OrderRequest",
    "ReasonCode",
    "Rejection",
    "SubmitAck",
    "TimeInForce",
    "check_request",
    "order_request_from",
]

Environment = Literal["sim", "demo", "live"]
TimeInForce = Literal["GTC", "DAY", "IOC"]


class ReasonCode(StrEnum):
    """Coduri de motiv stabile pentru respingerile adaptorului."""

    CAPABILITY_UNSUPPORTED_INSTRUMENT = "CAPABILITY_UNSUPPORTED_INSTRUMENT"
    CAPABILITY_UNSUPPORTED_ASSET_CLASS = "CAPABILITY_UNSUPPORTED_ASSET_CLASS"
    CAPABILITY_UNSUPPORTED_DERIVATIVE = "CAPABILITY_UNSUPPORTED_DERIVATIVE"
    CAPABILITY_UNSUPPORTED_LEVERAGE = "CAPABILITY_UNSUPPORTED_LEVERAGE"
    CAPABILITY_UNSUPPORTED_ORDER_TYPE = "CAPABILITY_UNSUPPORTED_ORDER_TYPE"
    CAPABILITY_UNSUPPORTED_TIME_IN_FORCE = "CAPABILITY_UNSUPPORTED_TIME_IN_FORCE"
    CAPABILITY_UNSUPPORTED_FRACTIONAL = "CAPABILITY_UNSUPPORTED_FRACTIONAL"
    CAPABILITY_UNSUPPORTED_SHORT = "CAPABILITY_UNSUPPORTED_SHORT"
    INVALID_QTY = "INVALID_QTY"
    INVALID_PRICE = "INVALID_PRICE"
    CLIENT_ORDER_ID_CONFLICT = "CLIENT_ORDER_ID_CONFLICT"
    UNKNOWN_ORDER = "UNKNOWN_ORDER"
    ORDER_NOT_OPEN = "ORDER_NOT_OPEN"


# --------------------------------------------------------------------------- modele


class BrokerCapabilities(Frozen):
    """Capabilitățile declarate ale brokerului. `instruments=None` înseamnă orice simbol."""

    order_types: frozenset[OrderType]
    time_in_force: frozenset[TimeInForce]
    asset_classes: frozenset[AssetClass]
    fractional: bool = False
    short: bool = False
    leverage: bool = False
    derivatives: bool = False
    instruments: frozenset[str] | None = None

    @model_validator(mode="after")
    def _check(self) -> BrokerCapabilities:
        if not self.order_types or not self.time_in_force:
            raise ValueError("brokerul trebuie să suporte cel puțin un tip de ordin și un TIF")
        return self


class OrderRequest(Frozen):
    """Cererea trimisă brokerului pentru un ordin aprobat (Req 3.2)."""

    client_order_id: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    side: Side
    qty: Dec
    order_type: OrderType = "MARKET"
    limit_price: Dec | None = None
    time_in_force: TimeInForce = "GTC"

    @model_validator(mode="after")
    def _check(self) -> OrderRequest:
        if self.qty <= 0:
            raise ValueError("qty trebuie să fie > 0")
        if self.order_type == "LIMIT":
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError("ordinul LIMIT necesită limit_price > 0")
        elif self.limit_price is not None:
            raise ValueError("ordinul MARKET nu acceptă limit_price")
        return self


class SubmitAck(Frozen):
    client_order_id: str
    accepted: bool
    ts: UtcDatetime
    broker_order_id: str | None = None
    reason_code: ReasonCode | None = None
    detail: str = ""

    @model_validator(mode="after")
    def _check(self) -> SubmitAck:
        if self.accepted == (self.reason_code is not None):
            raise ValueError("reason_code este obligatoriu exact pentru cererile respinse")
        return self


class CancelAck(Frozen):
    client_order_id: str
    accepted: bool
    ts: UtcDatetime
    reason_code: ReasonCode | None = None
    detail: str = ""

    @model_validator(mode="after")
    def _check(self) -> CancelAck:
        if self.accepted == (self.reason_code is not None):
            raise ValueError("reason_code este obligatoriu exact pentru anulările respinse")
        return self


class BrokerOrderStatus(Frozen):
    client_order_id: str
    instrument: str
    side: Side
    qty: Dec
    state: OrderState
    filled_qty: Dec = ZERO
    avg_fill_price: Dec | None = None
    broker_order_id: str | None = None
    reason_code: ReasonCode | None = None

    @property
    def remaining_qty(self) -> Decimal:
        return self.qty - self.filled_qty


class BrokerSnapshot(Frozen):
    """Starea completă raportată de broker: ordine, execuții, poziții și numerar."""

    ts: UtcDatetime
    environment: Environment
    account_id: str
    orders: tuple[BrokerOrderStatus, ...]
    execution_ids: tuple[str, ...]
    positions: dict[str, Dec]
    cash: Dec
    currency: str = REPORTING_CURRENCY
    complete: bool = True  # False: snapshot parțial, nu poate fi folosit la reconciliere


# --------------------------------------------------------------------------- protocol


@runtime_checkable
class BrokerAdapter(Protocol):
    @property
    def environment(self) -> Environment: ...

    @property
    def account_id(self) -> str: ...

    def capabilities(self) -> BrokerCapabilities: ...

    def submit(self, req: OrderRequest) -> SubmitAck: ...  # idempotent pe client_order_id

    def cancel(self, client_order_id: str) -> CancelAck: ...

    def snapshot(self) -> BrokerSnapshot: ...  # ordine, execuții, poziții, numerar

    def events(self) -> Iterator[ExecutionEvent]: ...


# --------------------------------------------------------------------------- verificări


class Rejection(Frozen):
    code: ReasonCode
    detail: str


def _is_integral(value: Decimal) -> bool:
    return value == value.to_integral_value()


def _is_multiple(value: Decimal, step: Decimal) -> bool:
    return _is_integral(value / step)


def check_request(
    caps: BrokerCapabilities,
    req: OrderRequest,
    instrument: Instrument | None,
    *,
    available_qty: Decimal = ZERO,
) -> Rejection | None:
    """Primul motiv de respingere pentru `req` sau `None` dacă cererea poate fi acceptată.

    `available_qty` este cantitatea deținută și nerezervată de alte vânzări deschise; o vânzare
    peste ea ar deschide o poziție scurtă.
    """
    if instrument is None or (
        caps.instruments is not None and req.instrument not in caps.instruments
    ):
        return Rejection(
            code=ReasonCode.CAPABILITY_UNSUPPORTED_INSTRUMENT,
            detail=f"instrumentul {req.instrument} nu este oferit de broker",
        )
    if instrument.asset_class not in caps.asset_classes:
        return Rejection(
            code=ReasonCode.CAPABILITY_UNSUPPORTED_ASSET_CLASS,
            detail=f"clasa de active {instrument.asset_class} nu este oferită",
        )
    if instrument.is_derivative and not caps.derivatives:
        return Rejection(
            code=ReasonCode.CAPABILITY_UNSUPPORTED_DERIVATIVE, detail="derivatele nu sunt oferite"
        )
    if instrument.requires_leverage and not caps.leverage:
        return Rejection(
            code=ReasonCode.CAPABILITY_UNSUPPORTED_LEVERAGE, detail="levierul nu este oferit"
        )
    if req.order_type not in caps.order_types:
        return Rejection(
            code=ReasonCode.CAPABILITY_UNSUPPORTED_ORDER_TYPE,
            detail=f"tipul de ordin {req.order_type} nu este oferit",
        )
    if req.time_in_force not in caps.time_in_force:
        return Rejection(
            code=ReasonCode.CAPABILITY_UNSUPPORTED_TIME_IN_FORCE,
            detail=f"time-in-force {req.time_in_force} nu este oferit",
        )
    if not _is_integral(req.qty) and not (caps.fractional and instrument.fractional):
        return Rejection(
            code=ReasonCode.CAPABILITY_UNSUPPORTED_FRACTIONAL,
            detail=f"cantitatea fracționară {req.qty} nu este oferită pentru {req.instrument}",
        )
    if req.qty < instrument.min_qty or not _is_multiple(req.qty, instrument.qty_step):
        return Rejection(
            code=ReasonCode.INVALID_QTY,
            detail=(
                f"cantitatea {req.qty} trebuie să fie >= {instrument.min_qty} și multiplu de "
                f"{instrument.qty_step}"
            ),
        )
    if req.limit_price is not None and not _is_multiple(req.limit_price, instrument.tick_size):
        return Rejection(
            code=ReasonCode.INVALID_PRICE,
            detail=f"limit_price {req.limit_price} nu este multiplu de {instrument.tick_size}",
        )
    shorting = req.side == "SELL" and req.qty > available_qty
    if (shorting or instrument.requires_short) and not caps.short:
        return Rejection(
            code=ReasonCode.CAPABILITY_UNSUPPORTED_SHORT,
            detail=f"vânzarea {req.qty} depășește cantitatea disponibilă {available_qty}",
        )
    return None


def order_request_from(
    order: Order, intent: OrderIntent, *, time_in_force: TimeInForce = "GTC"
) -> OrderRequest:
    """Transformă un ordin aprobat de OMS în cererea către broker (Req 3.2)."""
    if order.intent_id != intent.intent_id or order.instrument != intent.instrument:
        raise ValueError("ordinul și Order_Intent nu corespund")
    return OrderRequest(
        client_order_id=order.client_order_id,
        instrument=order.instrument,
        side=order.side,
        qty=order.qty,
        order_type=intent.order_type,
        limit_price=intent.limit_price,
        time_in_force=time_in_force,
    )
