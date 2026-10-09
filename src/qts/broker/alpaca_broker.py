"""`AlpacaBrokerAdapter`: broker Alpaca **paper** pentru modul Demo (Req 1.2, 2.1, 3.1–3.4).

Modul Demo = cont Alpaca cu bani simulați, dar API real: ordinele pleacă prin API-ul Alpaca
numai către endpoint-ul **paper** (`paper-api.alpaca.markets`). Nu există nicio cale către Live:
adaptorul refuză la construcție orice endpoint care nu este cel paper cunoscut (bariera de
siguranță locală, în plus față de `safety/stage.py` și `Fail_Safe_Block`).

Principii respectate din restul sistemului (ca `qts.data.alpaca_source`):

- **Client injectabil**: protocolul `AlpacaTradingClient` abstractizează apelurile de rețea, cu
  un transport neutru (dataclass-uri `Alpaca*`), deci adaptorul nu atinge niciodată tipurile
  `alpaca-py`. Testele injectează un client fals determinist, fără rețea și fără chei.
- **Fără float pe calea banilor**: valorile sosesc ca `str`/`float` de la Alpaca și devin
  `Decimal` prin reprezentarea text (`str(float)`), niciodată direct din `float` binar.
- **Timp UTC**: toate momentele sunt normalizate la UTC (`ensure_utc`).
- **Secrete**: cheile API vin din `Secret_Store` prin `from_secret_store`, dezvăluite o singură
  dată și pasate fabricii de client; nu sunt stocate în adaptor și nu apar în config.
- **Idempotență și semantică de protocol**: `submit`/`cancel`/`snapshot`/`events` reproduc exact
  contractul din `SimBroker` (dedupe pe `client_order_id`, `CLIENT_ORDER_ID_CONFLICT`,
  `UNKNOWN_ORDER`/`ORDER_NOT_OPEN`, `seq` contiguu pe ordin, `broker_exec_id` unic).

Snapshot fail-closed (Req 15.9)
    Dacă contul sau ordinele nu pot fi aduse complet de la Alpaca, `snapshot()` întoarce
    `complete=False`; reconcilierea tratează un snapshot incomplet drept declanșator de
    Kill_Switch, deci nu riscăm decizii pe o stare parțială.

Monedă (caveat FX)
    Conturile Alpaca paper sunt în USD. Numerarul și pozițiile sunt raportate în moneda contului,
    fără conversie silențioasă; `BrokerSnapshot.currency` reflectă moneda reală a contului, iar
    conversia în EUR (`REPORTING_CURRENCY`) ține de reconciliere/raportare, nu de adaptor.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Final, Protocol, runtime_checkable

from qts.core.clock import Clock, ensure_utc
from qts.core.models import (
    ExecKind,
    ExecutionEvent,
    Instrument,
    OrderState,
)
from qts.core.money import ZERO, dec
from qts.secrets.store import Identity, SecretRef, SecretStore

from .adapter import (
    BrokerCapabilities,
    BrokerOrderStatus,
    BrokerSnapshot,
    CancelAck,
    Environment,
    OrderRequest,
    ReasonCode,
    SubmitAck,
    check_request,
)

__all__ = [
    "ALPACA_PAPER_ENDPOINT",
    "DEFAULT_ALPACA_CAPABILITIES",
    "AlpacaAccount",
    "AlpacaBrokerAdapter",
    "AlpacaBrokerCredentials",
    "AlpacaClientFactory",
    "AlpacaExecution",
    "AlpacaOrderAck",
    "AlpacaPosition",
    "AlpacaTradingClient",
    "NonPaperEndpointError",
]

# Endpoint-ul paper canonic al Alpaca. Orice alt endpoint este refuzat la construcție.
ALPACA_PAPER_ENDPOINT: Final = "https://paper-api.alpaca.markets"

# Capabilități paper: acțiuni și ETF-uri cash US, fracționar, long-only (fără short, levier sau
# derivate), ca să corespundă Risk_Engine. MARKET/LIMIT și GTC/DAY sunt oferite de Alpaca.
DEFAULT_ALPACA_CAPABILITIES: Final = BrokerCapabilities(
    order_types=frozenset({"MARKET", "LIMIT"}),
    time_in_force=frozenset({"GTC", "DAY"}),
    asset_classes=frozenset({"etf", "stock"}),
    fractional=True,
    short=False,
    leverage=False,
    derivatives=False,
)

_OPEN_STATES: Final = frozenset({OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED})

# Starea raportată de Alpaca pentru un ordin, mapată pe `OrderState` intern. Numai stările pe
# care le poate întoarce un ordin paper long-only; necunoscutele devin `UNKNOWN` (fail-closed).
_STATE_BY_ALPACA: Final[dict[str, OrderState]] = {
    "new": OrderState.ACKNOWLEDGED,
    "accepted": OrderState.ACKNOWLEDGED,
    "pending_new": OrderState.ACKNOWLEDGED,
    "partially_filled": OrderState.PARTIALLY_FILLED,
    "filled": OrderState.FILLED,
    "canceled": OrderState.CANCELLED,
    "cancelled": OrderState.CANCELLED,
    "expired": OrderState.EXPIRED,
    "rejected": OrderState.REJECTED_BROKER,
}


# --------------------------------------------------------------------------- transport neutru


@dataclass(frozen=True, slots=True)
class AlpacaOrderAck:
    """Confirmarea unui `submit`, așa cum o întoarce clientul Alpaca (transport neutru)."""

    broker_order_id: str
    accepted: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AlpacaPosition:
    """O poziție deținută, cu cantitatea ca `str` (devine `Decimal` în adaptor)."""

    symbol: str
    qty: str


@dataclass(frozen=True, slots=True)
class AlpacaAccount:
    """Contul: numerar, moneda contului și pozițiile curente.

    `complete=False` înseamnă că Alpaca nu a putut raporta integral contul (de exemplu, lista de
    poziții a eșuat); adaptorul propagă aceasta într-un snapshot `complete=False`.
    """

    cash: str
    currency: str
    positions: Sequence[AlpacaPosition] = ()
    complete: bool = True


@dataclass(frozen=True, slots=True)
class AlpacaExecution:
    """Un eveniment de execuție/stare (trade update) normalizat de la Alpaca.

    `kind` folosește aceleași denumiri ca stările Alpaca (`fill`, `partial_fill`, `canceled`,
    `rejected`, `expired`, `new`/`accepted`); adaptorul le mapează pe `ExecKind` și emite
    `ExecutionEvent` cu `seq` contiguu pe ordin.
    """

    client_order_id: str
    kind: str
    ts: datetime
    qty: str | None = None
    price: str | None = None
    commission: str | None = None
    reason: str | None = None


@runtime_checkable
class AlpacaTradingClient(Protocol):
    """Clientul de tranzacționare Alpaca. Abstractizat pentru teste fără rețea."""

    @property
    def endpoint(self) -> str:
        """Endpoint-ul configurat; adaptorul verifică la construcție că este cel paper."""
        ...

    def submit_order(self, req: OrderRequest) -> AlpacaOrderAck: ...

    def cancel_order(self, broker_order_id: str) -> None: ...

    def get_account(self) -> AlpacaAccount: ...

    def poll_executions(self) -> Sequence[AlpacaExecution]:
        """Întoarce execuțiile noi (trade updates) de la ultimul apel, în ordine cronologică."""
        ...


# Fabrica de client real: construiește un `AlpacaTradingClient` din cheile deja dezvăluite și
# endpoint-ul paper. Definită înaintea adaptorului, ca adnotarea `from_secret_store` să nu
# aibă nevoie de ghilimele.
AlpacaClientFactory = Callable[[str, str, str], AlpacaTradingClient]


@dataclass(frozen=True, slots=True)
class AlpacaBrokerCredentials:
    """Referințele (nu valorile) cheilor API Alpaca, rezolvate din Secret_Store (Req 23)."""

    api_key_ref: SecretRef
    api_secret_ref: SecretRef


class NonPaperEndpointError(RuntimeError):
    """Adaptorul a fost îndreptat către un endpoint care nu este cel paper Alpaca (Req 2)."""


# --------------------------------------------------------------------------- stare internă


@dataclass(slots=True)
class _AlpacaOrder:
    """Starea unui ordin urmărit local: cererea, confirmarea și progresul execuțiilor."""

    req: OrderRequest
    ack: SubmitAck
    state: OrderState
    broker_order_id: str | None
    next_seq: int = 1
    filled_qty: Decimal = ZERO
    notional: Decimal = ZERO  # Σ preț × cantitate, moneda contului
    cancel_ack: CancelAck | None = None

    @property
    def remaining(self) -> Decimal:
        return self.req.qty - self.filled_qty

    def status(self) -> BrokerOrderStatus:
        avg = self.notional / self.filled_qty if self.filled_qty > 0 else None
        return BrokerOrderStatus(
            client_order_id=self.req.client_order_id,
            instrument=self.req.instrument,
            side=self.req.side,
            qty=self.req.qty,
            state=self.state,
            filled_qty=self.filled_qty,
            avg_fill_price=avg,
            broker_order_id=self.broker_order_id,
            reason_code=self.ack.reason_code,
        )


class AlpacaBrokerAdapter:
    """`BrokerAdapter` Alpaca paper pentru Demo. Mediul raportat este `demo`.

    Clientul (`client`) este injectat: la rulare un client real peste `alpaca-py` construit cu
    cheile din Secret_Store; în teste un client fals determinist. Instrumentele configurate
    permit `check_request` să valideze cereri față de capabilități și instrument.
    """

    def __init__(
        self,
        *,
        account_id: str,
        instruments: Iterable[Instrument],
        client: AlpacaTradingClient,
        clock: Clock,
        capabilities: BrokerCapabilities = DEFAULT_ALPACA_CAPABILITIES,
    ) -> None:
        endpoint = client.endpoint
        if not _is_paper_endpoint(endpoint):
            raise NonPaperEndpointError(
                f"endpoint {endpoint!r} nu este endpoint-ul paper Alpaca "
                f"({ALPACA_PAPER_ENDPOINT}); trading-ul live este interzis"
            )
        self._account_id = account_id
        self._instruments = {i.symbol: i for i in instruments}
        self._client = client
        self._clock = clock
        self._caps = capabilities
        self._orders: dict[str, _AlpacaOrder] = {}
        self._queue: deque[ExecutionEvent] = deque()
        self._exec_ids: list[str] = []
        self._seen_exec: set[str] = set()

    # ------------------------------------------------------------------ construcție

    @classmethod
    def from_secret_store(
        cls,
        *,
        account_id: str,
        instruments: Iterable[Instrument],
        store: SecretStore,
        credentials: AlpacaBrokerCredentials,
        requester: Identity,
        clock: Clock,
        environment: str = "demo",
        endpoint: str = ALPACA_PAPER_ENDPOINT,
        capabilities: BrokerCapabilities = DEFAULT_ALPACA_CAPABILITIES,
        client_factory: AlpacaClientFactory | None = None,
    ) -> AlpacaBrokerAdapter:
        """Construiește adaptorul rezolvând cheile din Secret_Store (Req 23.1, 23.4).

        `client_factory` creează clientul concret din cheile dezvăluite și endpoint; implicit
        `_default_client_factory`, care importă `alpaca-py` doar la rulare. Endpoint-ul trebuie
        să fie cel paper, altfel construcția eșuează cu `NonPaperEndpointError`.
        """
        if not _is_paper_endpoint(endpoint):
            raise NonPaperEndpointError(
                f"endpoint {endpoint!r} nu este endpoint-ul paper Alpaca ({ALPACA_PAPER_ENDPOINT})"
            )
        api_key = store.get(credentials.api_key_ref, requester, environment)
        api_secret = store.get(credentials.api_secret_ref, requester, environment)
        factory = client_factory if client_factory is not None else _default_client_factory
        client = factory(api_key.reveal(), api_secret.reveal(), endpoint)
        return cls(
            account_id=account_id,
            instruments=instruments,
            client=client,
            clock=clock,
            capabilities=capabilities,
        )

    # ------------------------------------------------------------------ BrokerAdapter

    @property
    def environment(self) -> Environment:
        return "demo"

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def endpoint(self) -> str:
        return self._client.endpoint

    def capabilities(self) -> BrokerCapabilities:
        return self._caps

    def submit(self, req: OrderRequest) -> SubmitAck:
        existing = self._orders.get(req.client_order_id)
        if existing is not None:
            if existing.req == req:
                return existing.ack
            return SubmitAck(
                client_order_id=req.client_order_id,
                accepted=False,
                ts=self._clock.now(),
                reason_code=ReasonCode.CLIENT_ORDER_ID_CONFLICT,
                detail="client_order_id refolosit cu alt conținut",
            )
        now = self._clock.now()
        rejection = check_request(
            self._caps,
            req,
            self._instruments.get(req.instrument),
            available_qty=self._available_to_sell(req.instrument),
        )
        if rejection is not None:
            ack = SubmitAck(
                client_order_id=req.client_order_id,
                accepted=False,
                ts=now,
                reason_code=rejection.code,
                detail=rejection.detail,
            )
            order = _AlpacaOrder(req, ack, OrderState.REJECTED_BROKER, None)
            self._orders[req.client_order_id] = order
            self._emit(order, ExecKind.REJECT, now, reason=rejection.code.value)
            return ack
        broker_ack = self._client.submit_order(req)
        if not broker_ack.accepted:
            ack = SubmitAck(
                client_order_id=req.client_order_id,
                accepted=False,
                ts=now,
                reason_code=ReasonCode.ORDER_NOT_OPEN,
                detail=broker_ack.reason or "ordin respins de broker",
            )
            order = _AlpacaOrder(req, ack, OrderState.REJECTED_BROKER, broker_ack.broker_order_id)
            self._orders[req.client_order_id] = order
            self._emit(order, ExecKind.REJECT, now, reason=ack.detail)
            return ack
        ack = SubmitAck(
            client_order_id=req.client_order_id,
            accepted=True,
            ts=now,
            broker_order_id=broker_ack.broker_order_id,
        )
        order = _AlpacaOrder(req, ack, OrderState.ACKNOWLEDGED, broker_ack.broker_order_id)
        self._orders[req.client_order_id] = order
        self._emit(order, ExecKind.ACK, now)
        return ack

    def cancel(self, client_order_id: str) -> CancelAck:
        now = self._clock.now()
        order = self._orders.get(client_order_id)
        if order is None:
            return CancelAck(
                client_order_id=client_order_id,
                accepted=False,
                ts=now,
                reason_code=ReasonCode.UNKNOWN_ORDER,
                detail="ordin necunoscut",
            )
        if order.cancel_ack is not None:
            return order.cancel_ack
        if order.state not in _OPEN_STATES:
            detail = f"ordinul este în starea {order.state.value}"
            self._emit(order, ExecKind.CANCEL_REJECTED, now, reason=ReasonCode.ORDER_NOT_OPEN.value)
            return CancelAck(
                client_order_id=client_order_id,
                accepted=False,
                ts=now,
                reason_code=ReasonCode.ORDER_NOT_OPEN,
                detail=detail,
            )
        if order.broker_order_id is not None:
            self._client.cancel_order(order.broker_order_id)
        remaining = order.remaining
        order.state = OrderState.CANCELLED
        order.cancel_ack = CancelAck(client_order_id=client_order_id, accepted=True, ts=now)
        self._emit(order, ExecKind.CANCELLED, now, qty=remaining)
        return order.cancel_ack

    def snapshot(self) -> BrokerSnapshot:
        now = self._clock.now()
        # Reconciliere: preluăm întâi execuțiile noi, ca stările ordinelor să reflecte brokerul.
        # Evenimentele produse rămân în coadă pentru următorul `events()`, deci nu se pierd.
        for raw in self._client.poll_executions():
            self._ingest(raw)
        try:
            account = self._client.get_account()
        except Exception:
            # Orice eșec de rețea/stare → snapshot parțial (fail-closed, Req 15.9).
            return self._incomplete_snapshot(now)
        positions: dict[str, Decimal] = {}
        try:
            for pos in account.positions:
                qty = dec(pos.qty)
                if qty != 0:
                    positions[pos.symbol] = qty
            cash = dec(account.cash)
        except (ValueError, ArithmeticError):
            return self._incomplete_snapshot(now)
        return BrokerSnapshot(
            ts=now,
            environment="demo",
            account_id=self._account_id,
            orders=tuple(o.status() for o in self._orders.values()),
            execution_ids=tuple(self._exec_ids),
            positions=dict(sorted(positions.items())),
            cash=cash,
            currency=account.currency,
            complete=account.complete,
        )

    def events(self) -> Iterator[ExecutionEvent]:
        """Aduce execuțiile noi de la client, le normalizează și golește coada."""
        for raw in self._client.poll_executions():
            self._ingest(raw)
        while self._queue:
            yield self._queue.popleft()

    # ------------------------------------------------------------------ intern

    def _incomplete_snapshot(self, now: datetime) -> BrokerSnapshot:
        """Snapshot parțial (fail-closed): ordinele locale, dar fără cont complet."""
        return BrokerSnapshot(
            ts=now,
            environment="demo",
            account_id=self._account_id,
            orders=tuple(o.status() for o in self._orders.values()),
            execution_ids=tuple(self._exec_ids),
            positions={},
            cash=ZERO,
            complete=False,
        )

    def _available_to_sell(self, instrument: str) -> Decimal:
        reserved = sum(
            (
                o.remaining
                for o in self._orders.values()
                if o.req.instrument == instrument
                and o.req.side == "SELL"
                and o.state in _OPEN_STATES
            ),
            ZERO,
        )
        held = sum(
            (
                _signed_filled(o)
                for o in self._orders.values()
                if o.req.instrument == instrument
            ),
            ZERO,
        )
        return held - reserved

    def _ingest(self, raw: AlpacaExecution) -> None:
        """Traduce un trade update Alpaca în `ExecutionEvent`, actualizând starea locală."""
        order = self._orders.get(raw.client_order_id)
        if order is None or order.cancel_ack is not None:
            return  # update pentru un ordin necunoscut local sau deja anulat
        kind = raw.kind.lower()
        ts = ensure_utc(raw.ts)
        if kind in ("partial_fill", "fill"):
            if order.state not in _OPEN_STATES:
                return
            qty = dec(raw.qty) if raw.qty is not None else ZERO
            price = dec(raw.price) if raw.price is not None else ZERO
            if qty <= 0 or price <= 0 or qty > order.remaining:
                return
            commission = dec(raw.commission) if raw.commission is not None else None
            order.filled_qty += qty
            order.notional += price * qty
            done = order.remaining == 0
            order.state = OrderState.FILLED if done else OrderState.PARTIALLY_FILLED
            self._emit(
                order,
                ExecKind.FILL if done else ExecKind.PARTIAL_FILL,
                ts,
                qty=qty,
                price=price,
                commission=commission,
            )
        elif kind in ("canceled", "cancelled"):
            if order.state not in _OPEN_STATES:
                return
            remaining = order.remaining
            order.state = OrderState.CANCELLED
            order.cancel_ack = CancelAck(
                client_order_id=order.req.client_order_id, accepted=True, ts=ts
            )
            self._emit(order, ExecKind.CANCELLED, ts, qty=remaining)
        elif kind == "expired":
            if order.state not in _OPEN_STATES:
                return
            remaining = order.remaining
            order.state = OrderState.EXPIRED
            self._emit(order, ExecKind.EXPIRED, ts, qty=remaining)
        # „new"/„accepted" și alte stări tranzitorii nu produc evenimente suplimentare: ACK a
        # fost deja emis la `submit`.

    def _emit(
        self,
        order: _AlpacaOrder,
        kind: ExecKind,
        ts: datetime,
        *,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        commission: Decimal | None = None,
        reason: str | None = None,
    ) -> None:
        seq = order.next_seq
        order.next_seq += 1
        coid = order.req.client_order_id
        exec_id = f"{self._account_id}:{coid}:{seq}"
        event = ExecutionEvent(
            broker_exec_id=exec_id,
            client_order_id=coid,
            kind=kind,
            qty=qty,
            price=price,
            commission=commission,
            broker_order_id=order.broker_order_id,
            reason=reason,
            ts_broker=ts,
            ts_receipt=ts,
            seq=seq,
        )
        self._exec_ids.append(exec_id)
        self._queue.append(event)


# --------------------------------------------------------------------------- ajutoare


def _signed_filled(order: _AlpacaOrder) -> Decimal:
    sign = Decimal(1) if order.req.side == "BUY" else Decimal(-1)
    return sign * order.filled_qty


def _is_paper_endpoint(endpoint: str) -> bool:
    """Adevărat numai pentru endpoint-ul paper Alpaca (gazdă `paper-api.alpaca.markets`)."""
    normalized = endpoint.strip().rstrip("/").lower()
    return normalized == ALPACA_PAPER_ENDPOINT.rstrip("/").lower()


def _default_client_factory(  # pragma: no cover - necesită rețea și pachetul extern
    api_key: str, api_secret: str, endpoint: str
) -> AlpacaTradingClient:
    """Construiește clientul real (necesită pachetul `alpaca-py` și acces la rețea)."""
    if not _is_paper_endpoint(endpoint):
        raise NonPaperEndpointError(f"endpoint {endpoint!r} nu este endpoint-ul paper Alpaca")
    return AlpacaPyTradingClient(api_key, api_secret)


class AlpacaPyTradingClient:  # pragma: no cover - necesită rețea și pachetul extern
    """Client real peste `alpaca-py`, cu `paper=True` forțat. Construit doar la rulare.

    Importul pachetului este amânat în constructor, ca modulul `qts.broker.alpaca_broker` să
    poată fi importat (și testat) fără `alpaca-py` instalat. `TradingClient(paper=True)` țintește
    exclusiv endpoint-ul paper; `endpoint` raportează acest fapt pentru verificarea adaptorului.
    """

    def __init__(self, api_key: str, api_secret: str) -> None:
        from alpaca.trading.client import TradingClient

        # paper=True → SDK-ul folosește exclusiv endpoint-ul paper (fără trading live).
        self._client = TradingClient(api_key, api_secret, paper=True)
        self._endpoint = ALPACA_PAPER_ENDPOINT
        self._last_seen: set[str] = set()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def submit_order(self, req: OrderRequest) -> AlpacaOrderAck:
        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = OrderSide.BUY if req.side == "BUY" else OrderSide.SELL
        tif = {
            "GTC": TimeInForce.GTC,
            "DAY": TimeInForce.DAY,
            "IOC": TimeInForce.IOC,
        }[req.time_in_force]
        qty = float(req.qty)
        if req.order_type == "LIMIT":
            if req.limit_price is None:  # garantat de OrderRequest; verificare defensivă
                raise ValueError("ordinul LIMIT necesită limit_price")
            request = LimitOrderRequest(
                symbol=req.instrument,
                qty=qty,
                side=side,
                time_in_force=tif,
                limit_price=float(req.limit_price),
                client_order_id=req.client_order_id,
                type=OrderType.LIMIT,
            )
        else:
            request = MarketOrderRequest(
                symbol=req.instrument,
                qty=qty,
                side=side,
                time_in_force=tif,
                client_order_id=req.client_order_id,
                type=OrderType.MARKET,
            )
        order = self._client.submit_order(request)
        status = str(getattr(order, "status", "")).lower()
        accepted = "rejected" not in status
        return AlpacaOrderAck(
            broker_order_id=str(order.id),
            accepted=accepted,
            reason="" if accepted else status,
        )

    def cancel_order(self, broker_order_id: str) -> None:
        self._client.cancel_order_by_id(broker_order_id)

    def get_account(self) -> AlpacaAccount:
        account = self._client.get_account()
        positions = self._client.get_all_positions()
        return AlpacaAccount(
            cash=str(account.cash),
            currency=str(getattr(account, "currency", "USD")),
            positions=tuple(AlpacaPosition(symbol=p.symbol, qty=str(p.qty)) for p in positions),
            complete=True,
        )

    def poll_executions(self) -> Sequence[AlpacaExecution]:
        from alpaca.trading.requests import GetOrdersRequest

        orders = self._client.get_orders(filter=GetOrdersRequest(status="all"))
        out: list[AlpacaExecution] = []
        for order in orders:
            key = str(order.id)
            if key in self._last_seen:
                continue
            self._last_seen.add(key)
            status = str(getattr(order, "status", "")).lower()
            filled = getattr(order, "filled_qty", None)
            price = getattr(order, "filled_avg_price", None)
            out.append(
                AlpacaExecution(
                    client_order_id=str(order.client_order_id),
                    kind=status,
                    ts=ensure_utc(order.updated_at),
                    qty=str(filled) if filled is not None else None,
                    price=str(price) if price is not None else None,
                )
            )
        return out
