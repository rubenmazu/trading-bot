"""Teste pentru `AlpacaBrokerAdapter`: broker Alpaca paper pentru Demo (Req 1.2, 2.1, 3.1–3.4).

Toate testele folosesc un client Alpaca fals, determinist (fără rețea, fără pachetul `alpaca-py`
și fără chei reale). Verificăm: idempotența pe `client_order_id`, conflictul de identificator,
respingerile de capabilități, semantica de anulare, snapshot complet vs. incomplet (fail-closed),
contiguitatea `seq` și unicitatea `broker_exec_id`, integrarea cu Secret_Store și refuzul
endpoint-ului care nu este cel paper.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from qts.broker.adapter import OrderRequest, ReasonCode
from qts.broker.alpaca_broker import (
    ALPACA_PAPER_ENDPOINT,
    AlpacaAccount,
    AlpacaBrokerAdapter,
    AlpacaBrokerCredentials,
    AlpacaExecution,
    AlpacaOrderAck,
    AlpacaPosition,
    NonPaperEndpointError,
)
from qts.core.clock import SimClock
from qts.core.models import ExecKind, Instrument, OrderState
from qts.secrets.store import (
    Identity,
    InMemorySecretStore,
    SecretAccessDeniedError,
    SecretRef,
)

D = Decimal
NOW = datetime(2025, 1, 2, 9, 0, tzinfo=UTC)
SYMBOL = "SPY"
_DEFAULT_QTY = D(10)


def _instrument(symbol: str = SYMBOL, **kw: object) -> Instrument:
    base: dict[str, object] = {
        "symbol": symbol,
        "venue": "XNAS",
        "asset_class": "etf",
        "currency": "USD",
        "tick_size": D("0.01"),
        "qty_step": D(1),
        "min_qty": D(1),
        "fractional": True,
        "calendar_id": "XNAS",
    }
    base.update(kw)
    return Instrument(**base)


class FakeTradingClient:
    """Client Alpaca fals, cu un „seam" de test pentru a forța execuții determinist.

    Oglindește rolul lui `FakeBroker.fill`: testul (sau driverul contract) cheamă `push_fill`,
    `push_cancel`, `push_expire` pentru a programa trade updates pe care adaptorul le aduce prin
    `poll_executions`. Starea contului (`cash`, poziții) este configurabilă pentru snapshot.
    """

    def __init__(
        self,
        *,
        endpoint: str = ALPACA_PAPER_ENDPOINT,
        clock: SimClock | None = None,
        cash: str = "100000",
        currency: str = "USD",
        account_complete: bool = True,
        account_raises: bool = False,
        reject: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._clock = clock
        self.cash = cash
        self.currency = currency
        self.account_complete = account_complete
        self.account_raises = account_raises
        self.reject = reject
        self.submitted: list[OrderRequest] = []
        self.cancelled: list[str] = []
        self.positions: dict[str, Decimal] = {}
        self._pending: list[AlpacaExecution] = []
        self._accepted = 0

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def _now(self) -> datetime:
        return self._clock.now() if self._clock is not None else NOW

    def submit_order(self, req: OrderRequest) -> AlpacaOrderAck:
        self.submitted.append(req)
        if self.reject:
            return AlpacaOrderAck(broker_order_id="", accepted=False, reason="rejected")
        self._accepted += 1
        return AlpacaOrderAck(broker_order_id=f"ALP-{self._accepted:08d}", accepted=True)

    def cancel_order(self, broker_order_id: str) -> None:
        self.cancelled.append(broker_order_id)

    def get_account(self) -> AlpacaAccount:
        if self.account_raises:
            raise ConnectionError("account unavailable")
        positions = tuple(
            AlpacaPosition(symbol=s, qty=str(q)) for s, q in sorted(self.positions.items())
        )
        return AlpacaAccount(
            cash=self.cash,
            currency=self.currency,
            positions=positions,
            complete=self.account_complete,
        )

    def poll_executions(self) -> Sequence[AlpacaExecution]:
        batch = self._pending
        self._pending = []
        return batch

    # ---- seam de test ---------------------------------------------------------

    def push_fill(
        self, client_order_id: str, qty: Decimal, price: Decimal, *, commission: str | None = None
    ) -> None:
        self._pending.append(
            AlpacaExecution(
                client_order_id=client_order_id,
                kind="fill",  # adaptorul decide PARTIAL/FILL după cantitatea rămasă
                ts=self._now(),
                qty=str(qty),
                price=str(price),
                commission=commission,
            )
        )

    def push_cancel(self, client_order_id: str) -> None:
        self._pending.append(
            AlpacaExecution(client_order_id=client_order_id, kind="canceled", ts=self._now())
        )

    def push_expire(self, client_order_id: str) -> None:
        self._pending.append(
            AlpacaExecution(client_order_id=client_order_id, kind="expired", ts=self._now())
        )


def _adapter(
    client: FakeTradingClient,
    *,
    instruments: Sequence[Instrument] | None = None,
    clock: SimClock | None = None,
) -> AlpacaBrokerAdapter:
    return AlpacaBrokerAdapter(
        account_id="ALPACA-PAPER-1",
        instruments=instruments if instruments is not None else [_instrument()],
        client=client,
        clock=clock if clock is not None else SimClock(NOW),
    )


def _req(
    coid: str = "c1", side: str = "BUY", qty: Decimal = _DEFAULT_QTY, **kw: object
) -> OrderRequest:
    return OrderRequest.model_validate(
        {"client_order_id": coid, "instrument": SYMBOL, "side": side, "qty": qty, **kw}
    )


# --------------------------------------------------------------------------- identitate


def test_environment_is_demo_and_endpoint_is_paper() -> None:
    adapter = _adapter(FakeTradingClient())
    assert adapter.environment == "demo"
    assert adapter.account_id == "ALPACA-PAPER-1"
    assert adapter.endpoint == ALPACA_PAPER_ENDPOINT
    caps = adapter.capabilities()
    assert caps.fractional and not caps.short and not caps.leverage and not caps.derivatives


# --------------------------------------------------------------------------- siguranță paper


def test_refuses_non_paper_endpoint() -> None:
    live = FakeTradingClient(endpoint="https://api.alpaca.markets")
    with pytest.raises(NonPaperEndpointError):
        _adapter(live)


def test_refuses_empty_or_other_endpoint() -> None:
    with pytest.raises(NonPaperEndpointError):
        _adapter(FakeTradingClient(endpoint="https://example.com"))


def test_paper_endpoint_trailing_slash_accepted() -> None:
    client = FakeTradingClient(endpoint=ALPACA_PAPER_ENDPOINT + "/")
    adapter = _adapter(client)
    assert adapter.environment == "demo"


# --------------------------------------------------------------------------- idempotență


def test_submit_is_idempotent_on_client_order_id() -> None:
    client = FakeTradingClient()
    adapter = _adapter(client)
    first = adapter.submit(_req())
    assert first.accepted and first.broker_order_id
    assert adapter.submit(_req()) == first
    assert len(client.submitted) == 1  # al doilea submit nu ajunge la broker
    assert [e.kind for e in adapter.events()] == [ExecKind.ACK]


def test_same_client_order_id_different_content_conflicts() -> None:
    adapter = _adapter(FakeTradingClient())
    assert adapter.submit(_req()).accepted
    list(adapter.events())
    conflict = adapter.submit(_req(qty=D(11)))
    assert not conflict.accepted
    assert conflict.reason_code is ReasonCode.CLIENT_ORDER_ID_CONFLICT
    assert list(adapter.events()) == []


def test_broker_rejection_maps_to_order_not_open() -> None:
    adapter = _adapter(FakeTradingClient(reject=True))
    ack = adapter.submit(_req())
    assert not ack.accepted and ack.reason_code is ReasonCode.ORDER_NOT_OPEN
    [event] = list(adapter.events())
    assert event.kind is ExecKind.REJECT
    assert adapter.snapshot().orders[0].state is OrderState.REJECTED_BROKER


# --------------------------------------------------------------------------- capabilități


def test_short_sale_rejected() -> None:
    adapter = _adapter(FakeTradingClient())
    ack = adapter.submit(_req(side="SELL"))
    assert not ack.accepted and ack.reason_code is ReasonCode.CAPABILITY_UNSUPPORTED_SHORT
    [event] = list(adapter.events())
    assert event.kind is ExecKind.REJECT and event.seq == 1


def test_derivative_rejected() -> None:
    deriv = _instrument("ESFUT", asset_class="stock", is_derivative=True)
    adapter = _adapter(FakeTradingClient(), instruments=[deriv])
    ack = adapter.submit(_req(coid="d1").model_copy(update={"instrument": "ESFUT"}))
    assert not ack.accepted
    assert ack.reason_code is ReasonCode.CAPABILITY_UNSUPPORTED_DERIVATIVE


def test_unsupported_instrument_rejected() -> None:
    adapter = _adapter(FakeTradingClient())
    ack = adapter.submit(_req().model_copy(update={"instrument": "NOT-OFFERED"}))
    assert not ack.accepted
    assert ack.reason_code is ReasonCode.CAPABILITY_UNSUPPORTED_INSTRUMENT


# --------------------------------------------------------------------------- anulare


def test_cancel_unknown_order() -> None:
    adapter = _adapter(FakeTradingClient())
    ack = adapter.cancel("nope")
    assert not ack.accepted and ack.reason_code is ReasonCode.UNKNOWN_ORDER
    assert list(adapter.events()) == []


def test_cancel_open_order_reaches_broker_and_reports_remaining() -> None:
    client = FakeTradingClient()
    adapter = _adapter(client)
    adapter.submit(_req())
    list(adapter.events())
    ack = adapter.cancel("c1")
    assert ack.accepted and ack.reason_code is None
    assert client.cancelled == ["ALP-00000001"]
    assert adapter.cancel("c1") == ack  # repetarea nu produce eveniment nou
    [cancelled] = list(adapter.events())
    assert cancelled.kind is ExecKind.CANCELLED and cancelled.qty == D(10)
    assert adapter.snapshot().orders[0].state is OrderState.CANCELLED


def test_cancel_terminal_order_rejected() -> None:
    client = FakeTradingClient()
    adapter = _adapter(client)
    adapter.submit(_req())
    client.push_fill("c1", D(10), D("10.00"))
    list(adapter.events())
    ack = adapter.cancel("c1")
    assert not ack.accepted and ack.reason_code is ReasonCode.ORDER_NOT_OPEN
    [event] = list(adapter.events())
    assert event.kind is ExecKind.CANCEL_REJECTED


# --------------------------------------------------------------------------- flux de evenimente


def test_events_seq_contiguous_and_unique_ids() -> None:
    client = FakeTradingClient()
    adapter = _adapter(client)
    adapter.submit(_req())
    client.push_fill("c1", D(4), D("10.00"))
    client.push_fill("c1", D(4), D("10.00"))
    client.push_fill("c1", D(2), D("10.00"))
    events = list(adapter.events())
    assert [e.kind for e in events] == [
        ExecKind.ACK,
        ExecKind.PARTIAL_FILL,
        ExecKind.PARTIAL_FILL,
        ExecKind.FILL,
    ]
    assert [e.seq for e in events] == [1, 2, 3, 4]
    ids = [e.broker_exec_id for e in events]
    assert len(ids) == len(set(ids))
    assert ids[0] == "ALPACA-PAPER-1:c1:1"


def test_overfill_update_ignored() -> None:
    client = FakeTradingClient()
    adapter = _adapter(client)
    adapter.submit(_req(qty=D(5)))
    client.push_fill("c1", D(5), D("10.00"))
    client.push_fill("c1", D(1), D("10.00"))  # peste cantitatea rămasă → ignorat
    events = list(adapter.events())
    assert [e.kind for e in events] == [ExecKind.ACK, ExecKind.FILL]
    assert adapter.snapshot().orders[0].filled_qty == D(5)


def test_fill_updates_position_and_avg_price() -> None:
    client = FakeTradingClient()
    adapter = _adapter(client)
    adapter.submit(_req(qty=D(10)))
    client.push_fill("c1", D(4), D("10.00"))
    client.push_fill("c1", D(6), D("11.00"))
    list(adapter.events())
    client.positions = {SYMBOL: D(10)}
    [status] = adapter.snapshot().orders
    assert status.filled_qty == D(10)
    assert status.avg_fill_price == (D(40) + D(66)) / D(10)


# --------------------------------------------------------------------------- snapshot


def test_snapshot_complete_reports_cash_currency_positions() -> None:
    client = FakeTradingClient(cash="12345.67", currency="USD")
    client.positions = {SYMBOL: D(3)}
    adapter = _adapter(client)
    snap = adapter.snapshot()
    assert snap.complete
    assert snap.cash == D("12345.67")
    assert snap.currency == "USD"  # fără conversie silențioasă în EUR
    assert snap.positions == {SYMBOL: D(3)}


def test_snapshot_incomplete_when_account_flag_false() -> None:
    adapter = _adapter(FakeTradingClient(account_complete=False))
    assert adapter.snapshot().complete is False


def test_snapshot_incomplete_when_account_retrieval_raises() -> None:
    adapter = _adapter(FakeTradingClient(account_raises=True))
    snap = adapter.snapshot()
    assert snap.complete is False
    assert snap.positions == {}


# --------------------------------------------------------------------------- Secret_Store


def _store() -> InMemorySecretStore:
    return InMemorySecretStore(
        values={
            "qts/demo/alpaca_key": "KEYVALUE123",
            "qts/demo/alpaca_key_secret": "SECRETVAL456",
        },
        acl={
            "qts/demo/alpaca_key": {("demo-runner:demo", "demo")},
            "qts/demo/alpaca_key_secret": {("demo-runner:demo", "demo")},
        },
    )


def test_from_secret_store_resolves_keys() -> None:
    captured: dict[str, str] = {}

    def fake_factory(api_key: str, api_secret: str, endpoint: str) -> FakeTradingClient:
        captured["key"] = api_key
        captured["secret"] = api_secret
        captured["endpoint"] = endpoint
        return FakeTradingClient(endpoint=endpoint)

    adapter = AlpacaBrokerAdapter.from_secret_store(
        account_id="ALPACA-PAPER-1",
        instruments=[_instrument()],
        store=_store(),
        credentials=AlpacaBrokerCredentials(
            api_key_ref=SecretRef("qts/demo/alpaca_key"),
            api_secret_ref=SecretRef("qts/demo/alpaca_key_secret"),
        ),
        requester=Identity("demo-runner:demo"),
        clock=SimClock(NOW),
        environment="demo",
        client_factory=fake_factory,
    )
    assert adapter.environment == "demo"
    assert captured == {
        "key": "KEYVALUE123",
        "secret": "SECRETVAL456",
        "endpoint": ALPACA_PAPER_ENDPOINT,
    }


def test_from_secret_store_denies_unauthorized_identity() -> None:
    with pytest.raises(SecretAccessDeniedError):
        AlpacaBrokerAdapter.from_secret_store(
            account_id="ALPACA-PAPER-1",
            instruments=[_instrument()],
            store=_store(),
            credentials=AlpacaBrokerCredentials(
                api_key_ref=SecretRef("qts/demo/alpaca_key"),
                api_secret_ref=SecretRef("qts/demo/alpaca_key_secret"),
            ),
            requester=Identity("intruder"),
            clock=SimClock(NOW),
            environment="demo",
            client_factory=lambda k, s, e: FakeTradingClient(endpoint=e),
        )


def test_from_secret_store_refuses_non_paper_endpoint_before_resolving_keys() -> None:
    with pytest.raises(NonPaperEndpointError):
        AlpacaBrokerAdapter.from_secret_store(
            account_id="ALPACA-PAPER-1",
            instruments=[_instrument()],
            store=_store(),
            credentials=AlpacaBrokerCredentials(
                api_key_ref=SecretRef("qts/demo/alpaca_key"),
                api_secret_ref=SecretRef("qts/demo/alpaca_key_secret"),
            ),
            requester=Identity("demo-runner:demo"),
            clock=SimClock(NOW),
            environment="demo",
            endpoint="https://api.alpaca.markets",
            client_factory=lambda k, s, e: FakeTradingClient(endpoint=e),
        )
