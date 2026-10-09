"""Teste pentru `AlpacaDataAdapter`: mapare bare → MarketEvent, Decimal, validare (Req 5.1–5.4).

Toate testele folosesc un client Alpaca fals, determinist (fără rețea, fără pachetul `alpaca-py`
și fără chei). Verificăm: conversia float→Decimal cu rotunjire la tick, timpii UTC și
`ts_close = ts_open + interval`, excluderea barelor invalide, ordonarea cronologică și integrarea
cu Secret_Store (chei dezvăluite o singură dată, printr-o fabrică injectată).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from qts.core.models import Bar
from qts.data.alpaca_source import (
    AlpacaBar,
    AlpacaCredentials,
    AlpacaDataAdapter,
)
from qts.secrets.store import Identity, InMemorySecretStore, SecretRef

D = Decimal
NOW = datetime(2024, 3, 26, 20, 0, tzinfo=UTC)


class FakeAlpacaClient:
    """Client determinist: întoarce barele date, fără rețea."""

    def __init__(self, bars: Sequence[AlpacaBar]) -> None:
        self._bars = list(bars)
        self.calls: list[tuple[str, int, datetime, datetime]] = []

    def get_bars(
        self, symbol: str, interval_min: int, start: datetime, end: datetime
    ) -> Sequence[AlpacaBar]:
        self.calls.append((symbol, interval_min, start, end))
        return self._bars


def _ab(ts: datetime, close: float, *, o: float = 100.0, h: float = 101.0, lo: float = 99.0,
        v: float = 1000.0) -> AlpacaBar:
    return AlpacaBar(ts_open=ts, open=o, high=h, low=lo, close=close, volume=v)


def _adapter(client: FakeAlpacaClient, **kw: object) -> AlpacaDataAdapter:
    base: dict[str, object] = {
        "symbol": "SPY",
        "interval_min": 15,
        "tick_size": D("0.01"),
        "client": client,
        "clock_now": NOW,
    }
    base.update(kw)
    return AlpacaDataAdapter(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- mapare


def test_maps_bars_to_market_events() -> None:
    t0 = datetime(2024, 3, 26, 15, 0, tzinfo=UTC)
    client = FakeAlpacaClient([_ab(t0, 100.5), _ab(t0 + timedelta(minutes=15), 100.7)])
    events = _adapter(client).load()
    assert len(events) == 2
    first = events[0]
    assert first.source_id == "alpaca:SPY"
    assert first.instrument == "SPY"
    assert first.kind == "bar"
    assert isinstance(first.payload, Bar)
    assert first.payload.ts_close == t0 + timedelta(minutes=15)
    assert first.payload.interval_min == 15


def test_float_prices_become_decimal_rounded_to_tick() -> None:
    t0 = datetime(2024, 3, 26, 15, 0, tzinfo=UTC)
    # 100.123 cu tick 0.01 → 100.12 (half-even); fără eroare binară de float.
    client = FakeAlpacaClient([_ab(t0, 100.123, o=100.126, h=100.129, lo=100.121)])
    bar = _adapter(client).load()[0].payload
    assert isinstance(bar, Bar)
    assert bar.close == D("100.12")
    assert bar.open == D("100.13")
    assert isinstance(bar.close, Decimal)


def test_naive_timestamps_treated_as_utc() -> None:
    naive = datetime(2024, 3, 26, 15, 0)  # noqa: DTZ001 - test intenționat
    client = FakeAlpacaClient([_ab(naive, 100.0)])
    bar = _adapter(client).load()[0].payload
    assert isinstance(bar, Bar)
    assert bar.ts_open == datetime(2024, 3, 26, 15, 0, tzinfo=UTC)


def test_bars_sorted_chronologically() -> None:
    t0 = datetime(2024, 3, 26, 15, 0, tzinfo=UTC)
    later = _ab(t0 + timedelta(minutes=15), 101.0)
    earlier = _ab(t0, 100.0)
    client = FakeAlpacaClient([later, earlier])  # ordine inversă la intrare
    events = _adapter(client).load()
    assert [e.payload.ts_open for e in events if isinstance(e.payload, Bar)] == [
        t0,
        t0 + timedelta(minutes=15),
    ]


# --------------------------------------------------------------------------- validare


def test_invalid_ohlc_bar_is_rejected() -> None:
    t0 = datetime(2024, 3, 26, 15, 0, tzinfo=UTC)
    # high < low: bară imposibilă, trebuie exclusă (Req 5.4).
    bad = AlpacaBar(ts_open=t0, open=100.0, high=98.0, low=99.0, close=100.0, volume=10.0)
    good = _ab(t0 + timedelta(minutes=15), 100.5)
    client = FakeAlpacaClient([bad, good])
    adapter = _adapter(client)
    events = adapter.load()
    assert len(events) == 1
    assert len(adapter.rejections) == 1


def test_negative_volume_rejected() -> None:
    t0 = datetime(2024, 3, 26, 15, 0, tzinfo=UTC)
    bad = AlpacaBar(ts_open=t0, open=100.0, high=101.0, low=99.0, close=100.0, volume=-5.0)
    client = FakeAlpacaClient([bad])
    adapter = _adapter(client)
    assert adapter.load() == []
    assert len(adapter.rejections) == 1


# --------------------------------------------------------------------------- interval


def test_rejects_interval_out_of_range() -> None:
    client = FakeAlpacaClient([])
    with pytest.raises(ValueError, match="interval_min"):
        _adapter(client, interval_min=1)
    with pytest.raises(ValueError, match="interval_min"):
        _adapter(client, interval_min=90)


def test_lookback_window_passed_to_client() -> None:
    client = FakeAlpacaClient([])
    _adapter(client, lookback=timedelta(days=2)).load()
    assert client.calls[0][0] == "SPY"
    _symbol, interval, start, end = client.calls[0]
    assert interval == 15
    assert end == NOW
    assert start == NOW - timedelta(days=2)


# --------------------------------------------------------------------------- Secret_Store


def test_from_secret_store_resolves_keys_and_builds_client() -> None:
    captured: dict[str, str] = {}

    def fake_factory(api_key: str, api_secret: str) -> FakeAlpacaClient:
        captured["key"] = api_key
        captured["secret"] = api_secret
        t0 = datetime(2024, 3, 26, 15, 0, tzinfo=UTC)
        return FakeAlpacaClient([_ab(t0, 100.0)])

    store = InMemorySecretStore(
        values={"qts/shadow/alpaca_key": "KEYVALUE123", "qts/shadow/alpaca_secret": "SECRETVAL456"},
        acl={
            "qts/shadow/alpaca_key": {("shadow-runner", "shadow")},
            "qts/shadow/alpaca_secret": {("shadow-runner", "shadow")},
        },
    )
    adapter = AlpacaDataAdapter.from_secret_store(
        symbol="SPY",
        interval_min=15,
        tick_size=D("0.01"),
        store=store,
        credentials=AlpacaCredentials(
            api_key_ref=SecretRef("qts/shadow/alpaca_key"),
            api_secret_ref=SecretRef("qts/shadow/alpaca_secret"),
        ),
        requester=Identity("shadow-runner"),
        environment="shadow",
        clock_now=NOW,
        client_factory=fake_factory,
    )
    events = adapter.load()
    assert len(events) == 1
    # Cheile au fost dezvăluite și pasate fabricii, dar nu sunt stocate în adaptor.
    assert captured == {"key": "KEYVALUE123", "secret": "SECRETVAL456"}


def test_from_secret_store_denies_unauthorized_identity() -> None:
    from qts.secrets.store import SecretAccessDeniedError

    store = InMemorySecretStore(
        values={"qts/shadow/alpaca_key": "KEYVALUE123", "qts/shadow/alpaca_secret": "SECRETVAL456"},
        acl={"qts/shadow/alpaca_key": {("other", "shadow")}},
    )
    with pytest.raises(SecretAccessDeniedError):
        AlpacaDataAdapter.from_secret_store(
            symbol="SPY",
            interval_min=15,
            tick_size=D("0.01"),
            store=store,
            credentials=AlpacaCredentials(
                api_key_ref=SecretRef("qts/shadow/alpaca_key"),
                api_secret_ref=SecretRef("qts/shadow/alpaca_secret"),
            ),
            requester=Identity("shadow-runner"),
            environment="shadow",
            clock_now=NOW,
            client_factory=lambda k, s: FakeAlpacaClient([]),
        )
