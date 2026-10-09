"""Teste pentru fabrica Shadow Alpaca și wiring-ul CLI `--source alpaca` (Req 1.1, 16.1, 23).

Deterministe: client Alpaca fals (fără rețea), Secret_Store în memorie și un ceas injectat. Nu
depind de pachetul `alpaca-py` și nici de chei reale.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from qts.config.loader import load_config
from qts.core.clock import Clock
from qts.data.alpaca_shadow import SHADOW_IDENTITY, alpaca_shadow_factory
from qts.data.alpaca_source import AlpacaBar, AlpacaDataAdapter
from qts.secrets.store import InMemorySecretStore
from tests.unit.test_bootstrap import CONFIG_DIR

T0 = datetime(2024, 3, 26, 15, 0, tzinfo=UTC)


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now


def _bars() -> list[AlpacaBar]:
    return [
        AlpacaBar(ts_open=T0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0),
        AlpacaBar(
            ts_open=T0 + timedelta(minutes=15),
            open=100.5,
            high=101.5,
            low=100.0,
            close=101.0,
            volume=1200.0,
        ),
    ]


class FakeClient:
    def __init__(self, bars: Sequence[AlpacaBar]) -> None:
        self._bars = list(bars)

    def get_bars(
        self, symbol: str, interval_min: int, start: datetime, end: datetime
    ) -> Sequence[AlpacaBar]:
        return self._bars


def _store(key: str = "KEYVALUE123", secret_val: str = "SECRETVAL456") -> InMemorySecretStore:  # noqa: S107
    return InMemorySecretStore(
        values={"qts/shadow/alpaca_key": key, "qts/shadow/alpaca_secret": secret_val},
        acl={
            "qts/shadow/alpaca_key": {(SHADOW_IDENTITY.name, "shadow")},
            "qts/shadow/alpaca_secret": {(SHADOW_IDENTITY.name, "shadow")},
        },
    )


def _shadow_config() -> object:
    return load_config(CONFIG_DIR / "shadow.toml")


# --------------------------------------------------------------------------- fabrică


def test_factory_builds_alpaca_adapter_from_config_and_secrets() -> None:
    config = _shadow_config()
    factory = alpaca_shadow_factory(
        _store(), client_factory=lambda k, s: FakeClient(_bars())
    )
    clock: Clock = FixedClock(T0 + timedelta(hours=1))
    adapter = factory(config, clock, Path("."))  # type: ignore[arg-type]
    assert isinstance(adapter, AlpacaDataAdapter)
    assert adapter.source_id.startswith("alpaca:")
    events = adapter.load()
    assert len(events) == 2
    assert all(e.kind == "bar" for e in events)


def test_factory_uses_first_instrument_symbol_and_interval() -> None:
    config = _shadow_config()
    captured: dict[str, object] = {}

    def client_factory(_k: str, _s: str) -> FakeClient:
        return FakeClient(_bars())

    factory = alpaca_shadow_factory(_store(), client_factory=client_factory)
    adapter = factory(config, FixedClock(T0), Path("."))  # type: ignore[arg-type]
    assert isinstance(adapter, AlpacaDataAdapter)
    # Simbolul și intervalul vin din config (primul instrument, data.bar_interval_min).
    assert adapter.symbol == config.instruments[0].symbol  # type: ignore[attr-defined]
    assert adapter.interval_min == config.data.bar_interval_min  # type: ignore[attr-defined]
    captured["ok"] = True
    assert captured["ok"] is True


def test_factory_denies_when_keys_missing() -> None:
    from qts.secrets.store import SecretUnavailableError

    empty = InMemorySecretStore(
        values={},
        acl={
            "qts/shadow/alpaca_key": {(SHADOW_IDENTITY.name, "shadow")},
            "qts/shadow/alpaca_secret": {(SHADOW_IDENTITY.name, "shadow")},
        },
    )
    factory = alpaca_shadow_factory(empty, client_factory=lambda k, s: FakeClient([]))
    with pytest.raises(SecretUnavailableError):
        factory(_shadow_config(), FixedClock(T0), Path("."))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- CLI


def test_cli_resolve_source_none_keeps_default_refusal() -> None:
    from qts.cli import _resolve_shadow_source

    assert _resolve_shadow_source(None) is None


def test_cli_resolve_source_unknown_rejected() -> None:
    from qts.bootstrap import BootstrapError
    from qts.cli import _resolve_shadow_source

    with pytest.raises(BootstrapError, match="necunoscut"):
        _resolve_shadow_source("mt5")


def test_cli_resolve_source_alpaca_builds_factory() -> None:
    from qts.cli import _resolve_shadow_source

    factory = _resolve_shadow_source("alpaca")
    assert factory is not None
    assert callable(factory)
