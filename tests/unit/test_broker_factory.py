"""Teste unitare pentru `broker/factory.py` (Req 1.2, 2.1, 2.2)."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

import pytest

from qts.broker import factory
from qts.broker.adapter import OrderRequest
from qts.broker.alpaca_broker import (
    AlpacaAccount,
    AlpacaBrokerAdapter,
    AlpacaExecution,
    AlpacaOrderAck,
)
from qts.broker.factory import (
    FAKE_DEFAULT_ACCOUNT,
    SIM_DEFAULT_ACCOUNT,
    BrokerFactoryError,
    BrokerNotAvailableError,
    LiveBrokerRefusedError,
    build_broker,
)
from qts.broker.fail_safe import FailSafeBlock, FailSafeReason, FailSafeRejectedError
from qts.broker.fake import FakeBroker
from qts.broker.sim import SimBroker
from qts.config.schema import AppConfig
from qts.core.clock import SimClock
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import CompleteCostModel, CostModelConfig
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.risk.context import KillSwitchScope
from qts.safety.kill_switch import JournalKillSwitchStore, KillSwitch
from qts.safety.stage import ProjectStage, StageInfo, StartupRefusedError
from qts.secrets.store import InMemorySecretStore
from tests.helpers import DEMO_BROKER, config_dict

T0 = datetime(2026, 3, 2, 10, tzinfo=UTC)
STAGE = StageInfo(ProjectStage.INITIAL, "stage.lock")
FAKE_BROKER: dict[str, Any] = {"kind": "fake", "endpoint": "fake://local"}
LIVE_BROKER: dict[str, Any] = {
    "kind": "live",
    "name": "alpaca",
    "endpoint": "https://api.alpaca.markets",
    "account_id": "LIVE1",
    "secret_ref": "qts/live/broker",
}


@dataclass
class _Env:
    conn: sqlite3.Connection
    journal: Journal
    ks: KillSwitch
    clock: SimClock


@pytest.fixture
def env(tmp_path: Path) -> _Env:
    conn = open_db(tmp_path / "factory.db")
    journal = Journal(conn)
    clock = SimClock(T0)
    return _Env(conn, journal, KillSwitch(JournalKillSwitchStore(journal), clock=clock), clock)


def _cost_model() -> CompleteCostModel:
    table = CommissionTable(
        broker="sim",
        version="v1",
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        currency="EUR",
        percent=Decimal("0.001"),
        minimum=Decimal("1"),
    )
    return CompleteCostModel(
        CostModelConfig(version="costs-v1", commissions=CommissionSchedule(tables=(table,)))
    )


def _cfg(**overrides: Any) -> AppConfig:
    return AppConfig.model_validate(config_dict(**overrides))


def _build(env: _Env, cfg: AppConfig) -> FailSafeBlock[OrderRequest, Any]:
    return build_broker(
        cfg,
        STAGE,
        clock=env.clock,
        kill_switch=env.ks,
        audit=env.journal,
        cost_model=_cost_model(),
    )


def _req(coid: str = "c1") -> OrderRequest:
    return OrderRequest(client_order_id=coid, instrument="XYZ", side="BUY", qty=Decimal("1"))


@pytest.mark.parametrize("mode", ["backtest", "shadow"])
def test_backtest_and_shadow_build_wrapped_sim_broker(env: _Env, mode: str) -> None:
    block = _build(env, _cfg(environment=mode))
    assert isinstance(block, FailSafeBlock)
    assert isinstance(block.inner, SimBroker)
    assert block.environment == "sim"
    assert block.account_id == SIM_DEFAULT_ACCOUNT


def test_demo_with_fake_endpoint_builds_wrapped_fake_broker(env: _Env) -> None:
    block = _build(env, _cfg(environment="demo", broker=FAKE_BROKER))
    assert isinstance(block, FailSafeBlock)
    assert isinstance(block.inner, FakeBroker)
    assert block.environment == "demo"
    assert block.account_id == FAKE_DEFAULT_ACCOUNT
    ack = block.submit(_req())
    assert ack.accepted


def test_fake_requires_fake_endpoint(env: _Env) -> None:
    cfg = _cfg(environment="demo", broker={**FAKE_BROKER, "endpoint": "https://example.test"})
    with pytest.raises(BrokerFactoryError):
        _build(env, cfg)


def test_non_alpaca_demo_broker_not_available(env: _Env) -> None:
    # Singurul broker demo disponibil este Alpaca paper; ibkr nu este implementat.
    with pytest.raises(BrokerNotAvailableError, match="nu este implementat"):
        _build(env, _cfg(environment="demo", broker=DEMO_BROKER))


ALPACA_DEMO_BROKER: dict[str, Any] = {
    "kind": "demo",
    "name": "alpaca",
    "endpoint": "https://paper-api.alpaca.markets",
    "account_id": "ALPACA-PAPER-1",
    "secret_ref": "qts/demo/alpaca_key",
}


def _alpaca_store() -> InMemorySecretStore:
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


class _StubAlpacaClient:
    """Client Alpaca minimal, fără rețea, doar pentru testul de fabrică."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def submit_order(self, req: OrderRequest) -> AlpacaOrderAck:
        return AlpacaOrderAck(broker_order_id="ALP-1", accepted=True)

    def cancel_order(self, broker_order_id: str) -> None:
        return None

    def get_account(self) -> AlpacaAccount:
        return AlpacaAccount(cash="100000", currency="USD", positions=(), complete=True)

    def poll_executions(self) -> list[AlpacaExecution]:
        return []


def test_demo_alpaca_builds_wrapped_paper_adapter(env: _Env) -> None:
    block = build_broker(
        _cfg(environment="demo", broker=ALPACA_DEMO_BROKER),
        STAGE,
        clock=env.clock,
        kill_switch=env.ks,
        audit=env.journal,
        secret_store=_alpaca_store(),
        alpaca_client_factory=lambda k, s, e: _StubAlpacaClient(e),
    )
    assert isinstance(block, FailSafeBlock)
    assert isinstance(block.inner, AlpacaBrokerAdapter)
    assert block.environment == "demo"
    assert block.account_id == "ALPACA-PAPER-1"
    assert block.submit(_req()).accepted


def test_demo_alpaca_requires_secret_store(env: _Env) -> None:
    with pytest.raises(BrokerFactoryError, match="Secret_Store"):
        build_broker(
            _cfg(environment="demo", broker=ALPACA_DEMO_BROKER),
            STAGE,
            clock=env.clock,
            kill_switch=env.ks,
            audit=env.journal,
        )


def test_sim_requires_cost_model(env: _Env) -> None:
    with pytest.raises(BrokerFactoryError):
        build_broker(_cfg(), STAGE, clock=env.clock, kill_switch=env.ks, audit=env.journal)


def _forbid_constructors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> NoReturn:
        raise AssertionError("adaptorul nu trebuie construit")

    monkeypatch.setattr(factory, "SimBroker", _boom)
    monkeypatch.setattr(factory, "FakeBroker", _boom)


def test_live_mode_refused_before_construction(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_constructors(monkeypatch)
    with pytest.raises(LiveBrokerRefusedError) as info:
        _build(env, _cfg(environment="live", broker=LIVE_BROKER))
    assert any("Live" in r for r in info.value.reasons)


@pytest.mark.parametrize(
    "broker",
    [
        {**FAKE_BROKER, "endpoint": "https://api.alpaca.markets"},  # endpoint live cunoscut
        {**DEMO_BROKER, "endpoint": "https://api.alpaca.markets", "name": "alpaca"},
        {**DEMO_BROKER, "account_id": "U1234567"},  # cont live IBKR
    ],
)
def test_live_endpoint_or_account_refused_before_construction(
    env: _Env, monkeypatch: pytest.MonkeyPatch, broker: dict[str, Any]
) -> None:
    _forbid_constructors(monkeypatch)
    with pytest.raises(StartupRefusedError):
        _build(env, _cfg(environment="demo", broker=broker))


def test_sim_submit_passes_and_kill_switch_blocks(env: _Env) -> None:
    block = _build(env, _cfg())
    assert block.submit(_req("c1")).accepted
    env.ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="R")
    with pytest.raises(FailSafeRejectedError) as info:
        block.submit(_req("c2"))
    assert info.value.reason is FailSafeReason.KILL_SWITCH_ACTIVE
    inner = block.inner
    assert isinstance(inner, SimBroker)
    assert [o.client_order_id for o in inner.snapshot().orders] == ["c1"]


def test_qts_live_is_never_imported(env: _Env) -> None:
    _build(env, _cfg())
    with pytest.raises(LiveBrokerRefusedError):
        _build(env, _cfg(environment="live", broker=LIVE_BROKER))
    assert "qts_live" not in sys.modules
    assert importlib.util.find_spec("qts_live") is None
