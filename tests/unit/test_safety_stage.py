"""Teste unitare pentru `qts.safety.stage` (Req 2.2, 2.3, 2.4)."""

from pathlib import Path

import pytest

from qts.config.loader import parse_config
from qts.safety.stage import (
    FAIL_CLOSED_SOURCE,
    ProjectStage,
    StageInfo,
    StartupRefusedError,
    check_startup,
    is_live_enabled,
    read_stage,
    startup_violations,
)
from tests.helpers import DEMO_BROKER, config_dict

INITIAL = StageInfo(ProjectStage.INITIAL, "stage.lock")


@pytest.mark.parametrize("env", ["backtest", "shadow"])
def test_sim_modes_allowed_in_initial(env: str) -> None:
    check_startup(parse_config(config_dict(environment=env)), INITIAL)


def test_live_never_enabled() -> None:
    assert is_live_enabled(INITIAL) is False
    assert is_live_enabled(StageInfo(ProjectStage.INITIAL, FAIL_CLOSED_SOURCE)) is False


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.alpaca.markets",
        "https://API.alpaca.markets/v2",
        "  127.0.0.1:7496 ",
        "tcp://gw.example:4001",
    ],
)
def test_known_live_endpoints_refused_even_in_demo(endpoint: str) -> None:
    cfg = parse_config(
        config_dict(environment="demo", broker={**DEMO_BROKER, "endpoint": endpoint})
    )
    with pytest.raises(StartupRefusedError) as exc:
        check_startup(cfg, INITIAL)
    assert any("endpoint live" in r for r in exc.value.reasons)


def test_live_endpoint_refused_regardless_of_broker_name() -> None:
    broker = {**DEMO_BROKER, "name": "other", "endpoint": "https://api.alpaca.markets"}
    cfg = parse_config(config_dict(environment="demo", broker=broker))
    assert any("endpoint live" in r for r in startup_violations(cfg, INITIAL))


@pytest.mark.parametrize("account", ["U1234567", "u1234567"])
def test_live_account_refused(account: str) -> None:
    cfg = parse_config(
        config_dict(environment="demo", broker={**DEMO_BROKER, "account_id": account})
    )
    assert any("cont live" in r for r in startup_violations(cfg, INITIAL))


def test_paper_alpaca_and_fake_endpoints_allowed() -> None:
    for broker in (
        {
            **DEMO_BROKER,
            "name": "alpaca",
            "endpoint": "https://paper-api.alpaca.markets",
            "account_id": "PA123",
        },
        {**DEMO_BROKER, "kind": "fake", "name": "fake", "endpoint": "fake://local"},
    ):
        check_startup(parse_config(config_dict(environment="demo", broker=broker)), INITIAL)


def test_live_config_reports_all_reasons() -> None:
    broker = {**DEMO_BROKER, "kind": "live", "endpoint": "127.0.0.1:7496", "account_id": "U1"}
    cfg = parse_config(config_dict(environment="live", broker=broker))
    reasons = startup_violations(cfg, INITIAL)
    assert len(reasons) == 4


@pytest.mark.parametrize(
    "content",
    [b"project_stage = ", b"\xff\xfe\x00garbage"],
)
def test_corrupt_stage_lock_fails_closed(tmp_path: Path, content: bytes) -> None:
    lock = tmp_path / "stage.lock"
    lock.write_bytes(content)
    info = read_stage(lock)
    assert info.stage is ProjectStage.INITIAL
    assert info.source == FAIL_CLOSED_SOURCE


def test_unreadable_stage_lock_fails_closed(tmp_path: Path) -> None:
    # Un director în locul fișierului nu poate fi citit -> tratat ca `initial`.
    lock = tmp_path / "stage.lock"
    lock.mkdir()
    assert read_stage(lock).source == FAIL_CLOSED_SOURCE


@pytest.mark.parametrize(
    "content",
    [
        'project_stage = "live"\n',
        "other = 1\n",
        "project_stage = 1\n",
        'project_stage = "INITIAL"\n',
    ],
)
def test_invalid_stage_value_refused(tmp_path: Path, content: str) -> None:
    lock = tmp_path / "stage.lock"
    lock.write_text(content, encoding="utf-8")
    with pytest.raises(StartupRefusedError):
        read_stage(lock)


def test_repo_stage_lock_is_initial() -> None:
    info = read_stage(Path(__file__).parents[2] / "stage.lock")
    assert info == StageInfo(ProjectStage.INITIAL, "stage.lock")
