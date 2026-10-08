"""Configurații invalide și refuzul Live în Initial_Stage (Req 2.4, 17.2).

Completează `test_config_loader.py`, `test_config_stage.py` și `test_safety_stage.py` cu:
fiecare abatere raportată separat, câmpuri necunoscute la fiecare nivel, secțiuni obligatorii
lipsă, limitele intervalului de bară, instrumente duplicate și verificări cap-coadă
fișier TOML → `load_config` → `check_startup`.
"""

from __future__ import annotations

import copy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from qts.config.loader import ConfigError, config_path, load_config, parse_config
from qts.safety.stage import (
    ProjectStage,
    StageInfo,
    StartupRefusedError,
    check_startup,
    read_stage,
)
from tests.helpers import DEMO_BROKER, config_dict

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "config"
INITIAL = StageInfo(ProjectStage.INITIAL, "stage.lock")


def _issues(data: dict[str, Any]) -> list[str]:
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    return exc.value.issues


def _locs(issues: list[str]) -> set[str]:
    return {i.split(":")[0] for i in issues}


# --- Req 17.2: fiecare abatere este raportată ------------------------------------------------


def test_every_out_of_bounds_risk_value_is_reported_separately() -> None:
    risk = {
        "reference_capital_eur": "200",
        "risk_per_trade_target_eur": "0.10",
        "risk_per_trade_max_eur": "0.75",
        "daily_loss_limit_eur": "5",
        "total_loss_limit_eur": "20",
    }
    risk_issues = [i for i in _issues(config_dict(risk=risk)) if i.startswith("risk")]
    for field in risk:
        assert any(i.startswith(f"risk: {field}") for i in risk_issues), field
    assert len(risk_issues) == 5


@pytest.mark.parametrize(
    "risk",
    [
        {"risk_per_trade_max_eur": "0"},
        {"risk_per_trade_max_eur": "-0.10"},
        {"daily_loss_limit_eur": "0"},
        {"daily_loss_limit_eur": "-1"},
        {"total_loss_limit_eur": "9.99"},
        {"max_open_positions": 0},
    ],
)
def test_non_positive_or_lowered_risk_values_rejected(risk: dict[str, Any]) -> None:
    issues = _issues(config_dict(risk=risk))
    assert any(i.startswith("risk") for i in issues)


def test_approved_risk_bounds_are_accepted() -> None:
    risk = {
        "risk_per_trade_target_eur": "0.50",
        "risk_per_trade_max_eur": "0.50",
        "daily_loss_limit_eur": "2",
        "total_loss_limit_eur": "10",
    }
    assert parse_config(config_dict(risk=risk)).risk.risk_per_trade_max_eur == Decimal("0.50")


@pytest.mark.parametrize(
    ("section", "loc"),
    [
        (None, "unknown_top"),
        ("run", "run.unknown_x"),
        ("data", "data.unknown_x"),
        ("broker", "broker.unknown_x"),
        ("risk", "risk.unknown_x"),
        ("strategy", "strategy.unknown_x"),
        ("kill_switch", "kill_switch.unknown_x"),
        ("instruments", "instruments.0.unknown_x"),
    ],
)
def test_unknown_field_rejected_at_every_level(section: str | None, loc: str) -> None:
    data = config_dict()
    if section is None:
        data["unknown_top"] = 1
    elif section == "instruments":
        data["instruments"][0]["unknown_x"] = 1
    else:
        data.setdefault(section, {})["unknown_x"] = 1
    issues = _issues(data)
    assert loc in _locs(issues)
    assert all("extra" in i for i in issues if i.startswith(loc))


def test_unknown_fields_at_several_levels_all_reported() -> None:
    data = config_dict(extra_top=1, run={"x": 1}, data={"y": 2}, broker={"z": 3})
    data["instruments"][0]["w"] = 4
    assert {"extra_top", "run.x", "data.y", "broker.z", "instruments.0.w"} <= _locs(_issues(data))


@pytest.mark.parametrize(
    "missing",
    ["schema_version", "environment", "run", "data", "broker", "strategy", "instruments"],
)
def test_missing_required_section_rejected(missing: str) -> None:
    data = config_dict()
    del data[missing]
    assert missing in _locs(_issues(data))


def test_all_missing_required_sections_reported_together() -> None:
    required = ["schema_version", "environment", "run", "data", "broker", "strategy", "instruments"]
    data = {k: v for k, v in config_dict().items() if k not in required}
    assert set(required) <= _locs(_issues(data))


def test_missing_required_nested_fields_reported() -> None:
    data = config_dict()
    del data["run"]["seed"]
    del data["data"]["source_id"]
    del data["instruments"][0]["symbol"]
    assert {"run.seed", "data.source_id", "instruments.0.symbol"} <= _locs(_issues(data))


def test_empty_instruments_rejected() -> None:
    assert "instruments" in _locs(_issues(config_dict(instruments=[])))


@pytest.mark.parametrize("minutes", [5, 15, 60])
def test_bar_interval_within_bounds_accepted(minutes: int) -> None:
    cfg = parse_config(config_dict(data={"bar_interval_min": minutes}))
    assert cfg.data.bar_interval_min == minutes


@pytest.mark.parametrize("minutes", [0, 4, 61, 1440, -5])
def test_bar_interval_outside_5_60_rejected(minutes: int) -> None:
    assert "data.bar_interval_min" in _locs(
        _issues(config_dict(data={"bar_interval_min": minutes}))
    )


def test_non_positive_freshness_thresholds_rejected() -> None:
    default_issues = _issues(config_dict(data={"default_freshness_seconds": 0}))
    assert "data.default_freshness_seconds" in _locs(default_issues)
    per_instrument = _issues(config_dict(data={"freshness_seconds": {"XYZ": -1, "ABC": 0}}))
    text = "\n".join(per_instrument)
    assert "XYZ" in text
    assert "ABC" in text


def test_duplicate_instruments_rejected() -> None:
    data = config_dict()
    data["instruments"].append(copy.deepcopy(data["instruments"][0]))
    assert any("duplicate" in i for i in _issues(data))


def test_backtest_without_dataset_reported_with_other_cross_field_issues() -> None:
    data = config_dict(broker={"kind": "demo"})
    del data["data"]["dataset_path"]
    data["instruments"].append(copy.deepcopy(data["instruments"][0]))
    text = "\n".join(_issues(data))
    assert "broker.kind=demo nu este permis" in text
    assert "data.dataset_path lipsă" in text
    assert "duplicate" in text


# --- Req 2.4: refuzul endpoint-ului / contului Live, cap-coadă din fișier TOML ----------------


def _write_demo(tmp_path: Path, replacements: dict[str, str]) -> Path:
    text = (CONFIG_DIR / "demo.toml").read_text(encoding="utf-8")
    for old, new in replacements.items():
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "demo.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repo_live_config_is_refused_end_to_end() -> None:
    cfg = load_config(config_path("live", CONFIG_DIR))
    with pytest.raises(StartupRefusedError) as exc:
        check_startup(cfg, read_stage(REPO / "stage.lock"))
    text = "\n".join(exc.value.reasons)
    assert "Live este interzis" in text
    assert "broker.kind=live" in text
    assert "endpoint live" in text
    assert "cont live" in text


@pytest.mark.parametrize("env", ["backtest", "shadow", "demo"])
def test_repo_initial_stage_configs_start_end_to_end(env: str) -> None:
    check_startup(load_config(config_path(env, CONFIG_DIR)), read_stage(REPO / "stage.lock"))


@pytest.mark.parametrize(
    ("replacements", "reason"),
    [
        ({'"127.0.0.1:7497"': '"127.0.0.1:7496"'}, "endpoint live"),
        ({'"127.0.0.1:7497"': '"localhost:4001"'}, "endpoint live"),
        (
            {
                'name = "ibkr"': 'name = "alpaca"',
                '"127.0.0.1:7497"': '"https://api.alpaca.markets"',
            },
            "endpoint live",
        ),
        ({'"DU0000000"': '"U0000000"'}, "cont live"),
    ],
)
def test_demo_file_with_live_endpoint_or_account_refused(
    tmp_path: Path, replacements: dict[str, str], reason: str
) -> None:
    cfg = load_config(_write_demo(tmp_path, replacements))
    with pytest.raises(StartupRefusedError) as exc:
        check_startup(cfg, INITIAL)
    assert any(reason in r for r in exc.value.reasons)


def test_demo_file_with_live_endpoint_and_account_reports_both(tmp_path: Path) -> None:
    cfg = load_config(
        _write_demo(tmp_path, {'"127.0.0.1:7497"': '"127.0.0.1:7496"', '"DU0000000"': '"U1"'})
    )
    with pytest.raises(StartupRefusedError) as exc:
        check_startup(cfg, INITIAL)
    text = "\n".join(exc.value.reasons)
    assert "endpoint live" in text
    assert "cont live" in text


def test_demo_file_relabelled_as_live_broker_is_rejected_by_schema(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"broker\.kind=live nu este permis în demo"):
        load_config(_write_demo(tmp_path, {'kind = "demo"': 'kind = "live"'}))


def test_live_endpoint_in_sim_backtest_config_refused() -> None:
    broker = {"kind": "sim", "endpoint": "https://api.alpaca.markets"}
    cfg = parse_config(config_dict(broker=broker))
    with pytest.raises(StartupRefusedError, match="endpoint live"):
        check_startup(cfg, INITIAL)


def test_live_account_in_shadow_config_refused() -> None:
    cfg = parse_config(
        config_dict(environment="shadow", broker={"kind": "sim", "account_id": "U9"})
    )
    with pytest.raises(StartupRefusedError, match="cont live"):
        check_startup(cfg, INITIAL)


def test_live_refused_when_stage_lock_missing(tmp_path: Path) -> None:
    broker = {**DEMO_BROKER, "kind": "live", "endpoint": "127.0.0.1:7496"}
    cfg = parse_config(config_dict(environment="live", broker=broker))
    with pytest.raises(StartupRefusedError):
        check_startup(cfg, read_stage(tmp_path / "missing.lock"))
