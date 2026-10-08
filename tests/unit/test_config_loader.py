"""Teste pentru schema și încărcarea configurației (Req 1.3, 17.1, 17.2, 17.6)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from qts.config.loader import ConfigError, config_path, load_config, parse_config
from qts.config.schema import ENVIRONMENTS
from tests.helpers import DEMO_BROKER, config_dict

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@pytest.mark.parametrize("env", ENVIRONMENTS)
def test_repo_config_per_environment_is_valid(env: str) -> None:
    cfg = load_config(config_path(env, CONFIG_DIR), expected_environment=env)
    assert cfg.environment == env


def test_config_path_rejects_unknown_environment() -> None:
    with pytest.raises(ConfigError, match="mediu necunoscut"):
        config_path("prod")


def test_file_declaring_other_environment_is_rejected(tmp_path: Path) -> None:
    raw = (CONFIG_DIR / "backtest.toml").read_text(encoding="utf-8")
    wrong = tmp_path / "demo.toml"
    wrong.write_text(raw, encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(wrong)
    assert any("declară 'backtest'" in i for i in exc.value.issues)


def test_expected_environment_mismatch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mediul așteptat este 'demo'"):
        load_config(CONFIG_DIR / "backtest.toml", expected_environment="demo")
    with pytest.raises(ConfigError, match="mediul așteptat este 'shadow'"):
        parse_config(config_dict(), expected_environment="shadow")


def test_environment_mismatch_reported_with_schema_issues() -> None:
    data = config_dict(run={"seed": -1})
    with pytest.raises(ConfigError) as exc:
        parse_config(data, expected_environment="demo")
    text = "\n".join(exc.value.issues)
    assert "mediul așteptat este 'demo'" in text
    assert "run.seed" in text


def test_each_cross_field_violation_is_a_separate_issue() -> None:
    risk = {"risk_per_trade_max_eur": "0.60", "daily_loss_limit_eur": "3"}
    with pytest.raises(ConfigError) as exc:
        parse_config(config_dict(risk=risk))
    risk_issues = [i for i in exc.value.issues if i.startswith("risk")]
    assert any("risk_per_trade_max_eur" in i for i in risk_issues)
    assert any("daily_loss_limit_eur" in i for i in risk_issues)
    assert len(risk_issues) >= 2


def test_unknown_nested_field_and_wrong_schema_version_rejected() -> None:
    data = config_dict(schema_version="2", broker={"kind": "sim", "password": "x"})
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    locs = {i.split(":")[0] for i in exc.value.issues}
    assert {"schema_version", "broker.password"} <= locs


def test_secret_ref_must_be_reference_and_value_is_not_echoed() -> None:
    leaked = "SuperSecret value 123"
    with pytest.raises(ConfigError) as exc:
        parse_config(config_dict(environment="demo", broker={**DEMO_BROKER, "secret_ref": leaked}))
    assert any(i.startswith("broker.secret_ref") for i in exc.value.issues)
    assert leaked not in str(exc.value)


def test_missing_mode_fields_all_reported() -> None:
    with pytest.raises(ConfigError) as exc:
        parse_config(config_dict(environment="demo", broker={"kind": "demo"}))
    missing = [i for i in exc.value.issues if "lipsă pentru broker.kind=demo" in i]
    assert len(missing) == 4


def test_repo_configs_contain_only_secret_references() -> None:
    for env in ENVIRONMENTS:
        broker = tomllib.loads(config_path(env, CONFIG_DIR).read_text(encoding="utf-8"))["broker"]
        assert set(broker) <= {"kind", "name", "endpoint", "account_id", "secret_ref"}
