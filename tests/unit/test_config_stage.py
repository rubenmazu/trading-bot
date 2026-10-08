import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from qts.config.loader import ConfigError, load_config, parse_config
from qts.config.snapshot import CodeVersion, SnapshotError, create_snapshot, read_code_version
from qts.safety.stage import (
    ProjectStage,
    StageInfo,
    StartupRefusedError,
    check_startup,
    read_stage,
    startup_violations,
)
from tests.helpers import DEMO_BROKER, config_dict

INITIAL = StageInfo(ProjectStage.INITIAL, "stage.lock")
CODE = CodeVersion(git_commit="abc", git_dirty=False, lock_sha256="00")
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_valid_backtest_config_parses() -> None:
    cfg = parse_config(config_dict())
    assert cfg.environment == "backtest"
    assert cfg.kill_switch.open_orders_policy == "keep"


def test_all_issues_are_reported_not_just_first() -> None:
    data = config_dict(run={"seed": -1}, data={"bar_interval_min": 1}, extra_section={})
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    text = "\n".join(exc.value.issues)
    assert "run.seed" in text
    assert "data.bar_interval_min" in text
    assert "extra_section" in text


def test_missing_fields_reported() -> None:
    data = config_dict()
    del data["strategy"]
    del data["run"]
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    locs = {issue.split(":")[0] for issue in exc.value.issues}
    assert {"strategy", "run"} <= locs


@pytest.mark.parametrize(
    "risk",
    [
        {"risk_per_trade_max_eur": "0.51"},
        {"daily_loss_limit_eur": "2.01"},
        {"total_loss_limit_eur": "11"},
        {"risk_per_trade_target_eur": "0.20"},
    ],
)
def test_risk_limits_cannot_exceed_approved_bounds(risk: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        parse_config(config_dict(risk=risk))


def test_broker_must_match_environment() -> None:
    with pytest.raises(ConfigError, match=r"broker\.kind=sim"):
        parse_config(config_dict(environment="demo"))
    with pytest.raises(ConfigError, match=r"broker\.endpoint lipsă"):
        parse_config(config_dict(environment="demo", broker={"kind": "demo"}))


def test_initial_stage_refuses_live_mode_and_endpoints() -> None:
    live_cfg = parse_config(config_dict(environment="live", broker={**DEMO_BROKER, "kind": "live"}))
    reasons = startup_violations(live_cfg, INITIAL)
    assert any("Live este interzis" in r for r in reasons)
    assert any("broker.kind=live" in r for r in reasons)

    demo_live_ep = parse_config(
        config_dict(environment="demo", broker={**DEMO_BROKER, "endpoint": "127.0.0.1:7496"})
    )
    with pytest.raises(StartupRefusedError):
        check_startup(demo_live_ep, INITIAL)

    demo_live_acct = parse_config(
        config_dict(environment="demo", broker={**DEMO_BROKER, "account_id": "U7654321"})
    )
    assert any("cont live" in r for r in startup_violations(demo_live_acct, INITIAL))


def test_unknown_demo_endpoint_is_refused() -> None:
    cfg = parse_config(
        config_dict(environment="demo", broker={**DEMO_BROKER, "endpoint": "https://x.example"})
    )
    assert any("demo verificate" in r for r in startup_violations(cfg, INITIAL))


def test_valid_demo_passes() -> None:
    cfg = parse_config(config_dict(environment="demo", broker=DEMO_BROKER))
    check_startup(cfg, INITIAL)


def test_stage_lock_fail_closed(tmp_path: Path) -> None:
    assert read_stage(tmp_path / "missing.lock").stage is ProjectStage.INITIAL
    bad = tmp_path / "stage.lock"
    bad.write_text('project_stage = "post_initial"\n', encoding="utf-8")
    with pytest.raises(StartupRefusedError):
        read_stage(bad)
    repo_lock = Path(__file__).parents[2] / "stage.lock"
    assert read_stage(repo_lock).source == "stage.lock"


def test_load_config_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="nu există"):
        load_config(tmp_path / "nope.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("schema_version = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="TOML invalid"):
        load_config(broken)


def test_snapshot_is_deterministic_and_content_addressed() -> None:
    cfg = parse_config(config_dict())
    a = create_snapshot(cfg, INITIAL, CODE, T0, ["d2", "d1"])
    b = create_snapshot(cfg, INITIAL, CODE, datetime(2026, 2, 1, tzinfo=UTC), ["d1", "d2"])
    assert a.snapshot_id == b.snapshot_id
    other = parse_config(config_dict(run={"seed": 7}))
    assert create_snapshot(other, INITIAL, CODE, T0, ["d1"]).snapshot_id != a.snapshot_id
    assert a.seed == 42 and a.rng_algorithm


def test_read_code_version_requires_lock_and_commit(tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match=r"uv\.lock"):
        read_code_version(tmp_path)
    (tmp_path / "uv.lock").write_text("x", encoding="utf-8")
    with pytest.raises(SnapshotError, match="git"):
        read_code_version(tmp_path)  # nu este repo git

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603, S607

    git("init", "-q")
    git("add", "uv.lock")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init")
    version = read_code_version(tmp_path)
    assert len(version.git_commit) == 40
    assert version.git_dirty is False
