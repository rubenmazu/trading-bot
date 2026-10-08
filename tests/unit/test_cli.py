"""Teste pentru comanda `qts backtest` (Req 1.2, 2.2, 17.7)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qts.cli import EXIT_REFUSED, PROFIT_WARNING, app
from tests.unit.test_bootstrap import CONFIG_DIR, REPO, STAGE_LOCK, write_backtest_config

runner = CliRunner()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=qts-test", "-c", "user.email=qts@test.invalid", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_live_config_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["backtest", "--config", str(CONFIG_DIR / "live.toml"), "--stage-lock", str(STAGE_LOCK)],
    )
    assert result.exit_code == EXIT_REFUSED
    assert "pornire refuzată" in result.output and "Live" in result.output
    assert list(tmp_path.iterdir()) == []


def test_snapshot_failure_without_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_backtest_config(tmp_path)
    shutil.copy(REPO / "uv.lock", tmp_path / "uv.lock")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["backtest", "--config", str(config), "--stage-lock", str(STAGE_LOCK)]
    )
    assert result.exit_code == EXIT_REFUSED
    assert "Configuration_Snapshot nu poate fi creat" in result.output
    assert not (tmp_path / "run.db").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="git indisponibil")
def test_backtest_success_in_committed_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_backtest_config(tmp_path)
    shutil.copy(REPO / "uv.lock", tmp_path / "uv.lock")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "uv.lock")
    _git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)
    db = (tmp_path / "cli.db").as_posix()
    result = runner.invoke(
        app,
        ["backtest", "--config", str(config), "--stage-lock", str(STAGE_LOCK), "--db", db],
    )
    assert result.exit_code == 0, result.output
    for label in ("brut total:", "cost comision:", "cost slippage:", "costuri totale:", "net:"):
        assert label in result.output
    assert "verificat" in result.output
    assert PROFIT_WARNING in result.output
    assert (tmp_path / "cli.db").exists() and not (tmp_path / "run.db").exists()


def test_cli_has_no_live_options() -> None:
    result = runner.invoke(app, ["backtest", "--help"])
    assert result.exit_code == 0
    assert "live" not in result.output.lower()


def test_evaluate_refuses_without_data_source_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Evaluarea necesită o pre-înregistrare validă și date partiționate; fără sursa reală
    # (Open_Decision, Req 30) comanda refuză pornirea, exact ca seam-ul lui `qts shadow`.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == EXIT_REFUSED
    assert "pornire refuzată" in result.output
    assert "Open_Decision" in result.output
    assert PROFIT_WARNING in result.output
    assert list(tmp_path.iterdir()) == []


def test_evaluate_has_no_live_options() -> None:
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "live" not in result.output.lower()
