"""Configuration_Snapshot (Req 1.4, 17.3, 17.4, 17.7, 23.2)."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from qts import ENGINE_VERSION
from qts.config.loader import parse_config
from qts.config.schema import AppConfig
from qts.config.snapshot import (
    RNG_ALGORITHM,
    CodeVersion,
    SnapshotError,
    create_snapshot,
    read_code_version,
    take_snapshot,
)
from qts.safety.stage import ProjectStage, StageInfo
from qts.secrets.store import Redactor
from tests.helpers import DEMO_BROKER, config_dict

STAGE = StageInfo(ProjectStage.INITIAL, "stage.lock")
CODE = CodeVersion("0" * 40, False, "f" * 64)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _cfg(**overrides: object) -> AppConfig:
    return parse_config(config_dict(**overrides))


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],  # noqa: S607
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    _git(tmp_path, "add", "uv.lock", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_snapshot_records_required_fields() -> None:
    snap = create_snapshot(_cfg(), STAGE, CODE, T0, ["ds-b", "ds-a"], "art-1")
    assert snap.engine_version == ENGINE_VERSION  # Req 1.4
    assert (snap.git_commit, snap.git_dirty, snap.lock_sha256) == ("0" * 40, False, "f" * 64)
    assert snap.seed == 42 and snap.rng_algorithm == RNG_ALGORITHM  # Req 17.4
    assert snap.timezone == "UTC"
    assert snap.dataset_ids == ["ds-a", "ds-b"] and snap.artifact_id == "art-1"
    assert snap.environment == "backtest" and snap.stage == "initial"
    assert snap.config["run"]["seed"] == 42
    assert len(snap.snapshot_id) == 64 and snap.verify()


def test_snapshot_id_is_deterministic_and_content_sensitive() -> None:
    a = create_snapshot(_cfg(), STAGE, CODE, T0, ["d1", "d2"])
    b = create_snapshot(_cfg(), STAGE, CODE, datetime(2027, 5, 5, tzinfo=UTC), ["d2", "d1"])
    assert a.snapshot_id == b.snapshot_id  # nu depinde de created_at sau ordinea datelor
    other_seed = create_snapshot(_cfg(run={"seed": 7}), STAGE, CODE, T0, ["d1", "d2"])
    dirty = create_snapshot(_cfg(), STAGE, CodeVersion("0" * 40, True, "f" * 64), T0, ["d1", "d2"])
    assert len({a.snapshot_id, other_seed.snapshot_id, dirty.snapshot_id}) == 3


def test_snapshot_is_immutable_and_tampering_is_detected() -> None:
    snap = create_snapshot(_cfg(), STAGE, CODE, T0, [])
    with pytest.raises(ValidationError):
        snap.seed = 1  # type: ignore[misc]
    tampered = snap.model_copy(update={"seed": 1})
    assert not tampered.verify()


def test_snapshot_contains_only_secret_reference() -> None:
    cfg = _cfg(environment="demo", broker=DEMO_BROKER)
    snap = create_snapshot(cfg, STAGE, CODE, T0, [], redactor=Redactor())
    assert snap.config["broker"]["secret_ref"] == DEMO_BROKER["secret_ref"]


def test_snapshot_refused_when_secret_value_in_content() -> None:
    redactor = Redactor()
    redactor.register("DU1234567")  # valoare cunoscută ca secret ajunsă în configurație
    cfg = _cfg(environment="demo", broker=DEMO_BROKER)
    with pytest.raises(SnapshotError) as exc:
        create_snapshot(cfg, STAGE, CODE, T0, [], redactor=redactor)
    assert "DU1234567" not in str(exc.value)


@pytest.mark.parametrize(
    ("code", "datasets", "created_at"),
    [
        (CodeVersion("", False, "f" * 64), [], T0),
        (CodeVersion("0" * 40, False, ""), [], T0),
        (CODE, ["d", "d"], T0),
        (CODE, [""], T0),
        (CODE, [], datetime(2026, 1, 1)),  # noqa: DTZ001 - fără fus orar
    ],
)
def test_snapshot_refused_on_invalid_inputs(
    code: CodeVersion, datasets: list[str], created_at: datetime
) -> None:
    with pytest.raises(SnapshotError):
        create_snapshot(_cfg(), STAGE, code, created_at, datasets)


def test_read_code_version_from_git_repo(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    code = read_code_version(repo)
    assert code.git_commit == head and not code.git_dirty
    assert code.lock_sha256 == hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest()
    (repo / "a.txt").write_text("b\n", encoding="utf-8")
    assert read_code_version(repo).git_dirty


def test_take_snapshot_end_to_end(repo: Path) -> None:
    snap = take_snapshot(_cfg(), STAGE, repo, T0, ["ds"])
    assert snap.git_commit == _git(repo, "rev-parse", "HEAD") and snap.verify()


def test_missing_lock_stops_run(repo: Path) -> None:
    (repo / "uv.lock").unlink()
    with pytest.raises(SnapshotError, match=r"uv\.lock"):
        take_snapshot(_cfg(), STAGE, repo, T0, [])


def test_no_git_repo_stops_run(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("x", encoding="utf-8")
    with pytest.raises(SnapshotError):
        read_code_version(tmp_path)


def test_repo_without_commits_stops_run(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "uv.lock").write_text("x", encoding="utf-8")
    with pytest.raises(SnapshotError):
        read_code_version(tmp_path)


def test_git_unavailable_stops_run(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    with pytest.raises(SnapshotError, match="indisponibil"):
        read_code_version(repo)
