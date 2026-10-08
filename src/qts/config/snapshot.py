"""Configuration_Snapshot: copie imuabilă și hash-uită a contextului unei rulări (Req 17.3–17.7).

Conținut: versiunea Trading_Engine (Req 1.4), commit-ul git și starea „dirty”, hash-ul
`uv.lock`, etapa, mediul, fusul orar, sămânța și algoritmul RNG (Req 17.4), identificatorii
datelor, artefactul și configurația efectivă. Configurația conține numai referințe la secrete
(`secret_ref`), nu valorile lor; în plus, înainte de creare se verifică faptul că nicio valoare
secretă cunoscută de `Redactor` nu a ajuns în conținut (Req 23.2).

Dacă versiunea codului, fișierul lock sau conținutul nu pot fi determinate, se ridică
`SnapshotError`, iar rularea trebuie oprită înaintea procesării (Req 17.7).
"""

from __future__ import annotations

import hashlib
import json
import subprocess  # apeluri git fixe, fără shell și fără input extern
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from qts import ENGINE_VERSION
from qts.config.schema import SCHEMA_VERSION, AppConfig
from qts.core.models import Frozen, UtcDatetime
from qts.safety.stage import StageInfo
from qts.secrets.store import REDACTOR, Redactor

RNG_ALGORITHM: Final = "numpy.random.PCG64"
SNAPSHOT_TIMEZONE: Final = "UTC"
GIT_TIMEOUT_SECONDS: Final = 10


class SnapshotError(Exception):
    """Snapshot-ul nu poate fi creat; procesarea nu trebuie să pornească (Req 17.7)."""


@dataclass(frozen=True)
class CodeVersion:
    git_commit: str
    git_dirty: bool
    lock_sha256: str


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607 - git din PATH, argumente constante
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # git lipsă din PATH, director inexistent, timeout etc.
        raise SnapshotError(f"git {' '.join(args)} indisponibil: {type(exc).__name__}") from exc
    if result.returncode != 0:
        raise SnapshotError(f"git {' '.join(args)} a eșuat: {result.stderr.strip()}")
    return result.stdout.strip()


def read_code_version(repo_root: Path) -> CodeVersion:
    """Citește commit-ul HEAD, starea „dirty” (fișiere urmărite) și hash-ul `uv.lock`."""
    lock = repo_root / "uv.lock"
    try:
        lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    except FileNotFoundError:
        raise SnapshotError(f"uv.lock lipsă în {repo_root}") from None
    except OSError as exc:
        raise SnapshotError(f"uv.lock ilizibil în {repo_root}: {type(exc).__name__}") from None
    commit = _git(repo_root, "rev-parse", "--verify", "HEAD")
    if not commit:
        raise SnapshotError("git rev-parse HEAD nu a întors un commit")
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no") != ""
    return CodeVersion(git_commit=commit, git_dirty=dirty, lock_sha256=lock_hash)


class ConfigurationSnapshot(Frozen):
    snapshot_id: str
    created_at: UtcDatetime
    schema_version: str
    engine_version: str
    git_commit: str
    git_dirty: bool
    lock_sha256: str
    stage: str
    environment: str
    timezone: str
    seed: int
    rng_algorithm: str
    dataset_ids: list[str]
    artifact_id: str | None
    config: dict[str, Any]

    def content(self) -> dict[str, Any]:
        """Conținutul hash-uit (tot în afară de `snapshot_id` și `created_at`)."""
        return self.model_dump(mode="json", exclude={"snapshot_id", "created_at"})

    def verify(self) -> bool:
        """Adevărat dacă `snapshot_id` corespunde conținutului (detectează modificări)."""
        return _hash_content(self.content()) == self.snapshot_id


def _hash_content(content: dict[str, Any]) -> str:
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def create_snapshot(
    config: AppConfig,
    stage: StageInfo,
    code: CodeVersion,
    created_at: datetime,
    dataset_ids: list[str],
    artifact_id: str | None = None,
    redactor: Redactor = REDACTOR,
) -> ConfigurationSnapshot:
    """Creează snapshot-ul. `snapshot_id` nu depinde de `created_at`, deci aceeași rulare
    repetată produce același identificator (Req 17.5)."""
    if not code.git_commit or not code.lock_sha256:
        raise SnapshotError("versiunea codului este incompletă (commit sau hash uv.lock lipsă)")
    if any(not d for d in dataset_ids) or len(set(dataset_ids)) != len(dataset_ids):
        raise SnapshotError("dataset_ids conține identificatori goi sau duplicați")
    content: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "git_commit": code.git_commit,
        "git_dirty": code.git_dirty,
        "lock_sha256": code.lock_sha256,
        "stage": stage.stage.value,
        "environment": config.environment,
        "timezone": SNAPSHOT_TIMEZONE,
        "seed": config.run.seed,
        "rng_algorithm": RNG_ALGORITHM,
        "dataset_ids": sorted(dataset_ids),
        "artifact_id": artifact_id,
        "config": config.model_dump(mode="json"),
    }
    # Fail-closed: nicio valoare secretă cunoscută nu intră în snapshot (Req 23.2).
    # Mesajul nu include conținutul, ca să nu propage secretul.
    if redactor.contains_secret_in(content):
        raise SnapshotError("configurația efectivă conține o valoare secretă; snapshot refuzat")
    try:
        snapshot = ConfigurationSnapshot(
            snapshot_id=_hash_content(content), created_at=created_at, **content
        )
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"snapshot-ul nu poate fi creat: {exc}") from exc
    if not snapshot.verify():  # serializarea trebuie să fie stabilă (round-trip)
        raise SnapshotError("conținutul snapshot-ului nu este serializabil determinist")
    return snapshot


def take_snapshot(
    config: AppConfig,
    stage: StageInfo,
    repo_root: Path,
    created_at: datetime,
    dataset_ids: list[str],
    artifact_id: str | None = None,
    redactor: Redactor = REDACTOR,
) -> ConfigurationSnapshot:
    """Punctul de intrare la începutul rulării: citește versiunea codului și creează
    snapshot-ul. Orice eșec ridică `SnapshotError`; apelantul nu pornește procesarea."""
    code = read_code_version(repo_root)
    return create_snapshot(config, stage, code, created_at, dataset_ids, artifact_id, redactor)
