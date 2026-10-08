"""Încărcarea configurației TOML cu raportarea tuturor abaterilor (Req 1.3, 17.1, 17.2, 17.6).

Există un fișier per mediu (`config/backtest.toml`, `shadow.toml`, `demo.toml`, `live.toml`).
Fiecare fișier își declară explicit mediul în câmpul `environment`; un fișier care declară alt
mediu decât cel așteptat (din numele fișierului sau din apelant) este refuzat.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from qts.config.schema import ENVIRONMENTS, AppConfig, ConfigIssuesError

DEFAULT_CONFIG_DIR = Path("config")


class ConfigError(Exception):
    """Configurație invalidă. `issues` conține fiecare abatere, nu doar prima."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("configurație invalidă:\n  - " + "\n  - ".join(issues))


def _format_errors(exc: ValidationError) -> list[str]:
    issues: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<rădăcină>"
        inner = err.get("ctx", {}).get("error")
        if isinstance(inner, ConfigIssuesError):
            issues.extend(f"{loc}: {issue} ({err['type']})" for issue in inner.issues)
        else:
            issues.append(f"{loc}: {err['msg']} ({err['type']})")
    return issues


def _environment_issues(data: dict[str, Any], expected: str | None, origin: str) -> list[str]:
    if expected is None:
        return []
    if expected not in ENVIRONMENTS:
        return [f"mediu așteptat necunoscut: {expected}; permis: {list(ENVIRONMENTS)}"]
    declared = data.get("environment")
    if declared is not None and declared != expected:
        return [
            f"environment: {origin} declară '{declared}', dar mediul așteptat este '{expected}'"
        ]
    return []


def parse_config(data: dict[str, Any], expected_environment: str | None = None) -> AppConfig:
    """Validează un dicționar de configurație.

    Dacă `expected_environment` este dat, un mediu declarat diferit este o abatere raportată
    împreună cu celelalte abateri ale schemei.
    """
    issues = _environment_issues(data, expected_environment, "configurația")
    try:
        config = AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(issues + _format_errors(exc)) from None
    if issues:
        raise ConfigError(issues)
    return config


def config_path(environment: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> Path:
    """Calea fișierului de configurație al unui mediu (`config/<mediu>.toml`)."""
    if environment not in ENVIRONMENTS:
        raise ConfigError([f"mediu necunoscut: {environment}; permis: {list(ENVIRONMENTS)}"])
    return config_dir / f"{environment}.toml"


def load_config(path: Path, expected_environment: str | None = None) -> AppConfig:
    """Încarcă și validează un fișier TOML.

    Mediul așteptat este `expected_environment` sau, dacă lipsește și numele fișierului este
    un mediu cunoscut (ex. `demo.toml`), numele fișierului.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError([f"fișierul de configurație nu există: {path}"]) from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError([f"TOML invalid în {path}: {exc}"]) from None

    issues: list[str] = []
    expected: str | None
    if path.stem in ENVIRONMENTS:
        if expected_environment is not None and expected_environment != path.stem:
            issues.append(
                f"fișierul {path.name} aparține mediului '{path.stem}', "
                f"dar mediul așteptat este '{expected_environment}'"
            )
        expected = expected_environment or path.stem
    else:
        expected = expected_environment
    try:
        config = parse_config(raw, expected_environment=expected)
    except ConfigError as exc:
        raise ConfigError(issues + exc.issues) from None
    if issues:
        raise ConfigError(issues)
    return config
