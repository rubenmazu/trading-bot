"""Etapa proiectului și verificările de pornire (Req 2.2, 2.3, 2.4).

Comportament fail-closed:
- `stage.lock` lipsă, ilizibil (permisiuni, director, codare) sau TOML invalid este tratat ca
  `initial`;
- orice valoare diferită de `initial` (inclusiv cheie lipsă) este refuzată, deoarece ieșirea din
  etapa inițială nu face parte din această versiune;
- în etapa inițială sunt permise numai Backtest, Shadow și Demo (allowlist); modul Live, brokerul
  `live`, endpoint-urile live cunoscute (oricărui broker) și conturile live cunoscute duc la
  refuzul pornirii;
- în Demo, endpoint-ul trebuie să fie în lista versionată de endpoint-uri demo (allowlist), deci
  un endpoint necunoscut este refuzat chiar dacă nu apare în lista live.

Live este dezactivat la instalare, actualizare, restaurare sau migrare (Req 2.3) prin construcție:
etapa vine exclusiv din `stage.lock` versionat în git, iar această versiune nu conține nicio cale
care să activeze Live (`is_live_enabled` întoarce mereu False).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from qts.config.schema import AppConfig

ENDPOINT_LIST_VERSION: Final = "2026-10-06.1"

# Liste versionate, pe broker. Le actualizăm doar după verificarea documentației brokerului.
# Endpoint-urile live sunt verificate față de TOATE brokerii, indiferent de `broker.name`,
# ca o configurație greșit etichetată să nu poată ocoli blocarea.
KNOWN_LIVE_ENDPOINTS_BY_BROKER: Final[dict[str, tuple[re.Pattern[str], ...]]] = {
    "alpaca": (re.compile(r"^https?://api\.alpaca\.markets(/|$)", re.IGNORECASE),),
    "ibkr": (re.compile(r"^(tcp://)?[^:/]+:(7496|4001)$", re.IGNORECASE),),  # TWS / Gateway live
}
KNOWN_DEMO_ENDPOINTS_BY_BROKER: Final[dict[str, tuple[re.Pattern[str], ...]]] = {
    "alpaca": (re.compile(r"^https?://paper-api\.alpaca\.markets(/|$)", re.IGNORECASE),),
    "ibkr": (re.compile(r"^(tcp://)?(127\.0\.0\.1|localhost):(7497|4002)$", re.IGNORECASE),),
    "fake": (re.compile(r"^fake://", re.IGNORECASE),),  # broker fals pentru teste
}
# IBKR: conturile paper încep cu "DU"; conturile live individuale încep cu "U".
KNOWN_LIVE_ACCOUNTS_BY_BROKER: Final[dict[str, tuple[re.Pattern[str], ...]]] = {
    "ibkr": (re.compile(r"^U\d+$", re.IGNORECASE),),
}

KNOWN_LIVE_ENDPOINT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    p for ps in KNOWN_LIVE_ENDPOINTS_BY_BROKER.values() for p in ps
)
KNOWN_DEMO_ENDPOINT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    p for ps in KNOWN_DEMO_ENDPOINTS_BY_BROKER.values() for p in ps
)
KNOWN_LIVE_ACCOUNT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    p for ps in KNOWN_LIVE_ACCOUNTS_BY_BROKER.values() for p in ps
)


class ProjectStage(StrEnum):
    INITIAL = "initial"


# Modurile permise pe etapă (Req 2.2). Allowlist: orice altceva este refuzat.
ALLOWED_ENVIRONMENTS_BY_STAGE: Final[dict[ProjectStage, frozenset[str]]] = {
    ProjectStage.INITIAL: frozenset({"backtest", "shadow", "demo"}),
}


class StartupRefusedError(Exception):
    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("pornire refuzată:\n  - " + "\n  - ".join(reasons))


@dataclass(frozen=True)
class StageInfo:
    stage: ProjectStage
    source: str  # "stage.lock" sau "implicit (fail-closed)"


FAIL_CLOSED_SOURCE: Final = "implicit (fail-closed)"


def read_stage(path: Path) -> StageInfo:
    """Citește etapa; orice abatere este refuzată sau tratată ca `initial`."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return StageInfo(ProjectStage.INITIAL, FAIL_CLOSED_SOURCE)
    value = raw.get("project_stage")
    if value != ProjectStage.INITIAL.value:
        raise StartupRefusedError(
            [
                f"project_stage={value!r} nu este suportat; această versiune rulează numai în "
                "Initial_Stage (Req 2)"
            ]
        )
    return StageInfo(ProjectStage.INITIAL, "stage.lock")


def is_live_enabled(stage: StageInfo) -> bool:
    """Live nu poate fi activat în această versiune (Req 2.2, 2.3)."""
    return "live" in ALLOWED_ENVIRONMENTS_BY_STAGE.get(stage.stage, frozenset())


def _matches(value: str | None, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return value is not None and any(p.search(value.strip()) for p in patterns)


def startup_violations(config: AppConfig, stage: StageInfo) -> list[str]:
    """Întoarce toate motivele pentru care pornirea trebuie refuzată (listă goală = permis)."""
    reasons: list[str] = []
    allowed_envs = ALLOWED_ENVIRONMENTS_BY_STAGE.get(stage.stage, frozenset())
    if config.environment not in allowed_envs:
        if config.environment == "live":
            reasons.append("modul Live este interzis în Initial_Stage (Req 2.2)")
        else:
            reasons.append(f"modul {config.environment} nu este permis în etapa {stage.stage}")
    if stage.stage is ProjectStage.INITIAL:
        if config.broker.kind == "live":
            reasons.append("broker.kind=live este interzis în Initial_Stage (Req 2.4)")
        if _matches(config.broker.endpoint, KNOWN_LIVE_ENDPOINT_PATTERNS):
            reasons.append(
                f"broker.endpoint este un endpoint live cunoscut (listă {ENDPOINT_LIST_VERSION})"
            )
        if _matches(config.broker.account_id, KNOWN_LIVE_ACCOUNT_PATTERNS):
            reasons.append("broker.account_id are formatul unui cont live cunoscut")
    if config.broker.kind == "demo" and not _matches(
        config.broker.endpoint, KNOWN_DEMO_ENDPOINT_PATTERNS
    ):
        reasons.append(
            "broker.endpoint nu este în lista endpoint-urilor demo verificate "
            f"(listă {ENDPOINT_LIST_VERSION})"
        )
    return reasons


def check_startup(config: AppConfig, stage: StageInfo) -> None:
    reasons = startup_violations(config, stage)
    if reasons:
        raise StartupRefusedError(reasons)
