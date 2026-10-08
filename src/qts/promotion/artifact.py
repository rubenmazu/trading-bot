"""`StrategyArtifact` imuabil: logica validată, hash-uită stabil (Req 16.2, 18.5).

Un `StrategyArtifact` capturează tot ce definește comportamentul unei strategii validate — codul
(`code_hash`), parametrii, regulile `Risk_Engine` (`risk_config_hash`), modelul complet de costuri
(`cost_model_version`), universul (`universe_version`), partițiile de date (cronologice, cu rol) și
hash-urile de pre-înregistrare și raport de validare. `artifact_id` = SHA-256 peste tot acest
conținut (totul în afară de `artifact_id` însuși), astfel încât:

- aceeași logică, construită de două ori, produce același `artifact_id` (determinism, Req 17.5);
- orice modificare a oricărui câmp de conținut schimbă `artifact_id`, ceea ce declanșează crearea
  unei versiuni noi și reluarea validării de la Backtest (Req 16.3).

Modelul este `frozen` și `extra="forbid"`: după construire nu mai poate fi mutat, iar câmpurile
necunoscute sunt respinse. La promovare, logica (acest artefact) rămâne neschimbată; doar
adaptoarele, configurația de mediu și referințele credentialelor se schimbă (Req 16.6), lucru
verificat de `promotion.promote.detect_disallowed_differences`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

from pydantic import model_validator

from qts.core.models import Frozen
from qts.research.partition import Partition

__all__ = [
    "ARTIFACT_VERSION",
    "StrategyArtifact",
    "artifact_hash",
    "create_artifact",
]

ARTIFACT_VERSION: Final = "artifact-v1"


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class StrategyArtifact(Frozen):
    """Artefactul imuabil al unei strategii validate (model din design.md, Data Models).

    `artifact_id` este SHA-256 peste conținut (toate câmpurile în afară de el însuși), calculat de
    `artifact_hash`. `verify` recompută hash-ul și confirmă că nimic nu s-a schimbat de la creare.
    """

    artifact_id: str
    version: str = ARTIFACT_VERSION
    strategy_id: str
    strategy_version: str
    code_hash: str
    params: dict[str, Any]
    risk_config_hash: str
    cost_model_version: str
    universe_version: str
    partitions: tuple[Partition, ...]
    preregistration_hash: str
    validation_report_hash: str | None = None

    @model_validator(mode="after")
    def _check(self) -> StrategyArtifact:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id este obligatoriu")
        if not self.strategy_version.strip():
            raise ValueError("strategy_version este obligatoriu")
        if not self.code_hash.strip():
            raise ValueError("code_hash este obligatoriu")
        if not self.risk_config_hash.strip():
            raise ValueError("risk_config_hash este obligatoriu")
        if not self.cost_model_version.strip():
            raise ValueError("cost_model_version este obligatoriu")
        if not self.universe_version.strip():
            raise ValueError("universe_version este obligatoriu")
        if not self.preregistration_hash.strip():
            raise ValueError("preregistration_hash este obligatoriu")
        if not self.partitions:
            raise ValueError("artefactul necesită cel puțin o partiție de date (Req 18.5)")
        dataset_ids = [p.dataset_id for p in self.partitions]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("partiții duplicate în artefact")
        expected = artifact_hash(self.content())
        if self.artifact_id != expected:
            raise ValueError("artifact_id nu corespunde conținutului artefactului")
        return self

    def content(self) -> dict[str, Any]:
        """Conținutul hash-uit (tot în afară de `artifact_id`)."""
        return self.model_dump(mode="json", exclude={"artifact_id"})

    def verify(self) -> bool:
        """Adevărat dacă `artifact_id` corespunde conținutului (detectează orice modificare)."""
        return artifact_hash(self.content()) == self.artifact_id


def artifact_hash(content: dict[str, Any]) -> str:
    """SHA-256 hex peste forma canonică a conținutului artefactului."""
    return hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


def create_artifact(
    *,
    strategy_id: str,
    strategy_version: str,
    code_hash: str,
    params: dict[str, Any],
    risk_config_hash: str,
    cost_model_version: str,
    universe_version: str,
    partitions: tuple[Partition, ...],
    preregistration_hash: str,
    validation_report_hash: str | None = None,
) -> StrategyArtifact:
    """Construiește un `StrategyArtifact` cu `artifact_id` derivat din conținut (Req 16.2, 17.5).

    Aceeași logică, construită de două ori, produce același identificator: hash-ul nu depinde de
    niciun element volatil (fără timestamp, fără ordine aleatorie). Orice schimbare de conținut
    (cod, parametri, risc, costuri, univers, partiții, pre-înregistrare, raport) schimbă
    `artifact_id`, ceea ce impune o versiune nouă și reluarea validării de la Backtest (Req 16.3).
    """
    content: dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "strategy_id": strategy_id,
        "strategy_version": strategy_version,
        "code_hash": code_hash,
        "params": params,
        "risk_config_hash": risk_config_hash,
        "cost_model_version": cost_model_version,
        "universe_version": universe_version,
        "partitions": [p.model_dump(mode="json") for p in partitions],
        "preregistration_hash": preregistration_hash,
        "validation_report_hash": validation_report_hash,
    }
    artifact = StrategyArtifact(
        artifact_id=artifact_hash(content),
        version=ARTIFACT_VERSION,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        code_hash=code_hash,
        params=params,
        risk_config_hash=risk_config_hash,
        cost_model_version=cost_model_version,
        universe_version=universe_version,
        partitions=partitions,
        preregistration_hash=preregistration_hash,
        validation_report_hash=validation_report_hash,
    )
    if not artifact.verify():  # serializarea trebuie să fie stabilă (round-trip)
        raise ValueError("conținutul artefactului nu este serializabil determinist")
    return artifact
