"""Manifestul unui set de date și verificarea sumei de control (Req 5.6, 5.8).

Fiecare fișier de date (CSV sau Parquet) are alături un manifest JSON `<fișier>.manifest.json`
care înregistrează sursa, intervalul temporal, fusul orar, calendarul, ajustările corporative
și SHA-256 al fișierului. Identificatorul setului (`dataset_id`) este derivat din sumă, deci
două seturi cu același conținut au același identificator, iar orice modificare îl schimbă.

La încărcare suma este recalculată. Dacă nu corespunde sau fișierul lipsește, setul este
invalidat: se ridică `DatasetInvalidError` și nu se emite niciun eveniment.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import Field, ValidationError, model_validator

from qts.core.models import Frozen, UtcDatetime, canonical_json

logger = logging.getLogger(__name__)

MANIFEST_SUFFIX: Final = ".manifest.json"
MANIFEST_SCHEMA_VERSION: Final = 1
_CHUNK: Final = 1 << 20

DataFormat = Literal["csv", "parquet"]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, Field(min_length=1)]


class DatasetInvalidError(Exception):
    """Setul de date nu poate fi folosit; `reason_code` este stabil pentru audit."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


class DatasetManifest(Frozen):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    source_id: NonEmpty
    data_file: NonEmpty  # numele fișierului, relativ la directorul manifestului
    format: DataFormat
    start: UtcDatetime
    end: UtcDatetime
    timezone: NonEmpty  # fusul în care sunt exprimați timpii fără offset din fișier
    calendar_id: NonEmpty
    corporate_adjustments: NonEmpty  # de exemplu "none" sau "split_dividend_adjusted"
    sha256: Sha256Hex

    @model_validator(mode="after")
    def _check(self) -> DatasetManifest:
        if self.end < self.start:
            raise ValueError("end trebuie să fie >= start")
        if Path(self.data_file).name != self.data_file:
            raise ValueError("data_file trebuie să fie un nume de fișier, fără director")
        return self

    @property
    def dataset_id(self) -> str:
        """Identificator stabil derivat din suma de control."""
        return f"ds_{self.sha256[:16]}"


def manifest_path_for(data_path: Path) -> Path:
    return data_path.with_name(data_path.name + MANIFEST_SUFFIX)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _format_of(path: Path) -> DataFormat:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".parquet", ".pq"):
        return "parquet"
    raise ValueError(f"format de fișier nesuportat: {path.name}")


def create_manifest(
    data_path: Path,
    *,
    source_id: str,
    start: datetime,
    end: datetime,
    timezone: str,
    calendar_id: str,
    corporate_adjustments: str,
) -> DatasetManifest:
    """Calculează suma de control și scrie manifestul alături de fișierul de date."""
    manifest = DatasetManifest(
        source_id=source_id,
        data_file=data_path.name,
        format=_format_of(data_path),
        start=start,
        end=end,
        timezone=timezone,
        calendar_id=calendar_id,
        corporate_adjustments=corporate_adjustments,
        sha256=file_sha256(data_path),
    )
    manifest_path_for(data_path).write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    logger.info("manifest scris dataset_id=%s file=%s", manifest.dataset_id, data_path.name)
    return manifest


def load_manifest(manifest_path: Path) -> DatasetManifest:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return DatasetManifest.model_validate(raw)
    except FileNotFoundError as exc:
        raise DatasetInvalidError("DATASET_MANIFEST_MISSING", str(manifest_path)) from exc
    except (ValidationError, ValueError) as exc:
        raise DatasetInvalidError("DATASET_MANIFEST_INVALID", str(exc).splitlines()[0]) from exc


def verify_dataset(manifest: DatasetManifest, data_path: Path) -> None:
    """Ridică `DatasetInvalidError` dacă fișierul lipsește sau suma nu corespunde (Req 5.8)."""
    if not data_path.is_file():
        logger.error("set invalidat: fișier lipsă dataset_id=%s", manifest.dataset_id)
        raise DatasetInvalidError("DATASET_FILE_MISSING", str(data_path))
    actual = file_sha256(data_path)
    if actual != manifest.sha256:
        logger.error(
            "set invalidat: sumă de control diferită dataset_id=%s expected=%s actual=%s",
            manifest.dataset_id,
            manifest.sha256,
            actual,
        )
        raise DatasetInvalidError(
            "DATASET_CHECKSUM_MISMATCH",
            f"{data_path.name}: expected {manifest.sha256}, actual {actual}",
        )


def load_verified(data_path: Path) -> DatasetManifest:
    """Încarcă manifestul fișierului și verifică suma de control."""
    manifest = load_manifest(manifest_path_for(data_path))
    if manifest.data_file != data_path.name:
        raise DatasetInvalidError(
            "DATASET_MANIFEST_INVALID",
            f"manifestul descrie {manifest.data_file}, nu {data_path.name}",
        )
    verify_dataset(manifest, data_path)
    return manifest
