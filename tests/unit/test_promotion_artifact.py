"""Teste pentru `promotion/artifact.py` (Req 16.2, 16.3, 17.5).

Acoperă: determinismul `artifact_id`, schimbarea `artifact_id` la orice modificare de conținut,
`verify`, respingerea unui `artifact_id` incoerent și invariantele modelului.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from qts.promotion.artifact import StrategyArtifact, artifact_hash, create_artifact
from qts.research.partition import DataPartitioner, Partition

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 1, tzinfo=UTC)
INSTRUMENTS = ("AAA", "BBB")


def _partitions() -> tuple[Partition, ...]:
    split = DataPartitioner(oos_fraction=0.25).split(
        start_ts=START, end_ts=END, instruments=INSTRUMENTS, label="study"
    )
    return (split.development, split.out_of_sample)


def _artifact(**overrides: Any) -> StrategyArtifact:
    kwargs: dict[str, Any] = {
        "strategy_id": "mean_reversion",
        "strategy_version": "1.0.0",
        "code_hash": "code-abc",
        "params": {"z": 2, "window": 15},
        "risk_config_hash": "risk-abc",
        "cost_model_version": "cost-v1",
        "universe_version": "universe-v1",
        "partitions": _partitions(),
        "preregistration_hash": "pre-abc",
        "validation_report_hash": "report-abc",
    }
    kwargs.update(overrides)
    return create_artifact(**kwargs)


def test_artifact_id_is_deterministic() -> None:
    # Req 17.5: aceeași logică, construită de două ori, produce același artifact_id.
    assert _artifact().artifact_id == _artifact().artifact_id


def test_artifact_verifies_its_own_hash() -> None:
    assert _artifact().verify() is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"strategy_version": "2.0.0"},
        {"code_hash": "code-xyz"},
        {"params": {"z": 3, "window": 15}},
        {"risk_config_hash": "risk-xyz"},
        {"cost_model_version": "cost-v2"},
        {"universe_version": "universe-v2"},
        {"preregistration_hash": "pre-xyz"},
        {"validation_report_hash": "report-xyz"},
    ],
)
def test_artifact_id_changes_on_any_content_change(overrides: dict[str, Any]) -> None:
    # Req 16.3: orice modificare de conținut schimbă artifact_id (impune versiune nouă).
    assert _artifact().artifact_id != _artifact(**overrides).artifact_id


def test_param_order_does_not_change_artifact_id() -> None:
    # Serializarea canonică sortează cheile: ordinea de inserare a parametrilor nu contează.
    a = _artifact(params={"z": 2, "window": 15})
    b = _artifact(params={"window": 15, "z": 2})
    assert a.artifact_id == b.artifact_id


def test_tampered_artifact_id_is_rejected() -> None:
    good = _artifact()
    tampered = good.content()
    with pytest.raises(ValueError, match="artifact_id"):
        StrategyArtifact(artifact_id="deadbeef", **tampered)


def test_empty_required_fields_rejected() -> None:
    with pytest.raises(ValueError, match="strategy_id"):
        _artifact(strategy_id="  ")
    with pytest.raises(ValueError, match="code_hash"):
        _artifact(code_hash="")


def test_requires_at_least_one_partition() -> None:
    with pytest.raises(ValueError, match="partiție"):
        _artifact(partitions=())


def test_duplicate_partitions_rejected() -> None:
    dev, _ = _partitions()
    with pytest.raises(ValueError, match="duplicate"):
        _artifact(partitions=(dev, dev))


def test_validation_report_hash_optional() -> None:
    without = _artifact(validation_report_hash=None)
    assert without.validation_report_hash is None
    assert without.verify() is True


def test_artifact_hash_matches_helper() -> None:
    a = _artifact()
    assert artifact_hash(a.content()) == a.artifact_id
