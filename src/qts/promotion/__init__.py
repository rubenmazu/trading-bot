"""Artefacte de strategie și promovare controlată între medii (Req 16).

Acest pachet conține `StrategyArtifact` imuabil, cu `artifact_id` = SHA-256 peste tot conținutul,
și mașina de promovare strict ordonată Backtest → Shadow → Demo (Live este în afara spec-ului).
"""

from qts.promotion.artifact import (
    ARTIFACT_VERSION,
    StrategyArtifact,
    artifact_hash,
    create_artifact,
)
from qts.promotion.promote import (
    ALLOWED_CHANGE_KEYS,
    PromotionBlockedError,
    PromotionController,
    PromotionError,
    PromotionOrderError,
    PromotionRecord,
    PromotionStage,
    PromotionState,
    StageStatus,
    allowed_environment_keys,
    detect_disallowed_differences,
)

__all__ = [
    "ALLOWED_CHANGE_KEYS",
    "ARTIFACT_VERSION",
    "PromotionBlockedError",
    "PromotionController",
    "PromotionError",
    "PromotionOrderError",
    "PromotionRecord",
    "PromotionStage",
    "PromotionState",
    "StageStatus",
    "StrategyArtifact",
    "allowed_environment_keys",
    "artifact_hash",
    "create_artifact",
    "detect_disallowed_differences",
]
