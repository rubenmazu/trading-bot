"""Cercetare: preînregistrare, walk-forward, robustețe și rapoarte."""

from qts.research.paper_qualification import (
    PAPER_QUALIFICATION_VERSION,
    PaperQualificationCriteria,
    PaperQualificationResult,
    PaperStage,
    StageObservations,
    evaluate_paper_qualification,
)
from qts.research.partition import (
    DataPartitioner,
    OosAlreadyConsumedError,
    OosNotReservedError,
    OosRecord,
    OosRegistry,
    OosStatus,
    Partition,
    PartitionRole,
    PartitionSplit,
    RegistryError,
    partition_hash,
)
from qts.research.report import (
    PROFIT_NOT_GUARANTEED_WARNING,
    CostAssumptions,
    EvaluationReport,
    UncertaintyReport,
    VariantReport,
    build_evaluation_report,
)

__all__ = [
    "PAPER_QUALIFICATION_VERSION",
    "PROFIT_NOT_GUARANTEED_WARNING",
    "CostAssumptions",
    "DataPartitioner",
    "EvaluationReport",
    "OosAlreadyConsumedError",
    "OosNotReservedError",
    "OosRecord",
    "OosRegistry",
    "OosStatus",
    "PaperQualificationCriteria",
    "PaperQualificationResult",
    "PaperStage",
    "Partition",
    "PartitionRole",
    "PartitionSplit",
    "RegistryError",
    "StageObservations",
    "UncertaintyReport",
    "VariantReport",
    "build_evaluation_report",
    "evaluate_paper_qualification",
    "partition_hash",
]
