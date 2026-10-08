"""Risk_Engine: limite, dimensionare și context de risc."""

from .capital import (
    CAPITAL_RULES_VERSION,
    CapitalChangeApproval,
    CapitalChangeDecision,
    CapitalChangeProposal,
    CapitalRejectReason,
    evaluate_capital_change,
    is_capital_change_authorized,
)
from .context import KillSwitchScope, KillSwitchState, MarketSnapshot, PositionRisk, RiskContext
from .engine import RISK_RULES_VERSION, RiskDecision, RiskEngine
from .limits import ExposureApproval, RejectReason

__all__ = [
    "CAPITAL_RULES_VERSION",
    "RISK_RULES_VERSION",
    "CapitalChangeApproval",
    "CapitalChangeDecision",
    "CapitalChangeProposal",
    "CapitalRejectReason",
    "ExposureApproval",
    "KillSwitchScope",
    "KillSwitchState",
    "MarketSnapshot",
    "PositionRisk",
    "RejectReason",
    "RiskContext",
    "RiskDecision",
    "RiskEngine",
    "evaluate_capital_change",
    "is_capital_change_authorized",
]
