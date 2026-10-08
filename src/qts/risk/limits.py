"""Coduri de respingere, interdicții de expunere și limitele monetare (Req 12.4, 13.2-13.10).

Fiecare verificare întoarce `None` (trece) sau o `Violation` cu valoarea calculată și limita
activă, pentru audit (12.4). Codurile de motiv sunt stabile: testele și auditul depind de ele.
Toate sumele sunt în EUR și includ costurile estimate (13.9).
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from enum import StrEnum

from pydantic import model_validator

from qts.core.models import Dec, Frozen, Instrument
from qts.core.money import ZERO

from .context import PositionRisk, ProjectStageName

__all__ = [
    "ExposureApproval",
    "RejectReason",
    "Violation",
    "check_daily_loss",
    "check_forbidden_exposure",
    "check_total_loss",
    "open_risk_eur",
    "position_open_risk_eur",
]


class RejectReason(StrEnum):
    # Pasul 1: condiții de blocare (12.6)
    KILL_SWITCH_ACTIVE = "RISK_KILL_SWITCH_ACTIVE"
    MODE_NOT_PERMITTED = "RISK_MODE_NOT_PERMITTED"
    DATA_MISSING = "RISK_DATA_MISSING"
    DATA_STALE = "RISK_DATA_STALE"
    # Pasul 2: interdicții de expunere (13.7, 13.10)
    LEVERAGE_FORBIDDEN = "RISK_LEVERAGE_FORBIDDEN"
    SHORT_FORBIDDEN = "RISK_SHORT_FORBIDDEN"
    DERIVATIVE_FORBIDDEN = "RISK_DERIVATIVE_FORBIDDEN"
    SHORT_UNSUPPORTED = "RISK_SHORT_UNSUPPORTED"
    MAX_OPEN_POSITIONS = "RISK_MAX_OPEN_POSITIONS"
    # Pasul 3: date de calcul
    STOP_MISSING = "RISK_STOP_MISSING"
    STOP_NOT_BELOW_ENTRY = "RISK_STOP_NOT_BELOW_ENTRY"
    FX_RATE_MISSING = "RISK_FX_RATE_MISSING"
    COST_MODEL_INCOMPLETE = "RISK_COST_MODEL_INCOMPLETE"
    INVALID_QTY = "RISK_INVALID_QTY"
    # Pasul 5: per tranzacție (13.2, 13.8)
    QTY_BELOW_MIN = "RISK_QTY_BELOW_MIN"
    MIN_NOTIONAL = "RISK_MIN_NOTIONAL"
    CASH_INSUFFICIENT = "RISK_CASH_INSUFFICIENT"
    TRADE_RISK_LIMIT = "RISK_TRADE_RISK_LIMIT"
    # Pașii 6-7: zilnic și total (13.3, 13.5)
    DAILY_LOSS_LIMIT = "RISK_DAILY_LOSS_LIMIT"
    TOTAL_LOSS_LIMIT = "RISK_TOTAL_LOSS_LIMIT"


class Violation(Frozen):
    reason: RejectReason
    value: Dec | None = None
    limit: Dec | None = None
    detail: str = ""


class ExposureApproval(Frozen):
    """Configurație de risc aprobată pentru expuneri interzise în Initial_Stage (Req 13.10).

    Are efect numai când `project_stage == "post_initial"`; în Initial_Stage este ignorată.
    """

    approved: bool
    approval_ref: str
    allow_leverage: bool = False
    allow_short: bool = False
    allow_derivatives: bool = False

    @model_validator(mode="after")
    def _ref_required(self) -> ExposureApproval:
        if self.approved and not self.approval_ref.strip():
            raise ValueError("o aprobare necesită approval_ref")
        return self


def _allowed(stage: ProjectStageName, approval: ExposureApproval | None, flag: str) -> bool:
    if stage != "post_initial" or approval is None or not approval.approved:
        return False
    return bool(getattr(approval, flag))


def check_forbidden_exposure(
    instrument: Instrument,
    *,
    is_short: bool,
    stage: ProjectStageName,
    approval: ExposureApproval | None,
) -> Violation | None:
    """Pasul 2: levier, short, futures/CFD/opțiuni (13.7); excepții numai aprobate (13.10)."""
    if instrument.is_derivative and not _allowed(stage, approval, "allow_derivatives"):
        return Violation(
            reason=RejectReason.DERIVATIVE_FORBIDDEN,
            detail=f"{instrument.symbol} este instrument derivat (futures/CFD/opțiuni)",
        )
    if instrument.requires_leverage and not _allowed(stage, approval, "allow_leverage"):
        return Violation(
            reason=RejectReason.LEVERAGE_FORBIDDEN,
            detail=f"{instrument.symbol} necesită levier",
        )
    if (instrument.requires_short or is_short) and not _allowed(stage, approval, "allow_short"):
        return Violation(
            reason=RejectReason.SHORT_FORBIDDEN,
            detail=f"ordinul pe {instrument.symbol} ar crea o vânzare în lipsă",
        )
    if is_short:
        # Aprobat formal, dar dimensionarea short nu există în această versiune: fail-closed.
        return Violation(
            reason=RejectReason.SHORT_UNSUPPORTED,
            detail="evaluarea expunerilor short nu este implementată",
        )
    return None


def position_open_risk_eur(pos: PositionRisk) -> Decimal:
    """Pierderea suplimentară cea mai defavorabilă, de la marcaj la stop, plus costul ieșirii.

    Fără stop, întreaga valoare marcată este considerată la risc (fail-closed).
    """
    floor = pos.stop_price if pos.stop_price is not None else ZERO
    price_risk = max(ZERO, pos.mark_price - floor) * pos.qty * pos.fx_rate_to_eur
    return price_risk + pos.exit_cost_eur


def open_risk_eur(positions: Iterable[PositionRisk]) -> Decimal:
    return sum((position_open_risk_eur(p) for p in positions), ZERO)


def check_daily_loss(
    daily_loss: Decimal, open_risk: Decimal, trade_risk: Decimal, limit: Decimal
) -> Violation | None:
    """Pasul 6: pierderea zilnică + riscul deschis + riscul tranzacției ≤ limita zilnică."""
    value = daily_loss + open_risk + trade_risk
    if value > limit:
        return Violation(
            reason=RejectReason.DAILY_LOSS_LIMIT,
            value=value,
            limit=limit,
            detail=f"pierdere zilnică {daily_loss}, risc deschis {open_risk}, "
            f"risc tranzacție {trade_risk}",
        )
    return None


def check_total_loss(
    total_loss: Decimal, open_risk: Decimal, trade_risk: Decimal, limit: Decimal
) -> Violation | None:
    """Pasul 7: pierderea totală + riscul deschis + riscul tranzacției ≤ 10 EUR (13.5)."""
    value = total_loss + open_risk + trade_risk
    if value > limit:
        return Violation(
            reason=RejectReason.TOTAL_LOSS_LIMIT,
            value=value,
            limit=limit,
            detail=f"pierdere totală {total_loss}, risc deschis {open_risk}, "
            f"risc tranzacție {trade_risk}",
        )
    return None
