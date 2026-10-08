"""Risk_Engine: evaluarea ordonată a fiecărui `Order_Intent` (Req 12, 13; design Risk_Engine).

Ordinea verificărilor (prima încălcare oprește evaluarea):

1. `Kill_Switch` aplicabil activ, mod nepermis pentru etapă, date lipsă sau expirate (12.6);
2. levier, short, futures/CFD/opțiuni (13.7; excepții numai aprobate după Initial_Stage, 13.10),
   apoi numărul maxim de poziții deschise (expunere agregată, 12.3);
3. stop obligatoriu și `entry > stop`, curs FX către EUR;
4. dimensionarea pe stop și costuri (`sizing.size_entry`);
5. per tranzacție: minim tranzacționabil, valoare minimă, numerar după costuri și riscul
   dus-întors ≤ `risk_per_trade_max_eur` (13.2, 13.8, 13.9);
6. pierderea zilnică (realizată + nerealizată) + risc deschis + risc tranzacție ≤ limita zilnică;
7. pierderea totală + risc deschis + risc tranzacție ≤ 10 EUR (13.5);
8. pașii 5-7 rulează întotdeauna pe cantitatea finală, deci și după orice reducere (12.7).

Ordinele de ieșire (SELL pentru cel mult cantitatea deținută) reduc riscul: trec prin pașii 1-2,
dar nu prin dimensionare și limitele 5-7, care constrâng numai riscul nou. Un `Kill_Switch`
activ blochează și ieșirile, ca orice ordin nou (design, proprietatea 13 „kill switch dominant”);
protecția pozițiilor deschise rămâne în seama ordinelor stop deja plasate și a politicii
`open_orders_policy` (14.3). Un SELL peste cantitatea deținută este vânzare în lipsă (13.7).

Modelul de costuri incomplet produce respingere (fail-closed, 8.3). Fiecare respingere conține
codul stabil, valoarea calculată și limita activă (12.4); fiecare decizie conține valorile
calculate (12.3) și versiunea regulilor (12.5).
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Final

from pydantic import Field, model_validator

from qts.config.schema import ENVIRONMENTS, RiskConfig
from qts.core.models import Dec, Frozen, OrderIntent
from qts.core.money import REPORTING_CURRENCY, ZERO
from qts.costs.errors import CostModelIncomplete
from qts.costs.model import CostModel
from qts.safety.stage import ALLOWED_ENVIRONMENTS_BY_STAGE, ProjectStage

from .context import MarketSnapshot, RiskContext
from .limits import (
    ExposureApproval,
    RejectReason,
    Violation,
    check_daily_loss,
    check_forbidden_exposure,
    check_total_loss,
    open_risk_eur,
)
from .sizing import TradeRisk, TradeRiskEstimator, size_entry

__all__ = ["RISK_RULES_VERSION", "RiskDecision", "RiskEngine"]

RISK_RULES_VERSION: Final = "risk-rules-v1"

_ALLOWED_MODES: Final[dict[str, frozenset[str]]] = {
    "initial": ALLOWED_ENVIRONMENTS_BY_STAGE[ProjectStage.INITIAL],
    # Live rămâne protejat separat de Live_Gate și Fail_Safe_Block (Req 2.6, 15).
    "post_initial": frozenset(ENVIRONMENTS),
}


class RiskDecision(Frozen):
    """`APPROVE(qty)` sau `REJECT(reason, value, limit)`."""

    intent_id: str
    approved: bool
    qty: Dec | None = None
    reason: RejectReason | None = None
    value: Dec | None = None
    limit: Dec | None = None
    detail: str = ""
    is_exit: bool = False
    reduced: bool = False
    rules_version: str = RISK_RULES_VERSION
    metrics: dict[str, Dec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _shape(self) -> RiskDecision:
        if self.approved:
            if self.qty is None or self.qty <= 0 or self.reason is not None:
                raise ValueError("APPROVE necesită qty > 0 și niciun motiv")
        elif self.reason is None or self.qty is not None:
            raise ValueError("REJECT necesită un motiv și nicio cantitate")
        return self


def _metrics(risk: TradeRisk, **extra: Decimal) -> dict[str, Decimal]:
    return {
        "qty": risk.qty,
        "entry_price": risk.entry_price,
        "stop_price": risk.stop_price,
        "fx_rate_to_eur": risk.fx_rate_to_eur,
        "notional_ccy": risk.notional_ccy,
        "notional_eur": risk.notional_eur,
        "price_risk_eur": risk.price_risk_eur,
        "entry_cost_eur": risk.entry_cost_eur,
        "exit_cost_eur": risk.exit_cost_eur,
        "trade_risk_eur": risk.trade_risk_eur,
        "cash_required_eur": risk.cash_required_eur,
        **extra,
    }


class RiskEngine:
    """Motor de risc independent de Strategy; aceleași reguli în toate modurile (12.2, 12.5)."""

    def __init__(
        self,
        config: RiskConfig,
        cost_model: CostModel,
        exposure_approval: ExposureApproval | None = None,
    ) -> None:
        self._config = config
        self._costs = cost_model
        self._approval = exposure_approval

    @property
    def config(self) -> RiskConfig:
        return self._config

    # ----------------------------------------------------------------- API public

    def evaluate(self, intent: OrderIntent, ctx: RiskContext) -> RiskDecision:
        def reject(v: Violation, metrics: dict[str, Decimal] | None = None) -> RiskDecision:
            return RiskDecision(
                intent_id=intent.intent_id,
                approved=False,
                reason=v.reason,
                value=v.value,
                limit=v.limit,
                detail=v.detail,
                is_exit=intent.side == "SELL",
                metrics=metrics or {},
            )

        # Pasul 1
        blocked = self._check_blocking(intent, ctx)
        if isinstance(blocked, Violation):
            return reject(blocked)
        snapshot = blocked

        # Pasul 2
        held = ctx.positions[intent.instrument].qty if intent.instrument in ctx.positions else ZERO
        if intent.side == "SELL":
            sell_qty = intent.requested_qty if intent.requested_qty is not None else held
            if sell_qty <= 0:
                return reject(
                    Violation(
                        reason=RejectReason.INVALID_QTY,
                        value=sell_qty,
                        limit=held,
                        detail="nicio cantitate deținută de vândut",
                    )
                )
            forbidden = check_forbidden_exposure(
                snapshot.instrument,
                is_short=sell_qty > held,
                stage=ctx.project_stage,
                approval=self._approval,
            )
            if forbidden is not None:
                return reject(forbidden.model_copy(update={"value": sell_qty, "limit": held}))
            return RiskDecision(
                intent_id=intent.intent_id,
                approved=True,
                qty=sell_qty,
                is_exit=True,
                metrics={"qty": sell_qty, "held_qty": held},
            )

        forbidden = check_forbidden_exposure(
            snapshot.instrument, is_short=False, stage=ctx.project_stage, approval=self._approval
        )
        if forbidden is not None:
            return reject(forbidden)
        if intent.instrument not in ctx.positions:
            open_count = Decimal(len(ctx.positions))
            limit = Decimal(self._config.max_open_positions)
            if open_count >= limit:
                return reject(
                    Violation(
                        reason=RejectReason.MAX_OPEN_POSITIONS,
                        value=open_count + 1,
                        limit=limit,
                        detail="o poziție nouă ar depăși numărul maxim de poziții deschise",
                    )
                )
        if intent.requested_qty is not None and intent.requested_qty <= 0:
            return reject(
                Violation(reason=RejectReason.INVALID_QTY, value=intent.requested_qty, limit=ZERO)
            )

        # Pasul 3
        prepared = self._prepare_estimator(intent, ctx, snapshot)
        if isinstance(prepared, Violation):
            return reject(prepared)
        est = prepared

        # Pașii 4-8
        try:
            return self._size_and_check(intent, ctx, snapshot, est, reject)
        except CostModelIncomplete as exc:
            return reject(
                Violation(
                    reason=RejectReason.COST_MODEL_INCOMPLETE,
                    detail=f"{exc.component}: {exc.detail}",
                ),
                None,
            )

    # ----------------------------------------------------------------- pași

    @staticmethod
    def _check_blocking(intent: OrderIntent, ctx: RiskContext) -> MarketSnapshot | Violation:
        scope = ctx.kill_switch.blocking_scope(intent.instrument)
        if scope is not None:
            return Violation(
                reason=RejectReason.KILL_SWITCH_ACTIVE, detail=f"Kill_Switch activ: {scope}"
            )
        stage = ctx.project_stage if ctx.project_stage == "post_initial" else "initial"
        if ctx.mode not in _ALLOWED_MODES[stage]:
            return Violation(
                reason=RejectReason.MODE_NOT_PERMITTED,
                detail=f"modul {ctx.mode} nu este permis în etapa {stage}",
            )
        snapshot = ctx.market.get(intent.instrument)
        if snapshot is None:
            return Violation(
                reason=RejectReason.DATA_MISSING,
                detail=f"fără date de piață pentru {intent.instrument}",
            )
        if not snapshot.data_fresh:
            return Violation(
                reason=RejectReason.DATA_STALE,
                detail=f"date expirate: {snapshot.freshness_reason or 'necunoscut'}",
            )
        return snapshot

    def _prepare_estimator(
        self, intent: OrderIntent, ctx: RiskContext, snapshot: MarketSnapshot
    ) -> TradeRiskEstimator | Violation:
        if intent.stop_price is None:
            return Violation(reason=RejectReason.STOP_MISSING, detail="intrarea necesită stop")
        if intent.order_type == "LIMIT" and intent.limit_price is not None:
            entry = intent.limit_price
        else:
            # Prețul cel mai defavorabil dintre referință și ask (fail-closed).
            entry = intent.ref_price
            if snapshot.quote is not None:
                entry = max(entry, snapshot.quote.ask)
        if not entry > intent.stop_price > 0:
            return Violation(
                reason=RejectReason.STOP_NOT_BELOW_ENTRY,
                value=intent.stop_price,
                limit=entry,
                detail="pentru long este necesar 0 < stop < entry",
            )
        if snapshot.instrument.currency == REPORTING_CURRENCY:
            fx = Decimal(1)
        elif snapshot.cost_ctx.fx_rate is not None:
            fx = snapshot.cost_ctx.fx_rate
        else:
            return Violation(
                reason=RejectReason.FX_RATE_MISSING,
                detail=f"lipsește cursul {snapshot.instrument.currency}→EUR",
            )
        return TradeRiskEstimator(
            self._costs,
            snapshot,
            entry_price=entry,
            stop_price=intent.stop_price,
            fx_rate_to_eur=fx,
            ts_decision=ctx.ts,
        )

    def _size_and_check(
        self,
        intent: OrderIntent,
        ctx: RiskContext,
        snapshot: MarketSnapshot,
        est: TradeRiskEstimator,
        reject: Callable[[Violation, dict[str, Decimal] | None], RiskDecision],
    ) -> RiskDecision:
        cfg = self._config
        inst = snapshot.instrument

        # Pasul 4: dimensionare pe bugetul țintă.
        sizing = size_entry(
            est,
            inst,
            budget_eur=cfg.risk_per_trade_target_eur,
            cash_eur=ctx.cash_eur,
            requested_qty=intent.requested_qty,
        )
        qty = sizing.qty
        sizing_metrics: dict[str, Decimal] = {
            "candidate_qty": sizing.candidate_qty,
            "fixed_costs_eur": sizing.fixed_costs_eur,
            "cost_per_unit_roundtrip_eur": sizing.cost_per_unit_roundtrip_eur,
            "risk_per_unit_eur": sizing.risk_per_unit_eur,
        }

        # Pasul 5 (și 8: totul pe cantitatea finală).
        if qty <= 0 or qty < inst.min_qty:
            return reject(
                Violation(
                    reason=RejectReason.QTY_BELOW_MIN,
                    value=qty,
                    limit=inst.min_qty,
                    detail="cantitatea dimensionată este sub minimul instrumentului",
                ),
                {"qty": qty, **sizing_metrics},
            )
        risk = est.at(qty)
        open_risk = open_risk_eur(ctx.positions.values())
        exposure = sum(
            (p.qty * p.mark_price * p.fx_rate_to_eur for p in ctx.positions.values()), ZERO
        )
        metrics = _metrics(
            risk,
            **sizing_metrics,
            cash_eur=ctx.cash_eur,
            open_risk_eur=open_risk,
            daily_loss_eur=ctx.daily_loss_eur,
            total_loss_eur=ctx.total_loss_eur,
            aggregate_exposure_eur=exposure + risk.notional_eur,
        )
        per_trade = self._check_per_trade(risk, ctx, inst.min_notional)
        if per_trade is not None:
            return reject(per_trade, metrics)

        # Pașii 6-7.
        for violation in (
            check_daily_loss(
                ctx.daily_loss_eur, open_risk, risk.trade_risk_eur, cfg.daily_loss_limit_eur
            ),
            check_total_loss(
                ctx.total_loss_eur, open_risk, risk.trade_risk_eur, cfg.total_loss_limit_eur
            ),
        ):
            if violation is not None:
                return reject(violation, metrics)

        return RiskDecision(
            intent_id=intent.intent_id,
            approved=True,
            qty=qty,
            reduced=sizing.reduced,
            metrics=metrics,
        )

    def _check_per_trade(
        self, risk: TradeRisk, ctx: RiskContext, min_notional: Decimal
    ) -> Violation | None:
        if risk.notional_ccy < min_notional:
            return Violation(
                reason=RejectReason.MIN_NOTIONAL,
                value=risk.notional_ccy,
                limit=min_notional,
                detail="valoarea nominală este sub minimul instrumentului",
            )
        if risk.cash_required_eur > ctx.cash_eur:
            return Violation(
                reason=RejectReason.CASH_INSUFFICIENT,
                value=risk.cash_required_eur,
                limit=ctx.cash_eur,
                detail="valoarea nominală plus costurile de intrare depășesc numerarul",
            )
        limit = self._config.risk_per_trade_max_eur
        if risk.trade_risk_eur > limit:
            return Violation(
                reason=RejectReason.TRADE_RISK_LIMIT,
                value=risk.trade_risk_eur,
                limit=limit,
                detail="pierderea la stop plus costurile dus-întors depășesc limita",
            )
        return None
