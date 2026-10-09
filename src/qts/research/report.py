"""Raportul de evaluare al unei strategii: brut, costuri, net, ipoteze, incertitudine, variante.

Req 21.3, 21.7, 29.1–29.5.

Acest modul nu rulează el însuși pipeline-ul de cercetare; primește rezultatele deja calculate
(brut, costuri pe categorii, net, distribuțiile bootstrap, lista variantelor evaluate) și le
asamblează într-un `EvaluationReport` imuabil și determinist. Raportul:

- prezintă rezultatul brut, costurile pe categorii (`CostBreakdown`) și rezultatul net (Req 29.1);
- enumeră ipotezele configurate de latență, lichiditate, spread, slippage, comisioane, conversie
  FX și taxe (Req 29.2);
- raportează incertitudinea estimării prin distribuțiile bootstrap (PnL net, drawdown maxim, cea
  mai lungă serie de pierderi, probabilitatea atingerii limitei totale de pierdere) (Req 21.7, 29);
- raportează **fiecare** variantă evaluată, inclusiv variantele respinse (Req 21.3);
- este marcat **incomplet** dacă vreo metrică obligatorie lipsește, iar promovarea este blocată în
  acest caz (Req 29.4; design: „Un raport cu o metrică lipsă este marcat incomplet");
- include întotdeauna avertismentul că profitul nu este garantat (Req 21.7, 29.5).

Toate valorile monetare sunt `Decimal`; moneda de raportare este unică (`REPORTING_CURRENCY`,
EUR), astfel încât comparațiile între moduri folosesc aceeași monedă și aceleași definiții
(Req 29.3). Raportul nu conține afirmații de profit garantat și nicio extrapolare fără ipoteze
(Req 29.5).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Final

from pydantic import Field, model_validator

from qts.core.models import CostBreakdown, Dec, Frozen
from qts.core.money import REPORTING_CURRENCY

from .bootstrap import Distribution

__all__ = [
    "PROFIT_NOT_GUARANTEED_WARNING",
    "REQUIRED_METRICS",
    "CostAssumptions",
    "EvaluationReport",
    "UncertaintyReport",
    "VariantReport",
    "build_evaluation_report",
]

# Avertismentul obligatoriu: profitul nu este garantat, pierderile sunt posibile (Req 21.7, 29.5).
PROFIT_NOT_GUARANTEED_WARNING: Final = (
    "AVERTISMENT: profitul nu este garantat; pierderile sunt posibile. Rezultatele sunt estimări "
    "cu incertitudine, fără extrapolări neînsoțite de ipoteze."
)

# Metricile obligatorii dintr-un raport de evaluare (Req 29.1). Dacă vreuna lipsește, raportul este
# marcat incomplet, iar promovarea este blocată (Req 29.4).
REQUIRED_METRICS: Final = (
    "period",
    "universe",
    "trade_count",
    "gross_result_eur",
    "net_result_eur",
    "costs",
    "max_drawdown_eur",
)


class CostAssumptions(Frozen):
    """Ipotezele configurate care intră în estimarea costurilor (Req 29.2).

    Fiecare câmp descrie, în text, ipoteza aplicată pentru fiecare componentă a modelului de
    costuri (latență, lichiditate, spread, slippage, comisioane, conversie FX, taxe). Textul este
    liber, dar prezența tuturor componentelor este obligatorie, pentru ca ipotezele raportului să
    fie complete (Req 29.2).
    """

    latency: str
    liquidity: str
    spread: str
    slippage: str
    commissions: str
    fx_conversion: str
    taxes: str

    @model_validator(mode="after")
    def _check(self) -> CostAssumptions:
        missing = [
            name
            for name in ("latency", "liquidity", "spread", "slippage", "commissions",
                         "fx_conversion", "taxes")
            if not getattr(self, name).strip()
        ]
        if missing:
            raise ValueError(f"ipoteze de cost lipsă: {sorted(missing)}")
        return self


class UncertaintyReport(Frozen):
    """Incertitudinea estimării, din distribuțiile bootstrap (Req 21.7, 29).

    Distribuțiile provin din `research.bootstrap`; sămânța și numărul de reeșantionări sunt
    raportate pentru reproducere.
    """

    seed: int
    resamples: Annotated[int, Field(ge=1)]
    net_pnl: Distribution
    max_drawdown: Distribution
    longest_losing_streak: Distribution
    prob_hit_total_loss_limit: Dec


class VariantReport(Frozen):
    """O variantă evaluată, raportată inclusiv dacă a fost respinsă (Req 21.3)."""

    label: str
    net_result_eur: Dec
    accepted: bool
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> VariantReport:
        if not self.label.strip():
            raise ValueError("varianta necesită o etichetă")
        if not self.accepted and not (self.rejection_reason or "").strip():
            raise ValueError("o variantă respinsă necesită un motiv de respingere")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("o variantă acceptată nu poate avea motiv de respingere")
        return self


class EvaluationReport(Frozen):
    """Raportul complet al unei evaluări (Req 21.3, 21.7, 29.1–29.5).

    Prezintă rezultatul brut, costurile pe categorii și rezultatul net, ipotezele, incertitudinea
    și toate variantele. Este marcat incomplet dacă o metrică obligatorie lipsește (Req 29.4), caz
    în care promovarea este blocată. Conține întotdeauna avertismentul că profitul nu este garantat.
    """

    strategy_id: str
    preregistration_id: str
    reporting_currency: str = REPORTING_CURRENCY

    # Metricile obligatorii (Req 29.1). Opționale la tip pentru a putea construi rapoarte parțiale
    # care sunt apoi marcate incomplete; `missing_metrics` le detectează.
    period: str | None = None
    universe: tuple[str, ...] | None = None
    trade_count: int | None = None
    gross_result_eur: Dec | None = None
    net_result_eur: Dec | None = None
    costs: CostBreakdown | None = None
    max_drawdown_eur: Dec | None = None

    assumptions: CostAssumptions
    uncertainty: UncertaintyReport | None = None
    variants: tuple[VariantReport, ...]

    warning: str = PROFIT_NOT_GUARANTEED_WARNING

    @model_validator(mode="after")
    def _check(self) -> EvaluationReport:
        # Req 29.3: o monedă unică de raportare per rulare (nu neapărat EUR global). Invariantul
        # este „o singură monedă coerentă", nu „întotdeauna EUR": implicitul rămâne EUR, deci
        # rapoartele EUR existente sunt neschimbate, dar o rulare în altă monedă (de exemplu Demo
        # în USD) raportează coerent în acea monedă.
        if not self.reporting_currency.strip():
            raise ValueError("moneda de raportare este obligatorie (Req 29.3)")
        if self.warning != PROFIT_NOT_GUARANTEED_WARNING:
            raise ValueError("avertismentul despre profit nu poate fi modificat (Req 21.7, 29.5)")
        if not self.variants:
            raise ValueError("raportul trebuie să enumere cel puțin varianta evaluată (Req 21.3)")
        labels = [v.label for v in self.variants]
        if len(labels) != len(set(labels)):
            raise ValueError("etichete de variantă duplicate")
        if self.universe is not None and len(self.universe) != len(set(self.universe)):
            raise ValueError("instrumente duplicate în universe")
        return self

    @property
    def missing_metrics(self) -> tuple[str, ...]:
        """Metricile obligatorii care lipsesc (Req 29.1, 29.4)."""
        missing: list[str] = []
        for name in REQUIRED_METRICS:
            value = getattr(self, name)
            if value is None:
                missing.append(name)
        # Un univers prezent dar gol nu acoperă metrica (fail-closed).
        if self.universe is not None and len(self.universe) == 0 and "universe" not in missing:
            missing.append("universe")
        return tuple(missing)

    @property
    def incomplete(self) -> bool:
        """Adevărat dacă lipsește o metrică obligatorie; raportul e marcat incomplet (Req 29.4)."""
        return bool(self.missing_metrics)

    @property
    def promotion_blocked(self) -> bool:
        """Promovarea este blocată când raportul este incomplet (Req 29.4)."""
        return self.incomplete

    @property
    def rejected_variants(self) -> tuple[VariantReport, ...]:
        """Variantele respinse, raportate explicit (Req 21.3)."""
        return tuple(v for v in self.variants if not v.accepted)


def build_evaluation_report(
    *,
    strategy_id: str,
    preregistration_id: str,
    assumptions: CostAssumptions,
    variants: tuple[VariantReport, ...],
    period: str | None = None,
    universe: tuple[str, ...] | None = None,
    trade_count: int | None = None,
    gross_result_eur: Decimal | None = None,
    net_result_eur: Decimal | None = None,
    costs: CostBreakdown | None = None,
    max_drawdown_eur: Decimal | None = None,
    uncertainty: UncertaintyReport | None = None,
) -> EvaluationReport:
    """Asamblează un `EvaluationReport` determinist din rezultatele deja calculate.

    Nu completează metrici lipsă și nu inventează valori: dacă o metrică obligatorie lipsește,
    raportul rezultat este marcat incomplet (`incomplete`) și blochează promovarea (Req 29.4).
    Avertismentul că profitul nu este garantat este inclus întotdeauna (Req 21.7, 29.5).
    """
    return EvaluationReport(
        strategy_id=strategy_id,
        preregistration_id=preregistration_id,
        period=period,
        universe=universe,
        trade_count=trade_count,
        gross_result_eur=gross_result_eur,
        net_result_eur=net_result_eur,
        costs=costs,
        max_drawdown_eur=max_drawdown_eur,
        assumptions=assumptions,
        uncertainty=uncertainty,
        variants=variants,
    )
