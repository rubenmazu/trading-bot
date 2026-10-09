"""Eligibilitatea instrumentelor și raportul versionat `Instrument_Universe`.

Req 4.1-4.4, 4.6, 4.7.

Un instrument este evaluat față de un set de praguri versionate (`EligibilityThresholds`):

- clasa de active permisă (4.1): ETF/ETP pe indici, ETP pe aur/mărfuri, acțiuni, ETF-uri și FX
  major fără levier; crypto numai în segmentul separat, cu configurație, buget și raport proprii
  (4.2), iar în segmentul crypto numai simbolurile din lista permisă (altcoins excluse, 4.3);
- excluderi dure (4.3, 4.4): derivate (futures, CFD, opțiuni), levier, vânzare în lipsă;
- praguri de volum, spread, preț, valoare minimă, fracționare, monedă și program (4.6);
- bugetul per tranzacție (4.7): pentru cantitatea minimă tranzacționabilă și un stop minim
  rezonabil, costul dus-întors (din `CostModel.estimate`, prin `TradeRiskEstimator`) plus
  pierderea la stop trebuie să încapă în `risk_per_trade_max_eur`. Un comision minim per ordin
  de peste ~0,25 EUR face instrumentul automat neeligibil la limita de 0,50 EUR.

Evaluarea este fail-closed: statistici lipsă, curs FX lipsă sau `CostModelIncomplete` produc
neeligibilitate. Toate motivele sunt raportate (nu doar primul), cu coduri stabile.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import AfterValidator, Field, model_validator

from qts.config.schema import RiskConfig
from qts.core.models import (
    AssetClass,
    Dec,
    Frozen,
    Instrument,
    Quote,
    UtcDatetime,
    canonical_hash,
)
from qts.core.money import PRECISION, REPORTING_CURRENCY, ceil_to_step, floor_to_step
from qts.costs.errors import CostModelIncomplete
from qts.costs.model import CostContext, CostModel

from .context import MarketSnapshot
from .sizing import TradeRiskEstimator, min_tradable_qty

__all__ = [
    "DEFAULT_MAJOR_FX_PAIRS",
    "ELIGIBILITY_RULES_VERSION",
    "CryptoUniverseConfig",
    "EligibilityReason",
    "EligibilityThresholds",
    "EligibilityVerdict",
    "IneligibleCode",
    "InstrumentMarketStats",
    "InstrumentUniverse",
    "UniverseSegmentReport",
    "evaluate_instrument",
    "evaluate_universe",
]

ELIGIBILITY_RULES_VERSION: Final = "eligibility-rules-v1"

_CTX: Final = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)
_BPS: Final = Decimal(10000)

DEFAULT_MAJOR_FX_PAIRS: Final = (
    "AUDUSD",
    "EURCHF",
    "EURGBP",
    "EURJPY",
    "EURUSD",
    "GBPUSD",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
)


def _sorted_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    # Tupluri sortate (nu frozenset): serializarea canonică și hash-ul rămân deterministe.
    return tuple(sorted(set(values)))


SortedStrs = Annotated[tuple[str, ...], AfterValidator(_sorted_unique)]
SortedAssetClasses = Annotated[tuple[AssetClass, ...], AfterValidator(_sorted_unique)]


class IneligibleCode(StrEnum):
    ASSET_CLASS_NOT_ALLOWED = "ASSET_CLASS_NOT_ALLOWED"
    DERIVATIVE = "DERIVATIVE"
    REQUIRES_LEVERAGE = "REQUIRES_LEVERAGE"
    REQUIRES_SHORT = "REQUIRES_SHORT"
    CRYPTO_ISOLATED = "CRYPTO_ISOLATED"
    ALTCOIN_EXCLUDED = "ALTCOIN_EXCLUDED"
    FX_NOT_MAJOR = "FX_NOT_MAJOR"
    CURRENCY_NOT_ALLOWED = "CURRENCY_NOT_ALLOWED"
    FRACTIONAL_REQUIRED = "FRACTIONAL_REQUIRED"
    CALENDAR_NOT_ALLOWED = "CALENDAR_NOT_ALLOWED"
    MARKET_STATS_MISSING = "MARKET_STATS_MISSING"
    FX_RATE_MISSING = "FX_RATE_MISSING"
    VOLUME_UNKNOWN = "VOLUME_UNKNOWN"
    VOLUME_BELOW_MIN = "VOLUME_BELOW_MIN"
    SPREAD_UNKNOWN = "SPREAD_UNKNOWN"
    SPREAD_ABOVE_MAX = "SPREAD_ABOVE_MAX"
    PRICE_BELOW_MIN = "PRICE_BELOW_MIN"
    PRICE_ABOVE_MAX = "PRICE_ABOVE_MAX"
    STOP_NOT_FEASIBLE = "STOP_NOT_FEASIBLE"
    MIN_ORDER_VALUE_TOO_HIGH = "MIN_ORDER_VALUE_TOO_HIGH"
    COST_MODEL_INCOMPLETE = "COST_MODEL_INCOMPLETE"
    COSTS_EXCEED_BUDGET = "COSTS_EXCEED_BUDGET"
    TRADE_RISK_EXCEEDS_BUDGET = "TRADE_RISK_EXCEEDS_BUDGET"


# --------------------------------------------------------------------------- configurație


class EligibilityThresholds(Frozen):
    """Pragurile versionate ale unui segment de univers (Req 4.6).

    Sumele `*_eur` sunt în EUR; prețul se convertește cu cursul din `CostContext`.
    """

    version: str
    segment: Literal["core", "crypto"] = "core"
    allowed_asset_classes: SortedAssetClasses = ("commodity_etp", "etf", "fx", "index_etp", "stock")
    allowed_currencies: SortedStrs
    allowed_calendars: SortedStrs  # programele de tranzacționare cunoscute și acceptate
    major_fx_pairs: SortedStrs = DEFAULT_MAJOR_FX_PAIRS
    allowed_crypto_symbols: SortedStrs = ()  # numai segmentul crypto; restul sunt altcoins
    min_adv_eur: Dec  # volum mediu zilnic minim, în valoare EUR
    max_spread_bps: Dec
    min_price_eur: Dec | None = None
    max_price_eur: Dec | None = None
    require_fractional: bool = False
    # Valoarea maximă a celei mai mici tranzacții posibile; None = capitalul de referință.
    max_min_order_value_eur: Dec | None = None
    # Stopul minim rezonabil pentru verificarea bugetului (4.7): cea mai mare dintre distanțe.
    min_stop_distance_fraction: Dec = Decimal("0.005")
    min_stop_distance_ticks: Annotated[int, Field(ge=0)] = 1
    min_stop_distance_sigma: Dec = Decimal(1)  # multiplu de sigma_bar (ignorat dacă lipsește)

    @model_validator(mode="after")
    def _check(self) -> EligibilityThresholds:
        if not self.version:
            raise ValueError("pragurile trebuie să aibă o versiune")
        if not self.allowed_asset_classes:
            raise ValueError("allowed_asset_classes nu poate fi gol")
        if self.segment == "core":
            if "crypto" in self.allowed_asset_classes:
                raise ValueError("crypto se evaluează numai în segmentul crypto separat (4.2)")
            if self.allowed_crypto_symbols:
                raise ValueError("allowed_crypto_symbols este permis numai în segmentul crypto")
        elif self.allowed_asset_classes != ("crypto",):
            raise ValueError("segmentul crypto permite numai clasa crypto")
        values = [
            self.min_adv_eur,
            self.max_spread_bps,
            self.min_stop_distance_fraction,
            self.min_stop_distance_sigma,
        ]
        for opt in (self.min_price_eur, self.max_price_eur, self.max_min_order_value_eur):
            if opt is not None:
                values.append(opt)
        if any(v < 0 for v in values):
            raise ValueError("pragurile nu pot fi negative")
        if (
            self.min_price_eur is not None
            and self.max_price_eur is not None
            and self.min_price_eur > self.max_price_eur
        ):
            raise ValueError("min_price_eur trebuie să fie <= max_price_eur")
        if self.min_stop_distance_fraction >= 1:
            raise ValueError("min_stop_distance_fraction trebuie să fie < 1")
        return self


class CryptoUniverseConfig(Frozen):
    """Segmentul crypto izolat: configurație, buget de risc și raport separate (Req 4.2)."""

    enabled: bool = False
    config_id: str
    thresholds: EligibilityThresholds
    risk: RiskConfig

    @model_validator(mode="after")
    def _check(self) -> CryptoUniverseConfig:
        if self.thresholds.segment != "crypto":
            raise ValueError("CryptoUniverseConfig necesită praguri cu segment='crypto'")
        return self


# --------------------------------------------------------------------------- intrări


class InstrumentMarketStats(Frozen):
    """Statisticile de piață la momentul evaluării, pentru un instrument."""

    instrument: str
    ts: UtcDatetime  # momentul evaluării (determină tabelul de comisioane aplicabil)
    price: Dec  # preț de referință, în moneda instrumentului
    quote: Quote | None = None
    spread_bps: Dec | None = None  # spread tipic (ex. median); altfel din cotație
    cost_ctx: CostContext  # broker, curs FX, sigma_bar, ADV (unități)

    @model_validator(mode="after")
    def _check(self) -> InstrumentMarketStats:
        if self.price <= 0:
            raise ValueError("price trebuie să fie > 0")
        if self.spread_bps is not None and self.spread_bps < 0:
            raise ValueError("spread_bps nu poate fi negativ")
        if self.quote is not None and self.quote.instrument != self.instrument:
            raise ValueError("cotația nu aparține instrumentului")
        if self.cost_ctx.adv is not None and self.cost_ctx.adv < 0:
            raise ValueError("adv nu poate fi negativ")
        return self


# --------------------------------------------------------------------------- rezultate


class EligibilityReason(Frozen):
    code: IneligibleCode
    detail: str = ""
    value: Dec | None = None
    limit: Dec | None = None


class EligibilityVerdict(Frozen):
    symbol: str
    segment: Literal["core", "crypto"]
    eligible: bool
    reasons: tuple[EligibilityReason, ...] = ()
    thresholds_version: str
    rules_version: str = ELIGIBILITY_RULES_VERSION
    cost_model_version: str
    computed: dict[str, Dec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _shape(self) -> EligibilityVerdict:
        if self.eligible == bool(self.reasons):
            raise ValueError("eligibil ⇔ niciun motiv de neeligibilitate")
        return self

    @property
    def codes(self) -> frozenset[IneligibleCode]:
        return frozenset(r.code for r in self.reasons)


class UniverseSegmentReport(Frozen):
    """Raportul unui segment (core sau crypto), cu versiune derivată din conținut."""

    segment: Literal["core", "crypto"]
    config_id: str
    ts: UtcDatetime
    thresholds: EligibilityThresholds
    risk_per_trade_max_eur: Dec
    rules_version: str = ELIGIBILITY_RULES_VERSION
    cost_model_version: str
    verdicts: tuple[EligibilityVerdict, ...]
    version: str = ""

    @property
    def eligible_symbols(self) -> tuple[str, ...]:
        return tuple(v.symbol for v in self.verdicts if v.eligible)


class InstrumentUniverse(Frozen):
    """Universul versionat: segmentul principal și, separat, segmentul crypto (Req 4.2)."""

    core: UniverseSegmentReport
    crypto: UniverseSegmentReport | None = None
    version: str = ""


def _with_version[M: (UniverseSegmentReport, InstrumentUniverse)](model: M, prefix: str) -> M:
    digest = canonical_hash(model.model_copy(update={"version": ""}))
    return model.model_copy(update={"version": f"{prefix}-{digest[:16]}"})


# --------------------------------------------------------------------------- evaluare


class _Collector:
    def __init__(self) -> None:
        self.reasons: list[EligibilityReason] = []

    def add(
        self,
        code: IneligibleCode,
        detail: str = "",
        value: Decimal | None = None,
        limit: Decimal | None = None,
    ) -> None:
        self.reasons.append(EligibilityReason(code=code, detail=detail, value=value, limit=limit))


def _check_static(inst: Instrument, th: EligibilityThresholds, out: _Collector) -> None:
    """Clasă, excluderi dure, monedă, fracționare și program (4.1-4.4, 4.6)."""
    if inst.is_derivative:
        out.add(IneligibleCode.DERIVATIVE, "futures, CFD și opțiuni sunt excluse (4.3)")
    if inst.requires_leverage:
        out.add(IneligibleCode.REQUIRES_LEVERAGE, "expunerea cu levier este exclusă (4.4)")
    if inst.requires_short:
        out.add(IneligibleCode.REQUIRES_SHORT, "vânzarea în lipsă este exclusă (4.4)")

    if inst.asset_class == "crypto":
        if th.segment == "core":
            out.add(
                IneligibleCode.CRYPTO_ISOLATED,
                "crypto se evaluează numai în segmentul crypto separat (4.2)",
            )
        elif inst.symbol not in th.allowed_crypto_symbols:
            out.add(IneligibleCode.ALTCOIN_EXCLUDED, f"{inst.symbol} nu este în lista permisă")
    elif inst.asset_class not in th.allowed_asset_classes:
        out.add(
            IneligibleCode.ASSET_CLASS_NOT_ALLOWED,
            f"clasa {inst.asset_class} nu este permisă în segmentul {th.segment}",
        )
    if inst.asset_class == "fx" and inst.symbol not in th.major_fx_pairs:
        out.add(IneligibleCode.FX_NOT_MAJOR, f"{inst.symbol} nu este o pereche FX majoră")

    if inst.currency not in th.allowed_currencies:
        out.add(IneligibleCode.CURRENCY_NOT_ALLOWED, f"moneda {inst.currency} nu este permisă")
    if th.require_fractional and not inst.fractional:
        out.add(IneligibleCode.FRACTIONAL_REQUIRED, "pragurile cer tranzacționare fracționată")
    if inst.calendar_id not in th.allowed_calendars:
        out.add(
            IneligibleCode.CALENDAR_NOT_ALLOWED,
            f"programul {inst.calendar_id} nu este cunoscut sau acceptat",
        )


def _spread_bps(stats: InstrumentMarketStats) -> Decimal | None:
    if stats.spread_bps is not None:
        return stats.spread_bps
    q = stats.quote
    if q is None or q.bid <= 0 or q.ask < q.bid:
        return None
    return (q.ask - q.bid) / ((q.ask + q.bid) / 2) * _BPS


def _check_market(
    inst: Instrument,
    stats: InstrumentMarketStats,
    th: EligibilityThresholds,
    fx: Decimal | None,
    out: _Collector,
    computed: dict[str, Decimal],
) -> None:
    """Volum, spread și preț (4.3 ilichide, 4.6)."""
    spread = _spread_bps(stats)
    if spread is None:
        out.add(IneligibleCode.SPREAD_UNKNOWN, "fără spread tipic sau cotație bid/ask")
    else:
        computed["spread_bps"] = spread
        if spread > th.max_spread_bps:
            out.add(IneligibleCode.SPREAD_ABOVE_MAX, value=spread, limit=th.max_spread_bps)

    if fx is None:
        return  # verificările în EUR nu se pot face; FX_RATE_MISSING este deja raportat
    price_eur = stats.price * fx
    computed["price_eur"] = price_eur
    adv = stats.cost_ctx.adv
    if adv is None:
        out.add(IneligibleCode.VOLUME_UNKNOWN, "fără volum mediu zilnic (ADV)")
    else:
        adv_eur = adv * price_eur
        computed["adv_eur"] = adv_eur
        if adv_eur < th.min_adv_eur:
            out.add(IneligibleCode.VOLUME_BELOW_MIN, value=adv_eur, limit=th.min_adv_eur)
    if th.min_price_eur is not None and price_eur < th.min_price_eur:
        out.add(IneligibleCode.PRICE_BELOW_MIN, value=price_eur, limit=th.min_price_eur)
    if th.max_price_eur is not None and price_eur > th.max_price_eur:
        out.add(IneligibleCode.PRICE_ABOVE_MAX, value=price_eur, limit=th.max_price_eur)


def _check_budget(
    inst: Instrument,
    stats: InstrumentMarketStats,
    th: EligibilityThresholds,
    cost_model: CostModel,
    risk: RiskConfig,
    fx: Decimal,
    out: _Collector,
    computed: dict[str, Decimal],
) -> None:
    """Cea mai mică tranzacție posibilă, cu stop minim, trebuie să încapă în buget (4.7)."""
    # Prețul cel mai defavorabil dintre referință și ask, ca în Risk_Engine.
    entry = stats.price if stats.quote is None else max(stats.price, stats.quote.ask)
    distance = max(
        entry * th.min_stop_distance_fraction,
        inst.tick_size * th.min_stop_distance_ticks,
    )
    if stats.cost_ctx.sigma_bar is not None:
        distance = max(distance, entry * stats.cost_ctx.sigma_bar * th.min_stop_distance_sigma)
    stop = floor_to_step(entry - distance, inst.tick_size)
    computed["entry_price"] = entry
    computed["stop_price"] = stop
    if stop <= 0 or stop >= entry:
        out.add(
            IneligibleCode.STOP_NOT_FEASIBLE,
            "stopul minim nu se poate plasa sub prețul de intrare",
            value=stop,
        )
        return

    qty = min_tradable_qty(inst)
    if inst.min_notional > 0:
        qty = max(qty, ceil_to_step(inst.min_notional / entry, inst.qty_step))
    computed["min_trade_qty"] = qty

    snapshot = MarketSnapshot(
        instrument=inst, data_fresh=True, quote=stats.quote, cost_ctx=stats.cost_ctx
    )
    est = TradeRiskEstimator(
        cost_model,
        snapshot,
        entry_price=entry,
        stop_price=stop,
        fx_rate_to_eur=fx,
        ts_decision=stats.ts,
    )
    try:
        tr = est.at(qty)
    except CostModelIncomplete as exc:
        out.add(IneligibleCode.COST_MODEL_INCOMPLETE, f"{exc.component}: {exc.detail}")
        return

    budget = risk.risk_per_trade_max_eur
    computed.update(
        min_trade_notional_eur=tr.notional_eur,
        roundtrip_cost_eur=tr.cost_eur,
        price_risk_eur=tr.price_risk_eur,
        trade_risk_eur=tr.trade_risk_eur,
        cash_required_eur=tr.cash_required_eur,
        budget_eur=budget,
    )
    max_value = (
        th.max_min_order_value_eur
        if th.max_min_order_value_eur is not None
        else risk.reference_capital_eur
    )
    if tr.cash_required_eur > max_value:
        out.add(
            IneligibleCode.MIN_ORDER_VALUE_TOO_HIGH,
            "cea mai mică tranzacție posibilă depășește valoarea maximă admisă",
            value=tr.cash_required_eur,
            limit=max_value,
        )
    if tr.cost_eur >= budget:
        out.add(
            IneligibleCode.COSTS_EXCEED_BUDGET,
            "costurile dus-întors minime consumă tot bugetul per tranzacție",
            value=tr.cost_eur,
            limit=budget,
        )
    elif tr.trade_risk_eur > budget:
        out.add(
            IneligibleCode.TRADE_RISK_EXCEEDS_BUDGET,
            "costurile plus pierderea la stopul minim depășesc bugetul per tranzacție",
            value=tr.trade_risk_eur,
            limit=budget,
        )


def evaluate_instrument(
    instrument: Instrument,
    stats: InstrumentMarketStats | None,
    thresholds: EligibilityThresholds,
    cost_model: CostModel,
    risk: RiskConfig,
    *,
    reporting_currency: str = REPORTING_CURRENCY,
) -> EligibilityVerdict:
    """Verdictul de eligibilitate, cu toate motivele de neeligibilitate și valorile calculate.

    `reporting_currency` (implicit EUR) este moneda rulării: un instrument în această monedă nu
    necesită curs FX (`fx=1`), deci calea EUR rămâne neschimbată.
    """
    out = _Collector()
    computed: dict[str, Decimal] = {}
    with localcontext(_CTX):
        _check_static(instrument, thresholds, out)
        if stats is None:
            out.add(IneligibleCode.MARKET_STATS_MISSING, "fără statistici de piață")
        else:
            if stats.instrument != instrument.symbol:
                raise ValueError("statisticile nu aparțin instrumentului evaluat")
            fx: Decimal | None
            if instrument.currency == reporting_currency:
                fx = Decimal(1)
            else:
                fx = stats.cost_ctx.fx_rate
                if fx is None:
                    out.add(
                        IneligibleCode.FX_RATE_MISSING,
                        f"lipsește cursul {instrument.currency}→{reporting_currency}",
                    )
            _check_market(instrument, stats, thresholds, fx, out, computed)
            if fx is not None:
                computed["fx_rate_to_eur"] = fx
                _check_budget(instrument, stats, thresholds, cost_model, risk, fx, out, computed)
    return EligibilityVerdict(
        symbol=instrument.symbol,
        segment=thresholds.segment,
        eligible=not out.reasons,
        reasons=tuple(out.reasons),
        thresholds_version=thresholds.version,
        cost_model_version=cost_model.version,
        computed=computed,
    )


def _segment(
    segment: Literal["core", "crypto"],
    config_id: str,
    ts: datetime,
    instruments: Iterable[Instrument],
    stats: Mapping[str, InstrumentMarketStats],
    thresholds: EligibilityThresholds,
    cost_model: CostModel,
    risk: RiskConfig,
    *,
    disabled_crypto: Iterable[Instrument] = (),
) -> UniverseSegmentReport:
    verdicts = [
        evaluate_instrument(i, stats.get(i.symbol), thresholds, cost_model, risk)
        for i in instruments
    ]
    # Crypto dezactivat: instrumentele rămân vizibile în raportul principal, ca neeligibile.
    verdicts += [
        evaluate_instrument(i, stats.get(i.symbol), thresholds, cost_model, risk)
        for i in disabled_crypto
    ]
    report = UniverseSegmentReport(
        segment=segment,
        config_id=config_id,
        ts=ts,
        thresholds=thresholds,
        risk_per_trade_max_eur=risk.risk_per_trade_max_eur,
        cost_model_version=cost_model.version,
        verdicts=tuple(sorted(verdicts, key=lambda v: v.symbol)),
    )
    return _with_version(report, f"universe-{segment}")


def evaluate_universe(
    instruments: Iterable[Instrument],
    stats: Mapping[str, InstrumentMarketStats],
    thresholds: EligibilityThresholds,
    cost_model: CostModel,
    risk: RiskConfig,
    *,
    ts: datetime,
    config_id: str = "core",
    crypto: CryptoUniverseConfig | None = None,
) -> InstrumentUniverse:
    """Raportul versionat `Instrument_Universe`; crypto este izolat în segmentul propriu (4.2)."""
    if thresholds.segment != "core":
        raise ValueError("pragurile principale trebuie să aibă segment='core'")
    items = list(instruments)
    symbols = [i.symbol for i in items]
    if len(symbols) != len(set(symbols)):
        raise ValueError("simboluri duplicate în univers")
    non_crypto = [i for i in items if i.asset_class != "crypto"]
    crypto_items = [i for i in items if i.asset_class == "crypto"]
    crypto_on = crypto is not None and crypto.enabled

    core = _segment(
        "core",
        config_id,
        ts,
        non_crypto,
        stats,
        thresholds,
        cost_model,
        risk,
        disabled_crypto=() if crypto_on else crypto_items,
    )
    crypto_report: UniverseSegmentReport | None = None
    if crypto is not None and crypto.enabled:
        crypto_report = _segment(
            "crypto",
            crypto.config_id,
            ts,
            crypto_items,
            stats,
            crypto.thresholds,
            cost_model,
            crypto.risk,
        )
    return _with_version(InstrumentUniverse(core=core, crypto=crypto_report), "universe")
