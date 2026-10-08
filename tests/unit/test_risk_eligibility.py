"""Teste unitare pentru eligibilitatea instrumentelor și `Instrument_Universe` (task 8.3).

Scenariul de bază (ETF în EUR, fracționat, pas 0.001):
- cotație 9.99/10.01 → spread 20 bps, entry 10.01;
- stop minim = max(0.5% · 10.01, 1 tick, 1 · σ 0.01 · 10.01) = 0.1001 → stop 9.90;
- cost pe picior: jumătate de spread 0.01/unitate + taxă fixă 0.05 → dus-întors ≈ 0.10;
- ADV 1000 unități · 10 EUR = 10 000 EUR ≥ pragul de 5 000 EUR.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from qts.config.schema import RiskConfig
from qts.core.models import Instrument, Quote
from qts.core.money import ceil_to_step
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import (
    CompleteCostModel,
    CostContext,
    CostModelConfig,
    FxConfig,
    LatencyConfig,
    SlippageConfig,
    TaxConfig,
)
from qts.risk.eligibility import (
    CryptoUniverseConfig,
    EligibilityThresholds,
    EligibilityVerdict,
    IneligibleCode,
    InstrumentMarketStats,
    InstrumentUniverse,
    evaluate_instrument,
    evaluate_universe,
)

D = Decimal
T0 = datetime(2025, 1, 2, 9, 30, tzinfo=UTC)
C = IneligibleCode


def _inst(symbol: str = "XYZ", **kw: Any) -> Instrument:
    data: dict[str, Any] = {
        "symbol": symbol,
        "venue": "XETR",
        "asset_class": "etf",
        "currency": "EUR",
        "tick_size": "0.01",
        "qty_step": "0.001",
        "min_qty": "0.001",
        "calendar_id": "XETR",
        "fractional": True,
    }
    data.update(kw)
    return Instrument.model_validate(data)


def _cost_model(table: dict[str, Any] | None = None, **cfg: Any) -> CompleteCostModel:
    t: dict[str, Any] = {
        "broker": "sim",
        "version": "v1",
        "valid_from": datetime(2024, 1, 1, tzinfo=UTC),
        "currency": "EUR",
        "percent": "0",
        "exchange_fee_fixed": "0.05",
    }
    t.update(table or {})
    data: dict[str, Any] = {
        "version": "costs-test",
        "commissions": CommissionSchedule(tables=(CommissionTable.model_validate(t),)),
        "slippage": SlippageConfig(k=D(0)),
        "latency": LatencyConfig(latency_ms=0),
        "fx": FxConfig(conversion_spread=D(0)),
        "taxes": TaxConfig(approved=True, reference="test"),
    }
    data.update(cfg)
    return CompleteCostModel(CostModelConfig(**data))


def _stats(symbol: str = "XYZ", price: str = "10", **kw: Any) -> InstrumentMarketStats:
    p = D(price)
    ctx: dict[str, Any] = {"broker": "sim", "sigma_bar": D("0.01"), "adv": D(1000)}
    ctx.update(kw.pop("ctx", {}))
    data: dict[str, Any] = {
        "instrument": symbol,
        "ts": T0,
        "price": p,
        "quote": Quote(instrument=symbol, ts=T0, bid=p - D("0.01"), ask=p + D("0.01")),
        "cost_ctx": CostContext(**ctx),
    }
    data.update(kw)
    return InstrumentMarketStats.model_validate(data)


def _th(**kw: Any) -> EligibilityThresholds:
    data: dict[str, Any] = {
        "version": "elig-v1",
        "allowed_currencies": ("EUR", "USD"),
        "allowed_calendars": ("XETR", "XNYS", "FX24", "CRYPTO24"),
        "min_adv_eur": "5000",
        "max_spread_bps": "50",
    }
    data.update(kw)
    return EligibilityThresholds.model_validate(data)


def _crypto_th(**kw: Any) -> EligibilityThresholds:
    return _th(
        version="elig-crypto-v1",
        segment="crypto",
        allowed_asset_classes=("crypto",),
        allowed_crypto_symbols=("BTCEUR",),
        **kw,
    )


RISK = RiskConfig()


def _eval(
    inst: Instrument | None = None,
    stats: InstrumentMarketStats | str | None = "default",
    th: EligibilityThresholds | None = None,
    model: CompleteCostModel | None = None,
) -> EligibilityVerdict:
    inst = inst or _inst()
    s = _stats(inst.symbol) if stats == "default" else stats
    assert not isinstance(s, str)
    return evaluate_instrument(inst, s, th or _th(), model or _cost_model(), RISK)


# --------------------------------------------------------------------------- instrument


def test_liquid_fractional_etf_is_eligible_with_computed_values() -> None:
    v = _eval()
    assert v.eligible, v.reasons
    assert v.thresholds_version == "elig-v1"
    assert v.cost_model_version == "costs-test"
    assert v.computed["stop_price"] == D("9.90")
    assert v.computed["spread_bps"] == D(20)
    assert v.computed["adv_eur"] == D(10000)
    assert D("0.10") <= v.computed["roundtrip_cost_eur"] < D("0.11")
    assert v.computed["trade_risk_eur"] <= RISK.risk_per_trade_max_eur


def test_min_commission_above_quarter_eur_makes_instrument_ineligible() -> None:
    # 0.30 EUR minim per ordin → ≥ 0.60 EUR dus-întors, peste bugetul de 0.50 (design, 4.7).
    v = _eval(model=_cost_model({"minimum": "0.30", "exchange_fee_fixed": "0"}))
    assert not v.eligible
    assert v.codes == {C.COSTS_EXCEED_BUDGET}
    reason = v.reasons[0]
    assert reason.value is not None and reason.value >= D("0.60")
    assert reason.limit == D("0.50")


def test_price_risk_of_whole_share_exceeds_budget() -> None:
    # Acțiune nefracționată la 50 EUR: stopul minim de ~0.50 EUR plus costuri depășește bugetul.
    inst = _inst("BIG", asset_class="stock", qty_step="1", min_qty="1", fractional=False)
    v = _eval(inst, _stats("BIG", price="50"))
    assert v.codes == {C.TRADE_RISK_EXCEEDS_BUDGET}
    assert v.computed["min_trade_qty"] == D(1)


def test_min_notional_raises_min_trade_and_cash_cap() -> None:
    inst = _inst(min_notional="150")
    v = _eval(inst)
    assert C.MIN_ORDER_VALUE_TOO_HIGH in v.codes
    assert v.computed["min_trade_qty"] == ceil_to_step(D(150) / D("10.01"), D("0.001"))


def test_all_hard_exclusions_are_reported_together() -> None:
    inst = _inst(
        "CFD1",
        is_derivative=True,
        requires_leverage=True,
        requires_short=True,
        currency="GBP",
        calendar_id="UNKNOWN",
    )
    v = _eval(inst, _stats("CFD1", ctx={"fx_rate": D("1.15")}))
    assert {
        C.DERIVATIVE,
        C.REQUIRES_LEVERAGE,
        C.REQUIRES_SHORT,
        C.CURRENCY_NOT_ALLOWED,
        C.CALENDAR_NOT_ALLOWED,
    } <= v.codes


def test_illiquid_instrument_reports_volume_and_spread() -> None:
    stats = _stats(ctx={"adv": D(10)}, spread_bps=D(120))
    v = _eval(stats=stats)
    assert {C.VOLUME_BELOW_MIN, C.SPREAD_ABOVE_MAX} <= v.codes


def test_missing_data_fails_closed() -> None:
    assert _eval(stats=None).codes == {C.MARKET_STATS_MISSING}

    no_adv = _eval(stats=_stats(ctx={"adv": None}))
    assert C.VOLUME_UNKNOWN in no_adv.codes

    no_spread = _eval(stats=_stats(quote=None, spread_bps=None))
    assert C.SPREAD_UNKNOWN in no_spread.codes

    usd = _inst("USX", currency="USD", calendar_id="XNYS")
    v = _eval(usd, _stats("USX"))
    assert C.FX_RATE_MISSING in v.codes
    assert "trade_risk_eur" not in v.computed


def test_incomplete_cost_model_is_ineligible() -> None:
    v = _eval(model=_cost_model(taxes=None))
    assert v.codes == {C.COST_MODEL_INCOMPLETE}


def test_fx_requires_major_pair_and_usd_conversion() -> None:
    minor = _inst("USDTRY", asset_class="fx", currency="TRY", calendar_id="FX24")
    v = _eval(minor, _stats("USDTRY", ctx={"fx_rate": D("0.03")}))
    assert {C.FX_NOT_MAJOR, C.CURRENCY_NOT_ALLOWED} <= v.codes

    major = _inst("EURUSD", asset_class="fx", currency="USD", calendar_id="FX24")
    ok = _eval(major, _stats("EURUSD", price="1.10", ctx={"fx_rate": D("0.91")}))
    assert C.FX_NOT_MAJOR not in ok.codes


def test_fractional_price_and_asset_class_thresholds() -> None:
    whole = _inst("W", qty_step="1", min_qty="1", fractional=False)
    th = _th(
        require_fractional=True,
        min_price_eur="20",
        allowed_asset_classes=("stock",),
    )
    v = _eval(whole, _stats("W"), th=th)
    assert {C.FRACTIONAL_REQUIRED, C.PRICE_BELOW_MIN, C.ASSET_CLASS_NOT_ALLOWED} <= v.codes


def test_crypto_is_isolated_from_core_and_altcoins_excluded() -> None:
    btc = _inst("BTCEUR", asset_class="crypto", calendar_id="CRYPTO24")
    alt = _inst("DOGEEUR", asset_class="crypto", calendar_id="CRYPTO24")
    assert _eval(btc, _stats("BTCEUR")).codes == {C.CRYPTO_ISOLATED}

    crypto_th = _crypto_th()
    assert _eval(btc, _stats("BTCEUR"), th=crypto_th).eligible
    assert C.ALTCOIN_EXCLUDED in _eval(alt, _stats("DOGEEUR"), th=crypto_th).codes
    assert C.ASSET_CLASS_NOT_ALLOWED in _eval(th=crypto_th).codes


def test_thresholds_validation() -> None:
    with pytest.raises(ValidationError):
        _th(allowed_asset_classes=("etf", "crypto"))
    with pytest.raises(ValidationError):
        _th(allowed_crypto_symbols=("BTCEUR",))
    with pytest.raises(ValidationError):
        _th(segment="crypto")  # segmentul crypto permite numai clasa crypto
    with pytest.raises(ValidationError):
        _th(max_spread_bps="-1")
    with pytest.raises(ValidationError):
        CryptoUniverseConfig(config_id="c", thresholds=_th(), risk=RISK)


# --------------------------------------------------------------------------- univers


def _universe(
    crypto: CryptoUniverseConfig | None = None, th: EligibilityThresholds | None = None
) -> InstrumentUniverse:
    insts = [
        _inst("XYZ"),
        _inst("CFD1", is_derivative=True),
        _inst("BTCEUR", asset_class="crypto", calendar_id="CRYPTO24"),
        _inst("DOGEEUR", asset_class="crypto", calendar_id="CRYPTO24"),
    ]
    stats = {i.symbol: _stats(i.symbol) for i in insts}
    return evaluate_universe(insts, stats, th or _th(), _cost_model(), RISK, ts=T0, crypto=crypto)


def test_universe_without_crypto_keeps_crypto_visible_as_isolated() -> None:
    u = _universe()
    assert u.crypto is None
    assert u.core.eligible_symbols == ("XYZ",)
    by_sym = {v.symbol: v for v in u.core.verdicts}
    assert by_sym["BTCEUR"].codes == {C.CRYPTO_ISOLATED}
    assert by_sym["CFD1"].codes == {C.DERIVATIVE}
    assert u.version.startswith("universe-")
    assert u.core.version.startswith("universe-core-")


def test_universe_with_crypto_enabled_uses_separate_segment_and_budget() -> None:
    crypto_risk = RiskConfig(risk_per_trade_target_eur=D("0.25"), risk_per_trade_max_eur=D("0.30"))
    cfg = CryptoUniverseConfig(
        enabled=True, config_id="crypto-1", thresholds=_crypto_th(), risk=crypto_risk
    )
    u = _universe(cfg)
    assert u.crypto is not None
    assert {v.symbol for v in u.core.verdicts} == {"XYZ", "CFD1"}
    assert u.crypto.config_id == "crypto-1"
    assert u.crypto.risk_per_trade_max_eur == D("0.30")
    assert u.crypto.eligible_symbols == ("BTCEUR",)
    assert {v.segment for v in u.crypto.verdicts} == {"crypto"}


def test_universe_version_is_deterministic_and_tracks_thresholds() -> None:
    assert _universe().version == _universe().version
    assert _universe().version != _universe(th=_th(version="elig-v2")).version


def test_universe_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_universe([_inst(), _inst()], {}, _th(), _cost_model(), RISK, ts=T0)
