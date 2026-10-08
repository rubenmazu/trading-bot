"""Clasificarea pe regimuri de piață și raportarea rezultatelor per regim (Req 20.1, 20.2, 20.5).

Un `Market_Regime` este definit prin două axe măsurabile, cu praguri fixate pe `Development_Set`
înaintea evaluării finale (Req 20.1):

- volatilitate realizată: deviația standard a randamentelor pe o fereastră glisantă, clasificată în
  tercile (JOASĂ / MEDIE / RIDICATĂ) cu praguri = cuantilele 1/3 și 2/3 ale volatilităților de pe
  Development_Set;
- trend: panta normalizată a prețului pe aceeași fereastră (regresie liniară a prețului în funcție
  de index, împărțită la prețul mediu), clasificată în DESCENDENT / LATERAL / ASCENDENT cu praguri
  simetrice `±trend_threshold`.

Pragurile sunt calculate o singură dată, cu `fit_regime_model`, pe randamentele/prețurile din
Development_Set, apoi aplicate neschimbat pe setul de validare. Fiecare bară de validare primește
o etichetă de regim, iar rezultatele nete per tranzacție sunt agregate pe regim (Req 20.2).

Promovarea este blocată dacă setul de validare conține zero sau un singur regim (Req 20.5):
`RegimeReport.distinct_regimes` numără regimurile efectiv prezente, iar `promotion_blocked` devine
adevărat când sunt mai puțin de două.

Statisticile numerice folosesc numpy (float) doar pentru clasificare; rezultatele nete per
tranzacție rămân `Decimal`, iar agregarea pe regim este exactă.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from enum import StrEnum

import numpy as np

from qts.core.models import Dec, Frozen
from qts.core.money import ZERO

__all__ = [
    "RegimeLabel",
    "RegimeModel",
    "RegimeReport",
    "RegimeStats",
    "TrendBand",
    "VolBand",
    "classify_regimes",
    "fit_regime_model",
    "realized_volatility",
    "regime_report",
    "trend_slope_normalized",
]


class VolBand(StrEnum):
    LOW = "vol_low"
    MID = "vol_mid"
    HIGH = "vol_high"


class TrendBand(StrEnum):
    DOWN = "trend_down"
    SIDEWAYS = "trend_sideways"
    UP = "trend_up"


class RegimeLabel(Frozen):
    """Eticheta de regim: combinația benzii de volatilitate cu banda de trend."""

    vol: VolBand
    trend: TrendBand

    @property
    def name(self) -> str:
        return f"{self.vol.value}+{self.trend.value}"


class RegimeModel(Frozen):
    """Praguri de regim fixate pe Development_Set, aplicate neschimbat la validare (Req 20.1)."""

    window: int
    vol_low_threshold: Dec
    vol_high_threshold: Dec
    trend_threshold: Dec

    def classify(self, vol: Decimal, trend: Decimal) -> RegimeLabel:
        if vol < self.vol_low_threshold:
            vol_band = VolBand.LOW
        elif vol < self.vol_high_threshold:
            vol_band = VolBand.MID
        else:
            vol_band = VolBand.HIGH
        if trend < -self.trend_threshold:
            trend_band = TrendBand.DOWN
        elif trend <= self.trend_threshold:
            trend_band = TrendBand.SIDEWAYS
        else:
            trend_band = TrendBand.UP
        return RegimeLabel(vol=vol_band, trend=trend_band)


class RegimeStats(Frozen):
    """Rezultatul net agregat al tranzacțiilor dintr-un regim (Req 20.2)."""

    regime: str
    trade_count: int
    net_result_eur: Dec


class RegimeReport(Frozen):
    """Rezultate nete raportate separat pe regim, plus poarta de acoperire (Req 20.2, 20.5)."""

    per_regime: tuple[RegimeStats, ...]

    @property
    def distinct_regimes(self) -> int:
        return len(self.per_regime)

    @property
    def promotion_blocked(self) -> bool:
        """Promovarea este blocată când sunt prezente < 2 regimuri (Req 20.5)."""
        return self.distinct_regimes < 2


def _to_float_array(values: Sequence[Decimal]) -> np.ndarray:
    return np.array([float(v) for v in values], dtype=np.float64)


def realized_volatility(prices: Sequence[Decimal]) -> Decimal:
    """Deviația standard a randamentelor simple ale `prices` (populație, ddof=0)."""
    if len(prices) < 2:
        return ZERO
    arr = _to_float_array(prices)
    if np.any(arr[:-1] == 0.0):
        raise ValueError("preț zero în calculul volatilității realizate")
    returns = arr[1:] / arr[:-1] - 1.0
    return Decimal(str(float(np.std(returns, ddof=0))))


def trend_slope_normalized(prices: Sequence[Decimal]) -> Decimal:
    """Panta regresiei liniare preț~index, normalizată la prețul mediu (adimensională)."""
    if len(prices) < 2:
        return ZERO
    arr = _to_float_array(prices)
    mean_price = float(np.mean(arr))
    if mean_price == 0.0:
        raise ValueError("preț mediu zero în calculul pantei de trend")
    index = np.arange(len(arr), dtype=np.float64)
    slope = float(np.polyfit(index, arr, 1)[0])
    return Decimal(str(slope / mean_price))


def _rolling_features(
    prices: Sequence[Decimal], window: int
) -> tuple[list[Decimal], list[Decimal]]:
    """Volatilitatea realizată și panta normalizată pe ferestre glisante de lungime `window`."""
    if window < 2:
        raise ValueError("fereastra de regim trebuie să fie >= 2")
    vols: list[Decimal] = []
    trends: list[Decimal] = []
    for end in range(window, len(prices) + 1):
        chunk = prices[end - window : end]
        vols.append(realized_volatility(chunk))
        trends.append(trend_slope_normalized(chunk))
    return vols, trends


def fit_regime_model(
    development_prices: Sequence[Decimal],
    *,
    window: int,
    trend_threshold: Decimal,
) -> RegimeModel:
    """Fixează pragurile de regim pe Development_Set (Req 20.1).

    Pragurile de volatilitate sunt cuantilele 1/3 și 2/3 ale volatilităților glisante observate pe
    Development_Set (tercile). Pragul de trend este simetric și furnizat explicit, fiind o decizie
    preînregistrată. Modelul rezultat este aplicat neschimbat la validare.
    """
    if trend_threshold < 0:
        raise ValueError("trend_threshold nu poate fi negativ")
    vols, _ = _rolling_features(development_prices, window)
    if not vols:
        raise ValueError("Development_Set este prea scurt pentru fereastra cerută")
    vol_arr = _to_float_array(vols)
    low = float(np.quantile(vol_arr, 1.0 / 3.0))
    high = float(np.quantile(vol_arr, 2.0 / 3.0))
    return RegimeModel(
        window=window,
        vol_low_threshold=Decimal(str(low)),
        vol_high_threshold=Decimal(str(high)),
        trend_threshold=trend_threshold,
    )


def classify_regimes(
    model: RegimeModel, prices: Sequence[Decimal]
) -> tuple[RegimeLabel | None, ...]:
    """Eticheta de regim pentru fiecare bară de validare.

    Primele `window - 1` bare nu au o fereastră completă și primesc `None`; de la a `window`-a bară
    înainte, eticheta corespunde ferestrei care se termină la acea bară.
    """
    vols, trends = _rolling_features(prices, model.window)
    labels: list[RegimeLabel | None] = [None] * min(model.window - 1, len(prices))
    for vol, trend in zip(vols, trends, strict=True):
        labels.append(model.classify(vol, trend))
    return tuple(labels[: len(prices)])


def regime_report(
    labels: Sequence[RegimeLabel | None],
    net_results_eur: Sequence[Decimal],
) -> RegimeReport:
    """Agregă rezultatele nete per regim, raportate separat (Req 20.2).

    `labels[i]` este regimul tranzacției `i`; etichetele `None` (fereastră incompletă) se ignoră.
    Regimurile sunt sortate după nume pentru raportare deterministă.
    """
    if len(labels) != len(net_results_eur):
        raise ValueError("labels și net_results_eur trebuie să aibă aceeași lungime")
    totals: dict[str, Decimal] = {}
    counts: dict[str, int] = {}
    for label, net in zip(labels, net_results_eur, strict=True):
        if label is None:
            continue
        name = label.name
        totals[name] = totals.get(name, ZERO) + net
        counts[name] = counts.get(name, 0) + 1
    per_regime = tuple(
        RegimeStats(regime=name, trade_count=counts[name], net_result_eur=totals[name])
        for name in sorted(totals)
    )
    return RegimeReport(per_regime=per_regime)
