"""Corecția pentru testarea multiplă: Deflated Sharpe Ratio și Holm–Bonferroni (Req 21.2, 21.5).

Când sunt comparate mai multe variante de strategie, un rezultat „bun" poate apărea din întâmplare.
Metoda preînregistrată (Req 21.2, 21.8) corectează acest risc; acest modul implementează cele două
metode din design:

- Deflated Sharpe Ratio (DSR, Bailey & López de Prado): pornind de la Sharpe-ul observat al
  variantei alese, de la numărul de variante încercate și de la variația Sharpe-urilor între
  variante, calculează probabilitatea ca Sharpe-ul adevărat să fie > 0 după ce se ține cont de
  numărul de încercări, de lungimea seriei și de asimetria/curtoza randamentelor. DSR este o
  probabilitate în (0, 1); strategia trece pragul dacă `DSR >= 1 - alpha` (Req 21.5).

- Holm–Bonferroni: controlează rata erorii de tip I pe setul de p-valori bootstrap ale variantelor,
  mai puțin conservator decât Bonferroni simplu. P-valorile sunt furnizate de apelant (de exemplu
  din bootstrap), nu calculate aici, pentru a nu depinde de implementarea bootstrap.

Metoda efectiv aplicată este citită din `PreRegistration.correction_method`; o metodă nepreînreg-
istrată este deja blocată de `require_preregistration` (Req 21.8). Statisticile folosesc numpy și
funcțiile normale implementate local (fără dependențe suplimentare).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal

import numpy as np

from qts.core.models import Dec, Frozen

from .preregistration import CorrectionMethod, PreRegistration

__all__ = [
    "DEFAULT_ALPHA",
    "DeflatedSharpeResult",
    "HolmResult",
    "HolmTest",
    "MultipleTestingResult",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "holm_bonferroni",
    "multiple_testing_correction",
    "normal_cdf",
    "normal_ppf",
    "sharpe_ratio",
]

DEFAULT_ALPHA = Decimal("0.05")
_EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------- funcții normale


def normal_cdf(x: float) -> float:
    """Funcția de repartiție a normalei standard, prin `math.erf`."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inversa CDF-ului normal standard (algoritmul Acklam, eroare < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("normal_ppf cere p în (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


# --------------------------------------------------------------------------- Sharpe / DSR


def sharpe_ratio(returns: Sequence[Decimal]) -> float:
    """Sharpe-ul (neanualizat) al unei serii de randamente per tranzacție."""
    if len(returns) < 2:
        raise ValueError("Sharpe cere cel puțin două observații")
    arr = np.array([float(r) for r in returns], dtype=np.float64)
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        raise ValueError("deviația standard zero: Sharpe nedefinit")
    return float(np.mean(arr)) / std


def expected_max_sharpe(variance_across_trials: float, trials: int) -> float:
    """Sharpe-ul maxim așteptat din `trials` încercări independente (benchmark DSR).

    `SR0 = sqrt(V) * [(1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e))]`, unde V este variația Sharpe-urilor
    între variante, N numărul de variante, γ constanta Euler–Mascheroni.
    """
    if trials < 1:
        raise ValueError("trials trebuie să fie >= 1")
    if variance_across_trials < 0:
        raise ValueError("variance_across_trials nu poate fi negativ")
    if trials == 1:
        return 0.0
    sqrt_v = math.sqrt(variance_across_trials)
    gamma = _EULER_MASCHERONI
    n = float(trials)
    term1 = (1.0 - gamma) * normal_ppf(1.0 - 1.0 / n)
    term2 = gamma * normal_ppf(1.0 - 1.0 / (n * math.e))
    return sqrt_v * (term1 + term2)


class DeflatedSharpeResult(Frozen):
    """Rezultatul Deflated Sharpe Ratio pentru varianta aleasă (Req 21.2, 21.5)."""

    observed_sharpe: Dec
    benchmark_sharpe: Dec
    dsr: Dec
    threshold: Dec
    passed: bool


def deflated_sharpe_ratio(
    chosen_returns: Sequence[Decimal],
    all_variant_sharpes: Sequence[Decimal],
    *,
    alpha: Decimal = DEFAULT_ALPHA,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio al variantei alese față de setul de variante încercate (Req 21.2).

    `chosen_returns` sunt randamentele per tranzacție ale variantei alese; `all_variant_sharpes`
    sunt Sharpe-urile tuturor variantelor evaluate (inclusiv cea aleasă), din care se estimează
    variația între încercări. Strategia trece dacă `DSR >= 1 - alpha` (Req 21.5).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha trebuie să fie în (0, 1)")
    trials = len(all_variant_sharpes)
    if trials < 1:
        raise ValueError("este nevoie de cel puțin o variantă")
    arr = np.array([float(r) for r in chosen_returns], dtype=np.float64)
    n_obs = arr.size
    if n_obs < 2:
        raise ValueError("DSR cere cel puțin două observații")
    observed = sharpe_ratio(chosen_returns)
    std = float(np.std(arr, ddof=1))
    mean = float(np.mean(arr))
    centered = arr - mean
    skew = float(np.mean(centered**3)) / std**3
    kurt = float(np.mean(centered**4)) / std**4  # curtoza totală (3 pentru normal)
    sharpes = np.array([float(s) for s in all_variant_sharpes], dtype=np.float64)
    variance_trials = float(np.var(sharpes, ddof=1)) if trials > 1 else 0.0
    benchmark = expected_max_sharpe(variance_trials, trials)
    denom = math.sqrt(max(1.0 - skew * observed + (kurt - 1.0) / 4.0 * observed**2, 1e-12))
    z = (observed - benchmark) * math.sqrt(n_obs - 1) / denom
    dsr = normal_cdf(z)
    threshold = Decimal(1) - alpha
    dsr_dec = Decimal(str(dsr))
    return DeflatedSharpeResult(
        observed_sharpe=Decimal(str(observed)),
        benchmark_sharpe=Decimal(str(benchmark)),
        dsr=dsr_dec,
        threshold=threshold,
        passed=dsr_dec >= threshold,
    )


# --------------------------------------------------------------------------- Holm–Bonferroni


class HolmTest(Frozen):
    """Rezultatul per variantă al corecției Holm–Bonferroni."""

    label: str
    p_value: Dec
    adjusted_threshold: Dec
    rejected_null: bool  # True = variantă semnificativă după corecție


class HolmResult(Frozen):
    """Agregatul Holm–Bonferroni peste toate variantele (Req 21.2, 21.5)."""

    alpha: Dec
    tests: tuple[HolmTest, ...]

    @property
    def any_significant(self) -> bool:
        return any(t.rejected_null for t in self.tests)


def holm_bonferroni(
    p_values: dict[str, Decimal], *, alpha: Decimal = DEFAULT_ALPHA
) -> HolmResult:
    """Procedura step-down Holm–Bonferroni pe p-valorile variantelor (Req 21.2).

    P-valorile (de ex. din bootstrap) sunt sortate crescător; a `i`-a (de la 1) este comparată cu
    `alpha / (m - i + 1)`. La prima p-valoare care depășește pragul, aceasta și toate cele
    ulterioare nu sunt respinse. Rezultatele sunt raportate cu eticheta originală, determinist.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha trebuie să fie în (0, 1)")
    if not p_values:
        raise ValueError("Holm–Bonferroni cere cel puțin o p-valoare")
    for label, p in p_values.items():
        if not 0 <= p <= 1:
            raise ValueError(f"p-valoare în afara [0, 1] pentru {label}: {p}")
    m = len(p_values)
    ordered = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    tests: dict[str, HolmTest] = {}
    still_rejecting = True
    for rank, (label, p) in enumerate(ordered):
        adjusted = alpha / Decimal(m - rank)
        if still_rejecting and p <= adjusted:
            rejected = True
        else:
            still_rejecting = False
            rejected = False
        tests[label] = HolmTest(
            label=label,
            p_value=p,
            adjusted_threshold=adjusted,
            rejected_null=rejected,
        )
    ordered_tests = tuple(tests[label] for label in sorted(p_values))
    return HolmResult(alpha=alpha, tests=ordered_tests)


# --------------------------------------------------------------------------- orchestrare


class MultipleTestingResult(Frozen):
    """Rezultatul metodei de corecție preînregistrate (Req 21.2, 21.5)."""

    method: CorrectionMethod
    chosen_label: str
    deflated_sharpe: DeflatedSharpeResult | None = None
    holm: HolmResult | None = None

    @property
    def passed(self) -> bool:
        """Strategia aleasă trece corecția preînregistrată.

        - DSR: trece dacă `DeflatedSharpeResult.passed`.
        - Holm: trece dacă varianta aleasă (`chosen_label`) este respinsă (semnificativă).
        - Combinat: ambele condiții.
        """
        dsr_ok = self.deflated_sharpe.passed if self.deflated_sharpe is not None else True
        holm_ok = self._chosen_rejected() if self.holm is not None else True
        return dsr_ok and holm_ok

    def _chosen_rejected(self) -> bool:
        if self.holm is None:
            return True
        return any(t.rejected_null and t.label == self.chosen_label for t in self.holm.tests)


def multiple_testing_correction(
    pre: PreRegistration,
    *,
    chosen_label: str,
    chosen_returns: Sequence[Decimal],
    all_variant_sharpes: Sequence[Decimal],
    bootstrap_p_values: dict[str, Decimal],
    alpha: Decimal = DEFAULT_ALPHA,
) -> MultipleTestingResult:
    """Aplică metoda de corecție din pre-înregistrare (Req 21.2, 21.5, 21.8).

    `chosen_returns` și `all_variant_sharpes` alimentează DSR; `bootstrap_p_values` (furnizate de
    apelant, de ex. din bootstrap) alimentează Holm–Bonferroni. Metoda `DEFLATED_SHARPE_HOLM`
    aplică ambele, iar strategia trece doar dacă trec amândouă.
    """
    method = pre.correction_method
    if chosen_label not in bootstrap_p_values and method in (
        CorrectionMethod.HOLM_BONFERRONI,
        CorrectionMethod.DEFLATED_SHARPE_HOLM,
    ):
        raise ValueError(f"lipsește p-valoarea bootstrap pentru varianta aleasă {chosen_label!r}")

    dsr: DeflatedSharpeResult | None = None
    holm: HolmResult | None = None
    if method in (CorrectionMethod.DEFLATED_SHARPE, CorrectionMethod.DEFLATED_SHARPE_HOLM):
        dsr = deflated_sharpe_ratio(chosen_returns, all_variant_sharpes, alpha=alpha)
    if method in (CorrectionMethod.HOLM_BONFERRONI, CorrectionMethod.DEFLATED_SHARPE_HOLM):
        holm = holm_bonferroni(bootstrap_p_values, alpha=alpha)

    return MultipleTestingResult(
        method=method,
        chosen_label=chosen_label,
        deflated_sharpe=dsr,
        holm=holm,
    )
