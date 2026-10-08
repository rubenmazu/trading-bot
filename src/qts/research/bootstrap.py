"""Stationary bootstrap peste rezultatele nete per tranzacție (Bootstrap_Monte_Carlo_Analysis).

Req 19.3, 19.4, 19.7.

Rezultatele nete per tranzacție sunt corelate serial (de exemplu grupări de câștiguri sau de
pierderi). Un bootstrap i.i.d. clasic ar distruge această autocorelație și ar subestima riscul
cozilor. Folosim **stationary bootstrap** (Politis & Romano, 1994): fiecare reeșantionare este
construită din blocuri contigue de lungime aleatorie geometrică (medie `1/p`), concatenate până
la lungimea seriei originale, cu indexare circulară (wrap-around). Blocurile contigue păstrează
autocorelația locală, iar lungimea aleatorie face procesul staționar.

Garanții:

- minimum 10.000 de reeșantionări (Req 19.3); un număr mai mic este refuzat;
- sămânța RNG este înregistrată în rezultat, deci rularea este reproductibilă bit-cu-bit
  (aceeași sămânță + aceleași intrări → aceleași distribuții);
- se produc distribuțiile pentru rezultatul net, drawdown-ul maxim, cea mai lungă serie de
  pierderi și probabilitatea atingerii limitei totale de pierdere (implicit 10 EUR) (Req 19.4);
- raportarea parțială (Req 19.7): `run_stationary_bootstrap` acceptă un `report_every`; la fiecare
  interval emite un `PartialReport` cu distribuțiile parțiale disponibile, calculate numai pe
  reeșantionările efectuate până atunci, în timp ce procesul continuă.

Banii sunt `Decimal`. Calculul vectorizat folosește `numpy` pe valori `float` doar intern, pentru
viteză; rezultatele raportate (medii, cuantile, praguri) sunt convertite înapoi în `Decimal`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Annotated, Final

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from qts.core.models import Dec, Frozen

__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_TOTAL_LOSS_LIMIT_EUR",
    "MIN_RESAMPLES",
    "BootstrapConfig",
    "BootstrapError",
    "BootstrapResult",
    "Distribution",
    "PartialReport",
    "collect_bootstrap",
    "longest_losing_streak",
    "max_drawdown",
    "run_stationary_bootstrap",
]

MIN_RESAMPLES: Final = 10_000
DEFAULT_RESAMPLES: Final = 10_000
# Limita totală de pierdere din configurația de risc (Req 13.5). Parametrizabilă aici pentru ca
# analiza să poată folosi exact pragul configurat al rulării.
DEFAULT_TOTAL_LOSS_LIMIT_EUR: Final = Decimal("10")

# Cuantilele raportate pentru fiecare distribuție (cozi + mediană).
_QUANTILES: Final = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


class BootstrapError(Exception):
    """Configurație invalidă pentru analiza bootstrap (fail-closed)."""


class BootstrapConfig(Frozen):
    """Parametrii fixați ai analizei bootstrap (parte din pre-înregistrare în practică).

    `resamples >= 10.000` (Req 19.3) și `seed` sunt impuse și înregistrate pentru reproducere.
    `avg_block_length` este media lungimii blocurilor geometrice (în tranzacții); probabilitatea
    de reîncepere a unui bloc este `p = 1 / avg_block_length`.
    """

    resamples: Annotated[int, Field(ge=MIN_RESAMPLES)] = DEFAULT_RESAMPLES
    avg_block_length: Dec = Decimal("5")
    seed: Annotated[int, Field(ge=0)]
    total_loss_limit_eur: Dec = DEFAULT_TOTAL_LOSS_LIMIT_EUR

    @model_validator(mode="after")
    def _check(self) -> BootstrapConfig:
        if self.avg_block_length < 1:
            raise ValueError("avg_block_length trebuie să fie >= 1 tranzacție")
        if self.total_loss_limit_eur <= 0:
            raise ValueError("total_loss_limit_eur trebuie să fie > 0")
        return self

    @property
    def restart_prob(self) -> float:
        """Probabilitatea geometrică de a începe un bloc nou, `p = 1 / avg_block_length`."""
        return 1.0 / float(self.avg_block_length)


class Distribution(Frozen):
    """Rezumatul unei distribuții empirice (dintr-un vector de reeșantionări).

    Toate valorile monetare sunt `Decimal`. `quantiles` mapează fiecare nivel la valoarea empirică.
    """

    count: Annotated[int, Field(ge=1)]
    mean: Dec
    minimum: Dec
    maximum: Dec
    quantiles: dict[str, Dec]


class PartialReport(Frozen):
    """Instantaneu intermediar emis în timpul rulării (Req 19.7).

    Conține distribuțiile calculate numai pe primele `completed` reeșantionări. `final` este
    adevărat doar pentru ultimul raport.
    """

    completed: Annotated[int, Field(ge=1)]
    total: Annotated[int, Field(ge=1)]
    final: bool
    net_pnl: Distribution
    max_drawdown: Distribution
    longest_losing_streak: Distribution
    prob_hit_total_loss_limit: Dec


class BootstrapResult(Frozen):
    """Rezultatul final al analizei bootstrap, cu sămânța înregistrată (Req 19.3, 19.4)."""

    seed: int
    resamples: int
    avg_block_length: Dec
    total_loss_limit_eur: Dec
    net_pnl: Distribution
    max_drawdown: Distribution
    longest_losing_streak: Distribution
    prob_hit_total_loss_limit: Dec


# --------------------------------------------------------------------------- metrici per cale


def max_drawdown(net_results: NDArray[np.float64]) -> float:
    """Drawdown-ul maxim (pozitiv) al curbei de capital cumulate dintr-o secvență de rezultate.

    Capitalul pornește de la 0 și acumulează rezultatele nete per tranzacție. Drawdown-ul la pasul
    `t` este `max_peak[:t] - equity[t]`; întoarcem maximul. Un capital monoton crescător dă 0.
    """
    if net_results.size == 0:
        return 0.0
    equity = np.cumsum(net_results)
    running_peak = np.maximum.accumulate(equity)
    # Vârful include capitalul inițial (0), deci drawdown-ul nu este niciodată negativ.
    drawdowns = np.maximum(running_peak, 0.0) - equity
    return float(np.max(drawdowns))


def longest_losing_streak(net_results: NDArray[np.float64]) -> int:
    """Cea mai lungă serie de tranzacții consecutive strict pierzătoare (rezultat net < 0)."""
    best = 0
    current = 0
    for value in net_results:
        if value < 0.0:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def _summarize(values: NDArray[np.float64]) -> Distribution:
    """Rezumă un vector de rezultate per reeșantionare într-o `Distribution` (Decimal)."""
    quantile_levels = np.array(_QUANTILES, dtype=np.float64)
    quantile_values = np.quantile(values, quantile_levels)
    quantiles = {
        f"{level:.2f}": _to_dec(value)
        for level, value in zip(_QUANTILES, quantile_values, strict=True)
    }
    return Distribution(
        count=int(values.size),
        mean=_to_dec(float(np.mean(values))),
        minimum=_to_dec(float(np.min(values))),
        maximum=_to_dec(float(np.max(values))),
        quantiles=quantiles,
    )


def _to_dec(value: float) -> Decimal:
    """Convertește un float intern într-un Decimal, prin reprezentarea text (fără eroare binară)."""
    return Decimal(format(round(float(value), 8), "f"))


# --------------------------------------------------------------------------- bootstrap staționar


def _stationary_indices(
    n: int, length: int, restart_prob: float, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Generează `length` indici pentru o reeșantionare staționară peste o serie de `n` elemente.

    Pornim de la un index uniform aleator. La fiecare pas, cu probabilitatea `restart_prob`
    reîncepem de la un index uniform nou; altfel avansăm cu 1 (circular, modulo `n`). Blocurile
    contigue astfel formate au lungime geometrică (medie `1/restart_prob`) și păstrează
    autocorelația (Politis & Romano).
    """
    idx = np.empty(length, dtype=np.intp)
    restarts = rng.random(length) < restart_prob
    starts = rng.integers(0, n, size=length)
    pos = int(rng.integers(0, n))
    for t in range(length):
        pos = int(starts[t]) if t == 0 or restarts[t] else (pos + 1) % n
        idx[t] = pos
    return idx


def _validate_inputs(net_results: Sequence[Decimal], config: BootstrapConfig) -> None:
    if not net_results:
        raise BootstrapError("sunt necesare rezultate nete per tranzacție pentru bootstrap")
    if config.resamples < MIN_RESAMPLES:
        raise BootstrapError(
            f"bootstrap-ul necesită minimum {MIN_RESAMPLES} reeșantionări (Req 19.3)"
        )


def run_stationary_bootstrap(
    net_results: Sequence[Decimal],
    config: BootstrapConfig,
    *,
    report_every: int | None = None,
) -> Iterator[PartialReport]:
    """Rulează stationary bootstrap și emite rapoarte (parțiale apoi final) ca iterator (Req 19.7).

    `net_results` sunt rezultatele nete per tranzacție (Decimal, EUR). Pentru fiecare dintre cele
    `config.resamples` reeșantionări calculăm patru metrici: suma rezultatelor nete (PnL net),
    drawdown-ul maxim, cea mai lungă serie de pierderi și dacă pierderea cumulată minimă a atins
    limita totală de pierdere. La fiecare `report_every` reeșantionări (și la final) emite un
    `PartialReport` cu distribuțiile disponibile până în acel moment (Req 19.7).

    Dacă `report_every` este `None`, se emite un singur raport final. Pentru rezultatul agregat
    comod, folosiți `collect_bootstrap`.

    Determinism: `config.seed` inițializează `numpy.random.default_rng`; aceeași sămânță și aceleași
    intrări produc exact aceleași rapoarte.
    """
    _validate_inputs(net_results, config)
    if report_every is not None and report_every < 1:
        raise BootstrapError("report_every trebuie să fie >= 1")

    series = np.array([float(x) for x in net_results], dtype=np.float64)
    n = series.size
    total = config.resamples
    rng = np.random.default_rng(config.seed)
    limit = float(config.total_loss_limit_eur)

    net_pnl = np.empty(total, dtype=np.float64)
    mdd = np.empty(total, dtype=np.float64)
    streaks = np.empty(total, dtype=np.float64)
    hit_limit = np.empty(total, dtype=np.float64)

    for i in range(total):
        idx = _stationary_indices(n, n, config.restart_prob, rng)
        sample = series[idx]
        equity = np.cumsum(sample)
        net_pnl[i] = float(equity[-1])
        mdd[i] = max_drawdown(sample)
        streaks[i] = float(longest_losing_streak(sample))
        # Atingerea limitei totale: cea mai adâncă pierdere cumulată <= -limit (Req 13.5, 19.4).
        hit_limit[i] = 1.0 if float(np.min(equity)) <= -limit else 0.0

        completed = i + 1
        is_final = completed == total
        should_report = is_final or (report_every is not None and completed % report_every == 0)
        if should_report:
            yield _build_report(
                completed=completed,
                total=total,
                final=is_final,
                net_pnl=net_pnl[:completed],
                mdd=mdd[:completed],
                streaks=streaks[:completed],
                hit_limit=hit_limit[:completed],
            )


def _build_report(
    *,
    completed: int,
    total: int,
    final: bool,
    net_pnl: NDArray[np.float64],
    mdd: NDArray[np.float64],
    streaks: NDArray[np.float64],
    hit_limit: NDArray[np.float64],
) -> PartialReport:
    return PartialReport(
        completed=completed,
        total=total,
        final=final,
        net_pnl=_summarize(net_pnl),
        max_drawdown=_summarize(mdd),
        longest_losing_streak=_summarize(streaks),
        prob_hit_total_loss_limit=_to_dec(float(np.mean(hit_limit))),
    )


def collect_bootstrap(
    net_results: Sequence[Decimal], config: BootstrapConfig
) -> BootstrapResult:
    """Rulează analiza complet și întoarce `BootstrapResult` (fără raportare parțială)."""
    final: PartialReport | None = None
    for report in run_stationary_bootstrap(net_results, config):
        final = report
    if final is None:  # imposibil: se emite întotdeauna raportul final (fail-closed)
        raise BootstrapError("bootstrap-ul nu a produs niciun raport")
    return BootstrapResult(
        seed=config.seed,
        resamples=config.resamples,
        avg_block_length=config.avg_block_length,
        total_loss_limit_eur=config.total_loss_limit_eur,
        net_pnl=final.net_pnl,
        max_drawdown=final.max_drawdown,
        longest_losing_streak=final.longest_losing_streak,
        prob_hit_total_loss_limit=final.prob_hit_total_loss_limit,
    )
