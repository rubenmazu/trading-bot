"""Metrici operaționale și benchmarkul de latență p50/p95/p99 (Req 25.5, 27.1–27.5).

Modulul colectează, determinist, metricile cerute de observabilitate (Req 25.5):

- **latențe** pe etape (`ingestion`, `strategy`, `risk`, `order_management`), în milisecunde;
- **prospețimea datelor** per instrument (vârsta ultimei bare față de un timp de referință);
- **ordine pe stare** (histogramă peste `OrderState`);
- **execuții** (contor);
- **costuri** pe categorii (`CostBreakdown`, în EUR);
- **PnL net** (realizat + nerealizat, în EUR);
- **utilizarea limitelor de risc** (raport observat/limită pentru pierderea zilnică și totală).

Nimic nu depinde de ceasul de perete: latențele sunt furnizate ca durate deja măsurate (ms), iar
prospețimea se calculează față de un timp de referință injectat. Astfel, aceleași intrări produc
mereu aceleași metrici (consecvent cu `canonical_hash` și testele de determinism, Req 6.1, 17.5).

`PercentileReport` folosește metoda „nearest-rank” peste eșantioane sortate, deci percentilele sunt
reproductibile exact, fără interpolare și fără dependență de float-uri de ceas. Benchmarkul
`run_load_benchmark` rulează un `callable` deterministic (care întoarce durate în ms) peste
`Supported_Load` inițial (20 de instrumente × bare de 5 minute, conform design) și raportează
p50/p95/p99 pe fiecare etapă, plus fracțiunea de bare procesate sub pragul de o secundă (Req 27.2).

În plus, `within_freshness_budget` verifică Req 27.3: procesarea Strategy + Risk pentru o bară
trebuie finalizată înainte de expirarea pragului de prospețime al instrumentului. Comparația este
deterministă (durată măsurată în ms față de prag în ms) și nu atinge ceasul de perete.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final

from qts.core.clock import ensure_utc
from qts.core.models import CostBreakdown, OrderState
from qts.core.money import ZERO, dec

__all__ = [
    "COMPONENT",
    "COMPONENT_VERSION",
    "SUPPORTED_LOAD",
    "WITHIN_THRESHOLD_MS",
    "BenchmarkReport",
    "LatencyStage",
    "MetricsCollector",
    "MetricsSnapshot",
    "PercentileReport",
    "SupportedLoad",
    "percentiles",
    "within_freshness_budget",
]

COMPONENT: Final = "metrics"
COMPONENT_VERSION: Final = "1"

# Pragul Req 27.2: 95% dintre bare procesate în maximum o secundă (1000 ms) de la recepție.
WITHIN_THRESHOLD_MS: Final = 1000.0


class LatencyStage(StrEnum):
    """Etapele pentru care se raportează percentilele de latență (Req 27.5)."""

    INGESTION = "ingestion"
    STRATEGY = "strategy"
    RISK = "risk"
    ORDER_MANAGEMENT = "order_management"


class SupportedLoad:
    """`Supported_Load`: volumul validat de instrumente, frecvență și resurse (Req 27.1).

    Definit prin numărul de instrumente, intervalul barelor (minute) și o descriere a resurselor
    de calcul. Valoarea inițială din design este 20 de instrumente cu bare de 5 minute.
    """

    __slots__ = ("_bar_interval_min", "_compute", "_instruments")

    def __init__(self, *, instruments: int, bar_interval_min: int, compute: str) -> None:
        if instruments <= 0:
            raise ValueError("instruments trebuie să fie > 0")
        if not 5 <= bar_interval_min <= 60:
            raise ValueError("bar_interval_min trebuie să fie în [5, 60] (Req 4.5)")
        self._instruments = instruments
        self._bar_interval_min = bar_interval_min
        self._compute = compute

    @property
    def instruments(self) -> int:
        return self._instruments

    @property
    def bar_interval_min(self) -> int:
        return self._bar_interval_min

    @property
    def compute(self) -> str:
        return self._compute

    @property
    def bars_per_hour(self) -> int:
        """Numărul de bare pe oră peste toate instrumentele (frecvența de evenimente)."""
        return self._instruments * (60 // self._bar_interval_min)


# `Supported_Load` inițial: 20 de instrumente, bare de 5 minute (design, Req 27.1).
SUPPORTED_LOAD: Final = SupportedLoad(
    instruments=20, bar_interval_min=5, compute="o singură mașină, un singur proces"
)


def percentiles(samples: Sequence[float], ranks: Iterable[int]) -> dict[int, float]:
    """Percentile deterministe prin „nearest-rank” peste eșantioane sortate.

    Pentru fiecare `p` din `ranks` (întreg în [1, 100]) întoarce valoarea de la poziția
    `ceil(p/100 * n)` (1-indexată) din eșantioanele sortate crescător. Fără interpolare, deci
    rezultatul este exact reproductibil. Ridică `ValueError` pentru eșantion gol sau rang invalid.
    """
    ordered = sorted(samples)
    n = len(ordered)
    if n == 0:
        raise ValueError("percentilele cer cel puțin un eșantion")
    result: dict[int, float] = {}
    for p in ranks:
        if not 1 <= p <= 100:
            raise ValueError(f"rang percentil invalid: {p} (așteptat 1..100)")
        # nearest-rank: ceil(p/100 * n), 1-indexat, limitat la n.
        rank = -(-p * n // 100)  # ceil fără float
        index = min(max(rank, 1), n) - 1
        result[p] = ordered[index]
    return result


class PercentileReport:
    """Percentilele p50/p95/p99 pentru un singur set de eșantioane (ms)."""

    __slots__ = ("_count", "_p50", "_p95", "_p99")

    def __init__(self, samples: Sequence[float]) -> None:
        pct = percentiles(samples, (50, 95, 99))
        self._p50 = pct[50]
        self._p95 = pct[95]
        self._p99 = pct[99]
        self._count = len(samples)

    @property
    def p50(self) -> float:
        return self._p50

    @property
    def p95(self) -> float:
        return self._p95

    @property
    def p99(self) -> float:
        return self._p99

    @property
    def count(self) -> int:
        return self._count

    def as_dict(self) -> dict[str, float]:
        return {"p50": self._p50, "p95": self._p95, "p99": self._p99}


class MetricsSnapshot:
    """Instantaneu imuabil al metricilor operaționale la un moment dat (Req 25.5)."""

    __slots__ = (
        "_costs",
        "_executions",
        "_freshness_ms",
        "_latency",
        "_net_pnl_eur",
        "_orders_by_state",
        "_ts",
        "_utilization",
    )

    def __init__(
        self,
        *,
        ts: datetime,
        latency: Mapping[LatencyStage, PercentileReport],
        freshness_ms: Mapping[str, float],
        orders_by_state: Mapping[OrderState, int],
        executions: int,
        costs: CostBreakdown,
        net_pnl_eur: Decimal,
        utilization: Mapping[str, Decimal],
    ) -> None:
        self._ts = ts
        self._latency = dict(latency)
        self._freshness_ms = dict(freshness_ms)
        self._orders_by_state = dict(orders_by_state)
        self._executions = executions
        self._costs = costs
        self._net_pnl_eur = net_pnl_eur
        self._utilization = dict(utilization)

    @property
    def ts(self) -> datetime:
        return self._ts

    @property
    def latency(self) -> dict[LatencyStage, PercentileReport]:
        return dict(self._latency)

    @property
    def freshness_ms(self) -> dict[str, float]:
        return dict(self._freshness_ms)

    @property
    def orders_by_state(self) -> dict[OrderState, int]:
        return dict(self._orders_by_state)

    @property
    def executions(self) -> int:
        return self._executions

    @property
    def costs(self) -> CostBreakdown:
        return self._costs

    @property
    def net_pnl_eur(self) -> Decimal:
        return self._net_pnl_eur

    @property
    def utilization(self) -> dict[str, Decimal]:
        return dict(self._utilization)


class MetricsCollector:
    """Agregă metricile operaționale fără a atinge ceasul de perete (Req 25.5).

    Latențele se adaugă ca durate deja măsurate (ms). Prospețimea se înregistrează sub forma
    timpului ultimei bare per instrument și se evaluează față de un timp de referință injectat în
    `snapshot(now=...)`. Costurile se cumulează pe categorii, iar utilizarea limitelor este un
    raport observat/limită (clamped la [0, +inf), cu limită zero tratată ca utilizare zero).
    """

    def __init__(self) -> None:
        self._latency: dict[LatencyStage, list[float]] = {stage: [] for stage in LatencyStage}
        self._last_bar_ts: dict[str, datetime] = {}
        self._orders_by_state: dict[OrderState, int] = {}
        self._executions = 0
        self._costs = CostBreakdown()
        self._realized_pnl_eur: Decimal = ZERO
        self._unrealized_pnl_eur: Decimal = ZERO
        self._utilization: dict[str, Decimal] = {}

    # ------------------------------------------------------------------ latențe

    def record_latency(self, stage: LatencyStage, duration_ms: float) -> None:
        """Adaugă o latență măsurată (ms) pentru o etapă. Negativă => eroare de contract."""
        if duration_ms < 0:
            raise ValueError(f"latență negativă pentru {stage}: {duration_ms}")
        self._latency[stage].append(float(duration_ms))

    def latency_samples(self, stage: LatencyStage) -> tuple[float, ...]:
        return tuple(self._latency[stage])

    # ------------------------------------------------------------------ prospețime

    def record_bar(self, instrument: str, ts_close: datetime) -> None:
        """Înregistrează timpul de închidere al ultimei bare pentru un instrument."""
        self._last_bar_ts[instrument] = ensure_utc(ts_close)

    def _freshness_ms(self, now: datetime) -> dict[str, float]:
        ref = ensure_utc(now)
        return {
            instrument: (ref - ts).total_seconds() * 1000.0
            for instrument, ts in self._last_bar_ts.items()
        }

    # ------------------------------------------------------------------ ordine pe stare

    def record_order_state(self, state: OrderState, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count nu poate fi negativ")
        self._orders_by_state[state] = self._orders_by_state.get(state, 0) + count

    # ------------------------------------------------------------------ execuții

    def record_execution(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count nu poate fi negativ")
        self._executions += count

    # ------------------------------------------------------------------ costuri

    def record_costs(self, costs: CostBreakdown) -> None:
        self._costs = self._costs + costs

    # ------------------------------------------------------------------ PnL

    def set_pnl(self, *, realized_eur: Decimal, unrealized_eur: Decimal) -> None:
        self._realized_pnl_eur = dec(realized_eur)
        self._unrealized_pnl_eur = dec(unrealized_eur)

    @property
    def net_pnl_eur(self) -> Decimal:
        """PnL net = realizat + nerealizat (Req 25.5)."""
        return self._realized_pnl_eur + self._unrealized_pnl_eur

    # ------------------------------------------------------------------ utilizarea limitelor

    def record_limit_utilization(self, name: str, *, observed: Decimal, limit: Decimal) -> None:
        """Înregistrează raportul observat/limită pentru o limită de risc (Req 25.5).

        O limită <= 0 este tratată ca utilizare zero (fail-safe: nu împărțim la zero). Valorile
        observate negative sunt limitate la zero (pierderea deja este ≥ 0 prin construcție).
        """
        obs = dec(observed)
        lim = dec(limit)
        if lim <= 0:
            self._utilization[name] = ZERO
            return
        ratio = max(ZERO, obs) / lim
        self._utilization[name] = ratio

    def utilization(self) -> dict[str, Decimal]:
        return dict(self._utilization)

    # ------------------------------------------------------------------ instantaneu

    def snapshot(self, *, now: datetime) -> MetricsSnapshot:
        """Produce un `MetricsSnapshot` imuabil evaluat față de timpul `now` injectat."""
        latency = {
            stage: PercentileReport(samples) for stage, samples in self._latency.items() if samples
        }
        return MetricsSnapshot(
            ts=ensure_utc(now),
            latency=latency,
            freshness_ms=self._freshness_ms(now),
            orders_by_state=dict(self._orders_by_state),
            executions=self._executions,
            costs=self._costs,
            net_pnl_eur=self.net_pnl_eur,
            utilization=dict(self._utilization),
        )


class BenchmarkReport:
    """Rezultatul benchmarkului de latență pe `Supported_Load` (Req 27.2, 27.5)."""

    __slots__ = ("_bars", "_load", "_stages", "_within_fraction")

    def __init__(
        self,
        *,
        load: SupportedLoad,
        bars: int,
        stages: Mapping[LatencyStage, PercentileReport],
        within_fraction: float,
    ) -> None:
        self._load = load
        self._bars = bars
        self._stages = dict(stages)
        self._within_fraction = within_fraction

    @property
    def load(self) -> SupportedLoad:
        return self._load

    @property
    def bars(self) -> int:
        return self._bars

    @property
    def stages(self) -> dict[LatencyStage, PercentileReport]:
        return dict(self._stages)

    def stage(self, stage: LatencyStage) -> PercentileReport:
        return self._stages[stage]

    @property
    def within_fraction(self) -> float:
        """Fracțiunea de bare procesate (total pe etape) sub `WITHIN_THRESHOLD_MS` (Req 27.2)."""
        return self._within_fraction

    def meets_target(self, *, min_fraction: float = 0.95) -> bool:
        """True dacă cel puțin `min_fraction` din bare sunt sub prag (Req 27.2)."""
        return self._within_fraction >= min_fraction


def within_freshness_budget(
    *, processing_ms: float, freshness_threshold_ms: float
) -> bool:
    """True dacă procesarea unei bare se încheie înainte de pragul de prospețime (Req 27.3).

    `processing_ms` este latența totală Strategy + Risk pentru bară (ms); `freshness_threshold_ms`
    este pragul de prospețime al instrumentului (ms), care trebuie să fie strict pozitiv. Comparația
    este strictă: procesarea trebuie să se încheie *înainte* de expirarea pragului. Fail-closed: o
    latență negativă este o eroare de contract.
    """
    if freshness_threshold_ms <= 0:
        raise ValueError("freshness_threshold_ms trebuie să fie > 0")
    if processing_ms < 0:
        raise ValueError(f"latență negativă: {processing_ms}")
    return processing_ms < freshness_threshold_ms


# O etapă de procesare pentru o bară: întoarce latența (ms) pentru (index bară, instrument).
StageTimer = Callable[[int, str], float]


def run_load_benchmark(
    *,
    load: SupportedLoad,
    bars_per_instrument: int,
    timer: StageTimer,
    stages: Sequence[LatencyStage] = tuple(LatencyStage),
) -> BenchmarkReport:
    """Rulează un benchmark determinist peste `Supported_Load` și raportează p50/p95/p99.

    Pentru fiecare bară (`bars_per_instrument` × `load.instruments`) și fiecare etadă, `timer`
    întoarce latența simulată în ms. `timer` este injectat (determinist, fără ceas de perete), deci
    benchmarkul este reproductibil. Pragul Req 27.2 se evaluează pe latența totală per bară (suma
    etapelor): o bară este „în buget” dacă suma latențelor etapelor este sub `WITHIN_THRESHOLD_MS`.
    """
    if bars_per_instrument <= 0:
        raise ValueError("bars_per_instrument trebuie să fie > 0")
    if not stages:
        raise ValueError("este nevoie de cel puțin o etapă")
    instrument_names = [f"INS{i:03d}" for i in range(load.instruments)]
    per_stage: dict[LatencyStage, list[float]] = {stage: [] for stage in stages}
    total_bars = 0
    within = 0
    for bar_index in range(bars_per_instrument):
        for instrument in instrument_names:
            total_bars += 1
            bar_total = 0.0
            for stage in stages:
                latency_ms = timer(bar_index, instrument)
                if latency_ms < 0:
                    raise ValueError(f"latență negativă în benchmark: {latency_ms}")
                per_stage[stage].append(latency_ms)
                bar_total += latency_ms
            if bar_total < WITHIN_THRESHOLD_MS:
                within += 1
    reports = {stage: PercentileReport(samples) for stage, samples in per_stage.items()}
    within_fraction = within / total_bars if total_bars else 0.0
    return BenchmarkReport(
        load=load, bars=total_bars, stages=reports, within_fraction=within_fraction
    )
