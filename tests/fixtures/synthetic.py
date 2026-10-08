"""Generator determinist de bare OHLCV sintetice pentru teste (Req 7.4, 7.6, 7.7).

Scenarii:
- `random_walk`: log-prețul urmează o plimbare aleatoare gaussiană, fără derivă;
- `trend`: plimbare aleatoare cu derivă constantă pe bară (`drift`);
- `mean_reverting`: proces Ornstein–Uhlenbeck discret pe log-preț, cu viteza `reversion`
  spre `mean_price`.

Golurile (`Gap`) sunt ortogonale scenariului: după `after_bar` bare emise lipsesc
`missing_bars` sloturi consecutive. Procesul de preț continuă în timpul golului, deci prima
bară de după gol se deschide la închiderea ultimului slot lipsă (ca un gol real de date).
`gaps_around_threshold` construiește o pereche de goluri, unul egal cu pragul (≤ prag, Req 7.7)
și unul imediat peste prag (Req 7.4).

Determinism: generatorul este `numpy.random.Generator(PCG64(seed))`, iar numărul și ordinea
extragerilor nu depind de scenariu. Calculul intern se face în float, apoi fiecare valoare este
convertită prin `repr(float)` → `Decimal` și rotunjită la `tick_size` cu `round_to_tick`.
Aceeași specificație produce aceleași bare și aceiași octeți CSV; sămânța este înregistrată
în `source_id` (`synthetic:<scenariu>:seed=<n>`).

Barele respectă invariantele din `qts.data.validate`: prețuri > 0,
`low ≤ min(open, close) ≤ max(open, close) ≤ high`, `volume ≥ 0`, timpi UTC aliniați la
interval și `ts_close - ts_open == interval_min`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from qts.core.models import Bar
from qts.core.money import round_to_tick
from qts.data.manifest import DatasetManifest, create_manifest

Scenario = Literal["random_walk", "trend", "mean_reverting"]
SCENARIOS: Final[tuple[Scenario, ...]] = ("random_walk", "trend", "mean_reverting")

DEFAULT_START: Final = datetime(2024, 1, 2, 8, 0, tzinfo=UTC)
CSV_COLUMNS: Final[tuple[str, ...]] = (
    "instrument",
    "ts_open",
    "ts_close",
    "interval_min",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_WICK_SCALE: Final = 0.5  # fitilul mediu, ca fracție din volatilitatea pe bară


@dataclass(frozen=True, slots=True)
class Gap:
    """Lipsesc `missing_bars` sloturi după primele `after_bar` bare emise."""

    after_bar: int
    missing_bars: int

    def __post_init__(self) -> None:
        if self.after_bar < 1:
            raise ValueError("after_bar trebuie să fie >= 1 (golul urmează unei bare)")
        if self.missing_bars < 1:
            raise ValueError("missing_bars trebuie să fie >= 1")

    def minutes(self, interval_min: int) -> int:
        """Durata golului: timpul dintre `ts_close` anterior și `ts_open` următor."""
        return self.missing_bars * interval_min


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    scenario: Scenario
    seed: int
    n_bars: int
    instrument: str = "XYZ"
    interval_min: int = 15
    start: datetime = DEFAULT_START
    start_price: Decimal = Decimal("100.00")
    tick_size: Decimal = Decimal("0.01")
    volatility: float = 0.002  # abaterea standard a log-randamentului pe bară
    drift: float = 0.0005  # derivă log pe bară, folosită numai de `trend`
    reversion: float = 0.05  # theta OU pe bară, folosită numai de `mean_reverting`
    mean_price: Decimal | None = None  # nivelul OU; implicit `start_price`
    volume_range: tuple[int, int] = (100, 10_000)  # [min, max)
    gaps: tuple[Gap, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenariu necunoscut: {self.scenario!r}")
        if self.seed < 0:
            raise ValueError("seed trebuie să fie >= 0")
        if self.n_bars < 1:
            raise ValueError("n_bars trebuie să fie >= 1")
        if not 5 <= self.interval_min <= 60:
            raise ValueError("interval_min trebuie să fie în [5, 60] (Req 4.5)")
        if self.start.tzinfo is None or self.start.utcoffset() != timedelta(0):
            raise ValueError("start trebuie să fie UTC")
        offset = self.start - _EPOCH
        if offset % timedelta(minutes=self.interval_min) != timedelta(0):
            raise ValueError("start trebuie să fie aliniat la interval_min")
        if self.start_price <= 0 or self.tick_size <= 0:
            raise ValueError("start_price și tick_size trebuie să fie > 0")
        if self.mean_price is not None and self.mean_price <= 0:
            raise ValueError("mean_price trebuie să fie > 0")
        if not (math.isfinite(self.volatility) and self.volatility >= 0):
            raise ValueError("volatility trebuie să fie finită și >= 0")
        if not math.isfinite(self.drift):
            raise ValueError("drift trebuie să fie finit")
        if not 0 < self.reversion <= 1:
            raise ValueError("reversion trebuie să fie în (0, 1]")
        low, high = self.volume_range
        if not 0 <= low < high:
            raise ValueError("volume_range trebuie să respecte 0 <= min < max")
        positions = [g.after_bar for g in self.gaps]
        if positions != sorted(set(positions)):
            raise ValueError("golurile trebuie să fie strict ordonate după after_bar")
        if positions and positions[-1] >= self.n_bars:
            raise ValueError("un gol trebuie urmat de cel puțin o bară")

    @property
    def source_id(self) -> str:
        return f"synthetic:{self.scenario}:seed={self.seed}"

    @property
    def n_slots(self) -> int:
        return self.n_bars + sum(g.missing_bars for g in self.gaps)


def gaps_around_threshold(
    threshold_minutes: int, interval_min: int, n_bars: int
) -> tuple[Gap, Gap]:
    """Un gol cu durata maximă ≤ prag (la n/3) și unul minim > prag (la 2n/3)."""
    if threshold_minutes < interval_min:
        raise ValueError("pragul trebuie să fie >= interval_min pentru un gol sub prag")
    if n_bars < 3:
        raise ValueError("n_bars trebuie să fie >= 3")
    below = threshold_minutes // interval_min
    return Gap(n_bars // 3, below), Gap(2 * n_bars // 3, below + 1)


# ----------------------------------------------------------------------------- generare


def _log_closes(spec: SyntheticSpec, shocks: NDArray[np.float64]) -> NDArray[np.float64]:
    x0 = math.log(float(spec.start_price))
    if spec.scenario == "random_walk":
        return x0 + np.cumsum(shocks)
    if spec.scenario == "trend":
        return x0 + np.cumsum(shocks + spec.drift)
    mu = math.log(float(spec.mean_price if spec.mean_price is not None else spec.start_price))
    out = np.empty_like(shocks)
    prev = x0
    for i, shock in enumerate(shocks.tolist()):
        prev = prev + spec.reversion * (mu - prev) + shock
        out[i] = prev
    return out


def _to_tick(value: float, tick: Decimal) -> Decimal:
    """float → Decimal prin `repr` (determinist), rotunjit la tick și cel puțin un tick."""
    return max(round_to_tick(Decimal(repr(value)), tick), tick)


def generate_bars(spec: SyntheticSpec) -> list[Bar]:
    """Barele emise (fără sloturile din goluri), în ordine temporală."""
    rng = np.random.Generator(np.random.PCG64(spec.seed))
    n = spec.n_slots
    # Ordinea extragerilor este fixă și independentă de scenariu.
    shocks = rng.standard_normal(n) * spec.volatility
    wicks = np.abs(rng.standard_normal((n, 2))) * (spec.volatility * _WICK_SCALE)
    volumes = rng.integers(spec.volume_range[0], spec.volume_range[1], size=n)

    closes_f = np.exp(_log_closes(spec, shocks)).tolist()
    up_f = np.exp(wicks[:, 0]).tolist()
    down_f = np.exp(-wicks[:, 1]).tolist()
    vol_i = volumes.tolist()

    tick = spec.tick_size
    closes = [_to_tick(c, tick) for c in closes_f]
    missing: set[int] = set()
    emitted = 0
    slot = 0
    gaps = iter(spec.gaps)
    next_gap = next(gaps, None)
    while slot < n:
        if next_gap is not None and emitted == next_gap.after_bar:
            missing.update(range(slot, slot + next_gap.missing_bars))
            slot += next_gap.missing_bars
            next_gap = next(gaps, None)
            continue
        emitted += 1
        slot += 1

    step = timedelta(minutes=spec.interval_min)
    first_open = _to_tick(float(spec.start_price), tick)
    bars: list[Bar] = []
    for i in range(n):
        if i in missing:
            continue
        o = first_open if i == 0 else closes[i - 1]
        c = closes[i]
        hi = max(_to_tick(float(max(o, c)) * up_f[i], tick), o, c)
        lo = min(_to_tick(float(min(o, c)) * down_f[i], tick), o, c)
        ts_open = spec.start + i * step
        bars.append(
            Bar(
                instrument=spec.instrument,
                ts_open=ts_open,
                ts_close=ts_open + step,
                interval_min=spec.interval_min,
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume=Decimal(int(vol_i[i])),
            )
        )
    return bars


def find_gaps(bars: Sequence[Bar]) -> list[tuple[str, datetime, datetime]]:
    """Golurile `(instrument, ts_close anterior, ts_open următor)`, per instrument."""
    last: dict[str, datetime] = {}
    found: list[tuple[str, datetime, datetime]] = []
    for bar in sorted(bars, key=lambda b: (b.instrument, b.ts_open)):
        prev = last.get(bar.instrument)
        if prev is not None and bar.ts_open > prev:
            found.append((bar.instrument, prev, bar.ts_open))
        last[bar.instrument] = bar.ts_close
    return found


# ----------------------------------------------------------------------------- CSV + manifest


def bars_to_csv_bytes(bars: Sequence[Bar]) -> bytes:
    """CSV UTF-8 cu terminator `\\n`, în formatul citit de `CsvSource`."""
    lines = [",".join(CSV_COLUMNS)]
    for b in bars:
        lines.append(
            ",".join(
                (
                    b.instrument,
                    b.ts_open.isoformat(),
                    b.ts_close.isoformat(),
                    str(b.interval_min),
                    str(b.open),
                    str(b.high),
                    str(b.low),
                    str(b.close),
                    str(b.volume),
                )
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_dataset(
    path: Path,
    bars: Sequence[Bar],
    *,
    source_id: str,
    calendar_id: str = "XETR",
    corporate_adjustments: str = "none",
    start: datetime | None = None,
    end: datetime | None = None,
) -> DatasetManifest:
    """Scrie CSV-ul și manifestul alăturat; intervalul implicit acoperă toate barele."""
    if not bars and (start is None or end is None):
        raise ValueError("pentru un set gol start și end sunt obligatorii")
    path.write_bytes(bars_to_csv_bytes(bars))
    return create_manifest(
        path,
        source_id=source_id,
        start=start if start is not None else min(b.ts_open for b in bars),
        end=end if end is not None else max(b.ts_close for b in bars),
        timezone="UTC",
        calendar_id=calendar_id,
        corporate_adjustments=corporate_adjustments,
    )


def write_synthetic_dataset(
    directory: Path, spec: SyntheticSpec, filename: str | None = None
) -> tuple[Path, DatasetManifest]:
    """Generează barele pentru `spec` și le scrie cu manifest în `directory`."""
    name = filename or f"{spec.instrument}_{spec.scenario}_s{spec.seed}.csv"
    path = directory / name
    manifest = write_dataset(path, generate_bars(spec), source_id=spec.source_id)
    return path, manifest
