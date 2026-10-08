"""Teste pentru generatorul de date sintetice din `tests/fixtures/synthetic.py` (Req 7.4, 7.7)."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from qts.core.models import Bar, canonical_bytes
from qts.data.csv_source import CsvSource
from qts.data.manifest import file_sha256
from qts.data.validate import filter_bars
from tests.fixtures.synthetic import (
    SCENARIOS,
    Gap,
    Scenario,
    SyntheticSpec,
    bars_to_csv_bytes,
    find_gaps,
    gaps_around_threshold,
    generate_bars,
    write_synthetic_dataset,
)


def _log_closes(bars: list[Bar]) -> NDArray[np.float64]:
    return np.log(np.array([float(b.close) for b in bars], dtype=np.float64))


# ----------------------------------------------------------------------------- determinism


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_same_seed_is_byte_identical(scenario: Scenario) -> None:
    spec = SyntheticSpec(scenario, seed=7, n_bars=300, gaps=(Gap(100, 3),))
    a, b = generate_bars(spec), generate_bars(spec)
    assert bars_to_csv_bytes(a) == bars_to_csv_bytes(b)
    assert [canonical_bytes(x) for x in a] == [canonical_bytes(x) for x in b]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_different_seed_differs(scenario: Scenario) -> None:
    a = generate_bars(SyntheticSpec(scenario, seed=1, n_bars=200))
    b = generate_bars(SyntheticSpec(scenario, seed=2, n_bars=200))
    assert bars_to_csv_bytes(a) != bars_to_csv_bytes(b)


def test_written_dataset_is_byte_identical_and_records_seed(tmp_path: Path) -> None:
    spec = SyntheticSpec("trend", seed=11, n_bars=50)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pa, ma = write_synthetic_dataset(tmp_path / "a", spec)
    pb, mb = write_synthetic_dataset(tmp_path / "b", spec)
    assert pa.read_bytes() == pb.read_bytes()
    assert ma.dataset_id == mb.dataset_id
    assert ma.source_id == "synthetic:trend:seed=11"


# ----------------------------------------------------------------------------- validitate


@given(
    scenario=st.sampled_from(SCENARIOS),
    seed=st.integers(0, 2**32 - 1),
    n_bars=st.integers(1, 120),
    interval=st.integers(5, 60),
    volatility=st.sampled_from([0.0, 0.001, 0.01, 0.2]),
)
def test_all_bars_pass_validator(
    scenario: Scenario, seed: int, n_bars: int, interval: int, volatility: float
) -> None:
    # Aliniere la epoca Unix, ca în `qts.data.bars` (contează pentru intervale ca 7 sau 13).
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    step = timedelta(minutes=interval)
    start = epoch + ((datetime(2024, 1, 2, tzinfo=UTC) - epoch) // step) * step
    gaps = (Gap(n_bars // 2, 2),) if n_bars >= 3 else ()
    spec = SyntheticSpec(
        scenario,
        seed=seed,
        n_bars=n_bars,
        interval_min=interval,
        start=start,
        volatility=volatility,
        gaps=gaps,
    )
    bars = generate_bars(spec)
    accepted, rejected = filter_bars(bars)
    assert rejected == []
    assert len(accepted) == n_bars
    for bar in bars:
        assert bar.ts_open.tzinfo == UTC
        assert (bar.ts_open - epoch) % step == timedelta(0)
        for price in (bar.open, bar.high, bar.low, bar.close):
            assert price == price.quantize(spec.tick_size)


def test_high_volatility_penny_prices_stay_positive() -> None:
    spec = SyntheticSpec("random_walk", seed=3, n_bars=500, start_price=Decimal("0.02"))
    bars = generate_bars(replace(spec, volatility=0.5))
    assert min(b.low for b in bars) >= spec.tick_size
    assert filter_bars(bars)[1] == []


def test_csv_round_trip_through_csv_source(tmp_path: Path) -> None:
    spec = SyntheticSpec("mean_reverting", seed=5, n_bars=80, gaps=(Gap(40, 4),))
    path, manifest = write_synthetic_dataset(tmp_path, spec)
    assert manifest.sha256 == file_sha256(path)
    source = CsvSource(path)
    events = source.load()
    assert source.rejections == []
    assert [e.payload for e in events] == generate_bars(spec)
    assert source.source_id == spec.source_id


def test_invalid_specs_are_refused() -> None:
    with pytest.raises(ValueError, match="interval_min"):
        SyntheticSpec("trend", seed=1, n_bars=10, interval_min=4)
    with pytest.raises(ValueError, match="aliniat"):
        SyntheticSpec("trend", seed=1, n_bars=10, start=datetime(2024, 1, 2, 8, 7, tzinfo=UTC))
    with pytest.raises(ValueError, match="urmat"):
        SyntheticSpec("trend", seed=1, n_bars=10, gaps=(Gap(10, 1),))


# ----------------------------------------------------------------------------- goluri


def test_gap_placement_and_duration() -> None:
    spec = SyntheticSpec("random_walk", seed=9, n_bars=30, gaps=(Gap(10, 2), Gap(20, 5)))
    bars = generate_bars(spec)
    assert len(bars) == 30
    found = find_gaps(bars)
    assert [(close, nxt) for _, close, nxt in found] == [
        (bars[9].ts_close, bars[10].ts_open),
        (bars[19].ts_close, bars[20].ts_open),
    ]
    assert [(nxt - close) for _, close, nxt in found] == [
        timedelta(minutes=30),
        timedelta(minutes=75),
    ]


def test_gaps_around_threshold() -> None:
    threshold = 60
    below, above = gaps_around_threshold(threshold, 15, n_bars=90)
    assert below.minutes(15) <= threshold < above.minutes(15)
    assert below.minutes(15) == threshold
    spec = SyntheticSpec("random_walk", seed=4, n_bars=90, gaps=(below, above))
    durations = [nxt - close for _, close, nxt in find_gaps(generate_bars(spec))]
    assert durations == [timedelta(minutes=60), timedelta(minutes=75)]


def test_gaps_do_not_change_surrounding_prices() -> None:
    plain = generate_bars(SyntheticSpec("trend", seed=2, n_bars=20))
    gapped = generate_bars(SyntheticSpec("trend", seed=2, n_bars=17, gaps=(Gap(10, 3),)))
    assert gapped[:10] == plain[:10]
    assert [b.close for b in gapped[10:]] == [b.close for b in plain[13:]]


# ----------------------------------------------------------------------------- caracter statistic


SEEDS = (101, 202, 303, 404, 505)


@pytest.mark.parametrize("seed", SEEDS)
def test_mean_reverting_stays_near_mean(seed: int) -> None:
    mr = _log_closes(generate_bars(SyntheticSpec("mean_reverting", seed=seed, n_bars=2000)))
    # OU staționar: σ/√(2θ) = 0.002/√0.1 ≈ 0.0063.
    assert float(np.std(mr)) < 0.02
    assert abs(float(np.mean(mr)) - math.log(100)) < 0.01
    # Regresia Δx pe (x - μ) are pantă ≈ -θ = -0.05.
    slope = float(np.polyfit(mr[:-1] - math.log(100), np.diff(mr), 1)[0])
    assert -0.12 < slope < -0.02


def test_mean_reverting_is_tighter_than_random_walk() -> None:
    def spread(scenario: Scenario) -> float:
        return float(
            np.mean(
                [
                    np.std(_log_closes(generate_bars(SyntheticSpec(scenario, s, 2000))))
                    for s in SEEDS
                ]
            )
        )

    assert spread("mean_reverting") < 0.5 * spread("random_walk")


@pytest.mark.parametrize("seed", SEEDS)
def test_trend_drifts_upward(seed: int) -> None:
    spec = SyntheticSpec("trend", seed=seed, n_bars=2000)
    x = _log_closes(generate_bars(spec))
    total = float(x[-1] - math.log(100))
    expected = spec.drift * spec.n_bars  # 1.0; zgomotul ≈ 0.002·√2000 ≈ 0.09
    assert 0.7 * expected < total < 1.3 * expected
    up = float(np.mean(np.diff(x) > 0))
    assert up > 0.55


def test_random_walk_has_no_meaningful_drift() -> None:
    totals = [
        float(_log_closes(generate_bars(SyntheticSpec("random_walk", seed=s, n_bars=2000)))[-1])
        - math.log(100)
        for s in SEEDS
    ]
    assert all(abs(t) < 0.4 for t in totals)  # ~4σ pentru σ ≈ 0.09
