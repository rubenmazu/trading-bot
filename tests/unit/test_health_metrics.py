"""Teste unitare și de proprietate pentru `health/metrics.py` (Req 25.5, 27.1–27.5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qts.core.models import CostBreakdown, OrderState
from qts.health.metrics import (
    SUPPORTED_LOAD,
    LatencyStage,
    MetricsCollector,
    PercentileReport,
    SupportedLoad,
    percentiles,
    run_load_benchmark,
    within_freshness_budget,
)

T0 = datetime(2026, 4, 1, 12, tzinfo=UTC)
D = Decimal


# --------------------------------------------------------------------------- percentile


def test_percentile_nearest_rank_known_values() -> None:
    samples = [float(x) for x in range(1, 101)]  # 1..100
    pct = percentiles(samples, (50, 95, 99))
    # nearest-rank: ceil(p/100 * 100) = p => valoarea p-a (1-indexată).
    assert pct[50] == 50.0
    assert pct[95] == 95.0
    assert pct[99] == 99.0


def test_percentile_single_sample() -> None:
    pct = percentiles([7.5], (50, 95, 99))
    assert pct[50] == pct[95] == pct[99] == 7.5


def test_percentile_report_fields() -> None:
    report = PercentileReport([10.0, 20.0, 30.0, 40.0])
    assert report.count == 4
    assert report.as_dict() == {"p50": report.p50, "p95": report.p95, "p99": report.p99}


def test_percentile_empty_rejected() -> None:
    with pytest.raises(ValueError, match="cel puțin un eșantion"):
        percentiles([], (50,))


def test_percentile_invalid_rank_rejected() -> None:
    with pytest.raises(ValueError, match="rang percentil invalid"):
        percentiles([1.0], (0,))
    with pytest.raises(ValueError, match="rang percentil invalid"):
        percentiles([1.0], (101,))


def test_percentile_is_order_independent() -> None:
    forward = percentiles([1.0, 2.0, 3.0, 4.0, 5.0], (50, 95, 99))
    backward = percentiles([5.0, 4.0, 3.0, 2.0, 1.0], (50, 95, 99))
    assert forward == backward


# --------------------------------------------------------------------------- Supported_Load


def test_supported_load_initial_definition() -> None:
    assert SUPPORTED_LOAD.instruments == 20
    assert SUPPORTED_LOAD.bar_interval_min == 5
    # 20 instrumente × (60 / 5) bare pe oră = 240 bare pe oră.
    assert SUPPORTED_LOAD.bars_per_hour == 240


def test_supported_load_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError, match="bar_interval_min"):
        SupportedLoad(instruments=20, bar_interval_min=4, compute="x")
    with pytest.raises(ValueError, match="instruments"):
        SupportedLoad(instruments=0, bar_interval_min=5, compute="x")


# --------------------------------------------------------------------------- latențe


def test_record_latency_and_snapshot_percentiles() -> None:
    collector = MetricsCollector()
    for ms in (100.0, 200.0, 300.0, 400.0, 500.0):
        collector.record_latency(LatencyStage.STRATEGY, ms)
    snap = collector.snapshot(now=T0)
    report = snap.latency[LatencyStage.STRATEGY]
    assert report.count == 5
    assert report.p50 == 300.0


def test_record_latency_negative_rejected() -> None:
    collector = MetricsCollector()
    with pytest.raises(ValueError, match="latență negativă"):
        collector.record_latency(LatencyStage.RISK, -1.0)


def test_snapshot_omits_stage_without_samples() -> None:
    collector = MetricsCollector()
    collector.record_latency(LatencyStage.INGESTION, 10.0)
    snap = collector.snapshot(now=T0)
    assert LatencyStage.INGESTION in snap.latency
    assert LatencyStage.STRATEGY not in snap.latency


# --------------------------------------------------------------------------- prospețime


def test_freshness_measured_against_injected_now() -> None:
    collector = MetricsCollector()
    collector.record_bar("INS001", T0 - timedelta(seconds=30))
    collector.record_bar("INS002", T0 - timedelta(seconds=5))
    snap = collector.snapshot(now=T0)
    assert snap.freshness_ms["INS001"] == pytest.approx(30_000.0)
    assert snap.freshness_ms["INS002"] == pytest.approx(5_000.0)


def test_freshness_requires_tz_aware_bar() -> None:
    collector = MetricsCollector()
    with pytest.raises(ValueError):
        collector.record_bar("INS001", datetime(2026, 4, 1, 12))  # noqa: DTZ001


# --------------------------------------------------------------------------- ordine și execuții


def test_orders_by_state_and_executions() -> None:
    collector = MetricsCollector()
    collector.record_order_state(OrderState.FILLED, 3)
    collector.record_order_state(OrderState.FILLED)
    collector.record_order_state(OrderState.REJECTED_RISK, 2)
    collector.record_execution(4)
    snap = collector.snapshot(now=T0)
    assert snap.orders_by_state[OrderState.FILLED] == 4
    assert snap.orders_by_state[OrderState.REJECTED_RISK] == 2
    assert snap.executions == 4


# --------------------------------------------------------------------------- costuri și PnL


def test_costs_accumulate_by_category() -> None:
    collector = MetricsCollector()
    collector.record_costs(CostBreakdown(spread=D("1.0"), commission=D("2.0")))
    collector.record_costs(CostBreakdown(spread=D("0.5"), slippage=D("3.0")))
    snap = collector.snapshot(now=T0)
    assert snap.costs.spread == D("1.5")
    assert snap.costs.commission == D("2.0")
    assert snap.costs.slippage == D("3.0")


def test_net_pnl_is_realized_plus_unrealized() -> None:
    collector = MetricsCollector()
    collector.set_pnl(realized_eur=D("10.0"), unrealized_eur=D("-4.0"))
    snap = collector.snapshot(now=T0)
    assert snap.net_pnl_eur == D("6.0")


# --------------------------------------------------------------------------- utilizarea limitelor


def test_limit_utilization_ratio() -> None:
    collector = MetricsCollector()
    collector.record_limit_utilization("daily_loss", observed=D("3.0"), limit=D("6.0"))
    snap = collector.snapshot(now=T0)
    assert snap.utilization["daily_loss"] == D("0.5")


def test_limit_utilization_zero_limit_is_zero() -> None:
    collector = MetricsCollector()
    collector.record_limit_utilization("total_loss", observed=D("5.0"), limit=D("0"))
    assert collector.utilization()["total_loss"] == D("0")


def test_limit_utilization_negative_observed_clamped() -> None:
    collector = MetricsCollector()
    collector.record_limit_utilization("daily_loss", observed=D("-2.0"), limit=D("10.0"))
    assert collector.utilization()["daily_loss"] == D("0")


# --------------------------------------------------------------------------- benchmark


def test_benchmark_reports_percentiles_per_stage() -> None:
    # Toate etapele au latență constantă de 100 ms => percentilele sunt 100 ms.
    report = run_load_benchmark(
        load=SUPPORTED_LOAD,
        bars_per_instrument=3,
        timer=lambda _bar, _ins: 100.0,
    )
    assert report.bars == SUPPORTED_LOAD.instruments * 3
    for stage in LatencyStage:
        pct = report.stage(stage)
        assert pct.p50 == pct.p95 == pct.p99 == 100.0
    # 4 etape × 100 ms = 400 ms < 1000 ms => toate barele sunt în buget.
    assert report.within_fraction == 1.0
    assert report.meets_target()


def test_benchmark_detects_breaching_bars() -> None:
    # Fiecare etapă 300 ms => 4 × 300 = 1200 ms > 1000 ms => nicio bară în buget.
    report = run_load_benchmark(
        load=SUPPORTED_LOAD,
        bars_per_instrument=2,
        timer=lambda _bar, _ins: 300.0,
    )
    assert report.within_fraction == 0.0
    assert not report.meets_target()


def test_benchmark_is_deterministic() -> None:
    def timer(bar: int, instrument: str) -> float:
        return float((bar * 7 + len(instrument)) % 50)

    first = run_load_benchmark(load=SUPPORTED_LOAD, bars_per_instrument=4, timer=timer)
    second = run_load_benchmark(load=SUPPORTED_LOAD, bars_per_instrument=4, timer=timer)
    assert first.within_fraction == second.within_fraction
    for stage in LatencyStage:
        assert first.stage(stage).as_dict() == second.stage(stage).as_dict()


def test_benchmark_rejects_nonpositive_bars() -> None:
    with pytest.raises(ValueError, match="bars_per_instrument"):
        run_load_benchmark(load=SUPPORTED_LOAD, bars_per_instrument=0, timer=lambda _b, _i: 1.0)


# --------------------------------------------------------------------------- buget de prospețime


def test_within_freshness_budget_true_when_faster_than_threshold() -> None:
    assert within_freshness_budget(processing_ms=400.0, freshness_threshold_ms=1000.0)


def test_within_freshness_budget_false_when_at_or_over_threshold() -> None:
    # Comparație strictă: egal cu pragul nu respectă bugetul.
    assert not within_freshness_budget(processing_ms=1000.0, freshness_threshold_ms=1000.0)
    assert not within_freshness_budget(processing_ms=1200.0, freshness_threshold_ms=1000.0)


def test_within_freshness_budget_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="freshness_threshold_ms"):
        within_freshness_budget(processing_ms=10.0, freshness_threshold_ms=0.0)


def test_within_freshness_budget_rejects_negative_processing() -> None:
    with pytest.raises(ValueError, match="latență negativă"):
        within_freshness_budget(processing_ms=-1.0, freshness_threshold_ms=1000.0)


# --------------------------------------------------------------------------- proprietăți


@given(
    samples=st.lists(
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=200,
    )
)
def test_percentiles_are_monotonic_and_in_sample(samples: list[float]) -> None:
    pct = percentiles(samples, (50, 95, 99))
    # Percentilele sunt nedescrescătoare: p50 <= p95 <= p99.
    assert pct[50] <= pct[95] <= pct[99]
    ordered = sorted(samples)
    # Fiecare percentilă este o valoare prezentă în eșantion (nearest-rank, fără interpolare).
    for value in pct.values():
        assert ordered[0] <= value <= ordered[-1]
        assert value in samples


@given(
    observed=st.decimals(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    limit=st.decimals(min_value="0.01", max_value=1000, allow_nan=False, allow_infinity=False),
)
def test_utilization_ratio_matches_division(observed: Decimal, limit: Decimal) -> None:
    collector = MetricsCollector()
    collector.record_limit_utilization("x", observed=observed, limit=limit)
    assert collector.utilization()["x"] == observed / limit
