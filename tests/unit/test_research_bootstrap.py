"""Teste pentru `research/bootstrap.py` (Req 19.3, 19.4, 19.7).

Verifică: minimum 10.000 de reeșantionări, determinism din sămânță, cele patru distribuții
raportate, raportarea parțială în timpul rulării și metricile per cale.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from qts.research.bootstrap import (
    MIN_RESAMPLES,
    BootstrapConfig,
    BootstrapError,
    collect_bootstrap,
    longest_losing_streak,
    max_drawdown,
    run_stationary_bootstrap,
)

# Serie mică de rezultate nete per tranzacție (EUR), cu grupări de pierderi (autocorelație).
_TRADES = [
    Decimal("1.5"),
    Decimal("-0.5"),
    Decimal("-0.8"),
    Decimal("-1.2"),
    Decimal("2.0"),
    Decimal("0.3"),
    Decimal("-0.4"),
    Decimal("1.1"),
]


def _config(**kwargs: object) -> BootstrapConfig:
    base: dict[str, object] = {"resamples": MIN_RESAMPLES, "seed": 42}
    base.update(kwargs)
    return BootstrapConfig(**base)


# --------------------------------------------------------------------------- metrici per cale


def test_max_drawdown_simple() -> None:
    arr = np.array([1.0, -0.5, -0.5, 2.0], dtype=np.float64)
    # equity: 1.0, 0.5, 0.0, 2.0 ; vârf 1.0 → drawdown max = 1.0
    assert max_drawdown(arr) == pytest.approx(1.0)


def test_max_drawdown_monotonic_increasing_is_zero() -> None:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert max_drawdown(arr) == pytest.approx(0.0)


def test_max_drawdown_empty_is_zero() -> None:
    assert max_drawdown(np.array([], dtype=np.float64)) == 0.0


def test_longest_losing_streak() -> None:
    arr = np.array([1.0, -1.0, -1.0, -1.0, 2.0, -1.0], dtype=np.float64)
    assert longest_losing_streak(arr) == 3


def test_longest_losing_streak_none() -> None:
    arr = np.array([1.0, 2.0, 0.0], dtype=np.float64)
    assert longest_losing_streak(arr) == 0


# --------------------------------------------------------------------------- configurație


def test_config_rejects_too_few_resamples() -> None:
    with pytest.raises(ValueError):
        BootstrapConfig(resamples=9_999, seed=1)


def test_run_rejects_empty_series() -> None:
    with pytest.raises(BootstrapError):
        list(run_stationary_bootstrap([], _config()))


def test_run_rejects_bad_report_every() -> None:
    with pytest.raises(BootstrapError):
        list(run_stationary_bootstrap(_TRADES, _config(), report_every=0))


# --------------------------------------------------------------------------- determinism (Req 19.3)


def test_same_seed_is_reproducible() -> None:
    a = collect_bootstrap(_TRADES, _config(seed=7))
    b = collect_bootstrap(_TRADES, _config(seed=7))
    assert a == b
    assert a.seed == 7


def test_different_seed_changes_result() -> None:
    a = collect_bootstrap(_TRADES, _config(seed=1))
    b = collect_bootstrap(_TRADES, _config(seed=2))
    # Distribuțiile nu trebuie să fie identice pentru semințe diferite.
    assert a != b


def test_records_configuration() -> None:
    result = collect_bootstrap(_TRADES, _config(seed=5))
    assert result.resamples == MIN_RESAMPLES
    assert result.total_loss_limit_eur == Decimal("10")


# --------------------------------------------------------------------------- distribuții (Req 19.4)


def test_all_four_distributions_present() -> None:
    result = collect_bootstrap(_TRADES, _config())
    assert result.net_pnl.count == MIN_RESAMPLES
    assert result.max_drawdown.count == MIN_RESAMPLES
    assert result.longest_losing_streak.count == MIN_RESAMPLES
    assert Decimal("0") <= result.prob_hit_total_loss_limit <= Decimal("1")
    # drawdown-ul este nenegativ
    assert result.max_drawdown.minimum >= Decimal("0")
    # seria de pierderi nu depășește lungimea seriei
    assert result.longest_losing_streak.maximum <= Decimal(len(_TRADES))


def test_probability_of_hitting_limit_depends_on_threshold() -> None:
    # O limită foarte mare (greu de atins) nu poate avea probabilitate mai mare decât una mică.
    high = collect_bootstrap(_TRADES, _config(total_loss_limit_eur=Decimal("1000")))
    low = collect_bootstrap(_TRADES, _config(total_loss_limit_eur=Decimal("1")))
    assert high.prob_hit_total_loss_limit <= low.prob_hit_total_loss_limit


# ----------------------------------------------------------------- raportare parțială (Req 19.7)


def test_partial_reports_are_emitted_while_running() -> None:
    reports = list(
        run_stationary_bootstrap(_TRADES, _config(), report_every=2_500)
    )
    # 10.000 / 2.500 = 4 rapoarte, ultimul final.
    assert [r.completed for r in reports] == [2_500, 5_000, 7_500, 10_000]
    assert [r.final for r in reports] == [False, False, False, True]
    # Fiecare raport parțial are deja toate distribuțiile disponibile (Req 19.7).
    for r in reports:
        assert r.net_pnl.count == r.completed
        assert r.max_drawdown.count == r.completed
        assert r.longest_losing_streak.count == r.completed


def test_single_final_report_when_no_interval() -> None:
    reports = list(run_stationary_bootstrap(_TRADES, _config()))
    assert len(reports) == 1
    assert reports[0].final is True
    assert reports[0].completed == MIN_RESAMPLES


def test_final_partial_report_matches_collect() -> None:
    reports = list(run_stationary_bootstrap(_TRADES, _config(seed=3), report_every=5_000))
    final = reports[-1]
    collected = collect_bootstrap(_TRADES, _config(seed=3))
    assert final.net_pnl == collected.net_pnl
    assert final.prob_hit_total_loss_limit == collected.prob_hit_total_loss_limit


# --------------------------------------------------------------------------- autocorelație


def test_stationary_bootstrap_preserves_mean_approximately() -> None:
    # Media reeșantionărilor PnL ≈ suma seriei originale (fiecare observație e echiprobabilă).
    result = collect_bootstrap(_TRADES, _config(seed=11))
    expected_total = Decimal(repr(round(float(sum(_TRADES)), 2)))
    assert result.net_pnl.mean == pytest.approx(expected_total, abs=Decimal("0.5"))
