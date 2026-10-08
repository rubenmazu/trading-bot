"""Teste pentru `research/walk_forward.py` (Req 19.1, 19.2 materializare, 19.7 indirect)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from qts.research.preregistration import WalkForwardSpec
from qts.research.walk_forward import (
    MIN_OOS_WINDOWS,
    InsufficientDataError,
    build_walk_forward_windows,
    min_bars_required,
)


def _spec(**kwargs: object) -> WalkForwardSpec:
    base: dict[str, object] = {
        "windows": 5,
        "train_bars": 100,
        "test_bars": 20,
        "step_bars": 20,
        "recalibrate": True,
    }
    base.update(kwargs)
    return WalkForwardSpec(**base)


def test_builds_exactly_the_requested_number_of_windows() -> None:
    spec = _spec(windows=5)
    windows = build_walk_forward_windows(spec, min_bars_required(spec))
    assert len(windows) == 5
    assert len(windows) >= MIN_OOS_WINDOWS


def test_oos_follows_development_without_overlap() -> None:
    # Req 19.1: fiecare OOS urmează imediat Development-ul asociat.
    spec = _spec(windows=6)
    windows = build_walk_forward_windows(spec, min_bars_required(spec) + 50)
    for w in windows:
        assert w.oos_start == w.dev_end
        assert w.dev_length == spec.train_bars
        assert w.oos_length == spec.test_bars
        assert w.recalibrate is spec.recalibrate


def test_windows_are_chronological_and_indexed() -> None:
    spec = _spec(windows=7)
    windows = build_walk_forward_windows(spec, min_bars_required(spec))
    for i, w in enumerate(windows):
        assert w.index == i
    for prev, curr in pairwise(windows):
        assert curr.dev_start == prev.dev_start + spec.step_bars


def test_insufficient_data_is_fail_closed() -> None:
    spec = _spec(windows=5)
    needed = min_bars_required(spec)
    with pytest.raises(InsufficientDataError):
        build_walk_forward_windows(spec, needed - 1)


def test_exact_minimum_bars_succeeds() -> None:
    spec = _spec(windows=5)
    windows = build_walk_forward_windows(spec, min_bars_required(spec))
    assert windows[-1].oos_end == min_bars_required(spec)


def test_spec_rejects_fewer_than_five_windows() -> None:
    # Req 19.1 impus în pre-înregistrare: minimum cinci ferestre.
    with pytest.raises(ValueError):
        _spec(windows=4)


def test_deterministic_output() -> None:
    spec = _spec(windows=5)
    a = build_walk_forward_windows(spec, 500)
    b = build_walk_forward_windows(spec, 500)
    assert a == b


def test_negative_total_bars_rejected() -> None:
    spec = _spec()
    with pytest.raises(ValueError):
        build_walk_forward_windows(spec, -1)
