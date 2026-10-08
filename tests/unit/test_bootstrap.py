"""Teste pentru `bootstrap.py`: compunere Backtest, separare nucleu / adaptoare (Req 1.2, 1.5)."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from qts.bootstrap import (
    BacktestResult,
    BootstrapError,
    ModeNotAvailableError,
    TrailingMarketEstimator,
    build_core,
    run_backtest,
)
from qts.config.loader import load_config, parse_config
from qts.config.snapshot import CodeVersion, ConfigurationSnapshot, SnapshotError
from qts.core.models import CostBreakdown
from qts.data.manifest import DatasetInvalidError
from qts.persistence.audit import verify_journal
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.safety.stage import StartupRefusedError
from tests.fixtures.synthetic import SyntheticSpec, generate_bars, write_synthetic_dataset

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "config"
STAGE_LOCK = REPO / "stage.lock"
CODE = CodeVersion(git_commit="0" * 40, git_dirty=False, lock_sha256="a" * 64)
SPEC = SyntheticSpec(scenario="mean_reverting", seed=11, n_bars=400, volatility=0.004)


def write_backtest_config(
    tmp_path: Path, spec: SyntheticSpec = SPEC, *, replacements: dict[str, str] | None = None
) -> Path:
    """Copia `config/backtest.toml` cu setul sintetic și baza de date în `tmp_path`."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    data_path, _ = write_synthetic_dataset(data_dir, spec)
    text = (CONFIG_DIR / "backtest.toml").read_text(encoding="utf-8")
    edits = {
        'dataset_path = "data/synthetic.csv"': f'dataset_path = "{data_path.as_posix()}"',
        'db_path = "runs/backtest.db"': f'db_path = "{(tmp_path / "run.db").as_posix()}"',
        **(replacements or {}),
    }
    for old, new in edits.items():
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "backtest.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _run(tmp_path: Path) -> BacktestResult:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return run_backtest(
        write_backtest_config(tmp_path),
        stage_lock=STAGE_LOCK,
        base_dir=tmp_path,
        code_version=CODE,
    )


def test_backtest_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.events_processed >= SPEC.n_bars
    assert result.orders > 0 and result.fills > 0
    assert result.journal_verified
    assert result.net_eur == result.gross_eur - result.costs.total
    assert result.costs.commission > 0

    conn = open_db(result.db_path)
    journal = Journal(conn)
    try:
        assert verify_journal(journal, expected_head=result.journal_head).ok
        records = list(journal.read())
        first = records[0]
        assert first.type == "config_snapshot" and first.correlation_id == result.snapshot_id
        snapshot = ConfigurationSnapshot.model_validate(first.payload)
        assert snapshot.verify() and snapshot.snapshot_id == result.snapshot_id
        assert snapshot.dataset_ids == [result.dataset_id]
        assert snapshot.environment == "backtest" and snapshot.git_commit == CODE.git_commit
        signals = [r for r in records if r.type == "signal"]
        assert len(signals) == SPEC.n_bars
        assert {r.payload["config_snapshot_id"] for r in signals} == {result.snapshot_id}
    finally:
        conn.close()


def test_backtest_is_repeatable_with_same_inputs(tmp_path: Path) -> None:
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    # Căile din configurație diferă (deci și snapshot-ul); rezultatele trebuie să coincidă.
    assert (a.orders, a.fills, a.gross_eur, a.costs, a.net_eur) == (
        b.orders,
        b.fills,
        b.gross_eur,
        b.costs,
        b.net_eur,
    )


def test_snapshot_failure_stops_before_processing(tmp_path: Path) -> None:
    config = write_backtest_config(tmp_path)
    with pytest.raises(SnapshotError):
        run_backtest(config, stage_lock=STAGE_LOCK, base_dir=tmp_path, repo_root=tmp_path)
    assert not (tmp_path / "run.db").exists()


def test_live_config_is_refused_before_composition(tmp_path: Path) -> None:
    with pytest.raises(StartupRefusedError):
        run_backtest(
            CONFIG_DIR / "live.toml", stage_lock=STAGE_LOCK, base_dir=tmp_path, code_version=CODE
        )
    assert list(tmp_path.iterdir()) == []


def test_corrupt_dataset_is_refused(tmp_path: Path) -> None:
    config = write_backtest_config(tmp_path)
    data = next((tmp_path / "data").glob("*.csv"))
    data.write_bytes(data.read_bytes() + b"XYZ,corrupt\n")
    with pytest.raises(DatasetInvalidError):
        run_backtest(config, stage_lock=STAGE_LOCK, base_dir=tmp_path, code_version=CODE)
    assert not (tmp_path / "run.db").exists()


def test_missing_cost_model_is_refused(tmp_path: Path) -> None:
    config = write_backtest_config(tmp_path)
    text = config.read_text(encoding="utf-8")
    config.write_text(text[: text.index("# Complete_Cost_Model")], encoding="utf-8")
    with pytest.raises(BootstrapError, match="costs"):
        run_backtest(config, stage_lock=STAGE_LOCK, base_dir=tmp_path, code_version=CODE)


@pytest.mark.parametrize("env", ["demo"])
def test_modes_without_composition_are_refused(env: str, tmp_path: Path) -> None:
    # Shadow are acum o compunere proprie (`run_shadow`, sarcina 14.3); numai Demo rămâne fără
    # compunere (adaptorul depinde de Open_Decision pentru broker).
    with pytest.raises(ModeNotAvailableError, match="not yet available"):
        run_backtest(
            CONFIG_DIR / f"{env}.toml",
            stage_lock=STAGE_LOCK,
            base_dir=tmp_path,
            code_version=CODE,
        )


def test_core_is_identical_across_modes(tmp_path: Path) -> None:
    """Req 1.5: schimbarea modului nu schimbă strategia, parametrii sau regulile de risc."""
    backtest = load_config(write_backtest_config(tmp_path))
    raw = backtest.model_dump(mode="json")
    raw["environment"] = "shadow"
    raw["run"]["db_path"] = "runs/shadow.db"
    raw["data"] = {**raw["data"], "source_id": "live_feed", "dataset_path": None}
    shadow = parse_config(raw, expected_environment="shadow")

    a = build_core(backtest, "snap-1")
    b = build_core(shadow, "snap-1")
    assert type(a.strategy) is type(b.strategy)
    assert vars(a.strategy) == vars(b.strategy)
    assert a.risk.config == b.risk.config
    assert a.cost_model.config == b.cost_model.config
    # Diferențele dintre configurații sunt numai de mediu / adaptoare.
    differing = {k for k in raw if backtest.model_dump(mode="json")[k] != raw[k]}
    assert differing == {"environment", "run", "data"}


def test_trailing_estimator_has_no_lookahead() -> None:
    bars = generate_bars(SyntheticSpec(scenario="random_walk", seed=1, n_bars=30))
    est = TrailingMarketEstimator(sigma_window=5)
    assert est.observe(bars[0]) == (None, bars[0].volume)
    est.observe(bars[1])
    sigma, _ = est.observe(bars[2])
    assert sigma is not None and sigma >= 0
    for bar in bars[3:]:
        before = est.before(bar)  # înainte de observare: estimarea curentă
        assert est.observe(bar) is not None
        assert est.before(bar) == before  # reținută pentru execuția la deschiderea barei
    # ADV = volumul rulant pe 24 de ore: o bară după o zi scoate barele vechi din fereastră.
    last = bars[-1]
    shift = timedelta(days=2)
    late = last.model_copy(
        update={"ts_open": last.ts_open + shift, "ts_close": last.ts_close + shift}
    )
    _, adv = est.observe(late)
    assert adv == late.volume


def test_estimator_rejects_too_small_window() -> None:
    with pytest.raises(ValueError):
        TrailingMarketEstimator(sigma_window=1)


def test_result_net_equals_gross_minus_costs() -> None:
    r = BacktestResult(
        snapshot_id="s",
        run_id="r",
        dataset_id="d",
        db_path="x",
        events_processed=0,
        orders=0,
        fills=0,
        realized_gross_eur=Decimal("1.5"),
        unrealized_gross_eur=Decimal("-0.5"),
        costs=CostBreakdown(commission=Decimal("0.3")),
        journal_head=(0, ""),
        journal_verified=True,
    )
    assert r.net_eur == Decimal("0.7")
    assert replace(r, realized_gross_eur=Decimal(0)).net_eur == Decimal("-0.8")
