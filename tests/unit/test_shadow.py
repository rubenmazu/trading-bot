"""Teste pentru compunerea Shadow: feed curent reluat + SimBroker + ceas real (Req 1.1, 16.1).

Determinismul este asigurat prin injectare: un ceas controlabil (`ReplayClock`) înlocuiește
`WallClock`, iar un `DataAdapter` de reluare peste date sintetice înlocuiește adaptorul de date
real (încă o Open_Decision, Req 30). Testele NU depind de timpul de perete.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from qts.bootstrap import (
    BacktestResult,
    BootstrapError,
    RealDataAdapterUnavailableError,
    ShadowDataFactory,
    ShadowResult,
    run_backtest,
    run_shadow,
)
from qts.config.schema import AppConfig
from qts.config.snapshot import ConfigurationSnapshot
from qts.core.clock import Clock, ensure_utc
from qts.core.models import MarketEvent
from qts.data.adapter import DataAdapter
from qts.data.csv_source import CsvSource
from qts.persistence.audit import verify_journal
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from tests.fixtures.synthetic import SyntheticSpec, write_synthetic_dataset
from tests.unit.test_bootstrap import CODE, CONFIG_DIR, STAGE_LOCK

SPEC = SyntheticSpec(scenario="mean_reverting", seed=11, n_bars=400, volatility=0.004)


class ReplayClock:
    """Ceas controlabil care imită timpul real: avansează odată cu fiecare eveniment reluat.

    Implementează portul `Clock`. Pornește la `start` și este dus înainte la `ts_receipt` al
    fiecărui eveniment pe măsură ce `ShadowReplay` îl emite, exact cum un feed curent livrează
    barele la închiderea lor. Nu merge niciodată înapoi; nu citește ceasul de perete.
    """

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, ts: datetime) -> None:
        ts = ensure_utc(ts)
        if ts > self._now:
            self._now = ts


class ShadowReplay:
    """`DataAdapter` de reluare peste date existente, care avansează un ceas la fiecare eveniment.

    Reutilizează `CsvSource` pentru parsarea și validarea datelor, dar livrează evenimentele
    „în timp real” față de `ReplayClock`: înainte de a emite un eveniment, duce ceasul la
    `ts_receipt`, astfel încât prospețimea să fie satisfăcută determinist (fără timp de perete).
    """

    def __init__(self, inner: CsvSource, clock: ReplayClock) -> None:
        self._inner = inner
        self._clock = clock
        self.source_id = inner.source_id

    def stream(self) -> Iterator[MarketEvent]:
        for event in self._inner.stream():
            self._clock.advance_to(event.ts_receipt)
            yield event


def _write_shadow_config(tmp_path: Path) -> tuple[Path, Path]:
    """Scrie `config/shadow.toml` cu o bază de date în `tmp_path` și setul sintetic alăturat."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path, _ = write_synthetic_dataset(data_dir, SPEC)
    text = (CONFIG_DIR / "shadow.toml").read_text(encoding="utf-8")
    text = text.replace(
        'db_path = "runs/shadow.db"', f'db_path = "{(tmp_path / "shadow.db").as_posix()}"'
    )
    config_path = tmp_path / "shadow.toml"
    config_path.write_text(text, encoding="utf-8")
    return config_path, data_path


def _replay_factory(data_path: Path) -> ShadowDataFactory:
    def factory(_config: AppConfig, clock: Clock, _base: Path) -> DataAdapter:
        assert isinstance(clock, ReplayClock)
        return ShadowReplay(CsvSource(data_path), clock)

    return factory


def _run_shadow(tmp_path: Path) -> ShadowResult:
    config_path, data_path = _write_shadow_config(tmp_path)
    clock = ReplayClock(datetime(2024, 1, 1, tzinfo=UTC))
    return run_shadow(
        config_path,
        stage_lock=STAGE_LOCK,
        base_dir=tmp_path,
        code_version=CODE,
        clock=clock,
        data_factory=_replay_factory(data_path),
    )


def test_shadow_runs_feed_through_sim_broker(tmp_path: Path) -> None:
    """Shadow procesează feed-ul reluat prin SimBroker și produce ordine și execuții."""
    result = _run_shadow(tmp_path)
    assert result.events_processed >= SPEC.n_bars
    assert result.orders > 0 and result.fills > 0
    assert result.journal_verified
    assert result.net_eur == result.gross_eur - result.costs.total
    assert result.run_id.startswith("sh-")
    assert result.source_id == SPEC.source_id


def test_shadow_journal_records_shadow_snapshot(tmp_path: Path) -> None:
    """Prima înregistrare este snapshot-ul mediului shadow, verificabil și cu source_id corect."""
    result = _run_shadow(tmp_path)
    conn = open_db(result.db_path)
    journal = Journal(conn)
    try:
        assert verify_journal(journal, expected_head=result.journal_head).ok
        first = next(iter(journal.read()))
        assert first.type == "config_snapshot"
        snapshot = ConfigurationSnapshot.model_validate(first.payload)
        assert snapshot.verify() and snapshot.snapshot_id == result.snapshot_id
        assert snapshot.environment == "shadow"
        assert snapshot.dataset_ids == [SPEC.source_id]
    finally:
        conn.close()


def test_shadow_is_deterministic_with_injected_clock(tmp_path: Path) -> None:
    """Cu aceleași intrări și ceas injectat, Shadow produce aceleași rezultate (Req 1.1)."""
    a = _run_shadow(tmp_path / "a")
    b = _run_shadow(tmp_path / "b")
    assert (a.orders, a.fills, a.gross_eur, a.costs, a.net_eur) == (
        b.orders,
        b.fills,
        b.gross_eur,
        b.costs,
        b.net_eur,
    )


def test_shadow_same_decisions_as_backtest(tmp_path: Path) -> None:
    """Req 1.1/16.1: aceeași secvență de pași → aceleași semnale în Backtest și Shadow.

    Backtest și Shadow folosesc aceeași strategie, risc și cost model peste exact aceleași bare;
    diferă numai ceasul (SimClock vs. ceas real injectat) și sursa de evenimente. Numărul de
    semnale generate trebuie să coincidă.
    """
    bt = _run_backtest_on_same_data(tmp_path / "bt")
    sh = _run_shadow(tmp_path / "sh")
    assert _signal_count(bt.db_path) == _signal_count(sh.db_path)
    assert sh.orders == bt.orders and sh.fills == bt.fills


def test_shadow_refuses_without_real_data_adapter(tmp_path: Path) -> None:
    """Fără `data_factory`, seam-ul sursei de date reale refuză pornirea (Open_Decision)."""
    config_path, _ = _write_shadow_config(tmp_path)
    with pytest.raises(RealDataAdapterUnavailableError, match="not yet available"):
        run_shadow(
            config_path,
            stage_lock=STAGE_LOCK,
            base_dir=tmp_path,
            code_version=CODE,
            clock=ReplayClock(datetime(2024, 1, 1, tzinfo=UTC)),
        )
    assert not (tmp_path / "shadow.db").exists()


def test_shadow_refuses_non_shadow_config(tmp_path: Path) -> None:
    """`run_shadow` pe o configurație care nu este shadow este refuzat înaintea procesării."""
    with pytest.raises(BootstrapError, match="environment=shadow"):
        run_shadow(
            CONFIG_DIR / "backtest.toml",
            stage_lock=STAGE_LOCK,
            base_dir=tmp_path,
            code_version=CODE,
            clock=ReplayClock(datetime(2024, 1, 1, tzinfo=UTC)),
        )


def test_shadow_live_config_is_refused(tmp_path: Path) -> None:
    """`run_shadow` acceptă numai configurații shadow; Live este refuzat înaintea procesării.

    Live rămâne interzis prin `check_startup` pe calea generală (vezi `test_cli`); aici
    verificăm că `run_shadow` nu procesează niciodată o configurație de alt mediu.
    """
    with pytest.raises(BootstrapError, match="environment=shadow"):
        run_shadow(
            CONFIG_DIR / "live.toml",
            stage_lock=STAGE_LOCK,
            base_dir=tmp_path,
            code_version=CODE,
            clock=ReplayClock(datetime(2024, 1, 1, tzinfo=UTC)),
        )
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- ajutoare


def _run_backtest_on_same_data(tmp_path: Path) -> BacktestResult:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path, _ = write_synthetic_dataset(data_dir, SPEC)
    text = (CONFIG_DIR / "backtest.toml").read_text(encoding="utf-8")
    text = text.replace(
        'dataset_path = "data/synthetic.csv"', f'dataset_path = "{data_path.as_posix()}"'
    ).replace('db_path = "runs/backtest.db"', f'db_path = "{(tmp_path / "bt.db").as_posix()}"')
    config_path = tmp_path / "backtest.toml"
    config_path.write_text(text, encoding="utf-8")
    return run_backtest(config_path, stage_lock=STAGE_LOCK, base_dir=tmp_path, code_version=CODE)


def _signal_count(db_path: str) -> int:
    conn = open_db(db_path)
    try:
        return sum(1 for r in Journal(conn).read() if r.type == "signal")
    finally:
        conn.close()


def test_replay_clock_never_goes_backwards() -> None:
    clock = ReplayClock(datetime(2024, 1, 2, tzinfo=UTC))
    clock.advance_to(datetime(2024, 1, 1, tzinfo=UTC))  # mai vechi: ignorat
    assert clock.now() == datetime(2024, 1, 2, tzinfo=UTC)
    clock.advance_to(datetime(2024, 1, 3, tzinfo=UTC))
    assert clock.now() == datetime(2024, 1, 3, tzinfo=UTC)


# --------------------------------------------------------------------------- CLI


def test_cli_shadow_refuses_without_real_data_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`qts shadow` refuză pornirea cât timp sursa de date reală nu există (Open_Decision)."""
    from typer.testing import CliRunner

    from qts.cli import EXIT_REFUSED, app

    config_path, _ = _write_shadow_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["shadow", "--config", str(config_path), "--stage-lock", str(STAGE_LOCK)]
    )
    assert result.exit_code == EXIT_REFUSED
    assert "pornire refuzată" in result.output
    assert "not yet available" in result.output


def test_cli_shadow_has_no_live_options() -> None:
    from typer.testing import CliRunner

    from qts.cli import app

    result = CliRunner().invoke(app, ["shadow", "--help"])
    assert result.exit_code == 0
    assert "live" not in result.output.lower()


def test_shadow_result_net_equals_gross_minus_costs() -> None:
    from qts.core.models import CostBreakdown

    r = ShadowResult(
        snapshot_id="s",
        run_id="sh-x",
        source_id="feed",
        db_path="x",
        events_processed=0,
        orders=0,
        fills=0,
        realized_gross_eur=Decimal("2.0"),
        unrealized_gross_eur=Decimal("-0.5"),
        costs=CostBreakdown(commission=Decimal("0.25")),
        journal_head=(0, ""),
        journal_verified=True,
    )
    assert r.gross_eur == Decimal("1.5")
    assert r.net_eur == Decimal("1.25")
