"""Teste pentru invalidarea subperioadelor cu goluri (Req 7.4, 7.7) și componența istorică a
universului (Req 7.5): `qts.data.gaps` și `qts.data.universe_history`, plus integrarea minimă în
`core/engine.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from qts.broker.adapter import CancelAck, OrderRequest, SubmitAck
from qts.config.schema import KillSwitchConfig, RiskConfig
from qts.core.clock import SimClock
from qts.core.engine import EngineConfig, TradingEngine
from qts.core.models import Bar, ExecutionEvent, Instrument, MarketEvent, Signal, SignalAction
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import (
    CompleteCostModel,
    CostContext,
    CostModelConfig,
    FxConfig,
    LatencyConfig,
    SlippageConfig,
    SpreadSchedule,
    TaxConfig,
)
from qts.data.freshness import FreshnessTracker
from qts.data.gaps import GAP_POLICY_VERSION, GapPolicy, SubperiodMark, missing_bars_between
from qts.data.universe_history import (
    UNIVERSE_HISTORY_VERSION,
    UniverseHistory,
    UniverseListing,
)
from qts.oms.manager import JournalOmsSink, OrderManager
from qts.persistence.audit import verify_journal
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.portfolio.portfolio import Portfolio
from qts.risk.engine import RiskEngine
from qts.safety.kill_switch import JournalKillSwitchStore, KillSwitch
from qts.strategy.base import StrategyState, build_signal
from qts.strategy.history_view import HistoryView
from tests.fixtures.synthetic import (
    Gap,
    SyntheticSpec,
    find_gaps,
    gaps_around_threshold,
    generate_bars,
)

D = Decimal
T0 = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)
INTERVAL = 15


def mk_bar(i: int, *, interval: int = INTERVAL, instrument: str = "XYZ", close: str = "10") -> Bar:
    ts_open = T0 + timedelta(minutes=interval * i)
    return Bar(
        instrument=instrument,
        ts_open=ts_open,
        ts_close=ts_open + timedelta(minutes=interval),
        interval_min=interval,
        open=close,
        high=close,
        low=close,
        close=close,
        volume="100",
    )


def gapped_bar(prev: Bar, missing: int, *, close: str = "10") -> Bar:
    """O bară care urmează lui `prev` după `missing` sloturi lipsă."""
    step = timedelta(minutes=prev.interval_min)
    ts_open = prev.ts_close + missing * step
    return Bar(
        instrument=prev.instrument,
        ts_open=ts_open,
        ts_close=ts_open + step,
        interval_min=prev.interval_min,
        open=close,
        high=close,
        low=close,
        close=close,
        volume="100",
    )


# --------------------------------------------------------------------------- măsurarea golului


def test_missing_bars_contiguous_is_zero() -> None:
    assert missing_bars_between(mk_bar(0), mk_bar(1)) == 0


def test_missing_bars_counts_absent_slots() -> None:
    prev = mk_bar(0)
    assert missing_bars_between(prev, gapped_bar(prev, 3)) == 3


def test_missing_bars_consistent_with_synthetic_gap() -> None:
    # Gap.minutes = missing_bars * interval; `missing_bars_between` trebuie să recupereze exact.
    gap = Gap(after_bar=1, missing_bars=4)
    prev = mk_bar(0)
    nxt = gapped_bar(prev, gap.missing_bars)
    assert gap.minutes(INTERVAL) == (nxt.ts_open - prev.ts_close) / timedelta(minutes=1)
    assert missing_bars_between(prev, nxt) == gap.missing_bars


def test_missing_bars_rejects_mixed_instruments_and_backwards() -> None:
    with pytest.raises(ValueError, match="instrumente"):
        missing_bars_between(mk_bar(0), mk_bar(1, instrument="ABC"))
    with pytest.raises(ValueError, match="precede"):
        missing_bars_between(mk_bar(2), mk_bar(0))


# --------------------------------------------------------------------------- GapPolicy


def test_gap_exactly_at_threshold_not_invalidated() -> None:
    """Req 7.7: un gol <= prag continuă subperioada, nimic nu se schimbă."""
    policy = GapPolicy(max_gap_bars=3)
    prev = mk_bar(0)
    first = policy.observe(prev)
    second = policy.observe(gapped_bar(prev, 3))  # exact la prag
    assert first.subperiod_id == 0 and not first.boundary
    assert second.subperiod_id == 0 and not second.boundary
    assert second.missing_before == 3
    assert policy.invalidated_subperiods("XYZ") == ()
    assert not policy.any_invalidated()


def test_gap_over_threshold_opens_new_subperiod_and_invalidates() -> None:
    """Req 7.4: un gol > prag deschide o subperioadă nouă și invalidează exclusiv cea afectată."""
    policy = GapPolicy(max_gap_bars=3)
    prev = mk_bar(0)
    policy.observe(prev)
    mark = policy.observe(gapped_bar(prev, 4))  # peste prag
    assert mark.boundary and mark.subperiod_id == 1 and mark.missing_before == 4
    assert policy.invalidated_subperiods("XYZ") == (0,)
    assert policy.current_subperiod("XYZ") == 1
    assert policy.any_invalidated()


def test_gap_policy_per_instrument_isolation() -> None:
    policy = GapPolicy(max_gap_bars=1)
    a0 = mk_bar(0, instrument="AAA")
    b0 = mk_bar(0, instrument="BBB")
    policy.observe(a0)
    policy.observe(b0)
    policy.observe(gapped_bar(a0, 5))  # AAA sare peste prag
    policy.observe(gapped_bar(b0, 1))  # BBB rămâne sub prag
    assert policy.invalidated_subperiods("AAA") == (0,)
    assert policy.invalidated_subperiods("BBB") == ()
    assert policy.current_subperiod("AAA") == 1
    assert policy.current_subperiod("BBB") == 0


def test_gap_policy_multiple_boundaries_increment_ids() -> None:
    policy = GapPolicy(max_gap_bars=2)
    b = mk_bar(0)
    marks: list[SubperiodMark] = [policy.observe(b)]
    for _ in range(3):
        b = gapped_bar(b, 5)
        marks.append(policy.observe(b))
    assert [m.subperiod_id for m in marks] == [0, 1, 2, 3]
    assert all(m.boundary for m in marks[1:])
    assert policy.invalidated_subperiods("XYZ") == (0, 1, 2)


def test_gap_policy_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="max_gap_bars"):
        GapPolicy(max_gap_bars=-1)


def test_gap_policy_version_is_stable() -> None:
    assert GAP_POLICY_VERSION == "gap-policy-v1"


def test_gaps_around_threshold_fixture_matches_policy() -> None:
    """Reutilizează fixtura: golul la n/3 este <= prag, cel la 2n/3 este > prag."""
    threshold_bars = 3
    below, above = gaps_around_threshold(threshold_bars * INTERVAL, INTERVAL, n_bars=9)
    assert below.missing_bars == threshold_bars
    assert above.missing_bars == threshold_bars + 1
    spec = SyntheticSpec(
        scenario="random_walk", seed=11, n_bars=9, interval_min=INTERVAL, gaps=(below, above)
    )
    bars = generate_bars(spec)
    policy = GapPolicy(max_gap_bars=threshold_bars)
    for bar in bars:
        policy.observe(bar)
    # Un singur gol depășește pragul (cel de la 2n/3), deci o singură subperioadă invalidată.
    assert len(policy.invalidated_subperiods(spec.instrument)) == 1
    assert policy.current_subperiod(spec.instrument) == 1
    # Fixtura chiar produce două goluri în bare.
    assert len(find_gaps(bars)) == 2


# --------------------------------------------------------------------------- UniverseHistory


def test_universe_active_before_listing_and_after_delisting() -> None:
    """Req 7.5: componența la timpul simulat, incl. delistate valabile atunci."""
    listed = T0
    delisted = T0 + timedelta(days=10)
    hist = UniverseHistory.build(
        [
            UniverseListing(symbol="OLD", listed_at=None, delisted_at=delisted),
            UniverseListing(symbol="NEW", listed_at=listed, delisted_at=None),
        ]
    )
    # Înainte de listarea NEW: numai OLD este activ.
    assert hist.active_universe(listed - timedelta(seconds=1)) == frozenset({"OLD"})
    # La fix momentul listării (inclusiv) ambele sunt active.
    assert hist.active_universe(listed) == frozenset({"OLD", "NEW"})
    # În ziua delistării (exclusiv): OLD iese, NEW rămâne.
    assert hist.active_universe(delisted) == frozenset({"NEW"})
    assert hist.active_universe(delisted - timedelta(seconds=1)) == frozenset({"OLD", "NEW"})


def test_universe_is_active_and_unknown_symbol() -> None:
    hist = UniverseHistory.build(
        [UniverseListing(symbol="A", listed_at=T0, delisted_at=T0 + timedelta(days=1))]
    )
    assert hist.is_active("A", T0)
    assert not hist.is_active("A", T0 - timedelta(seconds=1))
    assert not hist.is_active("A", T0 + timedelta(days=1))  # delistare exclusivă
    assert not hist.is_active("UNKNOWN", T0)


def test_universe_delisting_must_follow_listing() -> None:
    with pytest.raises(ValueError, match="delisted_at"):
        UniverseListing(symbol="X", listed_at=T0, delisted_at=T0)


def test_universe_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        UniverseHistory(listings=(UniverseListing(symbol="A"), UniverseListing(symbol="A")))


def test_universe_version_is_content_derived_and_order_independent() -> None:
    a = UniverseHistory.build(
        [UniverseListing(symbol="A"), UniverseListing(symbol="B", listed_at=T0)]
    )
    b = UniverseHistory.build(
        [UniverseListing(symbol="B", listed_at=T0), UniverseListing(symbol="A")]
    )
    assert a.version == b.version
    assert a.version.startswith(UNIVERSE_HISTORY_VERSION)


def test_universe_empty_is_always_empty() -> None:
    hist = UniverseHistory.build([])
    assert hist.active_universe(T0) == frozenset()


# --------------------------------------------------------------------------- integrare motor

INSTRUMENT = Instrument(
    symbol="XYZ",
    venue="XETR",
    asset_class="etf",
    currency="EUR",
    tick_size=D("0.01"),
    qty_step=D("0.001"),
    min_qty=D("0.001"),
    calendar_id="XETR",
    fractional=True,
)
SIGMA = D("0.002")
ADV = D(100_000)


def _cost_model() -> CompleteCostModel:
    table = CommissionTable.model_validate(
        {
            "broker": "sim",
            "version": "v1",
            "valid_from": datetime(2024, 1, 1, tzinfo=UTC),
            "currency": "EUR",
            "percent": "0",
            "exchange_fee_fixed": "0.05",
        }
    )
    return CompleteCostModel(
        CostModelConfig(
            version="costs-test",
            commissions=CommissionSchedule(tables=(table,)),
            spreads={"XYZ": SpreadSchedule(default=D("0.0002"))},
            slippage=SlippageConfig(k=D(0)),
            latency=LatencyConfig(latency_ms=0),
            fx=FxConfig(conversion_spread=D(0)),
            taxes=TaxConfig(approved=True, reference="test"),
        )
    )


def _market_context(inst: Instrument, bar: Bar) -> CostContext:
    return CostContext(broker="sim", sigma_bar=SIGMA, adv=ADV, bar_interval_min=bar.interval_min)


class _Count(StrategyState):
    n: int = 0


class _AlwaysEnter:
    """Emite ENTER_LONG pe fiecare bară.

    Pe bara-graniță motorul resetează starea și suprimă generarea de ordine; semnalul
    ENTER_LONG de pe acea bară nu produce niciun `client_order_id`, deci niciun ordin nu
    traversează golul.
    """

    strategy_id = "always_enter"
    version = "1"

    def initial_state(self) -> StrategyState:
        return _Count()

    def on_bar(
        self, bar: Bar, view: HistoryView, state: StrategyState
    ) -> tuple[Signal, StrategyState]:
        assert isinstance(state, _Count)
        action: SignalAction = "ENTER_LONG"
        signal = build_signal(
            self,
            bar,
            action=action,
            reason_code=f"S_{action}",
            config_snapshot_id="cfg-test",
            stop_price=bar.close - D(1),
        )
        return signal, _Count(n=state.n + 1)


@dataclass
class _ListData:
    events: list[MarketEvent]
    source_id: str = "synthetic"

    def stream(self) -> Iterator[MarketEvent]:
        return iter(self.events)


def _event(bar: Bar) -> MarketEvent:
    return MarketEvent(
        source_id="synthetic",
        instrument=bar.instrument,
        ts_source=bar.ts_close,
        ts_receipt=bar.ts_close,
        kind="bar",
        payload=bar,
    )


def _run_engine(bars: list[Bar], *, gap_policy: GapPolicy | None) -> tuple[TradingEngine, Journal]:
    clock = SimClock(bars[0].ts_open)
    journal = Journal(open_db(":memory:"))
    oms = OrderManager(JournalOmsSink(journal))
    kill_switch = KillSwitch(
        JournalKillSwitchStore(journal),
        clock=clock,
        config=KillSwitchConfig(),
        open_orders=lambda: [],
    )
    costs = _cost_model()

    def feed(event: MarketEvent) -> None:
        from qts.broker.sim import SimBarContext

        if isinstance(event.payload, Bar):
            sim.on_bar(event.payload, SimBarContext(sigma_bar=SIGMA, adv=ADV))

    from qts.broker.fail_safe import ApprovedTarget, FailSafeBlock
    from qts.broker.sim import SimBroker
    from qts.safety.stage import ProjectStage, StageInfo

    sim = SimBroker(account_id="SIM-LOCAL", instruments=[INSTRUMENT], cost_model=costs, clock=clock)
    guarded = FailSafeBlock(
        sim,
        approved=ApprovedTarget(environment="backtest", account_id="SIM-LOCAL"),
        stage=StageInfo(stage=ProjectStage.INITIAL, source="test"),
        kill_switch=kill_switch,
        audit=journal,
        clock=clock,
    )
    engine = TradingEngine(
        config=EngineConfig(run_id="run-1", mode="backtest", instruments=(INSTRUMENT,)),
        clock=clock,
        broker=guarded,
        journal=journal,
        strategy=_AlwaysEnter(),
        risk=RiskEngine(RiskConfig(), costs),
        oms=oms,
        portfolio=Portfolio.with_cash(D(100)),
        kill_switch=kill_switch,
        cost_model=costs,
        freshness=FreshnessTracker(clock, timedelta(days=5)),
        data=_ListData([_event(b) for b in bars]),
        market_context=_market_context,
        market_listeners=[feed],
        clock_driver=clock,
        gap_policy=gap_policy,
    )
    engine.run()
    return engine, journal


def _over_threshold_bars() -> list[Bar]:
    """Trei bare contigue, un gol peste prag, apoi trei bare contigue."""
    bars = [mk_bar(0), mk_bar(1), mk_bar(2)]
    nxt = gapped_bar(bars[-1], 5)  # gol de 5 bare > prag 3
    bars.append(nxt)
    for _ in range(2):
        nxt = gapped_bar(nxt, 0)
        bars.append(nxt)
    return bars


def test_engine_over_threshold_gap_splits_subperiods_and_journals() -> None:
    bars = _over_threshold_bars()
    _, journal = _run_engine(bars, gap_policy=GapPolicy(max_gap_bars=3))
    types = [r.type for r in journal.read()]
    invalidations = [r for r in journal.read() if r.type == "engine.subperiod_invalidated"]
    assert len(invalidations) == 1
    rec = invalidations[0]
    assert rec.payload["instrument"] == "XYZ"
    assert rec.payload["invalidated_subperiod_id"] == 0
    assert rec.payload["new_subperiod_id"] == 1
    assert rec.payload["missing_bars"] == 5
    # Ordinul de pe bara-graniță este suprimat (niciun semnal nu traversează golul).
    assert "engine.subperiod_order_suppressed" in types
    # Metricile de după gol sunt în continuare produse: strategia reemite după reset.
    assert verify_journal(journal).ok


def test_engine_resets_history_so_no_cross_gap_signal() -> None:
    bars = _over_threshold_bars()
    with_policy, journal = _run_engine(bars, gap_policy=GapPolicy(max_gap_bars=3))
    # Bara-graniță este a 4-a bară; strategia emite ENTER_LONG, dar generarea de ordine este
    # suprimată (nu se creează niciun client_order_id care să traverseze golul).
    suppressed = [r for r in journal.read() if r.type == "engine.subperiod_order_suppressed"]
    assert len(suppressed) == 1
    # Decizia de pe bara-graniță nu are ordin asociat.
    gap_decisions = [
        d for d in with_policy.decisions if d.action == "ENTER_LONG" and d.client_order_id is None
    ]
    assert gap_decisions, "ENTER_LONG de pe bara-graniță trebuie suprimat"
    # Înaintea golului au existat ordine create normal (metrici produse pe prima subperioadă).
    created_before = [
        d for d in with_policy.decisions if d.action == "ENTER_LONG" and d.client_order_id
    ]
    assert created_before


def test_engine_without_policy_unchanged_and_at_threshold_untouched() -> None:
    """Req 7.7: fără politică sau cu gol <= prag, nicio invalidare, niciun ordin suprimat."""
    bars = _over_threshold_bars()
    _, journal_none = _run_engine(bars, gap_policy=None)
    types_none = [r.type for r in journal_none.read()]
    assert "engine.subperiod_invalidated" not in types_none
    assert "engine.subperiod_order_suppressed" not in types_none

    # Gol de exact 3 bare (la prag), cu politică: tot nimic.
    at_threshold = [mk_bar(0), mk_bar(1), mk_bar(2)]
    nxt = gapped_bar(at_threshold[-1], 3)
    at_threshold.append(nxt)
    at_threshold.append(gapped_bar(nxt, 0))
    _, journal_eq = _run_engine(at_threshold, gap_policy=GapPolicy(max_gap_bars=3))
    types_eq = [r.type for r in journal_eq.read()]
    assert "engine.subperiod_invalidated" not in types_eq


class _StubBroker:
    """Broker inert: seam-ul universului nu procesează evenimente, deci nu îl folosește."""

    def submit(self, req: OrderRequest, /) -> SubmitAck:  # pragma: no cover - neapelat
        raise NotImplementedError

    def cancel(self, client_order_id: str, /) -> CancelAck:  # pragma: no cover - neapelat
        raise NotImplementedError

    def events(self) -> Iterator[ExecutionEvent]:  # pragma: no cover - neapelat
        return iter(())


def _idle_engine(*, universe_history: UniverseHistory | None) -> TradingEngine:
    clock = SimClock(T0)
    journal = Journal(open_db(":memory:"))
    return TradingEngine(
        config=EngineConfig(run_id="r", mode="backtest", instruments=(INSTRUMENT,)),
        clock=clock,
        broker=_StubBroker(),
        journal=journal,
        strategy=_AlwaysEnter(),
        risk=RiskEngine(RiskConfig(), _cost_model()),
        oms=OrderManager(JournalOmsSink(journal)),
        portfolio=Portfolio.with_cash(D(100)),
        kill_switch=KillSwitch(
            JournalKillSwitchStore(journal),
            clock=clock,
            config=KillSwitchConfig(),
            open_orders=lambda: [],
        ),
        cost_model=_cost_model(),
        freshness=FreshnessTracker(clock, timedelta(days=5)),
        universe_history=universe_history,
    )


def test_engine_active_universe_seam() -> None:
    hist = UniverseHistory.build(
        [UniverseListing(symbol="XYZ", listed_at=T0 + timedelta(minutes=20))]
    )
    engine = _idle_engine(universe_history=hist)
    assert engine.active_universe(T0) == frozenset()
    assert engine.active_universe(T0 + timedelta(minutes=20)) == frozenset({"XYZ"})


def test_engine_active_universe_none_without_history() -> None:
    engine = _idle_engine(universe_history=None)
    assert engine.active_universe(T0) is None
