"""P3: după atingerea limitei totale nu se mai aprobă niciun ordin (Req 13.5, 13.6, 13.11).

Pentru orice traiectorie de observații contabile (capital, retrageri, depuneri, PnL zilnic pe
mai multe `Trading_Day`), intercalată cu `Order_Intent` (intrări BUY și ieșiri SELL) într-un
context altfel aprobabil, încercări de „aprobare” a operatorului (autorizări invalide sau care
refolosesc configurația curentă ori una retrasă) și reporniri (`export_state` → JSON →
`LossMonitor(state=...)`):

    odată ce equity + retrageri − depuneri ≤ 90 EUR, `RiskEngine` nu mai aprobă niciun ordin
    în aceeași configurație de capital — nici după revenirea capitalului, nici după reporniri.

Oracolul ține independent evidența depășirii pragului pe configurație. Înainte de depășire
proprietatea nu constrânge: singurul blocaj permis este domeniul DAY. O autorizare validă
pentru o configurație nouă (niciodată folosită) este singura cale de reluare și demonstrează
că testul nu este vacuu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from qts.config.schema import RiskConfig
from qts.core.models import Instrument, OrderIntent, Quote
from qts.core.money import ZERO
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import (
    CompleteCostModel,
    CostContext,
    CostModelConfig,
    FxConfig,
    LatencyConfig,
    SlippageConfig,
    TaxConfig,
)
from qts.risk import (
    KillSwitchScope,
    KillSwitchState,
    MarketSnapshot,
    PositionRisk,
    RejectReason,
    RiskContext,
    RiskEngine,
)
from qts.risk.monitor import (
    CapitalConfigChangeError,
    KillSwitchActivation,
    LossMonitor,
    LossMonitorState,
    LossObservation,
)

D = Decimal
SYMBOL = "XYZ"
T0 = datetime(2026, 3, 2, 8, 0, tzinfo=UTC)
CONFIG = RiskConfig()
FLOOR = CONFIG.reference_capital_eur - CONFIG.total_loss_limit_eur  # 90 EUR


def _dec(lo: str, hi: str, places: int = 2) -> st.SearchStrategy[Decimal]:
    return st.decimals(
        min_value=D(lo), max_value=D(hi), places=places, allow_nan=False, allow_infinity=False
    )


# --------------------------------------------------------------------------- context aprobabil

_INST = Instrument.model_validate(
    {
        "symbol": SYMBOL,
        "venue": "XETR",
        "asset_class": "etf",
        "currency": "EUR",
        "tick_size": "0.01",
        "qty_step": "0.001",
        "min_qty": "0.001",
        "calendar_id": "XETR",
        "fractional": True,
    }
)
_COSTS = CompleteCostModel(
    CostModelConfig(
        version="costs-p3",
        commissions=CommissionSchedule(
            tables=(
                CommissionTable.model_validate(
                    {
                        "broker": "sim",
                        "version": "v1",
                        "valid_from": datetime(2024, 1, 1, tzinfo=UTC),
                        "currency": "EUR",
                        "percent": "0",
                        "exchange_fee_fixed": "0.05",
                    }
                ),
            )
        ),
        slippage=SlippageConfig(k=D(0)),
        latency=LatencyConfig(latency_ms=0),
        fx=FxConfig(conversion_spread=D(0)),
        taxes=TaxConfig(approved=True, reference="test"),
    )
)
ENGINE = RiskEngine(CONFIG, _COSTS)


def _ctx(ts: datetime, kill_switch: KillSwitchState) -> RiskContext:
    """Context favorabil: date proaspete, fără pierderi, o poziție XYZ deținută (pentru SELL)."""
    snap = MarketSnapshot(
        instrument=_INST,
        data_fresh=True,
        quote=Quote(instrument=SYMBOL, ts=ts, bid=D("9.99"), ask=D("10.01")),
        cost_ctx=CostContext(broker="sim", sigma_bar=D("0.01"), adv=D(1000)),
    )
    pos = PositionRisk(
        instrument=SYMBOL,
        qty=D(1),
        mark_price=D(10),
        stop_price=D("9.5"),
        fx_rate_to_eur=D(1),
        exit_cost_eur=D("0.05"),
    )
    return RiskContext(
        ts=ts,
        mode="backtest",
        cash_eur=D(100),
        positions={SYMBOL: pos},
        market={SYMBOL: snap},
        kill_switch=kill_switch,
    )


def _intent(n: int, side: str, requested: Decimal | None) -> OrderIntent:
    data: dict[str, Any] = {
        "intent_id": f"i{n}",
        "signal_id": f"s{n}",
        "instrument": SYMBOL,
        "side": side,
        "ref_price": "10",
        "stop_price": "9.5" if side == "BUY" else None,
        "requested_qty": requested,
    }
    return OrderIntent.model_validate(data)


# --------------------------------------------------------------------------- colaboratori


class RecordingSink:
    """Sink care poate eșua la cerere; activările rămân atunci în `pending` (fail-closed)."""

    def __init__(self) -> None:
        self.received: list[KillSwitchActivation] = []
        self.fail = False

    def activate(self, activation: KillSwitchActivation) -> None:
        if self.fail:
            raise RuntimeError("sink indisponibil")
        self.received.append(activation)


@dataclass(frozen=True)
class Authorization:
    new_capital_config_id: str
    valid: bool = True

    def is_valid(self) -> bool:
        return self.valid


class NotAnAuthorization:
    """Obiect care nu respectă protocolul `CapitalChangeAuthorization`."""

    new_capital_config_id = "cfg-forged"


# --------------------------------------------------------------------------- pași


@dataclass(frozen=True)
class Observe:
    advance_min: int
    equity: Decimal
    withdraw: Decimal
    deposit: Decimal
    realized: Decimal
    unrealized: Decimal
    sink_fails: bool


@dataclass(frozen=True)
class Intent:
    side: str
    requested: Decimal | None


@dataclass(frozen=True)
class Restart:
    pass


@dataclass(frozen=True)
class BadApproval:
    kind: str  # invalid | same | retired | blank | forged


@dataclass(frozen=True)
class NewConfig:
    pass


Step = Observe | Intent | Restart | BadApproval | NewConfig

_observe = st.builds(
    Observe,
    # Uneori trece la Trading_Day următor (24 h), de obicei câteva minute.
    advance_min=st.sampled_from([0, 1, 5, 15, 15, 60, 60 * 24]),
    equity=_dec("87", "110"),
    withdraw=st.sampled_from([ZERO] * 6 + [D("0.5"), D("2"), D("5")]),
    deposit=st.sampled_from([ZERO] * 6 + [D("0.5"), D("3"), D("10")]),
    realized=_dec("-2.5", "1"),
    unrealized=_dec("-1", "1"),
    sink_fails=st.sampled_from([False] * 9 + [True]),
)
_intent_step = st.builds(
    Intent,
    side=st.sampled_from(["BUY", "BUY", "SELL"]),
    requested=st.none() | st.sampled_from([D("0.1"), D("0.5"), D("1")]),
)
_single_step = st.one_of(
    _observe,
    _observe,
    _observe,
    _intent_step,
    _intent_step,
    _intent_step,
    st.just(Restart()),
    st.builds(BadApproval, st.sampled_from(["invalid", "same", "retired", "blank", "forged"])),
    st.just(NewConfig()),
)
# Rafală „șoc de capital”: o observație cu equity sub/în jurul pragului (valori discrete, ca
# distribuția să nu depindă de constantele colectate de Hypothesis din modulele importate),
# urmată imediat de câteva intenții. Retragerile cumulate pot încă menține capitalul ajustat
# peste prag, deci rafala nu garantează depășirea — doar o face frecventă.
_shock_observe = st.builds(
    Observe,
    advance_min=st.sampled_from([0, 1, 5, 15, 60, 60 * 24]),
    equity=st.sampled_from([D("80"), D("84.5"), D("87"), D("89"), D("90"), D("91.5"), D("95")]),
    withdraw=st.sampled_from([ZERO] * 4 + [D("0.5"), D("2")]),
    deposit=st.sampled_from([ZERO] * 4 + [D("0.5"), D("3")]),
    realized=_dec("-2.5", "1"),
    unrealized=_dec("-1", "1"),
    sink_fails=st.sampled_from([False] * 9 + [True]),
)
_shock_burst = st.tuples(_shock_observe, st.lists(_intent_step, min_size=1, max_size=3)).map(
    lambda t: [t[0], *t[1]]
)
_chunk = st.one_of(
    _single_step.map(lambda s: [s]),
    _single_step.map(lambda s: [s]),
    _single_step.map(lambda s: [s]),
    _shock_burst,
)
trajectories = st.lists(_chunk, min_size=1, max_size=30).map(
    lambda chunks: [step for chunk in chunks for step in chunk][:60]
)


# --------------------------------------------------------------------------- simulare


@dataclass
class Stats:
    breached: bool = False
    approved_before_breach: int = 0
    rejected_after_breach: int = 0
    rejected_after_recovery: int = 0
    rejected_after_restart: int = 0
    approved_after_new_config: int = 0
    refused_approvals_after_breach: int = 0


@dataclass
class Harness:
    sink: RecordingSink = field(default_factory=RecordingSink)
    config_n: int = 0
    used_ids: list[str] = field(default_factory=lambda: ["cfg-0"])
    ts: datetime = T0
    withdrawals: Decimal = ZERO
    deposits: Decimal = ZERO
    # Oracol independent: configurația curentă a atins pragul de 90 EUR.
    breached: bool = False
    recovered_since_breach: bool = False
    restarted_since_breach: bool = False
    any_new_config: bool = False
    intents: int = 0
    stats: Stats = field(default_factory=Stats)
    monitor: LossMonitor = field(init=False)

    def __post_init__(self) -> None:
        self.monitor = LossMonitor(CONFIG, self.sink, capital_config_id=self.config_id)

    @property
    def config_id(self) -> str:
        return f"cfg-{self.config_n}"

    def run(self, step: Step) -> None:
        match step:
            case Observe():
                self._observe(step)
            case Intent():
                self._intent(step)
            case Restart():
                self._restart()
            case BadApproval():
                self._bad_approval(step)
            case NewConfig():
                self._new_config()

    def _observe(self, s: Observe) -> None:
        self.ts += timedelta(minutes=s.advance_min)
        self.withdrawals += s.withdraw
        self.deposits += s.deposit
        obs = LossObservation(
            ts=self.ts,
            daily_realized_pnl_eur=s.realized,
            daily_unrealized_pnl_eur=s.unrealized,
            equity_eur=s.equity,
            cumulative_withdrawals_eur=self.withdrawals,
            cumulative_deposits_eur=self.deposits,
        )
        self.sink.fail = s.sink_fails
        try:
            self.monitor.observe(obs)
        except RuntimeError:
            assert s.sink_fails
        finally:
            self.sink.fail = False
        adjusted = s.equity + self.withdrawals - self.deposits
        if adjusted <= FLOOR:
            self.breached = True
            self.stats.breached = True
        elif self.breached:
            self.recovered_since_breach = True

    def _intent(self, s: Intent) -> None:
        self.intents += 1
        intent = _intent(self.intents, s.side, s.requested)
        # Fără Kill_Switch, contextul este aprobabil: singurul motiv de respingere e Kill_Switch.
        baseline = ENGINE.evaluate(intent, _ctx(self.ts, KillSwitchState()))
        assert baseline.approved, baseline
        ks = self.monitor.kill_switch_state(self.ts)
        decision = ENGINE.evaluate(intent, _ctx(self.ts, ks))
        if self.breached:
            assert ks.capital_config_active, "limita totală atinsă, dar Kill_Switch inactiv"
            assert ks.blocking_scope(SYMBOL) is KillSwitchScope.CAPITAL_CONFIG
            assert not decision.approved, f"aprobat după limita totală: {decision}"
            assert decision.reason is RejectReason.KILL_SWITCH_ACTIVE
            self.stats.rejected_after_breach += 1
            if self.recovered_since_breach:
                self.stats.rejected_after_recovery += 1
            if self.restarted_since_breach:
                self.stats.rejected_after_restart += 1
            event("intent:rejected_after_breach")
        else:
            # Înainte de depășire proprietatea nu constrânge; numai domeniul DAY poate bloca.
            assert not ks.capital_config_active, "Kill_Switch CAPITAL_CONFIG activ fără depășire"
            if decision.approved:
                self.stats.approved_before_breach += 1
                if self.any_new_config:
                    self.stats.approved_after_new_config += 1
                event("intent:approved_before_breach")
            else:
                assert decision.reason is RejectReason.KILL_SWITCH_ACTIVE
                assert ks.blocking_scope(SYMBOL) is KillSwitchScope.DAY
                event("intent:rejected_day")

    def _restart(self) -> None:
        raw = self.monitor.export_state().model_dump_json()
        state = LossMonitorState.model_validate_json(raw)
        self.monitor = LossMonitor(
            CONFIG, self.sink, capital_config_id=state.capital_config_id, state=state
        )
        if self.breached:
            self.restarted_since_breach = True

    def _bad_approval(self, s: BadApproval) -> None:
        auth: Any
        match s.kind:
            case "invalid":
                auth = Authorization(f"cfg-{len(self.used_ids)}", valid=False)
            case "same":
                auth = Authorization(self.config_id)
            case "retired":
                if len(self.used_ids) < 2:
                    auth = Authorization(self.config_id)
                else:
                    auth = Authorization(self.used_ids[0])
            case "blank":
                auth = Authorization("   ")
            case _:
                auth = NotAnAuthorization()
        before = self.monitor.export_state()
        with pytest.raises(CapitalConfigChangeError):
            self.monitor.start_new_capital_config(auth)
        assert self.monitor.export_state() == before
        if self.breached:
            self.stats.refused_approvals_after_breach += 1

    def _new_config(self) -> None:
        new_id = f"cfg-{len(self.used_ids)}"
        self.monitor.start_new_capital_config(Authorization(new_id))
        self.used_ids.append(new_id)
        self.config_n = len(self.used_ids) - 1
        assert self.monitor.export_state().capital_config_id == new_id
        # Configurație nouă: contabilitate și oracol pornesc de la zero.
        self.withdrawals = ZERO
        self.deposits = ZERO
        self.breached = False
        self.recovered_since_breach = False
        self.restarted_since_breach = False
        self.any_new_config = True


def _simulate(steps: list[Step]) -> Harness:
    h = Harness()
    for step in steps:
        h.run(step)
    # Activarea permanentă a fost emisă cel mult o dată pe configurație (idempotentă).
    ids = [a.activation_id for a in h.sink.received if a.scope is KillSwitchScope.CAPITAL_CONFIG]
    assert len(ids) == len(set(ids))
    return h


# --------------------------------------------------------------------------- proprietăți


@given(trajectories)
def test_property_3_no_approval_after_total_limit_in_same_capital_config(
    steps: list[Step],
) -> None:
    """**Validates: Requirements 13.5, 13.6, 13.11**"""
    h = _simulate(steps)
    event("breached" if h.stats.breached else "not_breached")


def test_property_3_generator_is_not_vacuous() -> None:
    """Depășiri frecvente, aprobări înainte de depășire și reluare după configurație nouă.

    **Validates: Requirements 13.5, 13.6, 13.11**
    """
    collected: list[Stats] = []

    @settings(max_examples=300, derandomize=True, database=None)
    @given(trajectories)
    def run(steps: list[Step]) -> None:
        collected.append(_simulate(steps).stats)

    run()
    n = len(collected)
    assert n >= 100
    assert sum(s.breached for s in collected) >= n // 5
    assert sum(s.approved_before_breach > 0 for s in collected) >= n // 5
    assert sum(s.rejected_after_breach > 0 for s in collected) >= n // 10
    assert any(s.rejected_after_recovery for s in collected)
    assert any(s.rejected_after_restart for s in collected)
    assert any(s.refused_approvals_after_breach for s in collected)
    assert any(s.approved_after_new_config for s in collected)
