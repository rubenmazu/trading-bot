"""Teste unitare pentru `LossMonitor` (Req 13.4, 13.6, 13.11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

from qts.config.schema import RiskConfig
from qts.risk.context import KillSwitchScope, KillSwitchState
from qts.risk.monitor import (
    CapitalConfigChangeError,
    KillSwitchActivation,
    LossMonitor,
    LossMonitorState,
    LossObservation,
    LossTriggerReason,
    zoned_trading_day,
)

D = Decimal
T0 = datetime(2024, 3, 4, 10, 0, tzinfo=UTC)


@dataclass
class RecordingSink:
    received: list[KillSwitchActivation] = field(default_factory=list)
    fail: bool = False

    def activate(self, activation: KillSwitchActivation) -> None:
        if self.fail:
            raise RuntimeError("sink indisponibil")
        self.received.append(activation)


@dataclass(frozen=True)
class FakeAuthorization:
    new_capital_config_id: str
    valid: bool = True

    def is_valid(self) -> bool:
        return self.valid


def obs(
    ts: datetime = T0,
    *,
    realized: str = "0",
    unrealized: str = "0",
    equity: str = "100",
    withdrawals: str = "0",
    deposits: str = "0",
) -> LossObservation:
    return LossObservation(
        ts=ts,
        daily_realized_pnl_eur=D(realized),
        daily_unrealized_pnl_eur=D(unrealized),
        equity_eur=D(equity),
        cumulative_withdrawals_eur=D(withdrawals),
        cumulative_deposits_eur=D(deposits),
    )


def make(sink: RecordingSink | None = None, **kw: object) -> tuple[LossMonitor, RecordingSink]:
    sink = sink or RecordingSink()
    cfg = RiskConfig(daily_loss_limit_eur=D("2"))
    return LossMonitor(cfg, sink, capital_config_id="cap-1", **kw), sink  # type: ignore[arg-type]


# ------------------------------------------------------------------ zilnic (13.4)


def test_below_daily_limit_does_nothing() -> None:
    mon, sink = make()
    assert mon.observe(obs(realized="-1.2", unrealized="-0.79", equity="98.01")) == ()
    assert sink.received == []
    assert not mon.kill_switch_state(T0).day_active


def test_daily_limit_reached_activates_day_once() -> None:
    mon, sink = make()
    first = mon.observe(obs(realized="-1.5", unrealized="-0.5", equity="98"))
    assert len(first) == 1
    act = first[0]
    assert act.scope is KillSwitchScope.DAY
    assert act.reason is LossTriggerReason.DAILY_LOSS_LIMIT_REACHED
    assert act.value == D("2") and act.limit == D("2")
    assert act.trading_day == date(2024, 3, 4) and not act.permanent
    # Idempotent în aceeași zi, chiar dacă pierderea crește.
    assert mon.observe(obs(T0 + timedelta(hours=1), realized="-3", equity="97")) == ()
    assert sink.received == [act]
    assert mon.kill_switch_state(T0 + timedelta(hours=2)).day_active


def test_day_scope_expires_next_trading_day_and_can_retrigger() -> None:
    mon, sink = make()
    mon.observe(obs(realized="-2", equity="98"))
    next_day = T0 + timedelta(days=1)
    assert not mon.kill_switch_state(next_day).day_active
    assert mon.observe(obs(next_day, realized="-0.5", equity="97.5")) == ()
    again = mon.observe(obs(next_day + timedelta(hours=1), unrealized="-2.1", equity="95.4"))
    assert len(again) == 1 and again[0].trading_day == date(2024, 3, 5)
    assert again[0].activation_id != sink.received[0].activation_id


def test_late_observation_from_past_day_does_not_activate() -> None:
    mon, sink = make()
    mon.observe(obs(T0 + timedelta(days=1)))
    assert mon.observe(obs(T0, realized="-5", equity="95")) == ()
    assert sink.received == []


def test_zoned_trading_day_uses_market_timezone() -> None:
    # 23:30 UTC = 01:30 în UTC+2: deja ziua următoare a pieței.
    fn = zoned_trading_day(timezone(timedelta(hours=2)))
    assert fn(datetime(2024, 3, 4, 23, 30, tzinfo=UTC)) == date(2024, 3, 5)
    rolled = zoned_trading_day(timezone(timedelta(hours=2)), rollover=time(2))
    assert rolled(datetime(2024, 3, 4, 23, 30, tzinfo=UTC)) == date(2024, 3, 4)
    mon, _ = make(trading_day=fn)
    mon.observe(obs(datetime(2024, 3, 4, 23, 30, tzinfo=UTC), realized="-2", equity="98"))
    assert mon.export_state().day_activation is not None
    assert mon.export_state().day_activation.trading_day == date(2024, 3, 5)  # type: ignore[union-attr]


# ------------------------------------------------------------------ total (13.6, 13.11)


def test_capital_floor_is_reference_minus_total_limit() -> None:
    mon, _ = make()
    assert mon.capital_floor_eur == D("90")


def test_adjusted_capital_at_floor_activates_permanent_kill_switch() -> None:
    mon, sink = make()
    assert mon.observe(obs(equity="90.01")) == ()
    acts = mon.observe(obs(T0 + timedelta(minutes=1), equity="90"))
    assert [a.scope for a in acts] == [KillSwitchScope.CAPITAL_CONFIG]
    act = acts[0]
    assert act.permanent and act.capital_config_id == "cap-1"
    assert act.reason is LossTriggerReason.TOTAL_LOSS_LIMIT_REACHED
    assert act.value == D("90") and act.limit == D("90")
    # Nu expiră la ziua următoare, nici când capitalul își revine; nu se reemite.
    later = T0 + timedelta(days=5)
    assert mon.observe(obs(later, equity="120")) == ()
    assert mon.kill_switch_state(later).capital_config_active
    assert sink.received == [act]


def test_withdrawals_and_deposits_adjust_capital() -> None:
    mon, _ = make()
    # 85 + 10 retras = 95 > 90: fără activare.
    assert mon.observe(obs(equity="85", withdrawals="10")) == ()
    # 100 - 15 depus = 85 ≤ 90: depunerea nu maschează pierderea.
    acts = mon.observe(obs(equity="100", deposits="15"))
    assert acts[0].scope is KillSwitchScope.CAPITAL_CONFIG


def test_both_scopes_can_trigger_on_same_observation() -> None:
    mon, sink = make()
    acts = mon.observe(obs(realized="-10", equity="90"))
    assert {a.scope for a in acts} == {KillSwitchScope.DAY, KillSwitchScope.CAPITAL_CONFIG}
    assert len(sink.received) == 2


def test_kill_switch_state_merges_with_base() -> None:
    mon, _ = make()
    mon.observe(obs(equity="89"))
    base = KillSwitchState(global_active=True, instruments=frozenset({"ABC"}))
    st = mon.kill_switch_state(T0, base)
    assert st.global_active and st.capital_config_active and st.instruments == {"ABC"}
    assert st.blocking_scope("XYZ") is KillSwitchScope.CAPITAL_CONFIG


def test_state_round_trip_keeps_permanent_activation() -> None:
    mon, _ = make()
    mon.observe(obs(realized="-3", equity="89"))
    raw = mon.export_state().model_dump_json()
    restored_state = LossMonitorState.model_validate_json(raw)
    assert restored_state == mon.export_state()
    sink2 = RecordingSink()
    mon2, _ = make(sink2, state=restored_state)
    assert mon2.kill_switch_state(T0).capital_config_active
    assert mon2.kill_switch_state(T0).day_active
    assert mon2.observe(obs(T0 + timedelta(minutes=5), realized="-4", equity="88")) == ()
    assert sink2.received == []


def test_restore_with_mismatched_config_id_fails() -> None:
    mon, _ = make()
    state = mon.export_state().model_copy(update={"capital_config_id": "other"})
    with pytest.raises(ValueError):
        make(state=state)


def test_new_capital_config_requires_valid_new_authorization() -> None:
    mon, _ = make()
    mon.observe(obs(equity="89"))
    with pytest.raises(CapitalConfigChangeError):
        mon.start_new_capital_config(FakeAuthorization("cap-2", valid=False))
    with pytest.raises(CapitalConfigChangeError):
        mon.start_new_capital_config(FakeAuthorization("cap-1"))  # aprobare în config existentă
    with pytest.raises(CapitalConfigChangeError):
        mon.start_new_capital_config(object())  # type: ignore[arg-type]
    assert mon.kill_switch_state(T0).capital_config_active

    mon.start_new_capital_config(FakeAuthorization("cap-2"))
    st = mon.export_state()
    assert st.capital_config_id == "cap-2" and not st.capital_config_active
    assert "cap-1" in st.retired_capital_config_ids
    with pytest.raises(CapitalConfigChangeError):
        mon.start_new_capital_config(FakeAuthorization("cap-1"))  # nu se poate reveni


def test_new_capital_config_keeps_day_scope() -> None:
    mon, _ = make()
    mon.observe(obs(realized="-10", equity="90"))
    mon.start_new_capital_config(FakeAuthorization("cap-2"))
    assert mon.kill_switch_state(T0).day_active


# ------------------------------------------------------------------ sink fail-closed


def test_sink_failure_keeps_state_active_and_retries() -> None:
    sink = RecordingSink(fail=True)
    mon, _ = make(sink)
    with pytest.raises(RuntimeError):
        mon.observe(obs(equity="89"))
    assert mon.kill_switch_state(T0).capital_config_active
    assert len(mon.export_state().pending) == 1
    sink.fail = False
    mon.flush()
    assert [a.scope for a in sink.received] == [KillSwitchScope.CAPITAL_CONFIG]
    assert mon.export_state().pending == ()


def test_negative_cumulative_flows_rejected() -> None:
    with pytest.raises(ValueError):
        obs(withdrawals="-1")
