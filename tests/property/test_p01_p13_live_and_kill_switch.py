"""P1 și P13: zero ordine Live în Initial_Stage; zero ordine noi sub un Kill_Switch activ.

P1 (Req 2.1, 2.4, 2.6). Pentru orice secvență generată de configurații, surse de etapă și comenzi,
cu etapa `initial` (inclusiv `stage.lock` lipsă, corupt sau ilizibil, tratat fail-closed ca
`initial`), numărul de apeluri `submit` către un adaptor cu `environment = "live"` este 0.
Rutele încercate:

a) `build_broker` pentru orice configurație (toate modurile și tipurile de broker, endpoint-uri
   și conturi live cunoscute cu variații de majuscule și spații), atât validată prin schemă, cât
   și construită ocolind schema (simulează un defect al validării): fabrica nu construiește și
   nu întoarce niciodată un adaptor live, iar `qts_live` nu este importat;
b) un adaptor-spion live învelit direct în `FailSafeBlock` cu orice `ApprovedTarget` (inclusiv
   `live` cu contul exact al spionului) și orice `Live_Gate` (implicit, cu toți furnizorii
   satisfăcuți, rău-intenționat care întoarce mereu True, care întoarce o valoare „truthy”,
   care ridică excepții): `submit` nu ajunge niciodată la spion;
c) eșecurile sink-ului de audit sau ale ceasului nu lasă cererea să treacă; fiecare tentativă
   blocată este auditată sau rămâne în `unaudited`.

P13 (Req 14.1, 14.2, 14.6). Cât timp un domeniu Kill_Switch aplicabil este activ, numărul de
ordine noi transmise pe acel domeniu este 0. Se generează secvențe de activări manuale și
automate (INSTRUMENT, GLOBAL, DAY), activări din `LossMonitor` (DAY la limita zilnică,
CAPITAL_CONFIG la pragul de capital), reluări valide și invalide, expirări la schimbarea
`Trading_Day`, înlocuirea configurației de capital și reporniri (reconstrucție din jurnal),
intercalate cu ordine pe instrumente arbitrare. Un oracol independent ține domeniile active;
la fiecare ordin: dacă un domeniu se aplică, spionul nu primește apelul; altfel ordinul trece
(non-vacuitate în ambele sensuri).
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from datetime import time as dtime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from qts.broker.factory import (
    BrokerFactoryError,
    LiveBrokerRefusedError,
    build_broker,
)
from qts.broker.fail_safe import (
    ApprovedTarget,
    FailSafeBlock,
    FailSafeReason,
    FailSafeRejectedError,
)
from qts.broker.fake import FakeBroker
from qts.broker.sim import SimBroker
from qts.config.schema import AppConfig, BrokerConfig, RiskConfig
from qts.core.clock import SimClock
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import CompleteCostModel, CostModelConfig
from qts.risk.context import KillSwitchScope
from qts.risk.monitor import (
    KillSwitchActivation,
    LossMonitor,
    LossObservation,
    TradingDayFn,
    utc_trading_day,
    zoned_trading_day,
)
from qts.safety.kill_switch import KillSwitch, KillSwitchEvent, ResumeRequest
from qts.safety.live_gate import ConditionCheck, LiveGate, LiveGateCondition
from qts.safety.stage import ProjectStage, StageInfo, StartupRefusedError, read_stage
from tests.helpers import config_dict

D = Decimal
T0 = datetime(2026, 3, 2, 10, tzinfo=UTC)
INITIAL = StageInfo(ProjectStage.INITIAL, "stage.lock")
SPY_LIVE_ACCOUNT = "U7654321"


# =========================================================================== dubluri comune


@dataclass(frozen=True)
class _Req:
    client_order_id: str
    instrument: str


class _MemStore:
    """`KillSwitchStore` în memorie (persistența reală este testată separat)."""

    def __init__(self) -> None:
        self.events: list[KillSwitchEvent] = []

    def append(self, event: KillSwitchEvent) -> None:
        self.events.append(event)

    def load(self) -> list[KillSwitchEvent]:
        return list(self.events)


class _Audit:
    """Sink de audit cu moduri de eșec: ok, fail_once, fail_always, alternate."""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.calls = 0
        self.records: list[dict[str, Any]] = []

    def append(self, **kwargs: Any) -> object:
        self.calls += 1
        failing = (
            self.mode == "fail_always"
            or (self.mode == "fail_once" and self.calls == 1)
            or (self.mode == "alternate" and self.calls % 2 == 1)
        )
        if failing:
            raise sqlite3.OperationalError("database is locked")
        self.records.append(kwargs)
        return None


class _RaisingClock:
    def now(self) -> datetime:
        raise RuntimeError("ceas indisponibil")


@dataclass
class _LiveSpy:
    """Adaptor live fals: numără orice apel `submit` care ar ajunge la rețea."""

    environment: str = "live"
    account_id: str | None = SPY_LIVE_ACCOUNT
    submit_calls: int = 0
    cancelled: list[str] = field(default_factory=list)

    def submit(self, req: _Req, /) -> str:
        self.submit_calls += 1
        return f"LIVE-ack:{req.client_order_id}"

    def cancel(self, client_order_id: str) -> str:
        self.cancelled.append(client_order_id)
        return f"cancel:{client_order_id}"


@dataclass
class _SimSpy:
    """Adaptor simulat fals: numără apelurile `submit` pe instrument."""

    environment: str = "sim"
    account_id: str | None = "SIM-LOCAL"
    calls: Counter[str] = field(default_factory=Counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def submit(self, req: _Req, /) -> str:
        with self.lock:
            self.calls[req.instrument] += 1
        return f"ack:{req.client_order_id}"


class _AlwaysOpenGate:
    def is_open(self) -> bool:
        return True


class _TruthyGate:
    def is_open(self) -> Any:
        return "yes"


class _RaisingGate:
    def is_open(self) -> bool:
        raise RuntimeError("poartă defectă")


def _gate(kind: str, stage: StageInfo) -> Any:
    if kind == "none":
        return None
    if kind == "default":
        return LiveGate(stage=lambda: stage)
    if kind == "all_providers":
        providers: dict[LiveGateCondition, Callable[[], ConditionCheck]] = {
            c: (lambda: ConditionCheck(satisfied=True, detail="ok"))
            for c in LiveGateCondition
            if c is not LiveGateCondition.STAGE_NOT_EXITED
        }
        return LiveGate(stage=lambda: stage, providers=providers)
    if kind == "always_open":
        return _AlwaysOpenGate()
    if kind == "truthy":
        return _TruthyGate()
    return _RaisingGate()


GATE_KINDS = ("none", "default", "all_providers", "always_open", "truthy", "raising")


# =========================================================================== surse de etapă

# None = conținutul trebuie refuzat de `read_stage` (StartupRefusedError).
STAGE_SOURCES: dict[str, tuple[bytes | None, bool]] = {
    # nume: (conținut stage.lock sau None = lipsă, refuzat?)
    "missing": (None, False),
    "directory": (None, False),  # calea este un director → OSError → fail-closed
    "initial": (b'project_stage = "initial"\n', False),
    "corrupt_toml": (b"project_stage = [[[", False),
    "undecodable": (b"\xff\xfe\x00\x81project_stage", False),
    "empty": (b"", True),  # TOML valid, cheie lipsă → refuzat
    "live": (b'project_stage = "live"\n', True),
    "upper": (b'project_stage = "INITIAL"\n', True),
}


def _read_stage(kind: str) -> StageInfo | None:
    """Etapa citită din `stage.lock` real; None dacă pornirea este refuzată (verificat)."""
    content, refused = STAGE_SOURCES[kind]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stage.lock"
        if kind == "directory":
            path.mkdir()
        elif content is not None:
            path.write_bytes(content)
        if refused:
            with pytest.raises(StartupRefusedError):
                read_stage(path)
            return None
        info = read_stage(path)
    assert info.stage is ProjectStage.INITIAL
    return info


# =========================================================================== P1a: fabrica


# (valoare, este endpoint live cunoscut?)
ENDPOINTS: tuple[tuple[str | None, bool], ...] = (
    (None, False),
    ("https://api.alpaca.markets", True),
    ("HTTPS://API.ALPACA.MARKETS/v2/orders", True),
    ("  http://api.alpaca.markets  ", True),
    ("ibkr-gw:7496", True),
    ("TCP://tws.local:4001", True),
    (" 10.0.0.5:7496 ", True),
    ("https://paper-api.alpaca.markets", False),
    ("127.0.0.1:7497", False),
    ("localhost:4002", False),
    ("fake://local", False),
    ("FAKE://Local", False),
    (" fake://local", False),
    ("https://example.test", False),
    ("", False),
)
ACCOUNTS: tuple[tuple[str | None, bool], ...] = (
    (None, False),
    ("U1234567", True),
    ("u42", True),
    (" U9 ", True),
    ("DU1234567", False),
    ("SIM-LOCAL", False),
    ("FAKE-DEMO-1", False),
    ("LIVE1", False),
    ("", False),
)


@dataclass(frozen=True)
class BuildCase:
    stage_kind: str
    environment: str
    kind: str
    name: str | None
    endpoint: str | None
    endpoint_live: bool
    account: str | None
    account_live: bool
    secret_ref: str | None
    bypass_schema: bool
    gate: str


@st.composite
def build_cases(draw: st.DrawFn) -> BuildCase:
    endpoint, endpoint_live = draw(
        st.one_of(
            st.sampled_from(ENDPOINTS),
            st.text("abcxyz.", max_size=8).map(lambda s: (s, False)),
        )
    )
    account, account_live = draw(st.sampled_from(ACCOUNTS))
    return BuildCase(
        stage_kind=draw(st.sampled_from(sorted(STAGE_SOURCES))),
        environment=draw(st.sampled_from(["backtest", "shadow", "demo", "live"])),
        kind=draw(st.sampled_from(["sim", "fake", "demo", "live"])),
        name=draw(st.sampled_from([None, "alpaca", "ibkr", "fake", "other"])),
        endpoint=endpoint,
        endpoint_live=endpoint_live,
        account=account,
        account_live=account_live,
        secret_ref=draw(st.sampled_from([None, "qts/demo/broker", "qts/live/broker"])),
        bypass_schema=draw(st.booleans()),
        gate=draw(st.sampled_from(GATE_KINDS)),
    )


def _cost_model() -> CompleteCostModel:
    table = CommissionTable(
        broker="sim",
        version="v1",
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        currency="EUR",
        percent=D("0.001"),
        minimum=D("1"),
    )
    return CompleteCostModel(
        CostModelConfig(version="costs-v1", commissions=CommissionSchedule(tables=(table,)))
    )


COST_MODEL = _cost_model()
_BASE_CONFIG = AppConfig.model_validate(config_dict())


def _config(case: BuildCase) -> AppConfig | None:
    broker: dict[str, Any] = {"kind": case.kind}
    for key, value in (
        ("name", case.name),
        ("endpoint", case.endpoint),
        ("account_id", case.account),
        ("secret_ref", case.secret_ref),
    ):
        if value is not None:
            broker[key] = value
    if case.bypass_schema:
        # Ocolește validarea schemei: fabrica trebuie să rămână sigură și fără ea.
        return _BASE_CONFIG.model_copy(
            update={
                "environment": case.environment,
                "broker": BrokerConfig.model_construct(
                    kind=cast(Any, case.kind),  # intenționat în afara schemei
                    name=case.name,
                    endpoint=case.endpoint,
                    account_id=case.account,
                    secret_ref=case.secret_ref,
                ),
            }
        )
    try:
        return AppConfig.model_validate(config_dict(environment=case.environment, broker=broker))
    except ValidationError:
        return None


def run_build_case(case: BuildCase) -> str:
    """Verifică P1 pe ruta fabricii; întoarce eticheta rezultatului."""
    stage = _read_stage(case.stage_kind)
    if stage is None:
        return "stage_refused"
    cfg = _config(case)
    if cfg is None:
        return "schema_refused"
    clock = SimClock(T0)
    ks = KillSwitch(_MemStore(), clock=clock)
    try:
        block = build_broker(
            cfg,
            stage,
            clock=clock,
            kill_switch=ks,
            audit=_Audit(),
            cost_model=COST_MODEL,
            live_gate=_gate(case.gate, stage),
        )
    except LiveBrokerRefusedError:
        assert cfg.environment == "live" or cfg.broker.kind == "live"
        return "live_refused"
    except (StartupRefusedError, BrokerFactoryError) as exc:
        assert cfg.environment != "live" and cfg.broker.kind != "live", (
            f"Live trebuia refuzat cu LiveBrokerRefusedError, nu {type(exc).__name__}"
        )
        return (
            "live_endpoint_or_account_refused"
            if (case.endpoint_live or case.account_live)
            else f"refused:{type(exc).__name__}"
        )
    # A fost construit un adaptor: nu poate fi live și nu poate proveni dintr-o cerere live.
    assert cfg.environment != "live" and cfg.broker.kind != "live", cfg
    assert not case.endpoint_live, f"endpoint live acceptat: {case.endpoint!r}"
    assert not case.account_live, f"cont live acceptat: {case.account!r}"
    assert isinstance(block, FailSafeBlock)
    assert isinstance(block.inner, SimBroker | FakeBroker), type(block.inner)
    assert block.environment in ("sim", "demo")
    return f"built:{type(block.inner).__name__}"


@given(build_cases())
def test_property_1_factory_never_builds_live_adapter(case: BuildCase) -> None:
    """**Validates: Requirements 2.1, 2.4, 2.6**"""
    event(run_build_case(case))
    assert "qts_live" not in sys.modules


# =========================================================================== P1b/c: Fail_Safe_Block


@dataclass(frozen=True)
class Cmd:
    kind: str  # submit | cancel | ks_global | ks_instr | flush
    coid: str = ""
    instrument: str = ""


_instruments = st.sampled_from(["AAA", "BBB", "XYZ", "", " "])
_coids = st.one_of(st.sampled_from(["", " "]), st.text("abc123-", min_size=1, max_size=6))


@st.composite
def commands(draw: st.DrawFn) -> Cmd:
    kind = draw(st.sampled_from(["submit", "submit", "submit", "cancel", "ks_global", "ks_instr"]))
    if draw(st.integers(0, 9)) == 0:
        kind = "flush"
    return Cmd(kind, draw(_coids), draw(_instruments))


@dataclass(frozen=True)
class BlockCase:
    stage_kind: str
    approved_env: str
    approved_account: str | None
    gate: str
    audit_mode: str
    clock_raises: bool
    commands: tuple[Cmd, ...]


@st.composite
def block_cases(draw: st.DrawFn) -> BlockCase:
    return BlockCase(
        stage_kind=draw(st.sampled_from([k for k, (_, ref) in STAGE_SOURCES.items() if not ref])),
        # `live` este dublat: cazul cel mai periculos (mod și cont identice cu ale spionului).
        approved_env=draw(st.sampled_from(["backtest", "shadow", "demo", "live", "live"])),
        approved_account=draw(
            st.sampled_from([SPY_LIVE_ACCOUNT, SPY_LIVE_ACCOUNT, "DU1234567", None])
        ),
        gate=draw(st.sampled_from(GATE_KINDS)),
        audit_mode=draw(st.sampled_from(["ok", "ok", "fail_once", "fail_always", "alternate"])),
        clock_raises=draw(st.integers(0, 5)) == 0,
        commands=tuple(draw(st.lists(commands(), min_size=1, max_size=25))),
    )


def run_block_case(case: BlockCase) -> Counter[str]:
    """Verifică P1 pe ruta directă prin `FailSafeBlock`; întoarce statistici."""
    stats: Counter[str] = Counter()
    stage = _read_stage(case.stage_kind)
    assert stage is not None
    stats[f"stage:{stage.source}"] += 1
    spy = _LiveSpy()
    audit = _Audit(case.audit_mode)
    ks = KillSwitch(_MemStore(), clock=SimClock(T0))
    block: FailSafeBlock[_Req, str] = FailSafeBlock(
        spy,
        approved=ApprovedTarget.model_validate(
            {"environment": case.approved_env, "account_id": case.approved_account}
        ),
        stage=stage,
        kill_switch=ks,
        audit=audit,
        clock=_RaisingClock() if case.clock_raises else SimClock(T0),
        live_gate=_gate(case.gate, stage),
    )
    for cmd in case.commands:
        if cmd.kind == "submit":
            before = len(audit.records) + len(block.unaudited)
            with pytest.raises(FailSafeRejectedError) as info:
                block.submit(_Req(cmd.coid, cmd.instrument))
            stats[f"reason:{info.value.reason}"] += 1
            if not info.value.audited:
                stats["not_audited"] += 1
            # Fiecare tentativă blocată lasă o urmă: Audit_Record sau coada `unaudited`.
            assert len(audit.records) + len(block.unaudited) >= before + 1
        elif cmd.kind == "cancel":
            assert block.cancel(cmd.coid) == f"cancel:{cmd.coid}"  # pass-through permis
        elif cmd.kind == "ks_global":
            ks.activate_manual(KillSwitchScope.GLOBAL, operator="alice", reason_code="R")
        elif cmd.kind == "ks_instr" and cmd.instrument.strip():
            ks.activate_automatic(
                KillSwitchScope.INSTRUMENT,
                component="health",
                reason_code="R",
                instrument=cmd.instrument,
            )
        elif cmd.kind == "flush":
            before = len(audit.records) + len(block.unaudited)
            block.flush_audit()
            assert len(audit.records) + len(block.unaudited) >= before
        assert spy.submit_calls == 0, f"submit a ajuns la adaptorul live după {cmd}"
    if case.approved_env == "live" and case.approved_account == SPY_LIVE_ACCOUNT:
        stats["live_target_matching_spy"] += 1
    return stats


@given(block_cases())
def test_property_1_fail_safe_never_reaches_live_adapter_in_initial_stage(
    case: BlockCase,
) -> None:
    """**Validates: Requirements 2.1, 2.4, 2.6**"""
    for key in run_block_case(case):
        event(key)


def test_property_1_not_vacuous() -> None:
    """Rutele relevante sunt atinse efectiv; proprietatea nu este vacuă.

    **Validates: Requirements 2.1, 2.4, 2.6**
    """
    build: Counter[str] = Counter()
    blocks: Counter[str] = Counter()

    @settings(max_examples=300, derandomize=True, database=None)
    @given(build_cases())
    def run_build(case: BuildCase) -> None:
        build[run_build_case(case)] += 1

    @settings(max_examples=300, derandomize=True, database=None)
    @given(block_cases())
    def run_blocks(case: BlockCase) -> None:
        blocks.update(run_block_case(case))

    run_build()
    run_blocks()
    for key in (
        "built:SimBroker",
        "built:FakeBroker",
        "live_refused",
        "live_endpoint_or_account_refused",
        "stage_refused",
        "schema_refused",
    ):
        assert build[key] > 0, (key, build)
    for key in (
        f"reason:{FailSafeReason.STAGE_FORBIDS_ENVIRONMENT}",
        f"reason:{FailSafeReason.ENVIRONMENT_MISMATCH}",
        f"reason:{FailSafeReason.ACCOUNT_MISMATCH}",
        f"reason:{FailSafeReason.INVALID_REQUEST}",
        "not_audited",
        "live_target_matching_spy",
        "stage:stage.lock",
        "stage:implicit (fail-closed)",
    ):
        assert blocks[key] > 0, (key, blocks)


# =========================================================================== P13: Kill_Switch

INSTRUMENTS = ("AAA", "BBB", "CCC", "DDD")
CC_PREFIX = "cc-"
SIM_TARGET = ApprovedTarget(environment="backtest", account_id="SIM-LOCAL")
ZERO_LOSS = D(0)
FULL_EQUITY = D(100)


@dataclass(frozen=True)
class _Auth:
    new_capital_config_id: str

    def is_valid(self) -> bool:
        return True


@dataclass(frozen=True)
class KsOp:
    kind: str
    instrument: str = "AAA"
    minutes: int = 0
    manual: bool = True
    loss: Decimal = ZERO_LOSS
    equity: Decimal = FULL_EQUITY
    valid: bool = True
    pick: int = 0


@st.composite
def ks_ops(draw: st.DrawFn) -> KsOp:
    kind = draw(
        st.sampled_from(
            [
                "submit",
                "submit",
                "submit",
                "submit",
                "advance",
                "advance",
                "act_instr",
                "act_instr",
                "act_global",
                "act_day",
                "observe",
                "observe",
                "resume",
                "resume",
                "resume",
                "expire",
                "replace_cc",
                "restart",
            ]
        )
    )
    return KsOp(
        kind=kind,
        instrument=draw(st.sampled_from(INSTRUMENTS)),
        minutes=draw(st.one_of(st.integers(0, 180), st.integers(600, 2000))),
        manual=draw(st.booleans()),
        loss=draw(st.sampled_from([D(0), D("1"), D("1.99"), D("2"), D("3")])),
        equity=draw(st.sampled_from([D(100), D(100), D(95), D("90.01"), D(90), D(85)])),
        valid=draw(st.integers(0, 3)) > 0,
        pick=draw(st.integers(0, 50)),
    )


@dataclass(frozen=True)
class KsCase:
    day_label: str
    day_fn: TradingDayFn
    start: datetime
    ops: tuple[KsOp, ...]


@st.composite
def ks_cases(draw: st.DrawFn) -> KsCase:
    fn: TradingDayFn
    if draw(st.booleans()):
        label, fn = "utc", utc_trading_day
    else:
        hours = draw(st.integers(-10, 12))
        rollover = dtime(draw(st.integers(0, 23)), draw(st.sampled_from([0, 30])))
        label = f"UTC{hours:+d}@{rollover}"
        fn = zoned_trading_day(timezone(timedelta(hours=hours)), rollover)
    return KsCase(
        day_label=label,
        day_fn=fn,
        start=T0 + timedelta(minutes=draw(st.integers(0, 24 * 60))),
        ops=tuple(draw(st.lists(ks_ops(), min_size=1, max_size=40))),
    )


@dataclass
class _Oracle:
    """Model independent al domeniilor active."""

    instruments: dict[str, str] = field(default_factory=dict)  # activation_id → instrument
    globals_: set[str] = field(default_factory=set)
    day_days: set[date] = field(default_factory=set)
    capital_active: bool = False

    def blocking(self, instrument: str, today: date) -> str | None:
        if self.capital_active:
            return "capital_config"
        if self.globals_:
            return "global"
        if today in self.day_days:
            return "day"
        if instrument in self.instruments.values():
            return "instrument"
        return None


def _resume_request(activation_id: str, ts: datetime, *, valid: bool) -> ResumeRequest:
    return ResumeRequest(
        activation_id=activation_id,
        operator="alice",
        approved=valid,
        reconciliation_ok=True,
        reconciliation_id="recon-1",
        reason_resolved=True,
        cause="cauză",
        correction="corecție",
        ts=ts,
    )


def run_ks_case(case: KsCase) -> Counter[str]:
    """Rulează secvența și verifică P13 la fiecare ordin; întoarce statistici."""
    stats: Counter[str] = Counter()
    clock = SimClock(case.start)
    store = _MemStore()
    holder: dict[str, KillSwitch] = {"ks": KillSwitch(store, clock=clock, trading_day=case.day_fn)}

    class _Sink:
        def activate(self, activation: KillSwitchActivation) -> None:
            holder["ks"].activate(activation)

    monitor = LossMonitor(
        RiskConfig(), _Sink(), capital_config_id=f"{CC_PREFIX}0", trading_day=case.day_fn
    )
    limit = RiskConfig().daily_loss_limit_eur
    floor = RiskConfig().reference_capital_eur - RiskConfig().total_loss_limit_eur
    spy = _SimSpy()

    def make_block() -> FailSafeBlock[_Req, str]:
        return FailSafeBlock(
            spy,
            approved=SIM_TARGET,
            stage=INITIAL,
            kill_switch=holder["ks"],
            audit=_Audit(),
            clock=clock,
        )

    block = make_block()
    oracle = _Oracle()
    cc_counter = 0
    ever_blocked = False

    for i, op in enumerate(case.ops):
        ks = holder["ks"]
        now = clock.now()
        today = case.day_fn(now)
        if op.kind == "advance":
            clock.advance_to(now + timedelta(minutes=op.minutes))
            if case.day_fn(clock.now()) != today:
                stats["day_rollover"] += 1
        elif op.kind == "expire":
            ks.expire_days()
        elif op.kind in ("act_instr", "act_global", "act_day"):
            scope = {
                "act_instr": KillSwitchScope.INSTRUMENT,
                "act_global": KillSwitchScope.GLOBAL,
                "act_day": KillSwitchScope.DAY,
            }[op.kind]
            instrument = op.instrument if scope is KillSwitchScope.INSTRUMENT else None
            if op.manual:
                out = ks.activate_manual(
                    scope, operator="alice", reason_code="R", instrument=instrument
                )
            else:
                out = ks.activate_automatic(
                    scope, component="reconciliation", reason_code="R", instrument=instrument
                )
            assert out.newly_activated
            if scope is KillSwitchScope.INSTRUMENT:
                oracle.instruments[out.activation_id] = op.instrument
            elif scope is KillSwitchScope.GLOBAL:
                oracle.globals_.add(out.activation_id)
            else:
                oracle.day_days.add(today)
        elif op.kind == "observe":
            monitor.observe(
                LossObservation(
                    ts=now,
                    daily_realized_pnl_eur=-op.loss,
                    daily_unrealized_pnl_eur=D(0),
                    equity_eur=op.equity,
                )
            )
            if op.loss >= limit:
                oracle.day_days.add(today)
            if op.equity <= floor:
                oracle.capital_active = True
        elif op.kind == "replace_cc":
            cc_counter += 1
            auth = _Auth(f"{CC_PREFIX}{cc_counter}")
            monitor.start_new_capital_config(auth)
            ks.replace_capital_config(auth, actor="operator:alice")
            if oracle.capital_active:
                stats["capital_replaced"] += 1
            oracle.capital_active = False
        elif op.kind == "resume":
            candidates = [t.activation_id for t in ks.active_activations()]
            if not candidates:
                continue
            target = candidates[op.pick % len(candidates)]
            decision = ks.resume(_resume_request(target, now, valid=op.valid))
            resumable = target in oracle.instruments or target in oracle.globals_
            assert decision.accepted == (op.valid and resumable), (target, decision)
            if decision.accepted:
                stats["resumed"] += 1
                oracle.instruments.pop(target, None)
                oracle.globals_.discard(target)
            else:
                stats["resume_refused"] += 1
        elif op.kind == "restart":
            holder["ks"] = KillSwitch(store, clock=clock, trading_day=case.day_fn)
            block = make_block()
        else:  # submit
            expected = oracle.blocking(op.instrument, today)
            before = spy.calls[op.instrument]
            req = _Req(f"o{i}", op.instrument)
            if expected is not None:
                with pytest.raises(FailSafeRejectedError) as info:
                    block.submit(req)
                assert info.value.reason is FailSafeReason.KILL_SWITCH_ACTIVE
                assert spy.calls[op.instrument] == before, f"ordin transmis sub domeniul {expected}"
                stats[f"blocked:{expected}"] += 1
                ever_blocked = True
            else:
                assert block.submit(req) == f"ack:o{i}", "ordin blocat fără domeniu activ"
                assert spy.calls[op.instrument] == before + 1
                stats["passed"] += 1
                if ever_blocked:
                    stats["passed_after_block"] += 1
                if oracle.instruments:
                    stats["passed_other_instrument"] += 1
    return stats


@given(ks_cases())
def test_property_13_no_new_orders_while_kill_switch_scope_active(case: KsCase) -> None:
    """**Validates: Requirements 14.1, 14.2, 14.6**"""
    for key in run_ks_case(case):
        event(key)


def test_property_13_not_vacuous() -> None:
    """Toate domeniile blochează efectiv, iar ordinele trec după ridicarea lor.

    **Validates: Requirements 14.1, 14.2, 14.6**
    """
    totals: Counter[str] = Counter()

    @settings(max_examples=300, derandomize=True, database=None)
    @given(ks_cases())
    def run(case: KsCase) -> None:
        totals.update(run_ks_case(case))

    run()
    for key in (
        "blocked:instrument",
        "blocked:global",
        "blocked:day",
        "blocked:capital_config",
        "passed",
        "passed_after_block",
        "passed_other_instrument",
        "resumed",
        "resume_refused",
        "capital_replaced",
        "day_rollover",
    ):
        assert totals[key] > 0, (key, totals)


def test_property_13_activation_from_other_thread_blocks_within_one_second() -> None:
    """Activarea GLOBAL dintr-un alt fir blochează ordinele în ≤ 1 s (Req 14.1, 14.2).

    **Validates: Requirements 14.1, 14.2**
    """
    clock = SimClock(T0)
    ks = KillSwitch(_MemStore(), clock=clock)
    spy = _SimSpy()
    block: FailSafeBlock[_Req, str] = FailSafeBlock(
        spy, approved=SIM_TARGET, stage=INITIAL, kill_switch=ks, audit=_Audit(), clock=clock
    )
    activated_at: list[float] = []

    def activator() -> None:
        time.sleep(0.05)
        ks.activate_automatic(KillSwitchScope.GLOBAL, component="health", reason_code="R")
        activated_at.append(time.perf_counter())

    results: list[tuple[float, bool]] = []
    thread = threading.Thread(target=activator)
    thread.start()
    deadline = time.perf_counter() + 5
    n = 0
    while time.perf_counter() < deadline:
        started = time.perf_counter()
        n += 1
        try:
            block.submit(_Req(f"t{n}", "AAA"))
            results.append((started, True))
        except FailSafeRejectedError:
            results.append((started, False))
            break
    thread.join(5)
    assert activated_at, "activarea nu s-a încheiat"
    t_act = activated_at[0]
    rejected = [s for s, ok in results if not ok]
    assert rejected, "niciun ordin nu a fost blocat"
    assert rejected[0] - t_act <= 1.0
    assert any(ok for _, ok in results)  # non-vacuitate: ordinele treceau înainte
    passed_before = spy.calls["AAA"]
    for k in range(20):  # după activare, niciun ordin nou nu mai ajunge la adaptor
        with pytest.raises(FailSafeRejectedError):
            block.submit(_Req(f"after{k}", "BBB"))
    assert spy.calls["AAA"] == passed_before and spy.calls["BBB"] == 0
    assert all(not ok for s, ok in results if s > t_act)
