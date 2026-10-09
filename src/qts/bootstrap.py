"""Compunerea porturilor per mod: singurul loc care cunoaște adaptoarele concrete (Req 1.2, 1.5).

Nucleul (`core/engine.py`) primește porturile prin constructor și nu importă niciun adaptor
concret (Req 1.1, 16.6). Acest modul le alege numai din mediul configurației:

| Mod      | Date                          | Broker                              | Ceas       |
|----------|-------------------------------|-------------------------------------|------------|
| Backtest | `CsvSource` (reluare istorică)| `SimBroker` în `FailSafeBlock`      | `SimClock` |
| Shadow   | feed curent (`DataAdapter`)   | `SimBroker` în `FailSafeBlock`      | `WallClock`|
| Demo     | feed Alpaca (`DataAdapter`)   | `AlpacaBrokerAdapter` paper învelit | `WallClock`|
| Live     | refuzat de `check_startup` în Initial_Stage și de `build_broker` (Req 2.2)   |

Demo este analog cu Shadow, dar brokerul este Alpaca paper (API real, bani simulați), NU
`SimBroker`: nu există alimentare `on_bar` și nici ascultători de piață simulați — brokerul își
produce singur execuțiile prin `events()`, pe care motorul le drenează. În plus, Demo conduce
`Reconciler` din bucla de rulare: la pornire (înaintea primului ordin) și periodic (≤ 60 s).

Separarea nucleu / adaptoare (Req 1.5, 16.6): `build_core` construiește Strategy, Risk_Engine
și modelul de costuri numai din secțiunile `strategy`, `risk`, `costs` și din identificatorul
snapshot-ului; nu primește modul și nu vede adaptoarele. `build_adapters` alege datele,
brokerul și ceasul numai din mediu, configurația de mediu și referințele de credențiale. La
schimbarea modului, codul și parametrii strategiei și regulile de risc rămân aceleași.

Secvența de pornire (fail-closed, totul înaintea primului eveniment procesat):

1. `load_config` (schema strictă) și, opțional, suprascrierea `run.db_path`;
2. `read_stage(stage.lock)` + `check_startup` (refuză Live și endpoint-urile/conturile live);
3. disponibilitatea modului (numai Backtest) și prezența secțiunii `costs`;
4. setul de date: manifest + sumă de control (`CsvSource`), înainte de snapshot, deoarece
   `dataset_id` intră în snapshot;
5. `Configuration_Snapshot` (git HEAD, hash `uv.lock`, configurație efectivă, sămânță). Dacă nu
   poate fi creat, rularea se oprește (Req 17.7). `code_version` este o injecție explicită
   *numai pentru teste* (un depozit temporar fără commit nu are HEAD); CLI-ul nu o expune;
6. baza SQLite + jurnal; snapshot-ul este prima înregistrare a rulării (`config_snapshot`);
7. Kill_Switch (persistat în jurnal), broker prin `build_broker` (învelit în `FailSafeBlock`),
   OMS, portofoliu (numerar inițial = `risk.reference_capital_eur`), Risk_Engine, LossMonitor,
   prospețime (relativă la `SimClock`), strategie, motor.

Timpul: în Backtest `SimClock` pornește la `manifest.start` și este avansat de motor. Și
`created_at` al snapshot-ului folosește acest timp simulat, deci jurnalul nu depinde de ceasul
de perete (determinism, Req 17.5).

Estimarea `σ_bar` și ADV (`TrailingMarketEstimator`), deterministă și fără look-ahead:
- `σ_bar` = abaterea standard de populație a ultimelor `SIGMA_WINDOW` randamente simple
  `close_i / close_{i-1} - 1` ale instrumentului (minimum 2 randamente);
- ADV = suma volumelor barelor cu `ts_close` în ultimele 24 de ore față de bara curentă
  (volum rulant pe o zi; la începutul setului este subestimat, deci slippage-ul este mai mare,
  adică conservator);
- lipsa unei estimări lasă câmpul `None`: modelul de costuri ridică `CostModelIncomplete`, iar
  Risk_Engine respinge ordinul (fail-closed);
- execuția simulată a barei *t+1* folosește estimarea disponibilă la închiderea barei *t*
  (înainte de a observa bara *t+1*), deci prețul de execuție nu folosește date viitoare.
Cursul FX nu este disponibil în Backtest: instrumentele non-EUR nu primesc `fx_rate`, deci
ordinele lor sunt respinse cu `COST_MODEL_INCOMPLETE`.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from itertools import pairwise
from pathlib import Path
from typing import Final

from qts.broker.alpaca_broker import AlpacaBrokerAdapter, AlpacaClientFactory
from qts.broker.factory import GuardedAdapter, build_broker
from qts.broker.sim import SimBarContext, SimBroker, SimBrokerConfig
from qts.config.loader import load_config
from qts.config.schema import AppConfig
from qts.config.snapshot import (
    CodeVersion,
    ConfigurationSnapshot,
    create_snapshot,
    read_code_version,
)
from qts.core.clock import Clock, SimClock, WallClock
from qts.core.engine import EngineConfig, MarketContextFn, MarketListener, TradingEngine
from qts.core.models import (
    Bar,
    CostBreakdown,
    ExecKind,
    Instrument,
    MarketEvent,
    Order,
    OrderState,
)
from qts.costs.errors import cost_context
from qts.costs.model import CompleteCostModel, CostContext
from qts.data.adapter import DataAdapter
from qts.data.csv_source import CsvSource
from qts.data.freshness import FreshnessTracker
from qts.oms.manager import JournalOmsSink, OrderManager, reporting_identity
from qts.persistence.audit import verify_journal
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.portfolio.portfolio import Portfolio, PortfolioState, pnl_report
from qts.recon.reconciler import (
    Reconciler,
    ReconciliationReason,
    ReconciliationTolerances,
)
from qts.risk.engine import RiskEngine
from qts.risk.monitor import LossMonitor
from qts.safety.kill_switch import JournalKillSwitchStore, KillSwitch
from qts.safety.stage import StageInfo, check_startup, read_stage
from qts.secrets.store import SecretStore
from qts.strategy.base import Strategy
from qts.strategy.mean_reversion import STRATEGY_ID as MEAN_REVERSION_ID
from qts.strategy.mean_reversion import MeanReversionStrategy

__all__ = [
    "AVAILABLE_MODES",
    "BOOTSTRAP_VERSION",
    "SIGMA_WINDOW",
    "BacktestResult",
    "BootstrapError",
    "CoreComponents",
    "DemoDataFactory",
    "DemoResult",
    "ModeAdapters",
    "ModeNotAvailableError",
    "RealDataAdapterUnavailableError",
    "ShadowDataFactory",
    "ShadowResult",
    "TrailingMarketEstimator",
    "build_adapters",
    "build_core",
    "build_demo_adapters",
    "build_shadow_adapters",
    "capital_config_id",
    "run_backtest",
    "run_demo",
    "run_shadow",
]

BOOTSTRAP_VERSION: Final = "1"
COMPONENT: Final = "bootstrap"
ACTOR: Final = "system:bootstrap"
SIGMA_WINDOW: Final = 20
ADV_WINDOW: Final = timedelta(days=1)
AVAILABLE_MODES: Final = frozenset({"backtest", "shadow", "demo"})

_OPEN_STATES: Final = frozenset(
    {
        OrderState.SUBMITTED,
        OrderState.ACKNOWLEDGED,
        OrderState.PARTIALLY_FILLED,
        OrderState.CANCEL_PENDING,
    }
)
_FILL_OUTCOMES: Final = frozenset({ExecKind.FILL.value, ExecKind.PARTIAL_FILL.value})

StrategyFactory = Callable[[AppConfig, str], Strategy]


def _mean_reversion(config: AppConfig, snapshot_id: str) -> Strategy:
    return MeanReversionStrategy.from_config(config.strategy, config_snapshot_id=snapshot_id)


# Registrul strategiilor cunoscute; aceeași intrare în toate modurile (Req 1.5).
STRATEGIES: Final[dict[str, StrategyFactory]] = {MEAN_REVERSION_ID: _mean_reversion}


class BootstrapError(Exception):
    """Compunerea nu poate fi făcută; nimic nu a fost procesat."""


class ModeNotAvailableError(BootstrapError):
    """Modul cerut nu are încă o compunere în această versiune."""


class RealDataAdapterUnavailableError(BootstrapError):
    """Seam-ul sursei de date reale (feed curent) din Shadow nu este încă implementat.

    Shadow compune ceasul real (`WallClock`), `SimBroker` și motorul comun, dar adaptorul de
    date *real* depinde de alegerea sursei (Open_Decision, Req 30). Până la acea alegere,
    compunerea Shadow cere injectarea explicită a unui `DataAdapter` (de exemplu o reluare peste
    date existente); fără el, pornirea este refuzată cu acest motiv clar pentru operator.
    """


# --------------------------------------------------------------------------- estimări piață


class TrailingMarketEstimator:
    """`σ_bar` și ADV din barele deja vizibile, per instrument (vezi docstring-ul modulului)."""

    def __init__(self, sigma_window: int = SIGMA_WINDOW, adv_window: timedelta = ADV_WINDOW):
        if sigma_window < 2:
            raise ValueError("sigma_window trebuie să fie >= 2")
        self._sigma_window = sigma_window
        self._adv_window = adv_window
        self._closes: dict[str, deque[Decimal]] = {}
        self._volumes: dict[str, deque[tuple[datetime, Decimal]]] = {}
        self._before: dict[str, tuple[datetime, Decimal | None, Decimal | None]] = {}

    def _estimate(self, symbol: str) -> tuple[Decimal | None, Decimal | None]:
        closes = list(self._closes.get(symbol, ()))
        sigma: Decimal | None = None
        with localcontext(cost_context()):
            returns = [b / a - 1 for a, b in pairwise(closes)]
            if len(returns) >= 2:
                n = Decimal(len(returns))
                mean = sum(returns, Decimal(0)) / n
                sigma = (sum(((r - mean) ** 2 for r in returns), Decimal(0)) / n).sqrt()
            total = sum((v for _, v in self._volumes.get(symbol, ())), Decimal(0))
        adv = total if total > 0 else None
        return sigma, adv

    def observe(self, bar: Bar) -> tuple[Decimal | None, Decimal | None]:
        """Adaugă bara; întoarce estimarea *după* bară. Estimarea dinainte rămâne reținută."""
        sym = bar.instrument
        sigma_before, adv_before = self._estimate(sym)
        self._before[sym] = (bar.ts_close, sigma_before, adv_before)
        closes = self._closes.setdefault(sym, deque(maxlen=self._sigma_window + 1))
        closes.append(bar.close)
        volumes = self._volumes.setdefault(sym, deque())
        volumes.append((bar.ts_close, bar.volume))
        while volumes and volumes[0][0] <= bar.ts_close - self._adv_window:
            volumes.popleft()
        return self._estimate(sym)

    def before(self, bar: Bar) -> tuple[Decimal | None, Decimal | None]:
        """Estimarea disponibilă înaintea barei `bar` (pentru execuția la deschiderea ei)."""
        stored = self._before.get(bar.instrument)
        if stored is not None and stored[0] == bar.ts_close:
            return stored[1], stored[2]
        return self._estimate(bar.instrument)  # bara nu a fost încă observată


# --------------------------------------------------------------------------- compunere


@dataclass(frozen=True, slots=True)
class CoreComponents:
    """Componentele independente de mod (Req 1.5): aceleași în Backtest, Shadow și Demo."""

    strategy: Strategy
    risk: RiskEngine
    cost_model: CompleteCostModel


@dataclass(frozen=True, slots=True)
class ModeAdapters:
    """Porturile dependente de mod: date, broker (învelit), ceas și alimentarea simulatorului.

    `clock` este `SimClock` în Backtest (avansat de evenimente) și `WallClock` în Shadow (ceas
    real). `data` este orice `DataAdapter`: `CsvSource` pentru reluare istorică (Backtest) sau
    reluare peste date existente (Shadow, până la adaptorul de date real, Req 30).
    """

    clock: Clock
    data: DataAdapter
    broker: GuardedAdapter
    market_listeners: tuple[MarketListener, ...]
    market_context: MarketContextFn
    dataset_id: str


def capital_config_id(config: AppConfig) -> str:
    """Identificatorul configurației de capital: hash-ul secțiunii `risk` (Req 13.6, 28)."""
    blob = config.risk.model_dump_json()
    return "cap-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_core(config: AppConfig, snapshot_id: str) -> CoreComponents:
    """Strategy, Risk_Engine și modelul de costuri; nu depind de mod și nu văd adaptoarele."""
    if config.costs is None:
        raise BootstrapError("secțiunea [costs] lipsește: Complete_Cost_Model este obligatoriu")
    factory = STRATEGIES.get(config.strategy.strategy_id)
    if factory is None:
        raise BootstrapError(
            f"strategie necunoscută: {config.strategy.strategy_id!r}; "
            f"cunoscute: {sorted(STRATEGIES)}"
        )
    reporting = config.reporting_currency
    cost_model = CompleteCostModel(config.costs, reporting_currency=reporting)
    return CoreComponents(
        strategy=factory(config, snapshot_id),
        risk=RiskEngine(config.risk, cost_model, reporting_currency=reporting),
        cost_model=cost_model,
    )


def _require_mode(config: AppConfig) -> None:
    if config.environment not in AVAILABLE_MODES:
        raise ModeNotAvailableError(
            f"modul {config.environment} is not yet available (mod nesuportat); "
            f"disponibile: {sorted(AVAILABLE_MODES)}"
        )


def _open_dataset(config: AppConfig, base_dir: Path) -> CsvSource:
    if config.data.dataset_path is None:  # pragma: no cover - impus de schemă pentru backtest
        raise BootstrapError("data.dataset_path lipsă")
    path = Path(config.data.dataset_path)
    return CsvSource(path if path.is_absolute() else base_dir / path)


def _build_sim_broker(
    config: AppConfig,
    stage: StageInfo,
    *,
    core: CoreComponents,
    clock: Clock,
    kill_switch: KillSwitch,
    journal: Journal,
) -> tuple[GuardedAdapter, SimBroker]:
    """`SimBroker` învelit în `FailSafeBlock` prin fabrică; comun Backtest și Shadow (Req 3.1)."""
    broker = build_broker(
        config,
        stage,
        clock=clock,
        kill_switch=kill_switch,
        audit=journal,
        cost_model=core.cost_model,
        sim_config=SimBrokerConfig(broker="sim", initial_cash=config.risk.reference_capital_eur),
        reporting_currency=config.reporting_currency,
    )
    sim = broker.inner
    if not isinstance(sim, SimBroker):  # pragma: no cover - garantat de build_broker(sim)
        raise BootstrapError("compunerea cu SimBroker necesită broker.kind=sim")
    return broker, sim


def _sim_wiring(sim: SimBroker) -> tuple[MarketContextFn, MarketListener]:
    """Estimatorul fără look-ahead, contextul de cost și alimentarea `SimBroker.on_bar`.

    Identic în Backtest și Shadow: estimarea folosită de execuția barei *t+1* este cea
    disponibilă la închiderea barei *t* (`estimator.before`), deci prețul de execuție nu
    folosește date viitoare. Diferă între moduri doar ceasul și sursa de evenimente.
    """
    estimator = TrailingMarketEstimator()

    def market_context(inst: Instrument, bar: Bar) -> CostContext:
        sigma, adv = estimator.observe(bar)
        return CostContext(
            broker="sim", sigma_bar=sigma, adv=adv, bar_interval_min=bar.interval_min
        )

    def feed_sim(event: MarketEvent) -> None:
        if isinstance(event.payload, Bar):
            sigma, adv = estimator.before(event.payload)
            sim.on_bar(event.payload, SimBarContext(sigma_bar=sigma, adv=adv))

    return market_context, feed_sim


def build_adapters(
    config: AppConfig,
    stage: StageInfo,
    *,
    core: CoreComponents,
    data: CsvSource,
    journal: Journal,
    kill_switch_factory: Callable[[SimClock], KillSwitch],
) -> tuple[ModeAdapters, KillSwitch]:
    """Porturile Backtest: reluare istorică `CsvSource`, `SimBroker` și `SimClock`."""
    _require_mode(config)
    clock = SimClock(data.manifest.start)
    kill_switch = kill_switch_factory(clock)
    broker, sim = _build_sim_broker(
        config, stage, core=core, clock=clock, kill_switch=kill_switch, journal=journal
    )
    market_context, feed_sim = _sim_wiring(sim)
    adapters = ModeAdapters(
        clock=clock,
        data=data,
        broker=broker,
        market_listeners=(feed_sim,),
        market_context=market_context,
        dataset_id=data.dataset_id,
    )
    return adapters, kill_switch


def build_shadow_adapters(
    config: AppConfig,
    stage: StageInfo,
    *,
    core: CoreComponents,
    data: DataAdapter,
    clock: Clock,
    journal: Journal,
    kill_switch_factory: Callable[[Clock], KillSwitch],
) -> tuple[ModeAdapters, KillSwitch]:
    """Porturile Shadow: feed curent reluat printr-un `DataAdapter`, `SimBroker` și ceas real.

    Spre deosebire de Backtest, ceasul este `WallClock` (timp real) și nu este avansat de motor:
    `run_shadow` nu transmite `clock_driver`. `SimBroker` execută local, deci nicio comandă
    reală nu părăsește procesul (Req 1.1: aceeași secvență de pași ca Backtest). Sursa `data`
    este injectată — adaptorul de date real rămâne un seam (vezi `RealDataAdapterUnavailableError`).
    """
    _require_mode(config)
    kill_switch = kill_switch_factory(clock)
    broker, sim = _build_sim_broker(
        config, stage, core=core, clock=clock, kill_switch=kill_switch, journal=journal
    )
    market_context, feed_sim = _sim_wiring(sim)
    adapters = ModeAdapters(
        clock=clock,
        data=data,
        broker=broker,
        market_listeners=(feed_sim,),
        market_context=market_context,
        dataset_id=data.source_id,
    )
    return adapters, kill_switch


# Numele brokerului Demo real, cheie în tabelele de comisioane (`CostContext.broker`).
DEMO_BROKER_NAME: Final = "alpaca"


def _demo_market_context(broker_name: str) -> MarketContextFn:
    """Context de cost pentru Demo: `σ_bar`/ADV estimate fără look-ahead, broker = Alpaca.

    Prețurile de execuție vin de la Alpaca paper (nu de la `SimBroker`), deci NU există alimentare
    `on_bar` și niciun ascultător simulat. Dar Risk_Engine folosește același `Complete_Cost_Model`
    pentru a estima costul ordinelor înaintea trimiterii (fail-closed): fără `σ_bar`/ADV, modelul
    de slippage ar fi incomplet și fiecare ordin ar fi respins. Reutilizăm
    `TrailingMarketEstimator` (deterministic, din barele deja vizibile) ca să alimentăm
    estimările — exact ca în Backtest/Shadow —, dar fără a executa nimic simulat; `broker` rămâne
    cheia tabelelor de comisioane Alpaca.
    """
    estimator = TrailingMarketEstimator()

    def market_context(inst: Instrument, bar: Bar) -> CostContext:
        sigma, adv = estimator.observe(bar)
        return CostContext(
            broker=broker_name, sigma_bar=sigma, adv=adv, bar_interval_min=bar.interval_min
        )

    return market_context


def build_demo_adapters(
    config: AppConfig,
    stage: StageInfo,
    *,
    core: CoreComponents,
    data: DataAdapter,
    clock: Clock,
    journal: Journal,
    kill_switch_factory: Callable[[Clock], KillSwitch],
    secret_store: SecretStore | None,
    alpaca_client_factory: AlpacaClientFactory | None = None,
) -> tuple[ModeAdapters, KillSwitch]:
    """Porturile Demo: feed curent (Alpaca) + broker Alpaca paper (API real) + ceas real.

    Spre deosebire de Shadow, brokerul NU este `SimBroker`, ci `AlpacaBrokerAdapter` (paper),
    construit prin `build_broker(...)` cu cheile din `Secret_Store`. Nu există alimentare de
    simulator: `market_listeners=()`, iar `market_context` este neutru (brokerul produce singur
    execuțiile prin `events()`, pe care motorul le drenează la fiecare pas). Ceasul este real
    (`WallClock` în producție; un ceas injectat în teste) și nu este avansat de motor.
    """
    _require_mode(config)
    kill_switch = kill_switch_factory(clock)
    broker = build_broker(
        config,
        stage,
        clock=clock,
        kill_switch=kill_switch,
        audit=journal,
        cost_model=core.cost_model,
        secret_store=secret_store,
        alpaca_client_factory=alpaca_client_factory,
    )
    inner = broker.inner
    if not isinstance(inner, AlpacaBrokerAdapter):  # pragma: no cover - garantat de build_broker
        raise BootstrapError(
            "compunerea Demo necesită broker.kind=demo cu broker.name=alpaca (Alpaca paper)"
        )
    adapters = ModeAdapters(
        clock=clock,
        data=data,
        broker=broker,
        market_listeners=(),
        market_context=_demo_market_context(DEMO_BROKER_NAME),
        dataset_id=data.source_id,
    )
    return adapters, kill_switch


# --------------------------------------------------------------------------- rulare


@dataclass(frozen=True, slots=True)
class BacktestResult:
    snapshot_id: str
    run_id: str
    dataset_id: str
    db_path: str
    events_processed: int
    orders: int
    fills: int
    realized_gross_eur: Decimal
    unrealized_gross_eur: Decimal
    costs: CostBreakdown
    journal_head: tuple[int, str]
    journal_verified: bool
    rejected_rows: int = 0
    order_states: dict[str, int] = field(default_factory=dict)

    @property
    def gross_eur(self) -> Decimal:
        return self.realized_gross_eur + self.unrealized_gross_eur

    @property
    def net_eur(self) -> Decimal:
        return self.gross_eur - self.costs.total


def _open_orders(oms: OrderManager) -> Callable[[], Iterable[Order]]:
    return lambda: [o for o in oms.orders.values() if o.state in _OPEN_STATES]


def run_backtest(
    config_path: Path,
    *,
    stage_lock: Path = Path("stage.lock"),
    repo_root: Path | None = None,
    base_dir: Path | None = None,
    db_path: str | None = None,
    code_version: CodeVersion | None = None,
) -> BacktestResult:
    """Pornește și rulează un Backtest complet; orice refuz apare înaintea procesării.

    `base_dir` (implicit directorul curent) rezolvă căile relative din configurație;
    `repo_root` (implicit `base_dir`) este depozitul git citit pentru snapshot.
    `code_version` este numai pentru teste: înlocuiește citirea git, dar snapshot-ul este
    creat și jurnalizat la fel.
    """
    base = base_dir if base_dir is not None else Path.cwd()
    # Mediul vine din fișier; Live este refuzat explicit de `check_startup`, iar celelalte
    # moduri fără compunere de `_require_mode`, cu motive clare pentru operator.
    config = load_config(config_path)
    if db_path is not None:
        config = config.model_copy(
            update={"run": config.run.model_copy(update={"db_path": db_path})}
        )
    stage = read_stage(stage_lock)
    check_startup(config, stage)
    _require_mode(config)
    if config.costs is None:
        raise BootstrapError("secțiunea [costs] lipsește: Complete_Cost_Model este obligatoriu")
    data = _open_dataset(config, base)

    code = code_version if code_version is not None else read_code_version(repo_root or base)
    snapshot = create_snapshot(
        config, stage, code, created_at=data.manifest.start, dataset_ids=[data.dataset_id]
    )
    core = build_core(config, snapshot.snapshot_id)

    resolved_db = Path(config.run.db_path)
    if not resolved_db.is_absolute():
        resolved_db = base / resolved_db
    conn = open_db(resolved_db)
    try:
        journal = Journal(conn)
        _record_snapshot(journal, snapshot)
        return _run(config, stage, snapshot, core, data, journal, str(resolved_db))
    finally:
        conn.close()


def _record_snapshot(journal: Journal, snapshot: ConfigurationSnapshot) -> None:
    journal.append(
        ts=snapshot.created_at,
        type="config_snapshot",
        correlation_id=snapshot.snapshot_id,
        component=COMPONENT,
        component_version=BOOTSTRAP_VERSION,
        actor=ACTOR,
        outcome="created",
        payload=snapshot.model_dump(mode="json"),
    )


def _make_kill_switch(
    config: AppConfig, journal: Journal, oms: OrderManager
) -> Callable[[Clock], KillSwitch]:
    def factory(clock: Clock) -> KillSwitch:
        return KillSwitch(
            JournalKillSwitchStore(journal),
            clock=clock,
            config=config.kill_switch,
            open_orders=_open_orders(oms),
        )

    return factory


@dataclass(frozen=True, slots=True)
class _EngineRun:
    """Rezultatul brut al unei rulări a motorului, independent de mod (Backtest / Shadow)."""

    processed: int
    orders: int
    fills: int
    realized_gross_eur: Decimal
    unrealized_gross_eur: Decimal
    costs: CostBreakdown
    journal_head: tuple[int, str]
    journal_verified: bool
    order_states: dict[str, int]


def _run_engine(
    config: AppConfig,
    snapshot: ConfigurationSnapshot,
    core: CoreComponents,
    adapters: ModeAdapters,
    kill_switch: KillSwitch,
    oms: OrderManager,
    journal: Journal,
    *,
    run_id: str,
    advance_clock: bool,
    broker_name: str = "sim",
    before_run: Callable[[Portfolio], None] | None = None,
    extra_listener_factory: Callable[[Portfolio], MarketListener] | None = None,
) -> _EngineRun:
    """Construiește și rulează motorul comun. `advance_clock` separă Backtest de Shadow.

    În Backtest ceasul este `SimClock`, avansat de motor (`clock_driver`). În Shadow ceasul este
    `WallClock` (timp real) și nu este avansat de motor (`advance_clock=False`): secvența de pași
    procesați rămâne identică — numai sursa timpului diferă (Req 1.1).

    `broker_name` este cheia tabelelor de comisioane din `CostContext` (implicit "sim" pentru
    Backtest/Shadow; "alpaca" pentru brokerul Demo real). `before_run` rulează înaintea primului
    eveniment (reconcilierea de pornire, Req 11.1). `extra_listener_factory` adaugă un ascultător
    de piață legat de portofoliu (reconcilierea periodică Demo, Req 11.3): motorul îl apelează la
    fiecare bară, înaintea strategiei, deci reconcilierea precede ordinele noi ale acelei bare.
    """
    clock = adapters.clock
    portfolio = Portfolio.with_cash(config.risk.reference_capital_eur)
    listeners: tuple[MarketListener, ...] = adapters.market_listeners
    if extra_listener_factory is not None:
        listeners = (*listeners, extra_listener_factory(portfolio))
    engine = TradingEngine(
        config=EngineConfig(
            run_id=run_id,
            mode=config.environment,
            # `ProjectStage` are numai INITIAL în această versiune: implicitul "initial".
            broker_name=broker_name,
            instruments=tuple(config.instruments),
            reporting_currency=config.reporting_currency,
        ),
        clock=clock,
        broker=adapters.broker,
        journal=journal,
        strategy=core.strategy,
        risk=core.risk,
        oms=oms,
        portfolio=portfolio,
        kill_switch=kill_switch,
        cost_model=core.cost_model,
        freshness=FreshnessTracker.from_config(config.data, clock),
        loss_monitor=LossMonitor(
            config.risk, kill_switch, capital_config_id=capital_config_id(config)
        ),
        data=adapters.data,
        market_context=adapters.market_context,
        market_listeners=listeners,
        clock_driver=clock if advance_clock and isinstance(clock, SimClock) else None,
    )
    if before_run is not None:
        before_run(portfolio)
    processed = engine.run()

    report = pnl_report(portfolio.snapshot())
    fills = sum(
        1 for r in journal.read() if r.type == "execution_event" and r.outcome in _FILL_OUTCOMES
    )
    states: dict[str, int] = {}
    for order in oms.orders.values():
        states[order.state.value] = states.get(order.state.value, 0) + 1
    return _EngineRun(
        processed=processed,
        orders=len(oms.orders),
        fills=fills,
        realized_gross_eur=report.realized_gross_eur,
        unrealized_gross_eur=report.unrealized_gross_eur,
        costs=report.costs,
        journal_head=journal.head,
        journal_verified=verify_journal(journal).ok,
        order_states=dict(sorted(states.items())),
    )


def _run(
    config: AppConfig,
    stage: StageInfo,
    snapshot: ConfigurationSnapshot,
    core: CoreComponents,
    data: CsvSource,
    journal: Journal,
    db_path: str,
) -> BacktestResult:
    oms = OrderManager(JournalOmsSink(journal))
    adapters, kill_switch = build_adapters(
        config,
        stage,
        core=core,
        data=data,
        journal=journal,
        kill_switch_factory=_make_kill_switch(config, journal, oms),
    )
    run_id = f"bt-{snapshot.snapshot_id[:16]}"
    run = _run_engine(
        config,
        snapshot,
        core,
        adapters,
        kill_switch,
        oms,
        journal,
        run_id=run_id,
        advance_clock=True,
    )
    return BacktestResult(
        snapshot_id=snapshot.snapshot_id,
        run_id=run_id,
        dataset_id=adapters.dataset_id,
        db_path=db_path,
        events_processed=run.processed,
        orders=run.orders,
        fills=run.fills,
        realized_gross_eur=run.realized_gross_eur,
        unrealized_gross_eur=run.unrealized_gross_eur,
        costs=run.costs,
        journal_head=run.journal_head,
        journal_verified=run.journal_verified,
        rejected_rows=len(data.rejections),
        order_states=run.order_states,
    )


# --------------------------------------------------------------------------- Shadow


@dataclass(frozen=True, slots=True)
class ShadowResult:
    """Rezultatul unei rulări Shadow (execuție simulată pe feed curent, ceas real).

    `net_eur = gross_eur - costs.total`, la fel ca Backtest. `source_id` identifică sursa de
    date injectată (reluare peste date existente până la adaptorul de date real, Req 30).
    """

    snapshot_id: str
    run_id: str
    source_id: str
    db_path: str
    events_processed: int
    orders: int
    fills: int
    realized_gross_eur: Decimal
    unrealized_gross_eur: Decimal
    costs: CostBreakdown
    journal_head: tuple[int, str]
    journal_verified: bool
    order_states: dict[str, int] = field(default_factory=dict)

    @property
    def gross_eur(self) -> Decimal:
        return self.realized_gross_eur + self.unrealized_gross_eur

    @property
    def net_eur(self) -> Decimal:
        return self.gross_eur - self.costs.total


# Seam-ul sursei de date reale: compus din mediu, ceas și directorul de bază, întoarce portul
# `DataAdapter` al feed-ului curent. Implicit nu există (Open_Decision, Req 30); `run_shadow`
# refuză pornirea cu `RealDataAdapterUnavailableError` dacă nu se injectează o fabrică.
ShadowDataFactory = Callable[[AppConfig, Clock, Path], DataAdapter]


def _no_real_data_adapter(config: AppConfig, clock: Clock, base_dir: Path) -> DataAdapter:
    raise RealDataAdapterUnavailableError(
        "shadow real data adapter is not yet available (sursa de date este Open_Decision, "
        "Req 30); injectați `data_factory` cu un DataAdapter de reluare peste date existente"
    )


def run_shadow(
    config_path: Path,
    *,
    stage_lock: Path = Path("stage.lock"),
    repo_root: Path | None = None,
    base_dir: Path | None = None,
    db_path: str | None = None,
    code_version: CodeVersion | None = None,
    clock: Clock | None = None,
    data_factory: ShadowDataFactory = _no_real_data_adapter,
) -> ShadowResult:
    """Pornește și rulează modul Shadow; orice refuz apare înaintea procesării.

    Compunere (design: tabelul modurilor): feed curent reluat printr-un `DataAdapter` + `SimBroker`
    (execuție locală) + `WallClock` (ceas real). Secvența de pași este identică cu Backtest
    (Req 1.1); diferă numai adaptoarele, configurația de mediu și referințele credențialelor
    (Req 16.1 — Shadow urmează imediat după Backtest în ordinea de promovare).

    `clock` (implicit `WallClock`) și `data_factory` sunt seam-uri injectabile: testele
    furnizează un ceas controlabil și o reluare deterministă, fără a depinde de timpul real.
    Adaptorul de date *real* nu face parte din această sarcină (Open_Decision, Req 30): fără
    `data_factory`, pornirea este refuzată cu `RealDataAdapterUnavailableError`.
    """
    base = base_dir if base_dir is not None else Path.cwd()
    wall = clock if clock is not None else WallClock()
    config = load_config(config_path)
    if config.environment != "shadow":
        raise BootstrapError(f"run_shadow necesită environment=shadow, nu {config.environment!r}")
    if db_path is not None:
        config = config.model_copy(
            update={"run": config.run.model_copy(update={"db_path": db_path})}
        )
    stage = read_stage(stage_lock)
    check_startup(config, stage)
    _require_mode(config)
    if config.costs is None:
        raise BootstrapError("secțiunea [costs] lipsește: Complete_Cost_Model este obligatoriu")
    data = data_factory(config, wall, base)

    code = code_version if code_version is not None else read_code_version(repo_root or base)
    # `created_at` folosește ceasul real; `snapshot_id` nu depinde de el (Req 17.5), deci
    # jurnalul rămâne verificabil și reproductibil.
    snapshot = create_snapshot(
        config, stage, code, created_at=wall.now(), dataset_ids=[data.source_id]
    )
    core = build_core(config, snapshot.snapshot_id)

    resolved_db = Path(config.run.db_path)
    if not resolved_db.is_absolute():
        resolved_db = base / resolved_db
    conn = open_db(resolved_db)
    try:
        journal = Journal(conn)
        _record_snapshot(journal, snapshot)
        return _run_shadow(config, stage, snapshot, core, data, wall, journal, str(resolved_db))
    finally:
        conn.close()


def _run_shadow(
    config: AppConfig,
    stage: StageInfo,
    snapshot: ConfigurationSnapshot,
    core: CoreComponents,
    data: DataAdapter,
    clock: Clock,
    journal: Journal,
    db_path: str,
) -> ShadowResult:
    oms = OrderManager(JournalOmsSink(journal))

    def make_kill_switch(ks_clock: Clock) -> KillSwitch:
        return KillSwitch(
            JournalKillSwitchStore(journal),
            clock=ks_clock,
            config=config.kill_switch,
            open_orders=_open_orders(oms),
        )

    adapters, kill_switch = build_shadow_adapters(
        config,
        stage,
        core=core,
        data=data,
        clock=clock,
        journal=journal,
        kill_switch_factory=make_kill_switch,
    )
    run_id = f"sh-{snapshot.snapshot_id[:16]}"
    run = _run_engine(
        config,
        snapshot,
        core,
        adapters,
        kill_switch,
        oms,
        journal,
        run_id=run_id,
        advance_clock=False,
    )
    return ShadowResult(
        snapshot_id=snapshot.snapshot_id,
        run_id=run_id,
        source_id=adapters.dataset_id,
        db_path=db_path,
        events_processed=run.processed,
        orders=run.orders,
        fills=run.fills,
        realized_gross_eur=run.realized_gross_eur,
        unrealized_gross_eur=run.unrealized_gross_eur,
        costs=run.costs,
        journal_head=run.journal_head,
        journal_verified=run.journal_verified,
        order_states=run.order_states,
    )


# --------------------------------------------------------------------------- Demo


@dataclass(frozen=True, slots=True)
class DemoResult:
    """Rezultatul unei rulări Demo (broker Alpaca paper, API real, bani simulați).

    Oglindește `ShadowResult`, dar execuțiile vin de la Alpaca paper, nu de la `SimBroker`.
    `net_eur = gross_eur - costs.total`. `source_id` identifică feed-ul (Alpaca).
    `kill_switch_active` semnalează dacă reconcilierea (de pornire sau periodică) a activat
    Kill_Switch GLOBAL —
    de exemplu la un snapshot incomplet (fail-closed, Req 11.6) — caz în care niciun ordin nou nu
    mai este aprobat.
    """

    snapshot_id: str
    run_id: str
    source_id: str
    db_path: str
    events_processed: int
    orders: int
    fills: int
    realized_gross_eur: Decimal
    unrealized_gross_eur: Decimal
    costs: CostBreakdown
    journal_head: tuple[int, str]
    journal_verified: bool
    reconciliations: int = 0
    startup_reconciled: bool = False
    kill_switch_active: bool = False
    order_states: dict[str, int] = field(default_factory=dict)

    @property
    def gross_eur(self) -> Decimal:
        return self.realized_gross_eur + self.unrealized_gross_eur

    @property
    def net_eur(self) -> Decimal:
        return self.gross_eur - self.costs.total


# Seam-ul feed-ului Demo: aceeași formă ca `ShadowDataFactory` — `(config, clock, base_dir) ->
# DataAdapter`. În producție este `alpaca_shadow_factory(store)` (același `AlpacaDataAdapter`);
# în teste o reluare deterministă peste bare sintetice.
DemoDataFactory = Callable[[AppConfig, Clock, Path], DataAdapter]


def _no_demo_data_adapter(config: AppConfig, clock: Clock, base_dir: Path) -> DataAdapter:
    raise RealDataAdapterUnavailableError(
        "demo real data adapter is not yet available; injectați `data_factory` (de exemplu "
        "`alpaca_shadow_factory(store)`) sau o reluare deterministă peste date existente"
    )


def run_demo(
    config_path: Path,
    *,
    stage_lock: Path = Path("stage.lock"),
    repo_root: Path | None = None,
    base_dir: Path | None = None,
    db_path: str | None = None,
    code_version: CodeVersion | None = None,
    clock: Clock | None = None,
    secret_store: SecretStore | None = None,
    data_factory: DemoDataFactory = _no_demo_data_adapter,
    alpaca_client_factory: AlpacaClientFactory | None = None,
) -> DemoResult:
    """Pornește și rulează modul Demo; orice refuz apare înaintea procesării (fail-closed).

    Compunere (design: tabelul modurilor): feed curent (Alpaca) + broker Alpaca **paper** (API
    real, bani simulați) + ceas real. Secvența de pași este identică cu Backtest/Shadow (Req 1.1);
    diferă brokerul (Alpaca paper prin `build_broker`, NU `SimBroker`) și reconcilierea activă
    (Req 11): la pornire, înaintea oricărui ordin nou, și periodic (interval ≤ 60 s).

    Seam-uri injectabile pentru determinism în teste: `clock` (implicit `WallClock`, nu este
    avansat de motor), `data_factory` (feed Alpaca sau reluare deterministă), `secret_store`
    (cheile brokerului) și `alpaca_client_factory` (client Alpaca fals, fără rețea). `code_version`
    este numai pentru teste.
    """
    base = base_dir if base_dir is not None else Path.cwd()
    wall = clock if clock is not None else WallClock()
    config = load_config(config_path)
    if config.environment != "demo":
        raise BootstrapError(f"run_demo necesită environment=demo, nu {config.environment!r}")
    if db_path is not None:
        config = config.model_copy(
            update={"run": config.run.model_copy(update={"db_path": db_path})}
        )
    stage = read_stage(stage_lock)
    check_startup(config, stage)  # refuză Live și endpoint-urile/conturile live (fail-closed)
    _require_mode(config)
    if config.costs is None:
        raise BootstrapError("secțiunea [costs] lipsește: Complete_Cost_Model este obligatoriu")
    data = data_factory(config, wall, base)

    code = code_version if code_version is not None else read_code_version(repo_root or base)
    snapshot = create_snapshot(
        config, stage, code, created_at=wall.now(), dataset_ids=[data.source_id]
    )
    core = build_core(config, snapshot.snapshot_id)

    resolved_db = Path(config.run.db_path)
    if not resolved_db.is_absolute():
        resolved_db = base / resolved_db
    conn = open_db(resolved_db)
    try:
        journal = Journal(conn)
        _record_snapshot(journal, snapshot)
        return _run_demo(
            config,
            stage,
            snapshot,
            core,
            data,
            wall,
            journal,
            str(resolved_db),
            secret_store=secret_store,
            alpaca_client_factory=alpaca_client_factory,
        )
    finally:
        conn.close()


def _run_demo(
    config: AppConfig,
    stage: StageInfo,
    snapshot: ConfigurationSnapshot,
    core: CoreComponents,
    data: DataAdapter,
    clock: Clock,
    journal: Journal,
    db_path: str,
    *,
    secret_store: SecretStore | None,
    alpaca_client_factory: AlpacaClientFactory | None,
) -> DemoResult:
    # Execuțiile Demo poartă moneda de raportare a rulării cu curs 1 (instrument și cont în aceeași
    # monedă, de exemplu SPY/USD cu cont USD): fără conversie, deci numerarul proiectat coincide cu
    # cel al contului și reconcilierea rămâne strictă (USD==USD).
    oms = OrderManager(
        JournalOmsSink(journal), fx_fn=reporting_identity(config.reporting_currency)
    )

    def make_kill_switch(ks_clock: Clock) -> KillSwitch:
        return KillSwitch(
            JournalKillSwitchStore(journal),
            clock=ks_clock,
            config=config.kill_switch,
            open_orders=_open_orders(oms),
        )

    adapters, kill_switch = build_demo_adapters(
        config,
        stage,
        core=core,
        data=data,
        clock=clock,
        journal=journal,
        kill_switch_factory=make_kill_switch,
        secret_store=secret_store,
        alpaca_client_factory=alpaca_client_factory,
    )
    reconciler = Reconciler(
        kill_switch,
        journal,
        clock,
        ReconciliationTolerances(reporting_currency=config.reporting_currency),
    )
    alpaca = adapters.broker.inner
    if not isinstance(alpaca, AlpacaBrokerAdapter):  # pragma: no cover - garantat de build_broker
        raise BootstrapError("compunerea Demo necesită brokerul Alpaca paper")

    # Reconcilierea este condusă din bucla de rulare (Reconciler nu pornește fire proprii):
    # - la pornire, înaintea primului ordin nou (Req 11.1);
    # - periodic, între blocuri de evenimente, când `should_run` o cere (Req 11.3, interval ≤ 60 s).
    driver = _DemoReconDriver(reconciler, alpaca, oms, clock, config.environment)

    def before_run(portfolio: Portfolio) -> None:
        driver.reconcile(ReconciliationReason.STARTUP, portfolio.snapshot())
        driver.startup = True

    def make_periodic_listener(portfolio: Portfolio) -> MarketListener:
        # Ascultătorul rulează la fiecare bară, înaintea strategiei (deci înaintea ordinelor noi
        # ale barei). Verifică hook-ul `should_run` al reconcilierului (interval ≤ 60 s) pe ceasul
        # injectat; reconcilierea periodică nu pornește fire proprii (Req 11.3).
        def listener(_event: MarketEvent) -> None:
            driver.maybe_periodic(portfolio.snapshot())

        return listener

    run_id = f"dm-{snapshot.snapshot_id[:16]}"
    run = _run_engine(
        config,
        snapshot,
        core,
        adapters,
        kill_switch,
        oms,
        journal,
        run_id=run_id,
        advance_clock=False,
        broker_name=DEMO_BROKER_NAME,
        before_run=before_run,
        extra_listener_factory=make_periodic_listener,
    )
    return DemoResult(
        snapshot_id=snapshot.snapshot_id,
        run_id=run_id,
        source_id=adapters.dataset_id,
        db_path=db_path,
        events_processed=run.processed,
        orders=run.orders,
        fills=run.fills,
        realized_gross_eur=run.realized_gross_eur,
        unrealized_gross_eur=run.unrealized_gross_eur,
        costs=run.costs,
        journal_head=run.journal_head,
        journal_verified=run.journal_verified,
        reconciliations=driver.count,
        startup_reconciled=driver.startup,
        kill_switch_active=kill_switch.state(clock.now()).global_active,
        order_states=run.order_states,
    )


class _DemoReconDriver:
    """Conduce `Reconciler` din bucla Demo: pornire + periodic, pe ceasul injectat (Req 11.1–11.3).

    `Reconciler` nu pornește fire proprii; acest driver citește `should_run`/`interval_seconds` și
    îi dă snapshot-ul brokerului împreună cu proiecția internă (portofoliu + OMS). Fail-closed:
    un snapshot incomplet activează deja Kill_Switch GLOBAL în `Reconciler` (Req 11.6).
    """

    def __init__(
        self,
        reconciler: Reconciler,
        broker: AlpacaBrokerAdapter,
        oms: OrderManager,
        clock: Clock,
        mode: str,
    ) -> None:
        self._reconciler = reconciler
        self._broker = broker
        self._oms = oms
        self._clock = clock
        self._mode = mode
        self.count = 0
        self.startup = False
        self._last_run: datetime | None = None

    def reconcile(self, reason: ReconciliationReason, portfolio_state: PortfolioState) -> None:
        result = self._reconciler.reconcile(
            self._broker.snapshot(), portfolio_state, self._oms, reason=reason
        )
        self.count += 1
        self._last_run = result.ts

    def maybe_periodic(self, portfolio_state: PortfolioState) -> None:
        if self._reconciler.should_run(self._clock.now(), self._mode, last_run=self._last_run):
            self.reconcile(ReconciliationReason.PERIODIC, portfolio_state)
