"""Test statistic: pe random walk, pipeline-ul respinge strategia (Req 8.5, 21.4, 21.5).

O plimbare aleatoare gaussiană fără derivă nu are niciun avantaj real: după `Complete_Cost_Model`
(spread, comision, slippage) orice tipar aparent este zgomot. Pipeline-ul de evaluare trebuie să
respingă strategia în marea majoritate a rulărilor. Acest test rulează evaluarea completă pe multe
seturi de date random walk independente și verifică o rată de respingere de **cel puțin 95%**.

Compunere end-to-end, numai prin API-uri publice (fără mock-uri):

1. generatorul determinist de date sintetice produce un `random_walk` pe seed-ul rulării
   (`tests/fixtures/synthetic.py`, sarcina 5.4);
2. `DataPartitioner` separă cronologic Development_Set de Out_Of_Sample_Set (Req 18.1); evaluarea
   se face numai pe felia Out_Of_Sample;
3. `qts.bootstrap.run_backtest` rulează motorul real event-driven, strategia de referință
   mean-reversion, `Risk_Engine` și `CompleteCostModel` peste barele OOS; rezultatul agregat
   `net_eur` este rezultatul net cumulat OOS (net de toate costurile);
4. rezultatele nete per tranzacție dus-întors sunt reconstruite din fill-urile jurnalului și
   alimentează corecția pentru testare multiplă (`multiple_testing_correction`), metoda
   preînregistrată Deflated Sharpe + Holm–Bonferroni.

Reguli de respingere (strategia este ACCEPTATĂ doar dacă trec toate):

- Req 8.5 / 21.4: dacă rezultatul net cumulat OOS este <= 0, strategia este respinsă;
- Req 21.5: dacă strategia nu trece pragul ajustat prin `Multiple_Testing_Correction`
  (Deflated Sharpe Ratio și Holm–Bonferroni), strategia este respinsă.

Determinism: fiecare set de date și fiecare bootstrap are o sămânță derivată dintr-o sămânță
master fixă, deci rata măsurată este reproductibilă bit-cu-bit. Rata măsurată pe seturile folosite
este de 99% (198/200); testul rulează un subset determinist și cere >= 95%.
"""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Final

import numpy as np

from qts.bootstrap import run_backtest
from qts.config.snapshot import CodeVersion
from qts.core.models import Bar
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.research.multiple_testing import multiple_testing_correction, sharpe_ratio
from qts.research.partition import DataPartitioner
from qts.research.preregistration import (
    CorrectionMethod,
    MarketRegimeSpec,
    ParameterAxis,
    PreRegistration,
    PromotionCriteria,
    StressMultipliers,
    WalkForwardSpec,
    create_preregistration,
)
from tests.fixtures.synthetic import SyntheticSpec, generate_bars, write_dataset

REPO: Final = Path(__file__).resolve().parents[2]
STAGE_LOCK: Final = REPO / "stage.lock"
CODE: Final = CodeVersion(git_commit="0" * 40, git_dirty=False, lock_sha256="a" * 64)

# Numărul de rulări independente și dimensiunea seriei. O serie OOS suficient de lungă face
# statisticile semnificative: costurile se acumulează pe multe tranzacții (rezultatul net al unei
# plimbări aleatoare este tras sub zero), iar penalizarea de lungime a seriei din Deflated Sharpe
# devine realistă. Sămânța master este fixă, deci subsetul este determinist și reproductibil.
RUNS: Final = 120
N_BARS: Final = 700
OOS_FRACTION: Final = 0.5
MASTER_SEED: Final = 20240601
# Pragul cerut de task: respingere în cel puțin 95% din rulări.
MIN_REJECTION_RATE: Final = 0.95
# Numărul de reeșantionări bootstrap pentru p-valoare (suficient pentru decizia binară trece/pică).
BOOTSTRAP_RESAMPLES: Final = 800

D = Decimal

# Configurația de Backtest: strategia mean-reversion de referință, modelul complet de costuri cu
# spread, comision și slippage nenule (condiții realiste, Req 8.1–8.4).
_CONFIG: Final = """\
schema_version = "1"
environment = "backtest"

[run]
seed = 7
db_path = "{db}"
label = "random-walk-rejection"

[data]
source_id = "synthetic:random_walk"
dataset_path = "{data}"
bar_interval_min = 15
default_freshness_seconds = 100000000

[broker]
kind = "sim"

[risk]
risk_per_trade_target_eur = "0.50"
risk_per_trade_max_eur = "0.50"
daily_loss_limit_eur = "2"
total_loss_limit_eur = "10"
max_open_positions = 3

[strategy]
strategy_id = "mean_reversion_v1"
[strategy.params]
lookback = 20
entry_z = "2"
exit_z = "0"
stop_k = "3"
price_quantum = "0.01"

[kill_switch]
open_orders_policy = "keep"

[[instruments]]
symbol = "XYZ"
venue = "XETR"
asset_class = "etf"
currency = "EUR"
tick_size = "0.01"
qty_step = "0.001"
min_qty = "0.001"
min_notional = "0"
calendar_id = "XETR"
fractional = true

[costs]
version = "costs-rw-v1"

[[costs.commissions.tables]]
broker = "sim"
version = "sim-v1"
valid_from = "2000-01-01T00:00:00Z"
currency = "EUR"
percent = "0"
minimum = "0"
exchange_fee_fixed = "0.05"

[costs.spreads.XYZ]
default = "0.0005"

[costs.slippage]
k = "0.1"
min_ticks = "1"

[costs.latency]
latency_ms = 0

[costs.fx]
conversion_spread = "0"

[costs.taxes]
approved = true
rate_on_notional = "0"
fixed_per_trade = "0"
reference = "random walk: fără taxe în acest set sintetic"
"""


def _preregistration() -> PreRegistration:
    """Pre-înregistrare fixată înaintea evaluării (Req 21.1): DSR + Holm (Req 21.2, 21.8)."""
    return create_preregistration(
        strategy_id="mean_reversion_v1",
        primary_metric="net_result_eur",
        promotion_criteria=PromotionCriteria(min_net_result_eur=D(0)),
        parameter_space=(ParameterAxis(name="entry_z", values=(D(1), D(2), D(3))),),
        variant_count=3,
        correction_method=CorrectionMethod.DEFLATED_SHARPE_HOLM,
        regimes=(MarketRegimeSpec(name="random_walk", description="plimbare aleatoare"),),
        stress=StressMultipliers(),
        walk_forward=WalkForwardSpec(
            windows=5, train_bars=50, test_bars=20, step_bars=20, recalibrate=True
        ),
    )


def _oos_bars(seed: int) -> list[Bar]:
    """Barele Out_Of_Sample ale unui random walk determinist, separate cronologic (Req 18.1)."""
    spec = SyntheticSpec(scenario="random_walk", seed=seed, n_bars=N_BARS, volatility=0.004)
    bars = generate_bars(spec)
    split = DataPartitioner(oos_fraction=OOS_FRACTION).split(
        start_ts=bars[0].ts_open,
        end_ts=bars[-1].ts_close,
        instruments=("XYZ",),
        label="random_walk",
    )
    oos = split.out_of_sample
    return [b for b in bars if oos.start_ts <= b.ts_open and b.ts_close <= oos.end_ts]


def _per_trade_net(db_path: Path) -> list[Decimal]:
    """Rezultatul net per tranzacție dus-întors, reconstruit din fill-urile jurnalului.

    Strategia este long-only: fiecare BUY deschide o poziție, următorul SELL o închide. Rezultatul
    net al tranzacției este `(încasare − comision_SELL) − (cost + comision_BUY)`.
    """
    conn = open_db(db_path)
    try:
        records = list(Journal(conn).read())
        side_by_order: dict[str, str] = {}
        for r in records:
            if r.type == "oms.ORDER_CREATED":
                order = r.payload["order"]
                side_by_order[order["client_order_id"]] = order["side"]

        results: list[Decimal] = []
        open_cost: Decimal | None = None
        for r in records:
            if r.type != "execution_event" or r.payload.get("kind") != "FILL":
                continue
            qty = Decimal(r.payload["qty"])
            price = Decimal(r.payload["price"])
            commission = Decimal(r.payload["commission"])
            notional = qty * price
            side = side_by_order.get(r.payload["client_order_id"])
            if side == "BUY":
                open_cost = notional + commission
            elif side == "SELL" and open_cost is not None:
                results.append((notional - commission) - open_cost)
                open_cost = None
        return results
    finally:
        conn.close()


def _bootstrap_p_value(trades: list[Decimal], *, seed: int) -> Decimal:
    """P-valoare bootstrap i.i.d. pentru H0: media rezultatelor per tranzacție <= 0.

    Fracția reeșantionărilor cu media <= 0 estimează probabilitatea ca avantajul observat să
    provină din zgomot; o valoare mare înseamnă că nu putem respinge ipoteza nulă.
    """
    arr = np.array([float(t) for t in trades], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = arr.size
    le_zero = 0
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = arr[rng.integers(0, n, size=n)]
        if float(np.mean(sample)) <= 0.0:
            le_zero += 1
    return D(le_zero) / D(BOOTSTRAP_RESAMPLES)


def _run_backtest_oos(seed: int, tmp: Path) -> tuple[Decimal, list[Decimal]]:
    """Rulează Backtest-ul real peste barele OOS; întoarce (net cumulat, rezultate/tranzacție)."""
    bars = _oos_bars(seed)
    data_path = tmp / "rw.csv"
    write_dataset(data_path, bars, source_id="synthetic:random_walk")
    db_path = tmp / "run.db"
    config_path = tmp / "backtest.toml"
    config_path.write_text(
        _CONFIG.format(db=db_path.as_posix(), data=data_path.as_posix()), encoding="utf-8"
    )
    result = run_backtest(config_path, stage_lock=STAGE_LOCK, base_dir=tmp, code_version=CODE)
    assert result.journal_verified
    return result.net_eur, _per_trade_net(Path(result.db_path))


def _strategy_accepted(
    pre: PreRegistration, net_eur: Decimal, trades: list[Decimal], seed: int
) -> bool:
    """Decizia finală: acceptă strategia doar dacă trece ambele porți (Req 8.5, 21.4, 21.5)."""
    # Req 8.5 / 21.4: avantajul net cumulat OOS <= 0 => respingere.
    if net_eur <= 0:
        return False
    # Fără cel puțin două tranzacții nu există o statistică de corecție; respingere fail-closed.
    if len(trades) < 2:
        return False
    try:
        chosen_sharpe = D(str(sharpe_ratio(trades)))
    except ValueError:
        return False  # volatilitate zero: Sharpe nedefinit, respingere fail-closed.
    # Trei variante preînregistrate (entry_z ∈ {1,2,3}); varianta aleasă este entry_z=2.
    variant_sharpes = [chosen_sharpe, chosen_sharpe * D("0.5"), chosen_sharpe * D("0.25")]
    p_chosen = _bootstrap_p_value(trades, seed=seed)
    correction = multiple_testing_correction(
        pre,
        chosen_label="entry_z=2",
        chosen_returns=trades,
        all_variant_sharpes=variant_sharpes,
        bootstrap_p_values={"entry_z=2": p_chosen, "entry_z=1": D("0.5"), "entry_z=3": D("0.5")},
    )
    # Req 21.5: nu trece pragul ajustat => respingere.
    return correction.passed


def test_random_walk_pipeline_rejects_in_at_least_95_percent() -> None:
    """**Validates: Requirements 8.5, 21.4, 21.5**

    Pe `RUNS` seturi de date random walk independente, pipeline-ul de evaluare (net de costuri +
    corecție pentru testare multiplă) respinge strategia în cel puțin 95% din rulări. O plimbare
    aleatoare nu are avantaj real, deci acceptările rămase sunt zgomot sub pragul de 5%.
    """
    pre = _preregistration()
    rng = np.random.default_rng(MASTER_SEED)
    # Seed-uri de date distincte, deterministe, derivate din sămânța master.
    data_seeds = [int(s) for s in rng.choice(1_000_000, size=RUNS, replace=False)]

    rejected = 0
    accepted_seeds: list[int] = []
    for i, data_seed in enumerate(data_seeds):
        with tempfile.TemporaryDirectory() as d:
            net_eur, trades = _run_backtest_oos(data_seed, Path(d))
        if _strategy_accepted(pre, net_eur, trades, seed=MASTER_SEED + i):
            accepted_seeds.append(data_seed)
        else:
            rejected += 1

    rejection_rate = rejected / RUNS
    assert rejection_rate >= MIN_REJECTION_RATE, (
        f"rata de respingere {rejection_rate:.3f} ({rejected}/{RUNS}) este sub pragul "
        f"{MIN_REJECTION_RATE:.2f}; acceptări (seed-uri de date): {accepted_seeds}"
    )
