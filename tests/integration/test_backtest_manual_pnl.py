"""Backtest de integrare pe date sintetice cu rezultat calculat MANUAL (Req 7.3, 8.4).

Acest test rulează un Backtest COMPLET prin compunerea reală (`bootstrap.run_backtest`:
schema de configurație, snapshot, jurnal cu lanț hash, motor event-driven, strategie, risc,
`SimBroker` învelit în `FailSafeBlock`, portofoliu) și verifică fiecare rezultat cu numere
derivate exclusiv cu creionul. Nimic nu este aproximat: toate aserțiunile sunt egalități
`Decimal`.

Setul de date (10 bare XYZ de 15 min, EUR, tick 0.01) este construit pe cheia strategiei
mean-reversion cu `lookback = 4`, `entry_z = 1`, `exit_z = 0`, `stop_k = 1`. Prețurile de
închidere alese sunt:

    idx:    0      1      2      3      4      5      6      7      8      9
    close: 100.00 100.00 100.00 100.00  99.00  98.00 101.00 100.00 100.00 100.00

iar deschiderea fiecărei bare este închiderea barei precedente (prima bară deschide la 100.00).

Modelul de costuri este ales cu numere rotunde ca fiecare componentă să fie exactă:
- slippage: `k = 0`, `min_ticks = 0`      → slippage = 0 (independent de σ_bar / ADV);
- latență:  `latency_ms = 0`              → latență = 0 (nu există `price_after_latency` în
  `estimate`; `SimBroker` folosește deschiderea barei *t+1*, fără întârziere suplimentară);
- FX:       instrument în EUR             → fx_conversion = 0 (fără conversie);
- taxe:     aprobate, rate 0, fix 0       → taxe = 0;
- spread:   fracție configurată 0.0002    → jumătate de spread = 0.0002 × preț / 2;
- comision: `exchange_fee_fixed = 0.05`, `percent = 0`, `minimum = 0` → 0.05 EUR fix per ordin.

Astfel singura categorie de cost care intră în numerar/PnL (prin lanțul OMS `commission_only`)
este comisionul; jumătatea de spread este încorporată în prețul de execuție al `SimBroker`
(deschiderea barei *t+1* ± jumătate de spread, rotunjit advers la tick), deci apare în brut.

Derivarea manuală a z-scorului (medie și abatere standard DE POPULAȚIE pe ultimele 4 închideri,
inclusiv bara curentă), per bară:

    idx 0 (08:15): 1 închidere vizibilă  → NONE  (INSUFFICIENT_HISTORY)
    idx 1 (08:30): 2 închideri           → NONE  (INSUFFICIENT_HISTORY)
    idx 2 (08:45): 3 închideri           → NONE  (INSUFFICIENT_HISTORY)
    idx 3 (09:00): [100,100,100,100]     → std = 0 → NONE (ZERO_VOLATILITY)
    idx 4 (09:15): [100,100,100,99]      → medie 99.75, std √0.1875 ≈ 0.4330127,
                   z = (99 − 99.75)/std ≈ −1.7320508 ≤ −1 → ENTER_LONG
                   stop = floor_quantum(close − stop_k·std) = floor_0.01(99 − 0.4330127)
                        = floor_0.01(98.5669873) = 98.56
    idx 5 (09:30): în poziție; stopul se evaluează ÎNTÂI: low[5] = 97.90 ≤ 98.56 → EXIT (STOP)
    idx 6..9     : fără poziție, z nu atinge pragul de intrare → NONE (NO_RULE_MATCHED)

Execuția (model `SimBroker`, Req 7.3 „Latență”: ordinul emis la închiderea barei *t* se execută
la DESCHIDEREA barei *t+1*, ajustată cu jumătatea de spread; slippage 0):

- intrare: semnal ENTER_LONG la închiderea barei 4 → BUY executat la deschiderea barei 5.
  open[5] = close[4] = 99.00; jumătate de spread = 0.0002 × 99.00 / 2 = 0.0099;
  preț BUY = ceil_0.01(99.00 + 0.0099) = ceil_0.01(99.0099) = 99.01.
- ieșire:  semnal EXIT la închiderea barei 5 → SELL executat la deschiderea barei 6.
  open[6] = close[5] = 98.00; jumătate de spread = 0.0002 × 98.00 / 2 = 0.0098;
  preț SELL = floor_0.01(98.00 − 0.0098) = floor_0.01(97.9902) = 97.99.

Dimensionarea (Risk_Engine, buget 0.50 EUR/tranzacție, numerar 100 EUR): entry = 99.00,
stop = 98.56. Costul dus-întors are o parte fixă (2 × 0.05 comision) și o parte proporțională
(jumătatea de spread × cantitate pe fiecare picior). Riscul de preț pe unitate este
99.00 − 98.56 = 0.44 EUR. Formula din design dă candidatul `qty = 0.870` (multiplu de
`qty_step = 0.001`), care se încadrează exact în buget și numerar fără reducere:
    trade_risk(0.870) = 0.870·0.44 + spread_in(0.870) + spread_out(0.870) + 0.10
                      = 0.3828 + 0.00861300 + 0.00857472 + 0.10 = 0.49998772 ≤ 0.50 ✓
    cash_required(0.870) = 0.870·99.00 + (0.00861300 + 0.05) = 86.18861300 ≤ 100 ✓
Cantitatea 0.870 este deci cea dimensionată de motorul de risc real.

PnL (metoda costului mediu, totul într-o singură zi de tranzacționare, poziție închisă la final):
    bază de cost  = 0.870 × 99.01 = 86.1387 EUR
    încasare      = 0.870 × 97.99 = 85.2513 EUR
    brut realizat = 85.2513 − 86.1387 = −0.8874 EUR
    nerealizat    = 0 (poziție închisă)
    comisioane    = 0.05 + 0.05 = 0.10 EUR  (singura categorie din `report.costs`)
    net           = brut − costuri = −0.8874 − 0.10 = −0.9874 EUR
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from qts.bootstrap import run_backtest
from qts.config.snapshot import CodeVersion
from qts.core.models import Bar, CostBreakdown, Instrument
from qts.costs.model import CompleteCostModel, CostContext, OrderSpec
from qts.persistence.audit import reconstruct_decision_chain, verify_journal
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from tests.fixtures.synthetic import write_dataset

REPO = Path(__file__).resolve().parents[2]
STAGE_LOCK = REPO / "stage.lock"
CODE = CodeVersion(git_commit="0" * 40, git_dirty=False, lock_sha256="a" * 64)

STEP = timedelta(minutes=15)
START = datetime(2024, 1, 2, 8, 0, tzinfo=UTC)

# (close, high, low) per bară; open = close-ul precedent (prima bară: open = 100.00).
# High/low aleși ca invariantele OHLC să fie respectate și ca stopul 98.56 să NU fie atins
# înainte de bara 5 (low[4] = 98.90 > 98.56), apoi atins exact la bara 5 (low[5] = 97.90).
_ROWS: tuple[tuple[str, str, str], ...] = (
    ("100.00", "100.10", "99.90"),  # 0
    ("100.00", "100.10", "99.90"),  # 1
    ("100.00", "100.10", "99.90"),  # 2
    ("100.00", "100.10", "99.90"),  # 3
    ("99.00", "100.10", "98.90"),  # 4  -> ENTER_LONG
    ("98.00", "99.10", "97.90"),  # 5  -> EXIT (stop)
    ("101.00", "101.10", "97.90"),  # 6
    ("100.00", "101.10", "99.90"),  # 7
    ("100.00", "100.10", "99.90"),  # 8
    ("100.00", "100.10", "99.90"),  # 9
)

_CONFIG = """\
schema_version = "1"
environment = "backtest"

[run]
seed = 7
db_path = "{db}"
label = "manual-pnl"

[data]
source_id = "synthetic:manual"
dataset_path = "{data}"
bar_interval_min = 15
# Prospețime foarte largă: barele istorice nu expiră în timpul reluării.
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
lookback = 4
entry_z = "1"
exit_z = "0"
stop_k = "1"
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
version = "costs-manual-v1"

[[costs.commissions.tables]]
broker = "sim"
version = "sim-v1"
valid_from = "2000-01-01T00:00:00Z"
currency = "EUR"
percent = "0"
minimum = "0"
exchange_fee_fixed = "0.05"

[costs.spreads.XYZ]
default = "0.0002"

[costs.slippage]
k = "0"
min_ticks = "0"

[costs.latency]
latency_ms = 0

[costs.fx]
conversion_spread = "0"

[costs.taxes]
approved = true
rate_on_notional = "0"
fixed_per_trade = "0"
reference = "manual: fără taxe în acest set sintetic"
"""

# Instrumentul exact din configurație, pentru verificarea separată a modelului de costuri.
_INSTRUMENT = Instrument(
    symbol="XYZ",
    venue="XETR",
    asset_class="etf",
    currency="EUR",
    tick_size=Decimal("0.01"),
    qty_step=Decimal("0.001"),
    min_qty=Decimal("0.001"),
    min_notional=Decimal("0"),
    calendar_id="XETR",
    fractional=True,
)


def _build_bars() -> list[Bar]:
    bars: list[Bar] = []
    prev_close = Decimal("100.00")
    for i, (close_s, high_s, low_s) in enumerate(_ROWS):
        ts_open = START + i * STEP
        open_ = Decimal("100.00") if i == 0 else prev_close
        close = Decimal(close_s)
        high = max(Decimal(high_s), open_, close)
        low = min(Decimal(low_s), open_, close)
        bars.append(
            Bar(
                instrument="XYZ",
                ts_open=ts_open,
                ts_close=ts_open + STEP,
                interval_min=15,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=Decimal("1000"),
            )
        )
        prev_close = close
    return bars


def _write_config_and_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_path = data_dir / "manual.csv"
    write_dataset(data_path, _build_bars(), source_id="synthetic:manual")
    db_path = tmp_path / "run.db"
    config_path = tmp_path / "backtest.toml"
    config_path.write_text(
        _CONFIG.format(db=db_path.as_posix(), data=data_path.as_posix()), encoding="utf-8"
    )
    return config_path


def test_manual_backtest_signals_executions_costs_and_net_pnl(tmp_path: Path) -> None:
    config_path = _write_config_and_data(tmp_path)

    result = run_backtest(config_path, stage_lock=STAGE_LOCK, base_dir=tmp_path, code_version=CODE)

    # ---- numărul de evenimente, ordine și execuții ---------------------------------------
    # Motorul procesează cele 10 bare plus evenimentele de execuție ale celor două ordine
    # (câte un ACK și un FILL fiecare): 10 + 4 = 14 evenimente.
    assert result.events_processed == len(_ROWS) + 4
    assert result.rejected_rows == 0
    assert result.orders == 2  # exact o intrare și o ieșire
    assert result.fills == 2
    assert result.order_states == {"FILLED": 2}

    # ---- PnL brut, nerealizat, costuri și net (toate exacte, Req 8.4) --------------------
    assert result.realized_gross_eur == Decimal("-0.8874")
    assert result.unrealized_gross_eur == Decimal("0")
    assert result.gross_eur == Decimal("-0.8874")
    # `report.costs` conține numai comisionul (lanțul OMS `commission_only`); celelalte
    # categorii sunt zero prin construcția modelului de costuri (vezi docstring).
    assert result.costs == CostBreakdown(commission=Decimal("0.10"))
    assert result.costs.total == Decimal("0.10")
    assert result.net_eur == Decimal("-0.9874")
    # net = brut − costuri, prin definiție.
    assert result.net_eur == result.gross_eur - result.costs.total

    # ---- jurnalul se verifică și lanțul deciziei se reconstruiește -----------------------
    assert result.journal_verified

    conn = open_db(result.db_path)
    journal = Journal(conn)
    try:
        assert verify_journal(journal, expected_head=result.journal_head).ok

        records = list(journal.read())
        # Prima înregistrare: snapshot-ul rulării.
        assert records[0].type == "config_snapshot"
        assert records[0].correlation_id == result.snapshot_id

        # Toate semnalele, în ordine, cu acțiunea și codul de motiv derivate manual.
        signals = [r for r in records if r.type == "signal"]
        observed = [(r.payload["action"], r.payload["reason_code"]) for r in signals]
        assert observed == [
            ("NONE", "INSUFFICIENT_HISTORY"),  # idx 0
            ("NONE", "INSUFFICIENT_HISTORY"),  # idx 1
            ("NONE", "INSUFFICIENT_HISTORY"),  # idx 2
            ("NONE", "ZERO_VOLATILITY"),  # idx 3 (std = 0)
            ("ENTER_LONG", "ENTRY_ZSCORE"),  # idx 4
            ("EXIT", "EXIT_STOP"),  # idx 5 (low <= stop)
            ("NONE", "NO_RULE_MATCHED"),  # idx 6
            ("NONE", "NO_RULE_MATCHED"),  # idx 7
            ("NONE", "NO_RULE_MATCHED"),  # idx 8
            ("NONE", "NO_RULE_MATCHED"),  # idx 9
        ]

        # Semnalul de intrare: stopul exact calculat manual.
        entry_signal = next(r for r in signals if r.payload["action"] == "ENTER_LONG")
        assert entry_signal.payload["stop_price"] == "98.56"
        assert entry_signal.ts == datetime(2024, 1, 2, 9, 15, tzinfo=UTC)

        # Order_Intent + decizia de risc pentru intrare: BUY, stop 98.56, qty 0.870.
        intents = {r.correlation_id: r for r in records if r.type == "order_intent"}
        risk = {r.correlation_id: r for r in records if r.type == "risk_decision"}
        entry_intent = intents[entry_signal.correlation_id]
        assert entry_intent.payload["side"] == "BUY"
        assert entry_intent.payload["ref_price"] == "99.00"
        assert entry_intent.payload["stop_price"] == "98.56"
        entry_risk = risk[entry_signal.correlation_id]
        assert entry_risk.payload["approved"] is True
        assert entry_risk.payload["qty"] == "0.870"

        # Order_Intent + decizia de risc pentru ieșire: SELL cu toată cantitatea deținută.
        exit_signal = next(r for r in signals if r.payload["action"] == "EXIT")
        exit_intent = intents[exit_signal.correlation_id]
        assert exit_intent.payload["side"] == "SELL"
        assert exit_intent.payload["requested_qty"] == "0.870"
        exit_risk = risk[exit_signal.correlation_id]
        assert exit_risk.payload["approved"] is True
        assert exit_risk.payload["qty"] == "0.870"

        # Execuțiile la deschiderea barei t+1, cu prețul exact ajustat cu jumătatea de spread.
        fills = [r for r in records if r.type == "execution_event" and r.payload["kind"] == "FILL"]
        assert len(fills) == 2
        buy_fill, sell_fill = fills
        assert buy_fill.payload["qty"] == "0.870"
        assert buy_fill.payload["price"] == "99.01"  # ceil_0.01(99.00 + 0.0099)
        assert buy_fill.payload["commission"] == "0.05000000"
        assert buy_fill.ts == datetime(2024, 1, 2, 9, 30, tzinfo=UTC)  # deschiderea barei 5
        assert sell_fill.payload["qty"] == "0.870"
        assert sell_fill.payload["price"] == "97.99"  # floor_0.01(98.00 − 0.0098)
        assert sell_fill.payload["commission"] == "0.05000000"
        assert sell_fill.ts == datetime(2024, 1, 2, 9, 45, tzinfo=UTC)  # deschiderea barei 6

        # Lanțul deciziei de intrare se reconstruiește complet (Req 24.5).
        chain = reconstruct_decision_chain(journal, entry_signal.correlation_id)
        assert chain.complete
        assert not chain.risk_rejected
        chain_types = [rec.type for rec in chain.records]
        for required in (
            "market_event",
            "signal",
            "order_intent",
            "risk_decision",
            "execution_event",
        ):
            assert required in chain_types
    finally:
        conn.close()


def test_cost_model_breakdown_matches_hand_computation() -> None:
    """Fiecare categorie de cost a modelului, pentru cantitatea și prețurile dimensionate.

    Sub-aserțiune focalizată (Req 8.1): confirmă că spread-ul este singura componentă nenulă
    în afara comisionului și că valoarea sa este exact jumătatea de spread × cantitate pe
    fiecare picior; slippage, latență, FX și taxe sunt zero prin construcția configurației.
    """
    model = _cost_model()
    qty = Decimal("0.870")
    ts = datetime(2024, 1, 2, 9, 15, tzinfo=UTC)
    # σ_bar și ADV sunt disponibile (modelul le cere chiar dacă `k = 0`), dar cu `k = 0`
    # slippage-ul este 0 indiferent de valorile lor.
    ctx = CostContext(
        broker="sim", sigma_bar=Decimal("0.01"), adv=Decimal("1000"), bar_interval_min=15
    )

    entry = model.estimate(
        OrderSpec(instrument=_INSTRUMENT, side="BUY", qty=qty, ts=ts, ref_price=Decimal("99.00")),
        None,
        ctx,
    )
    # jumătate de spread = 0.0002 × 99.00 / 2 = 0.0099; × 0.870 = 0.008613
    assert entry.spread == Decimal("0.00861300")
    assert entry.commission == Decimal("0.05000000")
    assert entry.slippage == Decimal("0")
    assert entry.latency == Decimal("0")
    assert entry.fx_conversion == Decimal("0")
    assert entry.taxes == Decimal("0")
    assert entry.total == Decimal("0.05861300")

    exit_ = model.estimate(
        OrderSpec(instrument=_INSTRUMENT, side="SELL", qty=qty, ts=ts, ref_price=Decimal("98.56")),
        None,
        ctx,
    )
    # jumătate de spread = 0.0002 × 98.56 / 2 = 0.009856; × 0.870 = 0.00857472
    assert exit_.spread == Decimal("0.00857472")
    assert exit_.commission == Decimal("0.05000000")
    assert exit_.slippage == Decimal("0")
    assert exit_.latency == Decimal("0")
    assert exit_.fx_conversion == Decimal("0")
    assert exit_.taxes == Decimal("0")
    assert exit_.total == Decimal("0.05857472")


def _cost_model() -> CompleteCostModel:
    from datetime import datetime as _dt

    from qts.costs.commission_tables import CommissionSchedule, CommissionTable
    from qts.costs.model import (
        CostModelConfig,
        FxConfig,
        LatencyConfig,
        SlippageConfig,
        SpreadSchedule,
        TaxConfig,
    )

    config = CostModelConfig(
        version="costs-manual-v1",
        commissions=CommissionSchedule(
            tables=(
                CommissionTable(
                    broker="sim",
                    version="sim-v1",
                    valid_from=_dt(2000, 1, 1, tzinfo=UTC),
                    currency="EUR",
                    percent=Decimal("0"),
                    minimum=Decimal("0"),
                    exchange_fee_fixed=Decimal("0.05"),
                ),
            )
        ),
        spreads={"XYZ": SpreadSchedule(default=Decimal("0.0002"))},
        slippage=SlippageConfig(k=Decimal("0"), min_ticks=Decimal("0")),
        latency=LatencyConfig(latency_ms=0),
        fx=FxConfig(conversion_spread=Decimal("0")),
        taxes=TaxConfig(approved=True, rate_on_notional=Decimal("0"), fixed_per_trade=Decimal("0")),
    )
    return CompleteCostModel(config)
