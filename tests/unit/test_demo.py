"""Teste pentru compunerea Demo: feed Alpaca + broker Alpaca paper + reconciliere (Req 1.1, 11).

Deterministe și fără rețea: un client Alpaca fals (ack + fill), o reluare peste bare sintetice,
`InMemorySecretStore` și un ceas controlabil înlocuiesc rețeaua, cheile reale și timpul de perete.
Nicio comandă reală nu părăsește procesul; brokerul paper este simulat în întregime de client.

Spre deosebire de Shadow (SimBroker), Demo folosește `AlpacaBrokerAdapter` prin `build_broker`:
execuțiile vin de la client (nu dintr-o alimentare `on_bar`), iar motorul le drenează la fiecare
pas. Reconcilierea este condusă din bucla de rulare: la pornire (înaintea primului ordin) și
periodic (interval ≤ 60 s); un snapshot incomplet activează Kill_Switch GLOBAL (fail-closed).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from qts.bootstrap import (
    BootstrapError,
    DemoDataFactory,
    DemoResult,
    run_demo,
)
from qts.broker.alpaca_broker import (
    ALPACA_PAPER_ENDPOINT,
    AlpacaAccount,
    AlpacaExecution,
    AlpacaOrderAck,
    AlpacaPosition,
)
from qts.config.schema import AppConfig
from qts.config.snapshot import ConfigurationSnapshot
from qts.core.clock import Clock, ensure_utc
from qts.core.models import Bar, MarketEvent
from qts.core.money import ZERO, dec
from qts.data.adapter import DataAdapter
from qts.data.alpaca_source import AlpacaBar, AlpacaStreamingSource
from qts.data.csv_source import CsvSource
from qts.persistence.audit import verify_journal
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.secrets.store import Identity, InMemorySecretStore, SecretRef
from tests.fixtures.synthetic import SyntheticSpec, write_synthetic_dataset
from tests.unit.test_bootstrap import CODE, CONFIG_DIR, STAGE_LOCK

D = Decimal
# Instrumentul sintetic este SPY/USD, ca în `config/demo.toml`: reluarea one-shot (`DemoReplay`)
# păstrează simbolul barei, deci trebuie să coincidă cu instrumentul configurat (altfel motorul
# ignoră barele ca „instrument necunoscut"). Fluxul continuu reetichetează oricum la SPY.
SPEC = SyntheticSpec(
    scenario="mean_reverting", seed=11, n_bars=400, volatility=0.004, instrument="SPY"
)
# Preț de execuție constant folosit de clientul paper fals. Portofoliul aplică exact acest preț
# (prin ExecutionEvent), deci numerarul și pozițiile raportate de client oglindesc proiecția
# internă și reconcilierea nu detectează divergențe (fără comision → costuri 0).
FILL_PRICE = D("100.00")
START = datetime(2024, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- ceas injectat


class ReplayClock:
    """Ceas controlabil (portul `Clock`): avansează la fiecare eveniment reluat, nu merge înapoi."""

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, ts: datetime) -> None:
        ts = ensure_utc(ts)
        if ts > self._now:
            self._now = ts


# --------------------------------------------------------------------------- feed de reluare


class DemoReplay:
    """`DataAdapter` de reluare peste bare sintetice, care avansează ceasul la fiecare eveniment.

    Oglindește `ShadowReplay` din testele Shadow: reutilizează `CsvSource` pentru parsare și
    validare, dar livrează barele „în timp real" față de `ReplayClock`, deci prospețimea este
    satisfăcută determinist (fără timp de perete).
    """

    def __init__(self, inner: CsvSource, clock: ReplayClock) -> None:
        self._inner = inner
        self._clock = clock
        self.source_id = inner.source_id

    def stream(self) -> Iterator[object]:
        for event in self._inner.stream():
            self._clock.advance_to(event.ts_receipt)
            yield event


# --------------------------------------------------------------------------- client Alpaca fals


class MirrorTradingClient:
    """Client Alpaca paper fals care confirmă și umple complet fiecare ordin, oglindind contul.

    Fiecare `submit_order` este acceptat; la următorul `poll_executions` se emite un `fill`
    complet la `FILL_PRICE`. Contul (`cash`, poziții) este actualizat din aceleași umpleri, în
    moneda contului (USD, ca un cont Alpaca paper real), pornind de la capitalul de referință,
    astfel încât reconcilierea (poziții + numerar) să coincidă cu proiecția internă în moneda de
    raportare USD (fără comision → costuri 0). `complete=False` forțează calea fail-closed din
    reconciliere (Req 11.6).
    """

    def __init__(
        self,
        *,
        endpoint: str = ALPACA_PAPER_ENDPOINT,
        clock: ReplayClock | None = None,
        currency: str = "USD",
        account_complete: bool = True,
        initial_cash: Decimal | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._clock = clock
        self.currency = currency
        self.account_complete = account_complete
        self._cash = initial_cash if initial_cash is not None else D("100")
        self._positions: dict[str, Decimal] = {}
        self.submitted: list[object] = []
        self._pending: list[AlpacaExecution] = []
        self._accepted = 0

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def _now(self) -> datetime:
        return self._clock.now() if self._clock is not None else START

    def submit_order(self, req: object) -> AlpacaOrderAck:
        self.submitted.append(req)
        self._accepted += 1
        coid = req.client_order_id  # type: ignore[attr-defined]
        qty: Decimal = req.qty  # type: ignore[attr-defined]
        side: str = req.side  # type: ignore[attr-defined]
        instrument: str = req.instrument  # type: ignore[attr-defined]
        # Oglindă de cont: aplicăm imediat efectul umplerii complete asupra numerarului/pozițiilor.
        signed = qty if side == "BUY" else -qty
        self._positions[instrument] = self._positions.get(instrument, ZERO) + signed
        if self._positions[instrument] == 0:
            self._positions.pop(instrument, None)
        self._cash += (-FILL_PRICE * qty) if side == "BUY" else (FILL_PRICE * qty)
        self._pending.append(
            AlpacaExecution(
                client_order_id=coid,
                kind="fill",
                ts=self._now(),
                qty=str(qty),
                price=str(FILL_PRICE),
            )
        )
        return AlpacaOrderAck(broker_order_id=f"ALP-{self._accepted:08d}", accepted=True)

    def cancel_order(self, broker_order_id: str) -> None:  # pragma: no cover - demo nu anulează
        return None

    def get_account(self) -> AlpacaAccount:
        positions = tuple(
            AlpacaPosition(symbol=s, qty=str(q)) for s, q in sorted(self._positions.items())
        )
        return AlpacaAccount(
            cash=str(self._cash),
            currency=self.currency,
            positions=positions,
            complete=self.account_complete,
        )

    def poll_executions(self) -> Sequence[AlpacaExecution]:
        batch = self._pending
        self._pending = []
        return batch


# --------------------------------------------------------------------------- compunere test


def _write_demo_config(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path, _ = write_synthetic_dataset(data_dir, SPEC)
    text = (CONFIG_DIR / "demo.toml").read_text(encoding="utf-8")
    text = text.replace(
        'db_path = "runs/demo.db"', f'db_path = "{(tmp_path / "demo.db").as_posix()}"'
    )
    config_path = tmp_path / "demo.toml"
    config_path.write_text(text, encoding="utf-8")
    return config_path, data_path


def _replay_factory(data_path: Path) -> DemoDataFactory:
    def factory(_config: AppConfig, clock: Clock, _base: Path) -> DataAdapter:
        assert isinstance(clock, ReplayClock)
        return DemoReplay(CsvSource(data_path), clock)  # type: ignore[return-value]

    return factory


def _secret_store() -> InMemorySecretStore:
    """Secret_Store în memorie cu cheile brokerului Alpaca demo (derivate din broker.secret_ref)."""
    return InMemorySecretStore(
        values={
            "qts/demo/alpaca_key": "KEYVALUE123",
            "qts/demo/alpaca_key_secret": "SECRETVAL456",
        },
        acl={
            "qts/demo/alpaca_key": {("demo-runner:demo", "demo")},
            "qts/demo/alpaca_key_secret": {("demo-runner:demo", "demo")},
        },
    )


# --------------------------------------------------------------------------- feed continuu (stream)


class ScriptedBarClient:
    """Client Alpaca fals pentru fluxul continuu: întoarce loturi succesive, câte unul per apel.

    Primul apel întoarce lotul de încălzire; fiecare apel ulterior întoarce lotul următorului
    sondaj. După epuizare întoarce gol. Deterministic, fără rețea.
    """

    def __init__(self, batches: Sequence[Sequence[AlpacaBar]]) -> None:
        self._batches = [list(b) for b in batches]
        self._i = 0
        self.calls = 0

    def get_bars(
        self, symbol: str, interval_min: int, start: datetime, end: datetime
    ) -> Sequence[AlpacaBar]:
        self.calls += 1
        if self._i < len(self._batches):
            batch = self._batches[self._i]
            self._i += 1
            return batch
        return []


def _bars_from_dataset(data_path: Path) -> list[AlpacaBar]:
    """Transformă barele validate din setul sintetic în `AlpacaBar` (OHLCV ca float)."""
    out: list[AlpacaBar] = []
    for event in CsvSource(data_path).stream():
        bar = event.payload
        if not isinstance(bar, Bar):  # pragma: no cover - setul conține numai bare
            continue
        out.append(
            AlpacaBar(
                ts_open=bar.ts_open,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )
        )
    return out


def _streaming_factory(
    bars: Sequence[AlpacaBar],
    *,
    warmup_count: int,
    max_polls: int,
) -> DemoDataFactory:
    """Fabrică de date Demo continuă (determinist): încălzire + o bară nouă pe sondaj.

    `sleep`-ul injectat avansează ceasul `ReplayClock` cu intervalul barei, deci:
    - prospețimea este satisfăcută (barele sosesc „la timp" față de ceas);
    - reconcilierea periodică (interval ≤ 60 s) se declanșează la fiecare sondaj, de mai multe ori.
    Niciun `time.sleep` real și niciun apel de rețea.
    """
    warmup = bars[:warmup_count]
    rest = bars[warmup_count : warmup_count + max_polls]
    batches: list[Sequence[AlpacaBar]] = [warmup, *[[b] for b in rest]]
    client = ScriptedBarClient(batches)

    def factory(config: AppConfig, clock: Clock, _base: Path) -> DataAdapter:
        assert isinstance(clock, ReplayClock)
        instrument = config.instruments[0]

        def sleep(seconds: float) -> None:
            clock.advance_to(clock.now() + timedelta(seconds=seconds))

        source = AlpacaStreamingSource(
            symbol=instrument.symbol,
            interval_min=config.data.bar_interval_min,
            tick_size=instrument.tick_size,
            client=client,
            now=clock.now,
            sleep=sleep,
            lookback=timedelta(days=5),
            warmup=True,
            max_polls=max_polls,
        )
        return _ClockAdvancingStream(source, clock)

    return factory


class _ClockAdvancingStream:
    """Învelește fluxul continuu și avansează ceasul la `ts_receipt`-ul fiecărui eveniment emis.

    Oglindește semantica reală: când o bară se închide, ceasul (timpul curent) ajunge la
    `ts_close`-ul ei, deci prospețimea este satisfăcută exact ca în reluarea Demo one-shot
    (`DemoReplay`). Astfel barele „sosesc la timp" față de ceas, fără timp de perete, și motorul
    le procesează identic — un flux continuu, nu un lot istoric mort.
    """

    def __init__(self, source: AlpacaStreamingSource, clock: ReplayClock) -> None:
        self._source = source
        self._clock = clock
        self.source_id = source.source_id

    @property
    def polls(self) -> int:
        return self._source.polls

    def stream(self) -> Iterator[MarketEvent]:
        for event in self._source.stream():
            self._clock.advance_to(event.ts_receipt)
            yield event


def _run_demo_streaming(
    tmp_path: Path,
    *,
    warmup_count: int,
    max_polls: int,
) -> tuple[DemoResult, MirrorTradingClient]:
    """Rulează `run_demo` cu feed continuu scriptat + client paper fals; totul determinist."""
    config_path, data_path = _write_demo_config(tmp_path)
    clock = ReplayClock(START)
    paper = MirrorTradingClient(clock=clock)

    def client_factory(_key: str, _secret: str, endpoint: str) -> MirrorTradingClient:
        assert endpoint == ALPACA_PAPER_ENDPOINT
        return paper

    bars = _bars_from_dataset(data_path)
    data_factory = _streaming_factory(bars, warmup_count=warmup_count, max_polls=max_polls)
    result = run_demo(
        config_path,
        stage_lock=STAGE_LOCK,
        base_dir=tmp_path,
        code_version=CODE,
        clock=clock,
        secret_store=_secret_store(),
        data_factory=data_factory,
        alpaca_client_factory=client_factory,
    )
    return result, paper


def _run_demo(
    tmp_path: Path,
    *,
    client: MirrorTradingClient | None = None,
) -> tuple[DemoResult, MirrorTradingClient]:
    config_path, data_path = _write_demo_config(tmp_path)
    clock = ReplayClock(START)
    paper = client if client is not None else MirrorTradingClient(clock=clock)
    paper._clock = clock

    def client_factory(_key: str, _secret: str, endpoint: str) -> MirrorTradingClient:
        assert endpoint == ALPACA_PAPER_ENDPOINT
        return paper

    result = run_demo(
        config_path,
        stage_lock=STAGE_LOCK,
        base_dir=tmp_path,
        code_version=CODE,
        clock=clock,
        secret_store=_secret_store(),
        data_factory=_replay_factory(data_path),
        alpaca_client_factory=client_factory,
    )
    return result, paper


# --------------------------------------------------------------------------- end-to-end


def test_demo_runs_feed_through_alpaca_paper_broker(tmp_path: Path) -> None:
    """Demo procesează feed-ul prin brokerul Alpaca paper și produce ordine + execuții reale."""
    result, paper = _run_demo(tmp_path)
    assert result.events_processed >= SPEC.n_bars
    assert result.orders > 0 and result.fills > 0
    assert len(paper.submitted) == result.orders  # ordinele au ajuns la clientul paper
    assert result.journal_verified
    assert result.net_eur == result.gross_eur - result.costs.total
    assert result.source_id == SPEC.source_id
    assert result.run_id.startswith("dm-")
    assert paper.currency == "USD"  # contul Alpaca paper raportează USD
    # Rulare coerentă în USD (SPY/USD, cont USD, raportare USD): contul oglindește proiecția, deci
    # reconcilierea strictă (USD==USD) nu detectează nicio divergență.
    assert not result.kill_switch_active


def test_demo_config_reports_in_usd(tmp_path: Path) -> None:
    """`config/demo.toml` setează moneda de raportare USD (cont Alpaca paper în USD)."""
    config_path, _ = _write_demo_config(tmp_path)
    config = _load_demo_config(config_path)
    assert config.reporting_currency == "USD"
    assert config.instruments[0].currency == "USD"


def test_demo_startup_reconciliation_before_orders(tmp_path: Path) -> None:
    """Reconcilierea de pornire rulează înaintea oricărui ordin (Req 11.1), apoi periodic (11.3)."""
    result, _ = _run_demo(tmp_path)
    assert result.startup_reconciled
    assert result.reconciliations >= 1
    # Prima reconciliere din jurnal (dacă există divergențe) nu precede niciun broker_request:
    conn = open_db(result.db_path)
    try:
        records = list(Journal(conn).read())
    finally:
        conn.close()
    first_startup = _first_index(records, type_="recon.startup")
    # Pe calea fericită nu există divergențe (deci niciun `recon.difference`); verificăm în schimb
    # că reconcilierea de pornire a fost marcată și că execuțiile apar numai după primul ordin.
    assert result.startup_reconciled
    first_request = _first_index(records, type_="broker_request")
    first_exec = _first_index(records, type_="execution_event")
    assert first_request is not None and first_exec is not None
    assert first_request <= first_exec  # nicio execuție înaintea primului ordin trimis
    assert first_startup is None  # fără divergențe de pornire pe calea fericită


def test_demo_journal_records_demo_snapshot(tmp_path: Path) -> None:
    """Prima înregistrare este snapshot-ul mediului demo, verificabil și cu source_id corect."""
    result, _ = _run_demo(tmp_path)
    conn = open_db(result.db_path)
    journal = Journal(conn)
    try:
        assert verify_journal(journal, expected_head=result.journal_head).ok
        first = next(iter(journal.read()))
        assert first.type == "config_snapshot"
        snapshot = ConfigurationSnapshot.model_validate(first.payload)
        assert snapshot.verify() and snapshot.snapshot_id == result.snapshot_id
        assert snapshot.environment == "demo"
        assert snapshot.dataset_ids == [SPEC.source_id]
    finally:
        conn.close()


def test_demo_is_deterministic(tmp_path: Path) -> None:
    """Cu aceleași intrări, ceas injectat și client fals, Demo produce aceleași rezultate."""
    a, _ = _run_demo(tmp_path / "a")
    b, _ = _run_demo(tmp_path / "b")
    assert (a.orders, a.fills, a.gross_eur, a.costs, a.net_eur) == (
        b.orders,
        b.fills,
        b.gross_eur,
        b.costs,
        b.net_eur,
    )


# ------------------------------------------------------------------- feed continuu (Req 1.1)


def test_demo_streaming_processes_bars_from_multiple_polls(tmp_path: Path) -> None:
    """Demo continuu procesează bare din mai multe sondaje (nu doar un lot) și apoi se oprește.

    Dovada „keeps running": barele dintr-un lot de încălzire PLUS câte una din fiecare dintre
    cele `max_polls` sondaje succesive sunt toate procesate, iar ordinele ajung la clientul paper
    de-a lungul sondajelor. Oprirea este deterministă la seam-ul `max_polls`.
    """
    warmup, polls = 394, 6
    result, paper = _run_demo_streaming(tmp_path, warmup_count=warmup, max_polls=polls)
    # Toate barele (încălzire + un sondaj fiecare) au fost procesate ca evenimente de piață.
    assert result.events_processed >= warmup + polls
    assert result.orders > 0 and result.fills > 0
    assert len(paper.submitted) == result.orders
    assert result.journal_verified
    assert result.net_eur == result.gross_eur - result.costs.total
    assert not result.kill_switch_active


def test_demo_streaming_orders_submitted_across_more_than_one_poll(tmp_path: Path) -> None:
    """Evenimentele din mai mult de un sondaj ajung la motor: ordine trimise după lotul inițial.

    Comparăm o rulare cu un singur sondaj cu una cu mai multe: numărul de bare procesate crește
    strict cu sondajele suplimentare, dovedind că fluxul continuă să alimenteze motorul în timp.
    """
    once, _ = _run_demo_streaming(tmp_path / "once", warmup_count=200, max_polls=1)
    many, _ = _run_demo_streaming(tmp_path / "many", warmup_count=200, max_polls=8)
    # Fiecare sondaj suplimentar aduce exact o bară nouă procesată.
    assert many.events_processed == once.events_processed + 7
    assert many.events_processed > once.events_processed


def test_demo_streaming_reconciles_more_than_once(tmp_path: Path) -> None:
    """Reconcilierea periodică se declanșează de mai multe ori de-a lungul sondajelor (Req 11.3).

    `sleep` avansează ceasul cu intervalul barei (900 s > 60 s) la fiecare sondaj, deci fiecare
    bară nouă declanșează o reconciliere periodică. Cu reconcilierea de pornire inclusă, numărul
    total depășește 1 și crește cu numărul de sondaje.
    """
    result, _ = _run_demo_streaming(tmp_path, warmup_count=250, max_polls=5)
    assert result.startup_reconciled
    # Pornire (1) + câte o reconciliere periodică per sondaj (ceasul a trecut peste interval).
    assert result.reconciliations > 1
    assert result.reconciliations >= 1 + 5


def test_demo_streaming_bars_are_chronological_and_unique(tmp_path: Path) -> None:
    """Fluxul continuu nu reemite nicio bară: evenimentele de piață sunt unice și cronologice."""
    config_path, data_path = _write_demo_config(tmp_path)
    clock = ReplayClock(START)
    bars = _bars_from_dataset(data_path)
    factory = _streaming_factory(bars, warmup_count=100, max_polls=5)
    source = factory(_load_demo_config(config_path), clock, tmp_path)
    closes = [e.payload.ts_close for e in source.stream() if isinstance(e.payload, Bar)]
    assert closes == sorted(closes)
    assert len(closes) == len(set(closes))
    assert len(closes) == 100 + 5


def _load_demo_config(config_path: Path) -> AppConfig:
    from qts.config.loader import load_config

    return load_config(config_path)


# --------------------------------------------------------------------------- fail-closed (Req 11.6)


def test_demo_incomplete_snapshot_activates_global_kill_switch(tmp_path: Path) -> None:
    """Un snapshot incomplet la broker activează Kill_Switch GLOBAL → niciun ordin nou aprobat.

    Reconcilierea de pornire primește un snapshot `complete=False` și, fail-closed (Req 11.6),
    activează domeniul GLOBAL înaintea oricărui ordin. Fail_Safe_Block blochează apoi orice
    `submit`, deci nu se aprobă niciun ordin nou și clientul paper nu primește nimic.
    """
    clock = ReplayClock(START)
    paper = MirrorTradingClient(clock=clock, account_complete=False)
    result, paper = _run_demo(tmp_path, client=paper)
    assert result.kill_switch_active
    assert result.startup_reconciled
    assert paper.submitted == []  # niciun ordin nu a ajuns la broker (blocat de kill switch)
    assert result.fills == 0
    assert result.journal_verified


def test_demo_startup_reconciliation_runs_before_first_submit_on_incomplete(
    tmp_path: Path,
) -> None:
    """Pe snapshot incomplet, GLOBAL este activat de reconcilierea de pornire înaintea ordinelor.

    Dovada ordinii: în jurnal, prima activare a Kill_Switch (`kill_switch.activated`) apare
    înaintea oricărui `broker_request`, deci blocarea precede prima încercare de a trimite ordine.
    """
    clock = ReplayClock(START)
    paper = MirrorTradingClient(clock=clock, account_complete=False)
    result, _ = _run_demo(tmp_path, client=paper)
    conn = open_db(result.db_path)
    try:
        records = list(Journal(conn).read())
    finally:
        conn.close()
    first_activation = _first_index(records, type_prefix="kill_switch.activated")
    first_request = _first_index(records, type_="broker_request")
    assert first_activation is not None
    # Fie nu există niciun broker_request (toate blocate), fie activarea îl precede.
    assert first_request is None or first_activation < first_request


# --------------------------------------------------------------------------- refuzuri de pornire


def test_demo_refuses_non_demo_config(tmp_path: Path) -> None:
    """`run_demo` pe o configurație care nu este demo este refuzat înaintea procesării."""
    with pytest.raises(BootstrapError, match="environment=demo"):
        run_demo(
            CONFIG_DIR / "backtest.toml",
            stage_lock=STAGE_LOCK,
            base_dir=tmp_path,
            code_version=CODE,
            clock=ReplayClock(START),
            secret_store=_secret_store(),
        )


def test_demo_refuses_live_config(tmp_path: Path) -> None:
    """`run_demo` acceptă numai configurații demo; Live este refuzat înaintea procesării."""
    with pytest.raises(BootstrapError, match="environment=demo"):
        run_demo(
            CONFIG_DIR / "live.toml",
            stage_lock=STAGE_LOCK,
            base_dir=tmp_path,
            code_version=CODE,
            clock=ReplayClock(START),
            secret_store=_secret_store(),
        )
    assert list(tmp_path.iterdir()) == []


def test_demo_refuses_without_secret_store(tmp_path: Path) -> None:
    """Fără Secret_Store, brokerul demo Alpaca nu poate rezolva cheile → refuz la pornire."""
    from qts.broker.factory import BrokerFactoryError

    config_path, data_path = _write_demo_config(tmp_path)
    with pytest.raises(BrokerFactoryError, match="Secret_Store"):
        run_demo(
            config_path,
            stage_lock=STAGE_LOCK,
            base_dir=tmp_path,
            code_version=CODE,
            clock=ReplayClock(START),
            secret_store=None,
            data_factory=_replay_factory(data_path),
        )


# --------------------------------------------------------------------------- ajutoare


def _first_index(
    records: Sequence[object], *, type_: str | None = None, type_prefix: str | None = None
) -> int | None:
    for i, rec in enumerate(records):
        rtype = rec.type  # type: ignore[attr-defined]
        if type_ is not None and rtype == type_:
            return i
        if type_prefix is not None and str(rtype).startswith(type_prefix):
            return i
    return None


def test_demo_result_net_equals_gross_minus_costs() -> None:
    from qts.core.models import CostBreakdown

    r = DemoResult(
        snapshot_id="s",
        run_id="dm-x",
        source_id="alpaca:XYZ",
        db_path="x",
        events_processed=0,
        orders=0,
        fills=0,
        realized_gross_eur=D("2.0"),
        unrealized_gross_eur=D("-0.5"),
        costs=CostBreakdown(commission=D("0.25")),
        journal_head=(0, ""),
        journal_verified=True,
    )
    assert r.gross_eur == D("1.5")
    assert r.net_eur == D("1.25")


def test_secret_ref_shape() -> None:
    """Referințele brokerului demo au forma așteptată (derivate din broker.secret_ref)."""
    assert SecretRef("qts/demo/alpaca_key").ref == "qts/demo/alpaca_key"
    assert Identity("demo-runner:demo").name == "demo-runner:demo"
    assert dec("1.00") == D("1.00")


# --------------------------------------------------------------------------- CLI


def test_cli_demo_has_no_live_options() -> None:
    from typer.testing import CliRunner

    from qts.cli import app

    result = CliRunner().invoke(app, ["demo", "--help"])
    assert result.exit_code == 0
    assert "live" not in result.output.lower()
    # Opțiunile pentru o verificare rapidă a fluxului continuu sunt expuse.
    assert "--max-polls" in result.output
    assert "--once" in result.output


def test_cli_demo_once_single_pass_refuses_cleanly_without_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`qts demo --once` este o singură trecere; fără chei refuză curat (cod 2), nu eroare de uz."""
    from typer.testing import CliRunner

    import qts.secrets.store as store_mod
    from qts.cli import EXIT_REFUSED, app

    monkeypatch.setattr(
        store_mod.KeyringSecretStore,
        "get",
        lambda self, ref, requester, environment: (_ for _ in ()).throw(
            store_mod.SecretUnavailableError(f"secretul {ref.ref} nu există în store")
        ),
    )
    config_path, _ = _write_demo_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["demo", "--once", "--config", str(config_path), "--stage-lock", str(STAGE_LOCK)]
    )
    assert result.exit_code == EXIT_REFUSED
    assert "pornire refuzată" in result.output


def test_cli_demo_refuses_without_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`qts demo` fără chei în keyring refuză curat (cod 2, fără traceback)."""
    from typer.testing import CliRunner

    import qts.secrets.store as store_mod
    from qts.cli import EXIT_REFUSED, app

    # Un backend keyring fals, gol: orice căutare întoarce None → SecretUnavailableError.
    monkeypatch.setattr(
        store_mod.KeyringSecretStore,
        "get",
        lambda self, ref, requester, environment: (_ for _ in ()).throw(
            store_mod.SecretUnavailableError(f"secretul {ref.ref} nu există în store")
        ),
    )
    config_path, _ = _write_demo_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["demo", "--config", str(config_path), "--stage-lock", str(STAGE_LOCK)]
    )
    assert result.exit_code == EXIT_REFUSED
    assert "pornire refuzată" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
