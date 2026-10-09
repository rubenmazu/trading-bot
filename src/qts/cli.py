"""Interfața CLI a operatorului. Comenzile se adaugă pe măsura implementării.

Nu există nicio opțiune Live: `qts backtest` acceptă numai configurații `backtest`, `qts shadow`
numai `shadow`, iar orice refuz de pornire (etapă, mod, snapshot, set de date) încheie comanda cu
cod nenul, înainte de procesarea vreunui eveniment.

`qts shadow` compune feed-ul curent (reluat printr-un `Data_Adapter`) cu `SimBroker` și ceasul
real (`WallClock`), aceeași secvență de pași ca Backtest (Req 1.1). Adaptorul de date *real*
depinde de alegerea sursei (Open_Decision, Req 30): până atunci comanda refuză pornirea cu un
motiv clar, deoarece nicio sursă de date reală nu este injectată din CLI.

`qts evaluate` produce raportul de evaluare al cercetării (brut, costuri, net, ipoteze,
incertitudine, toate variantele, marcare „incomplet", avertismentul că profitul nu este garantat;
Req 21.3, 21.7, 29). O evaluare completă necesită o pre-înregistrare validă și date partiționate
(Development_Set/Out_Of_Sample_Set); sursa de date reală este un Open_Decision (Req 30). Până la
injectarea unei surse și a unei pre-înregistrări din CLI, comanda refuză pornirea cu un motiv clar,
exact ca seam-ul folosit de `qts shadow`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import typer

from qts import ENGINE_VERSION
from qts.bootstrap import (
    BacktestResult,
    BootstrapError,
    DemoResult,
    ShadowResult,
    run_backtest,
    run_demo,
    run_shadow,
)

if TYPE_CHECKING:
    from qts.data.alpaca_shadow import ShadowDataFactory
    from qts.secrets.store import SecretStore
from qts.broker.factory import (
    BrokerFactoryError,
    BrokerNotAvailableError,
)
from qts.config.loader import ConfigError
from qts.config.snapshot import SnapshotError
from qts.data.manifest import DatasetInvalidError
from qts.research.report import (
    CostAssumptions,
    EvaluationReport,
    UncertaintyReport,
)
from qts.safety.stage import StartupRefusedError

app = typer.Typer(help="Quant_Trading_System (etapa inițială: fără ordine reale).")

EXIT_REFUSED: Final = 2
PROFIT_WARNING: Final = (
    "AVERTISMENT: rezultatele unui backtest nu garantează profitul viitor; "
    "pierderile sunt posibile."
)
# Portofoliul primește de la OMS numai comisionul raportat; spreadul, slippage-ul și latența
# simulate sunt deja în prețurile de execuție ale SimBroker, deci în rezultatul brut.
COST_NOTE: Final = (
    "spreadul, slippage-ul și latența simulate sunt incluse în prețurile de execuție "
    "(deci în brut); categoriile lor apar 0 până la descompunerea per execuție"
)
COST_LABELS: Final = (
    ("spread", "spread"),
    ("commission", "comision"),
    ("slippage", "slippage"),
    ("latency", "latență"),
    ("fx_conversion", "conversie FX"),
    ("taxes", "taxe"),
)


@app.command()
def version() -> None:
    """Afișează versiunea Trading_Engine."""
    typer.echo(ENGINE_VERSION)


def _summary(result: BacktestResult) -> list[str]:
    lines = [
        f"snapshot:            {result.snapshot_id}",
        f"run_id:              {result.run_id}",
        f"dataset:             {result.dataset_id} (rânduri excluse: {result.rejected_rows})",
        f"evenimente:          {result.events_processed}",
        f"ordine:              {result.orders} {result.order_states}",
        f"execuții:            {result.fills}",
        f"brut realizat:       {result.realized_gross_eur} EUR",
        f"brut nerealizat:     {result.unrealized_gross_eur} EUR",
        f"brut total:          {result.gross_eur} EUR",
    ]
    for attr, label in COST_LABELS:
        lines.append(f"{'cost ' + label + ':':<21}{getattr(result.costs, attr)} EUR")
    lines += [
        f"costuri totale:      {result.costs.total} EUR",
        f"notă:                {COST_NOTE}",
        f"net:                 {result.net_eur} EUR",
        f"jurnal:              seq={result.journal_head[0]} head={result.journal_head[1]} "
        f"({'verificat' if result.journal_verified else 'INTEGRITATE EȘUATĂ'})",
        f"baza de date:        {result.db_path}",
        PROFIT_WARNING,
    ]
    return lines


@app.command()
def backtest(
    config: Annotated[
        Path, typer.Option("--config", help="Fișierul de configurație Backtest (TOML).")
    ] = Path("config/backtest.toml"),
    stage_lock: Annotated[
        Path, typer.Option("--stage-lock", help="Fișierul cu etapa proiectului.")
    ] = Path("stage.lock"),
    db: Annotated[
        str | None,
        typer.Option("--db", help="Suprascrie run.db_path (intră în Configuration_Snapshot)."),
    ] = None,
) -> None:
    """Rulează un Backtest pe date istorice cu SimBroker și SimClock."""
    try:
        result = run_backtest(config, stage_lock=stage_lock, db_path=db)
    except (ConfigError, StartupRefusedError, BrokerFactoryError, BootstrapError) as exc:
        typer.echo(f"pornire refuzată: {exc}", err=True)
        raise typer.Exit(EXIT_REFUSED) from None
    except SnapshotError as exc:
        typer.echo(
            f"Configuration_Snapshot nu poate fi creat; rularea este oprită (Req 17.7): {exc}",
            err=True,
        )
        raise typer.Exit(EXIT_REFUSED) from None
    except DatasetInvalidError as exc:
        typer.echo(f"set de date invalid: {exc}", err=True)
        raise typer.Exit(EXIT_REFUSED) from None
    for line in _summary(result):
        typer.echo(line)
    if not result.journal_verified:
        raise typer.Exit(1)


def _cost_and_pnl_lines(result: BacktestResult | ShadowResult | DemoResult) -> list[str]:
    """Liniile de cost și PnL, comune pentru Backtest, Shadow și Demo (același model de costuri)."""
    lines = [
        f"brut realizat:       {result.realized_gross_eur} EUR",
        f"brut nerealizat:     {result.unrealized_gross_eur} EUR",
        f"brut total:          {result.gross_eur} EUR",
    ]
    for attr, label in COST_LABELS:
        lines.append(f"{'cost ' + label + ':':<21}{getattr(result.costs, attr)} EUR")
    lines += [
        f"costuri totale:      {result.costs.total} EUR",
        f"notă:                {COST_NOTE}",
        f"net:                 {result.net_eur} EUR",
        f"jurnal:              seq={result.journal_head[0]} head={result.journal_head[1]} "
        f"({'verificat' if result.journal_verified else 'INTEGRITATE EȘUATĂ'})",
        f"baza de date:        {result.db_path}",
        PROFIT_WARNING,
    ]
    return lines


def _shadow_summary(result: ShadowResult) -> list[str]:
    lines = [
        f"snapshot:            {result.snapshot_id}",
        f"run_id:              {result.run_id}",
        f"sursă:               {result.source_id}",
        f"evenimente:          {result.events_processed}",
        f"ordine:              {result.orders} {result.order_states}",
        f"execuții:            {result.fills}",
    ]
    return lines + _cost_and_pnl_lines(result)


@app.command()
def shadow(
    config: Annotated[
        Path, typer.Option("--config", help="Fișierul de configurație Shadow (TOML).")
    ] = Path("config/shadow.toml"),
    stage_lock: Annotated[
        Path, typer.Option("--stage-lock", help="Fișierul cu etapa proiectului.")
    ] = Path("stage.lock"),
    db: Annotated[
        str | None,
        typer.Option("--db", help="Suprascrie run.db_path (intră în Configuration_Snapshot)."),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            help="Sursa de date curentă. Momentan: 'alpaca' (feed real). Fără ea, pornirea "
            "este refuzată (sursa de date este Open_Decision, Req 30).",
        ),
    ] = None,
) -> None:
    """Rulează modul Shadow: feed curent cu SimBroker (execuție locală) și ceas real.

    Cu `--source alpaca`, feed-ul este `AlpacaDataAdapter` (bare reale, execuție simulată local).
    Cheile API vin din Secret_Store (keyring): `qts/shadow/alpaca_key` și
    `qts/shadow/alpaca_secret`. Fără `--source`, adaptorul de date real rămâne o Open_Decision
    (Req 30) și pornirea este refuzată.
    """
    from qts.secrets.store import SecretAccessDeniedError, SecretUnavailableError

    try:
        data_factory = _resolve_shadow_source(source)
        if data_factory is None:
            result = run_shadow(config, stage_lock=stage_lock, db_path=db)
        else:
            result = run_shadow(
                config, stage_lock=stage_lock, db_path=db, data_factory=data_factory
            )
    except (ConfigError, StartupRefusedError, BrokerFactoryError, BootstrapError) as exc:
        typer.echo(f"pornire refuzată: {exc}", err=True)
        raise typer.Exit(EXIT_REFUSED) from None
    except (SecretUnavailableError, SecretAccessDeniedError) as exc:
        typer.echo(
            "pornire refuzată: cheile Alpaca nu sunt disponibile în Secret_Store (keyring): "
            f"{exc}. Adaugă qts/shadow/alpaca_key și qts/shadow/alpaca_secret în keyring.",
            err=True,
        )
        raise typer.Exit(EXIT_REFUSED) from None
    except SnapshotError as exc:
        typer.echo(
            f"Configuration_Snapshot nu poate fi creat; rularea este oprită (Req 17.7): {exc}",
            err=True,
        )
        raise typer.Exit(EXIT_REFUSED) from None
    except DatasetInvalidError as exc:
        typer.echo(f"set de date invalid: {exc}", err=True)
        raise typer.Exit(EXIT_REFUSED) from None
    for line in _shadow_summary(result):
        typer.echo(line)
    if not result.journal_verified:
        raise typer.Exit(1)


def _resolve_shadow_source(source: str | None) -> ShadowDataFactory | None:
    """Alege fabrica de date Shadow din `--source`. `None` păstrează refuzul implicit (Req 30).

    Pentru `alpaca`, construiește un `Secret_Store` keyring cu ACL pentru identitatea feed-ului
    Shadow și întoarce fabrica `AlpacaDataAdapter`. Cheile reale nu apar niciodată aici: sunt
    citite din keyring doar în interiorul fabricii, la pornire.
    """
    if source is None:
        return None
    if source.lower() != "alpaca":
        raise BootstrapError(f"sursă de date necunoscută: {source!r} (acceptat: 'alpaca')")
    from qts.data.alpaca_shadow import SHADOW_IDENTITY, alpaca_shadow_factory
    from qts.secrets.store import KeyringSecretStore

    acl = {
        "qts/shadow/alpaca_key": {(SHADOW_IDENTITY.name, "shadow")},
        "qts/shadow/alpaca_secret": {(SHADOW_IDENTITY.name, "shadow")},
    }
    return alpaca_shadow_factory(KeyringSecretStore(acl))


def _demo_summary(result: DemoResult) -> list[str]:
    lines = [
        f"snapshot:            {result.snapshot_id}",
        f"run_id:              {result.run_id}",
        f"sursă:               {result.source_id}",
        f"evenimente:          {result.events_processed}",
        f"reconcilieri:        {result.reconciliations} "
        f"(pornire: {'da' if result.startup_reconciled else 'nu'})",
        f"kill switch global:  {'ACTIV' if result.kill_switch_active else 'inactiv'}",
        f"ordine:              {result.orders} {result.order_states}",
        f"execuții:            {result.fills}",
    ]
    return lines + _cost_and_pnl_lines(result)


@app.command()
def demo(
    config: Annotated[
        Path, typer.Option("--config", help="Fișierul de configurație Demo (TOML).")
    ] = Path("config/demo.toml"),
    stage_lock: Annotated[
        Path, typer.Option("--stage-lock", help="Fișierul cu etapa proiectului.")
    ] = Path("stage.lock"),
    db: Annotated[
        str | None,
        typer.Option("--db", help="Suprascrie run.db_path (intră în Configuration_Snapshot)."),
    ] = None,
    max_polls: Annotated[
        int | None,
        typer.Option(
            "--max-polls",
            help="Numărul de sondaje după lotul de încălzire (implicit: continuu, fără limită). "
            "Util pentru o verificare rapidă fără a rula la nesfârșit.",
        ),
    ] = None,
    once: Annotated[
        bool,
        typer.Option(
            "--once",
            help="O singură trecere (lot de încălzire + un sondaj), apoi oprire. "
            "Echivalent cu --max-polls 1.",
        ),
    ] = False,
) -> None:
    """Rulează modul Demo: broker Alpaca paper (API real, bani simulați) și ceas real.

    Brokerul este `AlpacaBrokerAdapter` (paper), construit prin `build_broker` cu cheile din
    Secret_Store (keyring); feed-ul este `AlpacaStreamingSource` continuu: reacționează la fiecare
    bară nou închisă în timp real, cu același motor și aceeași secvență de pași în toate modurile.
    Implicit rulează continuu; `--max-polls N` sau `--once` fac o verificare rapidă și se opresc.
    Reconcilierea rulează la pornire (înaintea primului ordin) și periodic (≤ 60 s): un snapshot
    incomplet activează Kill_Switch GLOBAL (fail-closed) și niciun ordin nou nu mai este aprobat.
    Comanda acceptă numai configurații demo; nu expune nicio țintă de producție.

    Cheile API vin din keyring: `qts/demo/alpaca_key`, `qts/demo/alpaca_key_secret` (brokerul) și
    `qts/demo/alpaca_key`, `qts/demo/alpaca_secret` (feed-ul). Fără ele, pornirea este refuzată.
    """
    from qts.secrets.store import SecretAccessDeniedError, SecretUnavailableError

    polls = 1 if once else max_polls
    try:
        secret_store, data_factory = _demo_dependencies(max_polls=polls)
        result = run_demo(
            config,
            stage_lock=stage_lock,
            db_path=db,
            secret_store=secret_store,
            data_factory=data_factory,
        )
    except (
        ConfigError,
        StartupRefusedError,
        BrokerFactoryError,
        BrokerNotAvailableError,
        BootstrapError,
    ) as exc:
        typer.echo(f"pornire refuzată: {exc}", err=True)
        raise typer.Exit(EXIT_REFUSED) from None
    except (SecretUnavailableError, SecretAccessDeniedError) as exc:
        typer.echo(
            "pornire refuzată: cheile Alpaca nu sunt disponibile în Secret_Store (keyring): "
            f"{exc}. Adaugă qts/demo/alpaca_key, qts/demo/alpaca_key_secret și "
            "qts/demo/alpaca_secret în keyring.",
            err=True,
        )
        raise typer.Exit(EXIT_REFUSED) from None
    except SnapshotError as exc:
        typer.echo(
            f"Configuration_Snapshot nu poate fi creat; rularea este oprită (Req 17.7): {exc}",
            err=True,
        )
        raise typer.Exit(EXIT_REFUSED) from None
    except DatasetInvalidError as exc:
        typer.echo(f"set de date invalid: {exc}", err=True)
        raise typer.Exit(EXIT_REFUSED) from None
    for line in _demo_summary(result):
        typer.echo(line)
    if not result.journal_verified:
        raise typer.Exit(1)


def _demo_dependencies(*, max_polls: int | None = None) -> tuple[SecretStore, ShadowDataFactory]:
    """Construiește Secret_Store (keyring) + fabrica de date Alpaca continuă pentru Demo.

    ACL-ul autorizează identitatea feed-ului (`shadow-feed`) pentru cheile feed-ului și
    identitatea runner-ului Demo (`demo-runner:demo`) pentru cheile brokerului, derivate din
    `broker.secret_ref` (`qts/demo/alpaca_key` → `qts/demo/alpaca_key` și
    `qts/demo/alpaca_key_secret`). Cheile reale nu apar niciodată aici: sunt citite din keyring
    doar în interiorul fabricilor.

    Feed-ul este `AlpacaStreamingSource` (continuu): `max_polls=None` rulează la nesfârșit, iar o
    valoare finită (din `--max-polls`/`--once`) oprește după acel număr de sondaje — același motor,
    aceeași secvență de pași, doar fluxul de date curge în timp real.
    """
    from qts.data.alpaca_shadow import SHADOW_IDENTITY, alpaca_streaming_factory
    from qts.secrets.store import KeyringSecretStore

    broker_ref = "qts/demo/alpaca_key"
    acl = {
        # Feed Alpaca (același seam ca Shadow): identitatea feed-ului, cheile de date.
        "qts/demo/alpaca_key": {
            (SHADOW_IDENTITY.name, "demo"),
            ("demo-runner:demo", "demo"),
        },
        "qts/demo/alpaca_secret": {(SHADOW_IDENTITY.name, "demo")},
        # Broker Alpaca paper: identitatea runner-ului Demo, cheia + secretul derivat.
        f"{broker_ref}_secret": {("demo-runner:demo", "demo")},
    }
    store = KeyringSecretStore(acl)
    return store, alpaca_streaming_factory(store, max_polls=max_polls)


def _evaluation_summary(report: EvaluationReport) -> list[str]:
    """Liniile raportului de evaluare: brut, costuri, net, ipoteze, incertitudine, variante.

    Metricile obligatorii lipsă sunt afișate ca `(lipsă)`; dacă raportul este incomplet, se adaugă
    marcajul explicit și blocarea promovării (Req 29.4). Avertismentul despre profit încheie
    raportul (Req 21.7, 29.5).
    """
    lines = [
        f"strategie:           {report.strategy_id}",
        f"pre-înregistrare:    {report.preregistration_id}",
        f"monedă raportare:    {report.reporting_currency}",
        f"perioadă:            {report.period if report.period is not None else '(lipsă)'}",
        f"univers:             "
        f"{', '.join(report.universe) if report.universe is not None else '(lipsă)'}",
        f"tranzacții:          "
        f"{report.trade_count if report.trade_count is not None else '(lipsă)'}",
        f"brut:                {_metric(report.gross_result_eur)}",
    ]
    if report.costs is not None:
        for attr, label in COST_LABELS:
            lines.append(f"{'cost ' + label + ':':<21}{getattr(report.costs, attr)} EUR")
        lines.append(f"costuri totale:      {report.costs.total} EUR")
    else:
        lines.append("costuri:             (lipsă)")
    lines += [
        f"net:                 {_metric(report.net_result_eur)}",
        f"drawdown maxim:      {_metric(report.max_drawdown_eur)}",
    ]
    lines += _assumption_lines(report.assumptions)
    lines += _uncertainty_lines(report.uncertainty)
    lines.append("variante:")
    for variant in report.variants:
        verdict = "acceptată" if variant.accepted else f"respinsă ({variant.rejection_reason})"
        lines.append(f"  - {variant.label}: {variant.net_result_eur} EUR [{verdict}]")
    if report.incomplete:
        lines.append(f"raport INCOMPLET; metrici lipsă: {', '.join(report.missing_metrics)}")
        lines.append("promovare blocată (Req 29.4)")
    else:
        lines.append("raport complet")
    lines.append(report.warning)
    return lines


def _metric(value: object) -> str:
    return f"{value} EUR" if value is not None else "(lipsă)"


def _assumption_lines(assumptions: CostAssumptions) -> list[str]:
    return [
        "ipoteze:",
        f"  latență:           {assumptions.latency}",
        f"  lichiditate:       {assumptions.liquidity}",
        f"  spread:            {assumptions.spread}",
        f"  slippage:          {assumptions.slippage}",
        f"  comisioane:        {assumptions.commissions}",
        f"  conversie FX:      {assumptions.fx_conversion}",
        f"  taxe:              {assumptions.taxes}",
    ]


def _uncertainty_lines(uncertainty: UncertaintyReport | None) -> list[str]:
    if uncertainty is None:
        return ["incertitudine:       (lipsă)"]
    return [
        "incertitudine (bootstrap):",
        f"  sămânță:           {uncertainty.seed}",
        f"  reeșantionări:     {uncertainty.resamples}",
        f"  PnL net (mediană): {uncertainty.net_pnl.quantiles.get('0.50', '?')} EUR",
        f"  drawdown (mediană):{uncertainty.max_drawdown.quantiles.get('0.50', '?')} EUR",
        f"  P(limită pierdere):{uncertainty.prob_hit_total_loss_limit}",
    ]


@app.command()
def evaluate(
    config: Annotated[
        Path, typer.Option("--config", help="Fișierul de configurație Backtest (TOML).")
    ] = Path("config/backtest.toml"),
    stage_lock: Annotated[
        Path, typer.Option("--stage-lock", help="Fișierul cu etapa proiectului.")
    ] = Path("stage.lock"),
) -> None:
    """Produce raportul de evaluare al cercetării (Req 21.3, 21.7, 29).

    O evaluare completă necesită o pre-înregistrare validă și date partiționate; sursa de date
    reală este un Open_Decision (Req 30). Până la injectarea acestora din CLI, comanda refuză
    pornirea cu un motiv clar, în același stil ca `qts shadow`.
    """
    typer.echo(
        "pornire refuzată: evaluarea cercetării necesită o pre-înregistrare validă și date "
        "partiționate (Development_Set/Out_Of_Sample_Set); sursa de date reală este un "
        "Open_Decision (Req 30) și nu este injectată din CLI.",
        err=True,
    )
    typer.echo(PROFIT_WARNING, err=True)
    raise typer.Exit(EXIT_REFUSED)
