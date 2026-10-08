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
from typing import Annotated, Final

import typer

from qts import ENGINE_VERSION
from qts.bootstrap import (
    BacktestResult,
    BootstrapError,
    ShadowResult,
    run_backtest,
    run_shadow,
)
from qts.broker.factory import BrokerFactoryError
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


def _cost_and_pnl_lines(result: BacktestResult | ShadowResult) -> list[str]:
    """Liniile de cost și PnL, comune pentru Backtest și Shadow (același model de costuri)."""
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
) -> None:
    """Rulează modul Shadow: feed curent cu SimBroker (execuție locală) și ceas real.

    Adaptorul de date real depinde de alegerea sursei (Open_Decision, Req 30); până atunci
    comanda refuză pornirea, deoarece CLI-ul nu injectează o sursă de date reală.
    """
    try:
        result = run_shadow(config, stage_lock=stage_lock, db_path=db)
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
    for line in _shadow_summary(result):
        typer.echo(line)
    if not result.journal_verified:
        raise typer.Exit(1)


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
