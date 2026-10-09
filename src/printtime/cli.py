"""Print Time command-line interface.

    printtime calendar build | validate
    printtime data plan-costs | download
    printtime panel build
    printtime analyze liquidity | reaction | regression | execution | strategy
    printtime report
    printtime replay
    printtime synth            synthetic release windows (tests, demos, CI)

Commands that belong to a later phase exit with a clear message instead of
pretending to work.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import typer

from printtime.config import Settings, load_config
from printtime.log import configure_logging, get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
calendar_app = typer.Typer(no_args_is_help=True, help="Release calendar and surprises (Phase 1)")
data_app = typer.Typer(no_args_is_help=True, help="Cost planning and budgeted downloads (Phase 2)")
panel_app = typer.Typer(no_args_is_help=True, help="Event-time panels (Phase 3)")
analyze_app = typer.Typer(no_args_is_help=True, help="Analyses (Phases 4-7)")
app.add_typer(calendar_app, name="calendar")
app.add_typer(data_app, name="data")
app.add_typer(panel_app, name="panel")
app.add_typer(analyze_app, name="analyze")

log = get_logger("printtime.cli")
_state: dict[str, Settings] = {}

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to config.yaml")]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override a config key, e.g. --set databento.budget_usd=30"),
]


@app.callback()
def main(
    config: ConfigOpt = Path("config.yaml"),
    set_: SetOpt = None,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
    pretty: Annotated[bool, typer.Option("--pretty", help="Readable logs instead of JSON")] = False,
) -> None:
    configure_logging(log_level, json=not pretty)
    _state["cfg"] = load_config(config, set_)


def cfg() -> Settings:
    return _state["cfg"]


def not_built(phase: int, what: str) -> None:
    typer.echo(f"{what} is built in Phase {phase}; it is not available yet.", err=True)
    raise typer.Exit(code=2)


@app.command()
def synth(
    event_type: Annotated[str, typer.Option("--event", help="CPI, NFP or FOMC")] = "CPI",
    date: Annotated[str, typer.Option(help="Release date, YYYY-MM-DD")] = "2025-03-12",
    surprise: Annotated[float, typer.Option(help="Standardised surprise")] = 1.0,
    control: Annotated[
        bool, typer.Option("--control", help="Control window: no release effects")
    ] = False,
    out: Annotated[Path, typer.Option(help="Output folder")] = Path("data/synthetic"),
) -> None:
    """Write one synthetic release window (one Parquet file per instrument)."""
    from printtime.synthetic.generator import Scenario, generate_event

    c = cfg()
    time_et = c.events[event_type].times_et[0]
    scenario = Scenario(event_type, dt.date.fromisoformat(date), time_et, surprise, control)
    folder = out / scenario.event_id
    folder.mkdir(parents=True, exist_ok=True)
    for name, book in generate_event(c, scenario).items():
        book.frame.write_parquet(folder / f"{name}.parquet")
        log.info("synthetic_written", instrument=name, rows=book.frame.height, path=str(folder))


# Phase 1 ---------------------------------------------------------------------
@calendar_app.command("build")
def calendar_build(
    tier2: Annotated[
        bool, typer.Option("--tier2", help="Also fetch PPI, retail sales and jobless claims")
    ] = False,
) -> None:
    """Fetch CPI and NFP release dates from FRED, add FOMC dates, write releases.csv."""
    from printtime.calendar.events import (
        build_calendar,
        fomc_event_rows,
        fred_event_rows,
        validate_fomc,
        write_fomc_template,
    )
    from printtime.calendar.fred import FredClient
    from printtime.calendar.surprises import write_surprises_template
    from printtime.secrets import require_key

    c = cfg()
    cal_dir = c.paths.calendar
    fomc_path = cal_dir / "fomc_dates.csv"
    surprises_path = cal_dir / "surprises.csv"
    for path, created in (
        (fomc_path, write_fomc_template(fomc_path)),
        (surprises_path, write_surprises_template(surprises_path)),
    ):
        if created:
            typer.echo(f"created template {path}: please fill it in")
    event_types = [
        e for e in c.tier(1) + (c.tier(2) if tier2 else []) if c.events[e].fred_release_search
    ]
    key = require_key(c.fred.api_key_env)
    with FredClient(key, c.fred.base_url, c.fred.timeout_s) as client:
        rows = fred_event_rows(c, client, event_types)
    fomc, rep = validate_fomc(c, fomc_path)
    if rep.ok:
        rows += fomc_event_rows(c, fomc)
    else:
        typer.echo(f"FOMC dates not included yet ({len(rep.errors)} issue(s) in {fomc_path.name}).")
    calendar = build_calendar(rows)
    out = cal_dir / "releases.csv"
    calendar.write_csv(out)
    counts = calendar.group_by(["event_type", "stage"]).len().sort(["event_type", "stage"])
    typer.echo(f"wrote {out} with {calendar.height} timestamps:")
    for r in counts.iter_rows(named=True):
        typer.echo(f"  {r['event_type']:<6} {r['stage']:<17} {r['len']}")


@calendar_app.command("validate")
def calendar_validate(
    skip_alfred: Annotated[
        bool, typer.Option("--skip-alfred", help="Do not cross-check against ALFRED")
    ] = False,
) -> None:
    """Validate fomc_dates.csv and surprises.csv, cross-check actuals against ALFRED, summarise."""
    from printtime.calendar.alfred import cross_check, fredapi_loader, mismatches
    from printtime.calendar.events import load_calendar, validate_fomc
    from printtime.calendar.surprises import summary, validate_surprises
    from printtime.secrets import has_key, require_key

    c = cfg()
    calendar = load_calendar(c)
    _, fomc_rep = validate_fomc(c, c.paths.calendar / "fomc_dates.csv")
    surprises, sur_rep = validate_surprises(c, c.paths.calendar / "surprises.csv", calendar)
    failed = False
    for rep in (fomc_rep, sur_rep):
        for w in rep.warnings:
            typer.echo(f"warning [{rep.name}]: {w}")
        for e in rep.errors:
            typer.echo(f"ERROR   [{rep.name}]: {e}")
        failed |= not rep.ok
    if not failed and not skip_alfred:
        if not has_key(c.fred.api_key_env):
            typer.echo("ALFRED cross-check skipped: FRED_API_KEY is not set.")
        else:
            check = cross_check(c, surprises, fredapi_loader(require_key(c.fred.api_key_env)))
            c.paths.tables.mkdir(parents=True, exist_ok=True)
            check.write_csv(c.paths.tables / "alfred_cross_check.csv")
            bad = mismatches(check)
            typer.echo(
                f"ALFRED cross-check: {check.height - bad.height} of {check.height} "
                "match after rounding."
            )
            for r in bad.iter_rows(named=True):
                typer.echo(
                    f"  {r['status']}: {r['event_id']} {r['variable']} "
                    f"actual={r['actual']} alfred={r['alfred_rounded']}"
                )
            failed |= bad.height > 0
    typer.echo("\nEvents in the calendar:")
    for r in (
        calendar.group_by(["event_type", "stage"])
        .len()
        .sort(["event_type", "stage"])
        .iter_rows(named=True)
    ):
        typer.echo(f"  {r['event_type']:<6} {r['stage']:<17} {r['len']}")
    if surprises.height:
        typer.echo("\nSurprise distributions (actual - consensus):")

        def fmt(x: float | None, spec: str) -> str:
            return "n/a" if x is None else format(x, spec)

        for r in summary(surprises).iter_rows(named=True):
            typer.echo(
                f"  {r['variable']:<18} n={r['events']:<3} mean={fmt(r['mean_surprise'], '+.3f')} "
                f"sd={fmt(r['sd_surprise'], '.3f')} "
                f"min={fmt(r['min_surprise'], '+.3f')} max={fmt(r['max_surprise'], '+.3f')} "
                f"|z|>1: {r['abs_z_over_1']}"
            )
    if failed:
        raise typer.Exit(code=1)
    typer.echo("\ncalendar validate: passed")


# Phase 2 ---------------------------------------------------------------------
@data_app.command("plan-costs")
def data_plan_costs() -> None:
    """Price a sample, extrapolate the full plan, and show trimming options."""
    not_built(2, "data plan-costs")


@data_app.command("download")
def data_download() -> None:
    """Budgeted, cached Databento downloads."""
    not_built(2, "data download")


# Phase 3 ---------------------------------------------------------------------
@panel_app.command("build")
def panel_build() -> None:
    """Align book and trade data onto event-time grids and compute baselines."""
    not_built(3, "panel build")


# Phases 4-7 ------------------------------------------------------------------
@analyze_app.command("liquidity")
def analyze_liquidity() -> None:
    """Depth withdrawal, spreads, activity and recovery vs control days."""
    not_built(4, "analyze liquidity")


@analyze_app.command("reaction")
def analyze_reaction() -> None:
    """Jump sizes, first mover and lead-lag."""
    not_built(5, "analyze reaction")


@analyze_app.command("regression")
def analyze_regression() -> None:
    """Reaction-function regressions with HC3 errors and FDR control."""
    not_built(6, "analyze regression")


@analyze_app.command("execution")
def analyze_execution() -> None:
    """Execution-cost curves from walking the 10-level book."""
    not_built(7, "analyze execution")


@analyze_app.command("strategy")
def analyze_strategy() -> None:
    """Chronologically validated surprise-direction strategy."""
    not_built(7, "analyze strategy")


# Phases 8-9 ------------------------------------------------------------------
@app.command()
def report() -> None:
    """Write reports/REPORT.md from the result tables."""
    not_built(9, "report")


@app.command()
def replay() -> None:
    """Replay a release window at market speed with Prometheus metrics."""
    not_built(8, "replay")


if __name__ == "__main__":
    app()
