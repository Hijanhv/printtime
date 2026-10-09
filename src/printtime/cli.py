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
from typing import TYPE_CHECKING, Annotated

import typer

from printtime.config import Settings, load_config
from printtime.log import configure_logging, get_logger

if TYPE_CHECKING:
    from printtime.data.plan import Window

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
def _windows(c: Settings) -> tuple[list[Window], list[Window]]:
    import polars as pl

    from printtime.calendar.events import load_calendar
    from printtime.data.plan import control_windows, event_windows

    calendar = load_calendar(c)
    ctrl_path = c.paths.calendar / "control_days.csv"
    if not ctrl_path.exists():
        typer.echo("no control days yet: run `printtime data controls` first", err=True)
        raise typer.Exit(code=2)
    controls = pl.read_csv(ctrl_path, try_parse_dates=True)
    return event_windows(calendar, c.tier(1)), control_windows(controls)


@data_app.command("controls")
def data_controls(
    allow_no_tier2: Annotated[
        bool,
        typer.Option(
            "--allow-no-tier2", help="Choose controls without Tier 2 dates (not recommended)"
        ),
    ] = False,
) -> None:
    """Choose control days (same clock time, no release) and write control_days.csv."""
    from printtime.calendar.events import load_calendar
    from printtime.data.controls import select_control_days

    c = cfg()
    controls = select_control_days(c, load_calendar(c), require_tier2=not allow_no_tier2)
    out = c.paths.calendar / "control_days.csv"
    controls.write_csv(out)
    for (t,), g in controls.group_by(["time_et"], maintain_order=True):
        typer.echo(f"  {t}: {g.height} control days ({', '.join(sorted(set(g['weekday'])))})")
    typer.echo(f"wrote {out}")


@data_app.command("plan-costs")
def data_plan_costs() -> None:
    """Price sample windows (free quotes), extrapolate the full plan, show trimming options."""
    from printtime.data import databento_io as dbio
    from printtime.data.plan import markdown, plan_costs

    c = cfg()
    events, controls = _windows(c)
    api = dbio.client(c)
    est = plan_costs(
        c,
        events,
        controls,
        lambda r: dbio.quote(c, r, api),
        lambda r: dbio.billable_bytes(c, r, api),
    )
    ledger = dbio.SpendLedger(c.databento.spend_ledger, c.databento.budget_usd)
    md = markdown(est, c.databento.budget_usd, ledger.remaining)
    c.paths.tables.mkdir(parents=True, exist_ok=True)
    (c.paths.tables / "cost_plan.md").write_text(md)
    est.samples.write_csv(c.paths.tables / "cost_plan_samples.csv")
    typer.echo(md)
    typer.echo(
        "Nothing was bought. Choose an option, set it in config.yaml, "
        "then run `printtime data download`."
    )


@data_app.command("download")
def data_download(
    yes: Annotated[
        bool, typer.Option("--yes", help="Actually buy; without it this is a priced dry run")
    ] = False,
    schema: Annotated[list[str] | None, typer.Option(help="Limit to these schemas")] = None,
) -> None:
    """Quote every request, check the total against the budget, and (with --yes) buy.

    Files already in data/raw are never bought again.
    """
    from printtime.data import databento_io as dbio
    from printtime.data.plan import control_windows_from, plan_settings, requests_for

    c = cfg()
    events, controls = _windows(c)
    ps = plan_settings(c)
    reqs = requests_for(c, events + control_windows_from(controls, ps.controls), ps)
    if schema:
        reqs = [r for r in reqs if r.schema in schema]
    todo = [r for r in reqs if not r.path(c.paths.raw).exists()]
    api = dbio.client(c)
    total = sum(dbio.quote(c, r, api) for r in todo)
    ledger = dbio.SpendLedger(c.databento.spend_ledger, c.databento.budget_usd)
    typer.echo(
        f"{len(reqs)} requests, {len(reqs) - len(todo)} cached, "
        f"{len(todo)} to buy for ${total:.2f}; "
        f"${ledger.remaining:.2f} of ${ledger.budget:.2f} left."
    )
    if total > ledger.remaining + 1e-9:
        typer.echo(
            "This exceeds the remaining budget. Trim the plan (see `data plan-costs`).", err=True
        )
        raise typer.Exit(code=1)
    if not yes:
        typer.echo("Dry run: nothing bought. Re-run with --yes to buy.")
        return
    for r in todo:
        dbio.download(c, r, api, confirm=True)
    typer.echo(f"done; spent ${ledger.spent:.2f} in total.")


# Phase 3 ---------------------------------------------------------------------
@panel_app.command("build")
def panel_build(
    synthetic: Annotated[
        int,
        typer.Option("--synthetic", help="Build a synthetic study with N events per type instead"),
    ] = 0,
    controls: Annotated[int, typer.Option(help="Control days per clock time (synthetic only)")] = 8,
) -> None:
    """Align book and trade data onto event-time grids, compute baselines, validate releases."""
    import polars as pl

    from printtime.panel.pipeline import run_panel_build, stages_for_study, synthetic_settings
    from printtime.panel.providers import DatabentoProvider, SyntheticProvider, stages_from

    c = cfg()
    if synthetic:
        from printtime.synthetic.generator import synthetic_study

        c = synthetic_settings(c)
        study = synthetic_study(c, synthetic, controls)
        stages = stages_for_study(c, study)
        c.paths.calendar.mkdir(parents=True, exist_ok=True)
        study.surprises.write_parquet(c.paths.calendar / "synthetic_surprises.parquet")
        provider: SyntheticProvider | DatabentoProvider = SyntheticProvider(c, study.scenarios)
        instruments = list(c.synthetic.instruments)
        typer.echo(
            f"SYNTHETIC study: {len(stages)} stages, outputs under {c.paths.processed.parent}"
        )
    else:
        from printtime.calendar.events import load_calendar

        calendar = load_calendar(c)
        ctrl_path = c.paths.calendar / "control_days.csv"
        ctrl = pl.read_csv(ctrl_path, try_parse_dates=True) if ctrl_path.exists() else None
        stages = stages_from(calendar, ctrl, c.tier(1))
        provider = DatabentoProvider(c, stages)
        instruments = list(c.instruments)
    result = run_panel_build(c, stages, provider, instruments)
    typer.echo(f"built panels for {result.stages_built} of {len(stages)} stages")
    if result.validation.height:
        bad = result.validation.filter(~pl.col("confirmed"))
        typer.echo(
            f"release validation: {result.validation.height - bad.height} confirmed, "
            f"{bad.height} not"
        )
        for r in bad.iter_rows(named=True):
            typer.echo(f"  CHECK BY HAND: {r['event_id']} (best spike ratio {r['best_ratio']})")
    for p in result.sanity_plots:
        typer.echo(f"sanity plot: {p}")


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
