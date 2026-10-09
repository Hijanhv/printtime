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
def calendar_build() -> None:
    """Fetch CPI and NFP release dates from FRED and write data/calendar/releases.csv."""
    not_built(1, "calendar build")


@calendar_app.command("validate")
def calendar_validate() -> None:
    """Validate fomc_dates.csv and surprises.csv, cross-check against ALFRED, summarise."""
    not_built(1, "calendar validate")


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
