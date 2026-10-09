"""The event calendar: one row per release timestamp, with t0 in UTC.

* CPI and NFP dates come from FRED (`fred.FredClient`).
* FOMC dates come from data/calendar/fomc_dates.csv, which you fill in from the
  Federal Reserve website. The statement (14:00 ET) and the press conference
  (14:30 ET) are two timestamps of the same event.

releases.csv columns:
    event_id, event_type, tier, date, stage, time_et, t0_utc_ns, t0_utc, source
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from printtime.calendar.fred import FredClient
from printtime.calendar.timezones import et_to_utc, et_to_utc_ns
from printtime.config import Settings
from printtime.log import get_logger

log = get_logger(__name__)

FOMC_COLUMNS = ["date", "statement_time_et", "press_conference_time_et", "source_url"]
RELEASE_COLUMNS = [
    "event_id",
    "event_type",
    "tier",
    "date",
    "stage",
    "time_et",
    "t0_utc_ns",
    "t0_utc",
    "source",
]
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_URL = re.compile(r"^https?://\S+$")


@dataclass
class ValidationReport:
    """Problems found in a hand-filled file. Errors block; warnings are shown."""

    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValueError(
                f"{self.name}: {len(self.errors)} error(s):\n  - " + "\n  - ".join(self.errors)
            )


def event_id(event_type: str, date: dt.date) -> str:
    return f"{event_type}_{date.isoformat()}"


def _row(
    cfg: Settings, event_type: str, date: dt.date, stage: str, time_et: str, source: str
) -> dict[str, object]:
    return {
        "event_id": event_id(event_type, date),
        "event_type": event_type,
        "tier": cfg.events[event_type].tier,
        "date": date,
        "stage": stage,
        "time_et": time_et,
        "t0_utc_ns": et_to_utc_ns(date, time_et, cfg.timezone),
        "t0_utc": et_to_utc(date, time_et, cfg.timezone).isoformat(),
        "source": source,
    }


def fred_event_rows(
    cfg: Settings, client: FredClient, event_types: list[str]
) -> list[dict[str, object]]:
    start = dt.date.fromisoformat(cfg.sample.start)
    end = dt.date.fromisoformat(cfg.sample.end)
    rows: list[dict[str, object]] = []
    for et in event_types:
        search = cfg.events[et].fred_release_search
        if search is None:
            continue
        rid = client.find_release_id(search)
        dates = client.release_dates(rid, start, end)
        time_et = cfg.events[et].times_et[0]
        rows += [_row(cfg, et, d, "release", time_et, f"FRED release {rid}") for d in dates]
        log.info("fred_release_dates", event_type=et, release_id=rid, n_dates=len(dates))
    return rows


def write_fomc_template(path: Path) -> bool:
    """Create the empty FOMC file for you to fill in. Never overwrites an existing file."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(FOMC_COLUMNS) + "\n")
    return True


def validate_fomc(cfg: Settings, path: Path) -> tuple[pl.DataFrame, ValidationReport]:
    rep = ValidationReport(path.name)
    if not path.exists():
        rep.errors.append(
            f"{path} does not exist; run `printtime calendar build` to create the template"
        )
        return pl.DataFrame(), rep
    df = pl.read_csv(path, infer_schema_length=0)  # read everything as text, then check
    missing = [c for c in FOMC_COLUMNS if c not in df.columns]
    if missing:
        rep.errors.append(f"missing columns: {missing}")
        return df, rep
    if df.height == 0:
        rep.errors.append(
            "no FOMC meetings yet: fill in one row per meeting from federalreserve.gov"
        )
        return df, rep
    start, end = cfg.sample.start, cfg.sample.end
    seen: set[str] = set()
    for i, r in enumerate(df.iter_rows(named=True), start=2):
        where = f"line {i}"
        try:
            d = dt.date.fromisoformat(str(r["date"]).strip())
        except ValueError:
            rep.errors.append(f"{where}: date {r['date']!r} is not YYYY-MM-DD")
            continue
        if d.isoformat() in seen:
            rep.errors.append(f"{where}: duplicate meeting date {d}")
        seen.add(d.isoformat())
        if not (start <= d.isoformat() <= end):
            rep.warnings.append(
                f"{where}: {d} is outside the sample {start} to {end} and will be ignored"
            )
        if d.weekday() >= 5:
            rep.errors.append(f"{where}: {d} is a weekend")
        st = str(r["statement_time_et"] or "").strip()
        if not _HHMM.match(st):
            rep.errors.append(f"{where}: statement_time_et {st!r} is not HH:MM")
        pc = str(r["press_conference_time_et"] or "").strip()
        if pc and not _HHMM.match(pc):
            rep.errors.append(
                f"{where}: press_conference_time_et {pc!r} is not HH:MM (leave empty if none)"
            )
        if pc and _HHMM.match(st) and pc <= st:
            rep.errors.append(f"{where}: press conference {pc} is not after the statement {st}")
        if not _URL.match(str(r["source_url"] or "").strip()):
            rep.errors.append(f"{where}: source_url must be an http(s) link to the Fed's page")
    return df, rep


def fomc_event_rows(cfg: Settings, df: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for r in df.iter_rows(named=True):
        d = dt.date.fromisoformat(str(r["date"]).strip())
        if not (cfg.sample.start <= d.isoformat() <= cfg.sample.end):
            continue
        src = str(r["source_url"]).strip()
        rows.append(_row(cfg, "FOMC", d, "statement", str(r["statement_time_et"]).strip(), src))
        pc = str(r["press_conference_time_et"] or "").strip()
        if pc:
            rows.append(_row(cfg, "FOMC", d, "press_conference", pc, src))
    return rows


def build_calendar(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=dict.fromkeys(RELEASE_COLUMNS, pl.String))
    df = pl.DataFrame(rows).select(RELEASE_COLUMNS).sort(["t0_utc_ns", "event_type"])
    dup = df.group_by(["event_id", "stage"]).len().filter(pl.col("len") > 1)
    if dup.height:
        raise ValueError(f"duplicate events in calendar: {dup['event_id'].to_list()}")
    return df


def load_calendar(cfg: Settings) -> pl.DataFrame:
    path = cfg.paths.calendar / "releases.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `printtime calendar build`")
    return pl.read_csv(path, try_parse_dates=True)
