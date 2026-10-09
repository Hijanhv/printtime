"""data/calendar/surprises.csv: template, validator and standardised surprises.

You fill this file in by hand from public news sources; the code never
writes values into it. The validator catches the mistakes hand-typed data
usually has: missing numbers, wrong units, duplicated rows, a date that does
not match any release, a typo that makes core CPI +3.0% instead of +0.3%.

Standardised surprise (Balduzzi, Elton & Green 2001):

    z = (actual - consensus) / std(actual - consensus over the sample)

computed per variable. `standardise(..., expanding=True)` uses only earlier
events' surprises for the scale, which the strategy needs so that it never
looks at the future.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import polars as pl

from printtime.calendar.events import ValidationReport, event_id
from printtime.config import Settings

SURPRISE_COLUMNS = [
    "event_id",
    "event_type",
    "release_date",
    "variable",
    "actual",
    "consensus",
    "prior",
    "unit",
    "source_url",
    "notes",
]
_URL = re.compile(r"^https?://\S+$")
# A surprise scale below this is treated as zero: the z-score is undefined.
MIN_SCALE = 1e-9


def write_surprises_template(path: Path) -> bool:
    """Create the empty file for you to fill in. Never overwrites an existing file."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(SURPRISE_COLUMNS) + "\n")
    return True


def _num(x: object) -> float | None:
    s = "" if x is None else str(x).strip().replace(",", "")
    if s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return v


def _decimals(x: object) -> int:
    s = str(x).strip()
    return len(s.split(".", 1)[1]) if "." in s else 0


def validate_surprises(
    cfg: Settings, path: Path, calendar: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, ValidationReport]:
    """Check the hand-filled file.

    Returns typed rows (actual, consensus and prior as floats) and a report.
    """
    rep = ValidationReport(path.name)
    if not path.exists():
        rep.errors.append(
            f"{path} does not exist; run `printtime calendar build` to create the template"
        )
        return pl.DataFrame(), rep
    raw = pl.read_csv(path, infer_schema_length=0)
    missing = [c for c in SURPRISE_COLUMNS if c not in raw.columns]
    if missing:
        rep.errors.append(f"missing columns: {missing}")
        return raw, rep
    if raw.height == 0:
        rep.errors.append("no rows yet: add one row per release and variable")
        return raw, rep

    known_events = (
        set(calendar["event_id"].to_list()) if calendar is not None and calendar.height else None
    )
    vars_cfg = cfg.surprise_variables
    seen: set[tuple[str, str]] = set()
    rows = []
    for i, r in enumerate(raw.iter_rows(named=True), start=2):
        where = f"line {i}"
        et = str(r["event_type"] or "").strip()
        var = str(r["variable"] or "").strip()
        eid = str(r["event_id"] or "").strip()
        bad = False
        if et not in ("CPI", "NFP"):
            rep.errors.append(
                f"{where}: event_type {et!r} must be CPI or NFP "
                "(FOMC uses a market-implied surprise)"
            )
            bad = True
        if var not in vars_cfg:
            rep.errors.append(
                f"{where}: unknown variable {var!r}; expected one of {sorted(vars_cfg)}"
            )
            bad = True
        elif vars_cfg[var].event_type != et:
            rep.errors.append(
                f"{where}: variable {var} belongs to {vars_cfg[var].event_type}, not {et}"
            )
            bad = True
        try:
            d = dt.date.fromisoformat(str(r["release_date"] or "").strip())
        except ValueError:
            rep.errors.append(f"{where}: release_date {r['release_date']!r} is not YYYY-MM-DD")
            continue
        if not (cfg.sample.start <= d.isoformat() <= cfg.sample.end):
            rep.errors.append(f"{where}: release_date {d} is outside the sample")
        if et in ("CPI", "NFP") and eid != event_id(et, d):
            rep.errors.append(f"{where}: event_id {eid!r} should be {event_id(et, d)!r}")
        if known_events is not None and eid not in known_events:
            rep.errors.append(
                f"{where}: {eid} is not in releases.csv (wrong date, or a rescheduled release?)"
            )
        key = (eid, var)
        if key in seen:
            rep.errors.append(f"{where}: duplicate row for {eid} / {var}")
        seen.add(key)

        values = {k: _num(r[k]) for k in ("actual", "consensus", "prior")}
        for k in ("actual", "consensus"):
            if values[k] is None:
                rep.errors.append(f"{where}: {k} is missing")
                bad = True
        for k, v in values.items():
            if v is not None and np.isnan(v):
                rep.errors.append(f"{where}: {k} {r[k]!r} is not a number")
                bad = True
        unit = str(r["unit"] or "").strip()
        if var in vars_cfg and unit != vars_cfg[var].unit:
            rep.errors.append(
                f"{where}: unit for {var} must be {vars_cfg[var].unit!r}, got {unit!r}"
            )
        if not _URL.match(str(r["source_url"] or "").strip()):
            rep.errors.append(
                f"{where}: source_url must be an http(s) link to where the figure came from"
            )
        if bad or var not in vars_cfg:
            continue
        lo, hi = vars_cfg[var].plausible
        for k in ("actual", "consensus"):
            v = values[k]
            if v is not None and not (lo <= v <= hi):
                rep.warnings.append(
                    f"{where}: {var} {k} = {v} is outside the plausible range [{lo}, {hi}]; typo?"
                )
        for k in ("actual", "consensus"):
            if _decimals(r[k]) > vars_cfg[var].decimals + 1:
                rep.warnings.append(
                    f"{where}: {var} {k} has more decimals than published figures ({r[k]})"
                )
        rows.append(
            {
                "event_id": eid,
                "event_type": et,
                "release_date": d,
                "variable": var,
                "actual": values["actual"],
                "consensus": values["consensus"],
                "prior": values["prior"],
                "unit": unit,
                "source_url": str(r["source_url"]).strip(),
                "notes": r["notes"],
            }
        )

    out = pl.DataFrame(rows, schema_overrides={"prior": pl.Float64}) if rows else pl.DataFrame()
    if calendar is not None and calendar.height and out.height:
        for et in ("CPI", "NFP"):
            primary = cfg.primary_variable(et)
            have = set(out.filter(pl.col("variable") == primary)["event_id"].to_list())
            need = set(calendar.filter(pl.col("event_type") == et)["event_id"].to_list())
            missing_ev = sorted(need - have)
            if missing_ev:
                rep.warnings.append(
                    f"{len(missing_ev)} {et} release(s) have no {primary} row yet: "
                    + ", ".join(missing_ev[:6])
                    + (" ..." if len(missing_ev) > 6 else "")
                )
    return out, rep


def standardise(df: pl.DataFrame, expanding: bool = False) -> pl.DataFrame:
    """Add `surprise` (actual - consensus) and `z` (standardised) per variable.

    expanding=False: scale = std over the whole sample (the BEG convention,
    used for the reaction-function regressions).
    expanding=True: scale = std over strictly earlier releases only; NaN until
    at least two earlier surprises exist (used by the strategy, no look-ahead).
    """
    # Round away float noise (0.4 - 0.3 != 0.3 - 0.2 in binary); published
    # figures have at most a few decimals, so 10 decimals loses nothing.
    df = df.sort(["variable", "release_date"]).with_columns(
        (pl.col("actual") - pl.col("consensus")).round(10).alias("surprise")
    )
    if not expanding:
        sd = pl.col("surprise").std(ddof=1).over("variable")
        return df.with_columns(
            pl.when(sd > MIN_SCALE).then(pl.col("surprise") / sd).otherwise(None).alias("z")
        )
    parts = []
    for _, g in df.group_by("variable", maintain_order=True):
        s = g["surprise"].to_numpy()
        z = np.full(s.size, np.nan)
        for k in range(s.size):
            if k >= 2:
                scale = float(np.std(s[:k], ddof=1))
                z[k] = s[k] / scale if scale > MIN_SCALE else np.nan
        parts.append(g.with_columns(pl.Series("z", z)))
    return pl.concat(parts)


def summary(df: pl.DataFrame) -> pl.DataFrame:
    """Per variable: number of events and the distribution of surprises."""
    z = standardise(df)
    return (
        z.group_by(["event_type", "variable"])
        .agg(
            pl.len().alias("events"),
            pl.col("surprise").mean().alias("mean_surprise"),
            pl.col("surprise").std(ddof=1).alias("sd_surprise"),
            pl.col("surprise").min().alias("min_surprise"),
            pl.col("surprise").max().alias("max_surprise"),
            (pl.col("z").abs() > 1).sum().alias("abs_z_over_1"),
        )
        .sort(["event_type", "variable"])
    )
