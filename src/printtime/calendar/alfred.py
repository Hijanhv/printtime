"""Cross-check hand-entered `actual` figures against ALFRED first releases.

ALFRED is FRED's archive of every published vintage of a series. A release
on date D creates a vintage whose realtime_start is D. For that vintage:

* the newest observation is the reference month m being announced;
* the headline figure is derived from m and m-1 *as both appeared on D*
  (prior months are often revised in the same release, and the published
  month-over-month change uses the revised prior month).

So for each event we rebuild the published number from the vintage dated on
the release day, round it like the official figure (0.1% or 1k), and flag any
`actual` that differs after rounding. A mismatch usually means a typo, a
wrong month, or a release that was rescheduled.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import polars as pl

from printtime.config import Settings
from printtime.log import get_logger

log = get_logger(__name__)

# series id -> DataFrame with columns realtime_start (date), date (date), value (float)
VintageLoader = Callable[[str], pl.DataFrame]


def fredapi_loader(api_key: str) -> VintageLoader:
    from fredapi import Fred

    fred = Fred(api_key=api_key)

    def load(series: str) -> pl.DataFrame:
        pdf = fred.get_series_all_releases(series)
        return pl.DataFrame(
            {
                "realtime_start": [d.date() for d in pdf["realtime_start"]],
                "date": [d.date() for d in pdf["date"]],
                "value": pdf["value"].astype(float).to_numpy(),
            }
        )

    return load


def published_value(
    vintages: pl.DataFrame, release_date: dt.date, transform: str
) -> tuple[dt.date, float] | None:
    """The figure published on release_date, rebuilt from ALFRED. None if no vintage that day."""
    new = vintages.filter(pl.col("realtime_start") == release_date)
    if new.height == 0:
        return None
    month = new["date"].max()
    if not isinstance(month, dt.date):
        raise TypeError(f"ALFRED dates should be dates, got {type(month).__name__}")
    as_of = vintages.filter(pl.col("realtime_start") <= release_date)

    def value_at(d: dt.date) -> float | None:
        v = as_of.filter(pl.col("date") == d).sort("realtime_start")
        return float(v["value"][-1]) if v.height else None

    cur = value_at(month)
    if cur is None:
        return None
    if transform == "level":
        return month, cur
    prev_month = (month.replace(day=1) - dt.timedelta(days=1)).replace(day=1)
    prev = value_at(prev_month)
    if prev is None:
        return None
    if transform == "pct_change":
        return month, (cur / prev - 1.0) * 100.0
    return month, cur - prev


def cross_check(cfg: Settings, surprises: pl.DataFrame, load: VintageLoader) -> pl.DataFrame:
    """One row per (event, variable): your actual, ALFRED's figure, and whether they agree."""
    cache: dict[str, pl.DataFrame] = {}
    rows = []
    for r in surprises.iter_rows(named=True):
        var = cfg.surprise_variables[r["variable"]]
        if var.series not in cache:
            cache[var.series] = load(var.series)
        got = published_value(cache[var.series], r["release_date"], var.transform)
        if got is None:
            rows.append(
                {
                    **_base(r),
                    "reference_month": None,
                    "alfred": None,
                    "alfred_rounded": None,
                    "status": "no ALFRED vintage on the release date",
                }
            )
            continue
        month, value = got
        rounded = round(value, var.decimals)
        tol = 0.5 * 10 ** (-var.decimals) + 1e-9
        status = "ok" if abs(rounded - r["actual"]) <= tol else "MISMATCH"
        rows.append(
            {
                **_base(r),
                "reference_month": month,
                "alfred": value,
                "alfred_rounded": rounded,
                "status": status,
            }
        )
    out = pl.DataFrame(rows)
    n_bad = int((out["status"] != "ok").sum()) if out.height else 0
    log.info("alfred_cross_check", rows=out.height, not_ok=n_bad)
    return out


def _base(r: dict[str, object]) -> dict[str, object]:
    return {"event_id": r["event_id"], "variable": r["variable"], "actual": r["actual"]}


def mismatches(check: pl.DataFrame) -> pl.DataFrame:
    return check.filter(pl.col("status") != "ok") if check.height else check


__all__ = ["VintageLoader", "cross_check", "fredapi_loader", "mismatches", "published_value"]
