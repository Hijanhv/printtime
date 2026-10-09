"""Control days: the same clock window on days with no scheduled release.

08:30 and 14:00 ET are busy times even without news (the cash Treasury
open, scheduled auctions, the equity open an hour later), so every liquidity
effect is measured against control windows at the same clock time.

Rules:
* no Tier 1 or Tier 2 release at that time on that day (weekly jobless claims
  are every Thursday 08:30, so Tier 2 dates must be in the calendar);
* not a U.S. market holiday, not a weekend, not in the optional exclusion file;
* weekdays matched to the events' weekday mix where possible, then drawn at
  random with a fixed seed so the selection is reproducible.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter

import numpy as np
import polars as pl

from printtime.calendar.timezones import et_to_utc, et_to_utc_ns
from printtime.config import Settings
from printtime.log import get_logger

log = get_logger(__name__)


class MissingTier2Error(RuntimeError):
    """Raised when control days would be chosen without knowing Tier 2 release dates."""


def _easter(year: int) -> dt.date:
    """Gregorian Easter (anonymous algorithm), needed for Good Friday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1) if month < 12 else dt.date(year, 12, 31)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def us_market_holidays(year: int) -> set[dt.date]:
    """U.S. exchange holidays by rule (NYSE calendar), used only to avoid empty control days.

    CME sometimes trades with early closes on these days; excluding them is
    conservative. A day with missing data is caught later by the quality checks.
    """
    mon, thu = 0, 3
    return {
        _observed(dt.date(year, 1, 1)),
        _nth_weekday(year, 1, mon, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, mon, 3),  # Presidents' Day
        _easter(year) - dt.timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, mon),  # Memorial Day
        _observed(dt.date(year, 6, 19)),  # Juneteenth
        _observed(dt.date(year, 7, 4)),
        _nth_weekday(year, 9, mon, 1),  # Labor Day
        _nth_weekday(year, 11, thu, 4),  # Thanksgiving
        _nth_weekday(year, 11, thu, 4)
        + dt.timedelta(days=1),  # day after Thanksgiving (early close)
        _observed(dt.date(year, 12, 25)),
        dt.date(year, 12, 24),  # Christmas Eve: early close, thin markets
    }


def load_exclusions(cfg: Settings) -> set[dt.date]:
    path = cfg.control_days.exclude_file
    if not path.exists():
        return set()
    df = pl.read_csv(path, infer_schema_length=0)
    return {dt.date.fromisoformat(str(d).strip()) for d in df["date"].to_list() if str(d).strip()}


def _allocate(total: int, weights: Counter[int]) -> dict[int, int]:
    """Split `total` across weekdays in proportion to `weights` (largest remainder)."""
    n = sum(weights.values())
    if n == 0:
        return {}
    exact = {k: total * v / n for k, v in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True)[
        : total - sum(out.values())
    ]:
        out[k] += 1
    return out


def select_control_days(
    cfg: Settings,
    calendar: pl.DataFrame,
    *,
    require_tier2: bool = True,
    extra_exclusions: set[dt.date] | None = None,
) -> pl.DataFrame:
    """One row per control window: control_id, time_et, date, weekday, t0_utc_ns, t0_utc."""
    have_tier2 = set(calendar["event_type"].unique().to_list()) & set(cfg.tier(2))
    if require_tier2 and not have_tier2:
        raise MissingTier2Error(
            "the calendar has no Tier 2 releases (PPI, retail sales, jobless claims). "
            "Without them control "
            "days could land on a claims Thursday at 08:30. Run `printtime calendar build --tier2`."
        )
    start = dt.date.fromisoformat(cfg.sample.start)
    end = dt.date.fromisoformat(cfg.sample.end)
    holidays: set[dt.date] = set()
    for y in range(start.year, end.year + 1):
        holidays |= us_market_holidays(y)
    excluded = load_exclusions(cfg) | (extra_exclusions or set())

    busy: dict[str, set[dt.date]] = {}  # any Tier 1 or 2 release at that time: not a control
    studied: dict[
        str, list[dt.date]
    ] = {}  # Tier 1 events: their weekday mix is what controls match
    tier1 = set(cfg.tier(1))
    for r in calendar.iter_rows(named=True):
        d = r["date"] if isinstance(r["date"], dt.date) else dt.date.fromisoformat(str(r["date"]))
        busy.setdefault(str(r["time_et"]), set()).add(d)
        if r["event_type"] in tier1:
            studied.setdefault(str(r["time_et"]), []).append(d)

    rng = np.random.default_rng(cfg.control_days.seed)
    rows = []
    for time_et, n_wanted in cfg.control_days.per_time.items():
        event_days = busy.get(time_et, set())
        candidates = [
            start + dt.timedelta(days=k)
            for k in range((end - start).days + 1)
            if (start + dt.timedelta(days=k)).weekday() < 5
        ]
        candidates = [
            d for d in candidates if d not in event_days and d not in holidays and d not in excluded
        ]
        weights = Counter(d.weekday() for d in studied.get(time_et, []))
        if cfg.control_days.match_weekday and weights:
            plan = _allocate(n_wanted, weights)
        else:
            plan = {wd: 0 for wd in range(5)}
            plan.update(_allocate(n_wanted, Counter(range(5))))
        chosen: list[dt.date] = []
        shortfall = 0
        for wd, k in sorted(plan.items()):
            pool = [d for d in candidates if d.weekday() == wd]
            take = min(k, len(pool))
            shortfall += k - take
            pick: list[int] = (
                rng.choice(len(pool), size=take, replace=False).tolist() if take else []
            )
            chosen += [pool[i] for i in sorted(pick)]
        if shortfall:
            # A weekday ran out of free days (every Thursday has jobless claims at
            # 08:30, for example): fill the gap from the remaining free days.
            rest = [d for d in candidates if d not in set(chosen)]
            take = min(shortfall, len(rest))
            pick = rng.choice(len(rest), size=take, replace=False).tolist() if take else []
            chosen += [rest[i] for i in sorted(pick)]
            log.warning(
                "control_weekday_shortfall",
                time_et=time_et,
                refilled=take,
                missing=shortfall - take,
            )
        for d in sorted(chosen):
            rows.append(
                {
                    "control_id": f"CONTROL{time_et.replace(':', '')}_{d.isoformat()}",
                    "time_et": time_et,
                    "date": d,
                    "weekday": d.strftime("%a"),
                    "t0_utc_ns": et_to_utc_ns(d, time_et, cfg.timezone),
                    "t0_utc": et_to_utc(d, time_et, cfg.timezone).isoformat(),
                }
            )
        log.info("controls_selected", time_et=time_et, wanted=n_wanted, chosen=len(chosen))
    return pl.DataFrame(rows)
