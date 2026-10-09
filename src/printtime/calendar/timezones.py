"""Release times: New York local time to UTC, safe across daylight saving.

U.S. releases are scheduled in New York time (08:30 ET for CPI and NFP,
14:00 ET for FOMC). New York is UTC-4 in summer and UTC-5 in winter, so a
fixed offset would put t0 an hour wrong for part of every year. zoneinfo
applies the correct offset for each date.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

NS = 1_000_000_000
NEW_YORK = "America/New_York"


def et_to_utc(date: dt.date, hhmm: str, tz: str = NEW_YORK) -> dt.datetime:
    """Local release time on `date` as an aware UTC datetime."""
    hour, minute = (int(x) for x in hhmm.split(":"))
    local = dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=ZoneInfo(tz))
    return local.astimezone(dt.UTC)


def et_to_utc_ns(date: dt.date, hhmm: str, tz: str = NEW_YORK) -> int:
    """Same as et_to_utc, as integer nanoseconds since the epoch (Databento ts_event units)."""
    return int(et_to_utc(date, hhmm, tz).timestamp()) * NS


def utc_ns_to_et(ns: int, tz: str = NEW_YORK) -> dt.datetime:
    return dt.datetime.fromtimestamp(ns / NS, dt.UTC).astimezone(ZoneInfo(tz))
