"""Confirm each release happened when the calendar says (spec 3.3).

Releases are occasionally rescheduled, for example during a government
shutdown. If the calendar's t0 is wrong, every result for that event is wrong,
so each event is checked against the market itself: price moves in the first
minute after t0 must be clearly larger than in the quiet stretch before it.

    spike ratio = mean |mid change| per second in [t0, t0 + 60 s]
                / mean |mid change| per second in [t0 - 10 min, t0 - 1 min]

An event is confirmed if any reference instrument (ZN, ES by default) shows a
ratio of at least `validation.min_spike_ratio`. Events that fail are listed
for you to check by hand; they are not deleted.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from printtime.config import Settings
from printtime.panel.grid import last_index, offsets_ns, sample
from printtime.schema import NO_PRICE


def mid_on_second_grid(frame: pl.DataFrame, t0_ns: int, start_s: int, end_s: int) -> np.ndarray:
    ts = frame["ts_event"].to_numpy()
    b = frame["bid_px_0"].to_numpy().astype(np.float64)
    a = frame["ask_px_0"].to_numpy().astype(np.float64)
    mid = np.where((b == NO_PRICE) | (a == NO_PRICE), np.nan, (b + a) / 2)
    grid = t0_ns + offsets_ns(start_s, end_s, 1000)
    return sample(mid, last_index(ts, grid))


def spike_ratio(cfg: Settings, frame: pl.DataFrame, t0_ns: int) -> float:
    v = cfg.validation
    lo = min(v.quiet_window_s[0], v.spike_window_s[0])
    hi = max(v.quiet_window_s[1], v.spike_window_s[1])
    mid = mid_on_second_grid(frame, t0_ns, lo, hi)
    moves = np.abs(np.diff(mid))  # moves[k] is the change into second lo + k + 1
    secs = np.arange(lo + 1, hi + 1)

    def mean_in(w: tuple[int, int]) -> float:
        m = moves[(secs > w[0]) & (secs <= w[1])]
        return float(np.nanmean(m)) if m.size and not np.all(np.isnan(m)) else float("nan")

    quiet = mean_in(v.quiet_window_s)
    spike = mean_in(v.spike_window_s)
    if np.isnan(quiet) or np.isnan(spike):
        return float("nan")
    if quiet == 0:
        return float("inf") if spike > 0 else float("nan")
    return spike / quiet


def validate_events(cfg: Settings, ratios: pl.DataFrame) -> pl.DataFrame:
    """ratios: event_id, instrument, ratio. Returns one row per event with a verdict."""
    # NaN (no data) must not win a max(): treat it as missing.
    ref = ratios.filter(
        pl.col("instrument").is_in(cfg.validation.reference_instruments)
    ).with_columns(pl.col("ratio").fill_nan(None))
    return (
        ref.group_by("event_id")
        .agg(
            pl.col("ratio").max().alias("best_ratio"),
            pl.col("instrument").sort_by("ratio").last().alias("best_instrument"),
        )
        .with_columns(
            (pl.col("best_ratio") >= cfg.validation.min_spike_ratio)
            .fill_null(False)
            .alias("confirmed")
        )
        .sort("event_id")
    )
