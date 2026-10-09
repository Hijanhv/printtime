"""Event-time panels (spec 4): book and trade data aligned on grids around t0.

For every window (an event stage or a control day) and instrument:

* fine grid:   100 ms steps from -60 s to +120 s (first mover, jumps);
* coarse grid: 1 s steps from -30 min to +60 min (liquidity dynamics);
* long grid:   1 s ohlcv bars from -2 h to +4 h (longer reactions), when available.

Per grid point (state = last update at or before that instant):
    bid/ask price and size, spread (ticks), mid (ticks), depth at 1/3/5/10
    levels (where the source has them), and per interval since the previous
    grid point: book updates, trades, traded volume, signed (aggressor) volume.

`mid_move` is the mid minus the last mid strictly before t0, so a jump at t0
shows up exactly at offset 0. It is the one deliberate exception to "only
past data": before t0 it is measured against the pre-release mid, which lies
in the future of those grid points. It is a description of the path, never a
predictor; every other measure uses only data at or before its grid point
(tested), and `mid_move` does too from t0 onward.

Baselines: the median of each measure over -30 min to -10 min (coarse grid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import polars as pl

from printtime.config import Settings
from printtime.panel.grid import counts_between, last_index, offsets_ns, sample
from printtime.schema import BID, FILL, NO_PRICE, TRADE

NS = 1_000_000_000
MS = 1_000_000
F64 = npt.NDArray[np.float64]

MEASURES = [
    "bid_px",
    "ask_px",
    "bid_sz",
    "ask_sz",
    "spread",
    "mid",
    "mid_move",
    "depth_1",
    "depth_3",
    "depth_5",
    "depth_10",
    "updates",
    "trades",
    "volume",
    "signed_volume",
]
BASELINE_MEASURES = [
    "spread",
    "depth_1",
    "depth_3",
    "depth_5",
    "depth_10",
    "updates",
    "trades",
    "volume",
]


@dataclass(frozen=True)
class Stage:
    """One alignment point: an event stage (CPI release, FOMC statement, ...) or a control."""

    window_id: str
    event_type: str  # CPI, NFP, FOMC, or CONTROL
    stage: str  # release, statement, press_conference, control
    time_et: str
    t0_ns: int


@dataclass(frozen=True)
class Book:
    frame: pl.DataFrame
    tick_size: float
    source: str  # mbp-1, mbp-10, synthetic


class FrameProvider(Protocol):
    def book(self, window_id: str, instrument: str) -> Book | None: ...


def _levels(frame: pl.DataFrame) -> int:
    return sum(1 for c in frame.columns if c.startswith("bid_px_"))


def panel_for(
    frame: pl.DataFrame, t0_ns: int, start_s: int, end_s: int, step_ms: int, depth_levels: list[int]
) -> pl.DataFrame:
    """Measures on one grid for one book frame (rows sorted by ts_event)."""
    ts = frame["ts_event"].to_numpy()
    offs = offsets_ns(start_s, end_s, step_ms)
    grid = t0_ns + offs
    idx = last_index(ts, grid)
    b = frame["bid_px_0"].to_numpy()
    a = frame["ask_px_0"].to_numpy()
    valid = (b != NO_PRICE) & (a != NO_PRICE)
    mid = np.where(valid, (b + a) / 2.0, np.nan)
    pre = last_index(ts, np.array([t0_ns - 1], dtype=np.int64))[0]
    pre_mid = float(mid[pre]) if pre >= 0 else np.nan

    out: dict[str, npt.NDArray[np.generic]] = {"offset_ms": (offs // MS).astype(np.int64)}
    out["bid_px"] = sample(np.where(valid, b, np.nan), idx)
    out["ask_px"] = sample(np.where(valid, a, np.nan), idx)
    out["bid_sz"] = sample(frame["bid_sz_0"].to_numpy(), idx)
    out["ask_sz"] = sample(frame["ask_sz_0"].to_numpy(), idx)
    out["spread"] = out["ask_px"] - out["bid_px"]  # type: ignore[operator]
    out["mid"] = sample(mid, idx)
    out["mid_move"] = out["mid"] - pre_mid  # type: ignore[operator]
    levels = _levels(frame)
    for k in depth_levels:
        if k <= levels:
            depth = np.zeros(frame.height)
            for i in range(k):
                depth = depth + frame[f"bid_sz_{i}"].to_numpy() + frame[f"ask_sz_{i}"].to_numpy()
            out[f"depth_{k}"] = sample(depth, idx)
        else:
            out[f"depth_{k}"] = np.full(offs.size, np.nan)
    action = frame["action"].to_numpy()
    is_trade = np.isin(action, [TRADE, FILL])
    size = frame["size"].to_numpy().astype(np.float64)
    sign = np.where(frame["side"].to_numpy() == BID, 1.0, -1.0)
    out["updates"] = counts_between(ts, grid)
    out["trades"] = counts_between(ts, grid, is_trade.astype(np.float64))
    out["volume"] = counts_between(ts, grid, np.where(is_trade, size, 0.0))
    out["signed_volume"] = counts_between(ts, grid, np.where(is_trade, size * sign, 0.0))
    return pl.DataFrame(out)


def ohlcv_panel(frame: pl.DataFrame, t0_ns: int, start_s: int, end_s: int) -> pl.DataFrame:
    """Long-horizon close and volume on a 1 s grid from ohlcv-1s bars (ts_event = bar open)."""
    ts = frame["ts_event"].to_numpy() + NS  # a bar is complete at the end of its second
    offs = offsets_ns(start_s, end_s, 1000)
    grid = t0_ns + offs
    idx = last_index(ts, grid)
    close = sample(frame["close"].to_numpy(), idx)
    pre = last_index(ts, np.array([t0_ns], dtype=np.int64))[0]
    pre_close = float(frame["close"][int(pre)]) if pre >= 0 else np.nan
    return pl.DataFrame(
        {
            "offset_ms": (offs // MS).astype(np.int64),
            "close": close,
            "close_move": close - pre_close,
            "volume": counts_between(ts, grid, frame["volume"].to_numpy()),
        }
    )


def tag(df: pl.DataFrame, stage: Stage, instrument: str, book: Book) -> pl.DataFrame:
    return df.with_columns(
        pl.lit(stage.window_id).alias("window_id"),
        pl.lit(stage.event_type).alias("event_type"),
        pl.lit(stage.stage).alias("stage"),
        pl.lit(stage.time_et).alias("time_et"),
        pl.lit(instrument).alias("instrument"),
        pl.lit(book.tick_size).alias("tick_size"),
        pl.lit(book.source).alias("source"),
    )


def baselines(coarse: pl.DataFrame, window_s: tuple[int, int]) -> pl.DataFrame:
    """Median of each measure over the baseline window, per window, stage and instrument."""
    lo, hi = window_s[0] * 1000, window_s[1] * 1000
    keys = ["window_id", "event_type", "stage", "instrument"]
    return (
        coarse.filter((pl.col("offset_ms") >= lo) & (pl.col("offset_ms") < hi))
        .group_by(keys)
        .agg(
            [
                # NaN (no book yet at that grid point) counts as missing: polars
                # sorts NaN above every number, so a plain median is biased up.
                pl.col(m).fill_nan(None).median().alias(f"base_{m}")
                for m in BASELINE_MEASURES
                if m in coarse.columns
            ]
        )
        .sort(keys)
    )


MOVES_WINDOW_S = (-60, 120)


def mid_changes(
    frame: pl.DataFrame, t0_ns: int, window_s: tuple[int, int] = MOVES_WINDOW_S
) -> pl.DataFrame:
    """Every mid-price change near the release, at its exact exchange timestamp.

    The fine grid is 100 ms; first-mover timing needs millisecond precision, so
    it is measured from these exact times instead. offset_ns is relative to t0,
    mid_move to the last mid strictly before t0.
    """
    ts = frame["ts_event"].to_numpy()
    b = frame["bid_px_0"].to_numpy()
    a = frame["ask_px_0"].to_numpy()
    mid = np.where((b != NO_PRICE) & (a != NO_PRICE), (b + a) / 2.0, np.nan)
    pre = last_index(ts, np.array([t0_ns - 1], dtype=np.int64))[0]
    pre_mid = float(mid[pre]) if pre >= 0 else np.nan
    lo, hi = t0_ns + window_s[0] * NS, t0_ns + window_s[1] * NS
    keep = (ts >= lo) & (ts <= hi)
    t, m = ts[keep], mid[keep]
    changed = np.concatenate([[True], np.diff(m) != 0]) & ~np.isnan(m)
    return pl.DataFrame(
        {"offset_ns": (t[changed] - t0_ns).astype(np.int64), "mid_move": m[changed] - pre_mid}
    )


@dataclass
class Panels:
    fine: pl.DataFrame
    coarse: pl.DataFrame
    baselines: pl.DataFrame
    moves: pl.DataFrame


def build_panels(
    cfg: Settings, stages: list[Stage], provider: FrameProvider, instruments: list[str]
) -> Panels:
    p = cfg.panel
    fine_parts, coarse_parts, move_parts = [], [], []
    for st in stages:
        for inst in instruments:
            book = provider.book(st.window_id, inst)
            if book is None or book.frame.height == 0:
                continue
            f = panel_for(
                book.frame,
                st.t0_ns,
                p.fine_grid.start_s,
                p.fine_grid.end_s,
                p.fine_grid.step_ms,
                p.depth_levels,
            )
            c = panel_for(
                book.frame,
                st.t0_ns,
                p.coarse_grid.start_s,
                p.coarse_grid.end_s,
                p.coarse_grid.step_ms,
                p.depth_levels,
            )
            fine_parts.append(tag(f, st, inst, book))
            coarse_parts.append(tag(c, st, inst, book))
            move_parts.append(tag(mid_changes(book.frame, st.t0_ns), st, inst, book))
    fine = pl.concat(fine_parts) if fine_parts else pl.DataFrame()
    coarse = pl.concat(coarse_parts) if coarse_parts else pl.DataFrame()
    base = baselines(coarse, p.baseline_window_s) if coarse.height else pl.DataFrame()
    moves = pl.concat(move_parts) if move_parts else pl.DataFrame()
    return Panels(fine, coarse, base, moves)
