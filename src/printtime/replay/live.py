"""Release-day replay with Prometheus metrics (spec 9.2).

Plays one release window for every instrument, merged in exchange-time
order, at market speed (or faster), and exports per instrument:

    printtime_spread_ticks               current bid-ask spread
    printtime_top_depth_lots             best bid size + best ask size
    printtime_depth_pct_of_baseline      the same, as % of the pre-release baseline
    printtime_mid_move_ticks             mid minus the pre-release mid
    printtime_trades_per_second          trades in the last second
    printtime_seconds_from_release       the replay clock, relative to t0
    printtime_last_event_wall_timestamp_seconds   for the "no data" alert

The baseline is the median top-of-book depth from -30 min to -10 min when the
book reaches back that far, otherwise over the first minute of the replay.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import polars as pl
from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from printtime.panel.build import Book
from printtime.schema import FILL, NO_PRICE, TRADE

NS = 1_000_000_000


@dataclass
class ReplayStats:
    events: int = 0
    wall_seconds: float = 0.0


class ReplayMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r, lbl = self.registry, ["instrument"]
        self.spread = Gauge("printtime_spread_ticks", "Bid-ask spread in ticks", lbl, registry=r)
        self.depth = Gauge(
            "printtime_top_depth_lots", "Best bid size plus best ask size", lbl, registry=r
        )
        self.depth_pct = Gauge(
            "printtime_depth_pct_of_baseline", "Top-of-book depth as % of baseline", lbl, registry=r
        )
        self.mid_move = Gauge(
            "printtime_mid_move_ticks", "Mid minus the pre-release mid, ticks", lbl, registry=r
        )
        self.trade_rate = Gauge(
            "printtime_trades_per_second", "Trades in the last second", lbl, registry=r
        )
        self.events = Counter("printtime_events", "Book updates replayed", lbl, registry=r)
        self.clock = Gauge(
            "printtime_seconds_from_release", "Replay clock relative to t0", registry=r
        )
        self.last_wall = Gauge(
            "printtime_last_event_wall_timestamp_seconds",
            "Wall time of the last update",
            registry=r,
        )
        self.in_window = Gauge(
            "printtime_replay_active", "1 while a release window is being replayed", registry=r
        )

    def serve(self, port: int) -> None:
        start_http_server(port, registry=self.registry)


def baseline_depth(frame: pl.DataFrame, t0_ns: int, replay_start_ns: int) -> float:
    """Normal top-of-book depth before the release.

    Prefer -30 min to -10 min (the panels' baseline). If the data do not reach that
    far back, use anything from -30 min up to one minute before t0, never later:
    the last minute can already contain the liquidity withdrawal. With neither,
    there is no honest baseline and the % metric is not reported.
    """
    ts = frame["ts_event"].to_numpy()
    depth = (frame["bid_sz_0"] + frame["ask_sz_0"]).to_numpy().astype(np.float64)
    for lo_s, hi_s in ((-1800, -600), (-1800, -60)):
        sel = (ts >= t0_ns + lo_s * NS) & (ts < t0_ns + hi_s * NS)
        if sel.sum() >= 50:
            return float(np.median(depth[sel]))
    return float("nan")


def merged_stream(books: dict[str, Book], start_ns: int, end_ns: int) -> pl.DataFrame:
    parts = []
    for inst, b in books.items():
        f = b.frame.filter(pl.col("ts_event").is_between(start_ns, end_ns))
        parts.append(
            f.select(
                "ts_event", "action", "bid_px_0", "ask_px_0", "bid_sz_0", "ask_sz_0"
            ).with_columns(pl.lit(inst).alias("instrument"))
        )
    return pl.concat(parts).sort("ts_event", maintain_order=True) if parts else pl.DataFrame()


def replay(
    books: dict[str, Book],
    t0_ns: int,
    metrics: ReplayMetrics,
    *,
    start_s: int = -300,
    end_s: int = 600,
    realtime: bool = True,
    speed: float = 1.0,
) -> ReplayStats:
    if speed <= 0:
        raise ValueError("speed must be positive")
    start_ns, end_ns = t0_ns + start_s * NS, t0_ns + end_s * NS
    base = {i: baseline_depth(b.frame, t0_ns, start_ns) for i, b in books.items()}
    pre_mid = {}
    for i, b in books.items():
        f = b.frame.filter(pl.col("ts_event") < t0_ns).tail(1)
        pre_mid[i] = float((f["bid_px_0"][0] + f["ask_px_0"][0]) / 2) if f.height else float("nan")
    stream = merged_stream(books, start_ns, end_ns)
    trades: dict[str, deque[int]] = {i: deque() for i in books}
    stats = ReplayStats()
    wall0 = time.perf_counter()
    metrics.in_window.set(1)
    cols = [
        stream[c].to_numpy()
        for c in ("ts_event", "action", "bid_px_0", "ask_px_0", "bid_sz_0", "ask_sz_0")
    ]
    insts = stream["instrument"].to_list()
    for k in range(stream.height):
        ts, action, bp, ap, bs, asz = (int(c[k]) for c in cols)
        inst = insts[k]
        if realtime:
            due = wall0 + (ts - start_ns) / NS / speed
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        if bp != NO_PRICE and ap != NO_PRICE:
            metrics.spread.labels(inst).set(ap - bp)
            metrics.mid_move.labels(inst).set((ap + bp) / 2 - pre_mid[inst])
        metrics.depth.labels(inst).set(bs + asz)
        if base[inst] and np.isfinite(base[inst]):
            metrics.depth_pct.labels(inst).set(100.0 * (bs + asz) / base[inst])
        q = trades[inst]
        if action in (TRADE, FILL):
            q.append(ts)
        while q and q[0] <= ts - NS:
            q.popleft()
        metrics.trade_rate.labels(inst).set(len(q))
        metrics.events.labels(inst).inc()
        metrics.clock.set((ts - t0_ns) / NS)
        metrics.last_wall.set(time.time())
        stats.events += 1
    metrics.in_window.set(0)
    stats.wall_seconds = time.perf_counter() - wall0
    return stats
