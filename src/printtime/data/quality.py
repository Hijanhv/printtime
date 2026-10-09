"""Per-window data-quality checks (spec 3.5).

Every check is counted and logged; nothing is silently dropped. A window is:

* `ok`      no problems;
* `warn`    usable, with issues worth knowing (a duplicated record, an
            instrument switch in a quiet part of the window);
* `reject`  not usable for analysis: no data, a crossed book, or a data gap
            longer than `quality.max_gap_inside_window_s` around the release.

Note on sequence numbers: Databento sequence numbers belong to the exchange
channel, which carries many instruments. A single-symbol download therefore
skips numbers all the time, so only *decreasing* sequences are an error.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from printtime.config import Settings
from printtime.log import get_logger
from printtime.schema import FILL, NO_PRICE, TRADE

log = get_logger(__name__)
NS = 1_000_000_000
CORE_WINDOW_S = (-60, 120)  # the part of every event window that must be complete


@dataclass
class WindowQuality:
    window_id: str
    instrument: str
    schema: str
    rows: int = 0
    out_of_order_timestamps: int = 0
    out_of_order_sequence: int = 0
    duplicate_records: int = 0
    crossed_book_rows: int = 0
    locked_book_rows: int = 0
    one_sided_rows: int = 0
    bad_size_rows: int = 0
    max_gap_s: float = 0.0
    max_gap_core_s: float = 0.0
    starts_late_s: float = 0.0
    ends_early_s: float = 0.0
    instrument_ids: list[int] = field(default_factory=list)
    raw_symbols: list[str] = field(default_factory=list)
    roll_week: bool = False
    status: str = "ok"
    reasons: list[str] = field(default_factory=list)

    def write(self, root: Path) -> Path:
        path = root / self.window_id / f"{self.instrument}_{self.schema}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path


def _gap(ts: np.ndarray, lo: int, hi: int) -> float:
    """Longest stretch without an update inside [lo, hi], counting the edges."""
    inside = ts[(ts >= lo) & (ts <= hi)]
    points = np.concatenate([[lo], inside, [hi]])
    return float(np.diff(points).max() / NS)


def check_window(
    cfg: Settings,
    frame: pl.DataFrame,
    *,
    window_id: str,
    instrument: str,
    schema: str,
    start_ns: int,
    end_ns: int,
    t0s_ns: list[int],
    release_date: dt.date,
    definitions: pl.DataFrame | None = None,
) -> WindowQuality:
    q = WindowQuality(window_id, instrument, schema, rows=frame.height)
    if frame.height == 0:
        q.status, q.reasons = "reject", ["no data in the window"]
        return q
    ts = frame["ts_event"].to_numpy()
    q.out_of_order_timestamps = int(np.sum(np.diff(ts) < 0))
    if "sequence" in frame.columns:
        q.out_of_order_sequence = int(np.sum(np.diff(frame["sequence"].to_numpy()) < 0))
    q.duplicate_records = int(frame.is_duplicated().sum()) // 2 if frame.height else 0
    if "bid_px_0" in frame.columns:
        b, a = frame["bid_px_0"].to_numpy(), frame["ask_px_0"].to_numpy()
        both = (b != NO_PRICE) & (a != NO_PRICE)
        q.crossed_book_rows = int(np.sum(both & (b > a)))
        q.locked_book_rows = int(np.sum(both & (b == a)))
        q.one_sided_rows = int(np.sum(~both))
        sizes = [frame[c].to_numpy() for c in frame.columns if c.startswith(("bid_sz_", "ask_sz_"))]
        bad = np.zeros(frame.height, dtype=bool)
        for s in sizes:
            bad |= s < 0
        if "action" in frame.columns:
            is_trade = np.isin(frame["action"].to_numpy(), [TRADE, FILL])
            bad |= is_trade & (frame["size"].to_numpy() <= 0)
        q.bad_size_rows = int(bad.sum())
    q.max_gap_s = _gap(ts, start_ns, end_ns)
    q.max_gap_core_s = max(
        _gap(ts, t0 + CORE_WINDOW_S[0] * NS, t0 + CORE_WINDOW_S[1] * NS) for t0 in t0s_ns
    )
    q.starts_late_s = max(0.0, (ts.min() - start_ns) / NS)
    q.ends_early_s = max(0.0, (end_ns - ts.max()) / NS)
    if "instrument_id" in frame.columns:
        q.instrument_ids = sorted(int(x) for x in frame["instrument_id"].unique().to_list())
    if definitions is not None and definitions.height and q.instrument_ids:
        d = definitions.filter(pl.col("instrument_id").is_in(q.instrument_ids))
        q.raw_symbols = sorted(d["raw_symbol"].to_list())
        days = cfg.roll.days_before_expiration.get(instrument)
        if days is not None and d.height:
            exp = min(
                dt.datetime.fromtimestamp(x / NS, dt.UTC).date()
                for x in d["expiration_ns"].to_list()
            )
            q.roll_week = 0 <= (exp - release_date).days <= days

    if q.crossed_book_rows > cfg.quality.crossed_book_tolerance:
        q.reasons.append(f"{q.crossed_book_rows} crossed-book rows")
    if q.max_gap_core_s > cfg.quality.max_gap_inside_window_s:
        q.reasons.append(f"{q.max_gap_core_s:.1f} s without data around the release")
    if q.reasons:
        q.status = "reject"
    else:
        warns = []
        if q.out_of_order_timestamps or q.out_of_order_sequence:
            warns.append("records out of order")
        if q.duplicate_records:
            warns.append(f"{q.duplicate_records} duplicate records")
        if q.bad_size_rows:
            warns.append(f"{q.bad_size_rows} rows with impossible sizes")
        if len(q.instrument_ids) > 1:
            warns.append("the front contract changed inside the window")
        if q.roll_week:
            warns.append("release falls in the roll period")
        if q.max_gap_s > cfg.quality.max_gap_inside_window_s:
            warns.append(f"{q.max_gap_s:.1f} s gap away from the release")
        if warns:
            q.status, q.reasons = "warn", warns
    if q.status != "ok":
        log.warning(
            "window_quality",
            window=window_id,
            instrument=instrument,
            schema=schema,
            status=q.status,
            reasons=q.reasons,
        )
    return q


def summary_table(reports: list[WindowQuality]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "window_id": r.window_id,
                "instrument": r.instrument,
                "schema": r.schema,
                "rows": r.rows,
                "status": r.status,
                "roll_week": r.roll_week,
                "max_gap_core_s": r.max_gap_core_s,
                "reasons": "; ".join(r.reasons),
            }
            for r in reports
        ]
    )
