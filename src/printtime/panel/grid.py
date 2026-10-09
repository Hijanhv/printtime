"""Event-time grids: sampling a book onto fixed offsets from t0.

The book in force at a grid point is the last update *at or before* that
instant, found with a binary search. Using "at or before" (never "nearest")
means a grid point never sees an update from its future, which is what the
no-look-ahead tests check.

All arithmetic is in integer nanoseconds. Event timestamps are ~1.7e18 ns,
where float64 only resolves ~256 ns, so mixing in floats would shift times.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]
NS = 1_000_000_000
MS = 1_000_000


def offsets_ns(start_s: int, end_s: int, step_ms: int) -> I64:
    """Grid offsets from t0 in ns, inclusive of both ends."""
    n = (end_s - start_s) * 1000 // step_ms
    out: I64 = start_s * NS + np.arange(n + 1, dtype=np.int64) * (step_ms * MS)
    return out


def last_index(ts: I64, at: I64) -> I64:
    """Index of the last row with ts <= at, or -1 when no row precedes that instant."""
    out: I64 = np.searchsorted(ts, at, side="right").astype(np.int64) - 1
    return out


def sample(values: npt.NDArray[np.generic], idx: I64, fill: float = np.nan) -> F64:
    """values[idx] as float, with `fill` where idx == -1 (before the first update)."""
    out: F64 = np.where(idx >= 0, values[np.maximum(idx, 0)].astype(np.float64), fill)
    return out


def counts_between(ts: I64, edges: I64, weights: npt.NDArray[np.generic] | None = None) -> F64:
    """Per grid interval (edges[k-1], edges[k]]: number of rows, or the sum of weights.

    The first entry covers everything up to edges[0] that is not counted elsewhere
    and is reported as NaN, because the interval before the grid starts is unknown.
    """
    pos = np.searchsorted(ts, edges, side="right")
    if weights is None:
        cum = pos.astype(np.float64)
        out: F64 = np.diff(cum, prepend=np.nan)
        return out
    csum = np.concatenate([[0.0], np.cumsum(weights.astype(np.float64))])
    out = np.diff(csum[pos], prepend=np.nan)
    return out
