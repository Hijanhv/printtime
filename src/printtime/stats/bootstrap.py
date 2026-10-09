"""Bootstrap confidence intervals that resample events, never individual seconds.

Seconds inside one release window are strongly dependent: if depth is low at
+3 s it is low at +4 s. Resampling seconds would pretend there are thousands
of independent observations when there are really ~20 events, and the
intervals would be far too narrow. So every interval here resamples whole
events (rows), with replacement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CurveCI:
    mean: F64
    lo: F64
    hi: F64
    n_events: int


def event_indices(n_events: int, reps: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """reps x n_events matrix of resampled event (row) indices."""
    out: npt.NDArray[np.int64] = rng.integers(0, n_events, size=(reps, n_events))
    return out


def curve_ci(matrix: F64, reps: int, seed: int, level: float = 0.95) -> CurveCI:
    """matrix: events x time. Mean across events with a by-event bootstrap band."""
    m = np.asarray(matrix, dtype=np.float64)
    n = m.shape[0]
    # Grid points where no event has data give NaN, which is the right answer;
    # numpy's "mean of empty slice" warning about them is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(m, axis=0) if n else np.full(m.shape[1], np.nan)
        if n < 2:
            return CurveCI(mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan), n)
        idx = event_indices(n, reps, np.random.default_rng(seed))
        boot = np.nanmean(m[idx], axis=1)  # reps x time
        a = (1 - level) / 2
        lo, hi = np.nanquantile(boot, [a, 1 - a], axis=0)
    return CurveCI(mean, lo, hi, n)


def scalar_ci(
    values: F64, reps: int, seed: int, stat: str = "median", level: float = 0.95
) -> tuple[float, float, float, int]:
    """(estimate, lo, hi, n) for the mean or median across events."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    n = v.size
    f = np.median if stat == "median" else np.mean
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    est = float(f(v))
    if n < 2:
        return est, float("nan"), float("nan"), n
    idx = event_indices(n, reps, np.random.default_rng(seed))
    boot = f(v[idx], axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    return est, float(lo), float(hi), n
