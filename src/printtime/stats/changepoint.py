"""When does liquidity withdrawal start? Change-point detection with ruptures.

Withdrawal is a ramp: depth is roughly flat, then declines into the release.
A change-in-mean model would put the break in the middle of the ramp; a
continuous piecewise-linear model ("clinear": flat or sloped segments that
join up) puts it where the decline begins, which is the question asked.

Per event: take depth as % of baseline on the 1 s grid over a search window
ending at t0, smooth it with a short rolling median (single snapshots are
noisy), and find the single best break. The break only counts as a
withdrawal start if the line after it falls by at least `min_drop_pct`
percentage points by t0; otherwise the event shows no withdrawal (NaN).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import ruptures as rpt

F64 = npt.NDArray[np.float64]


def rolling_median(x: F64, width: int) -> F64:
    if width <= 1:
        return x.copy()
    pad = width // 2
    padded = np.pad(x, (pad, width - 1 - pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    out: F64 = np.nanmedian(windows, axis=1)
    return out


def withdrawal_start(
    offsets_s: F64,
    depth_pct: F64,
    search_s: tuple[int, int] = (-180, 0),
    smooth: int = 5,
    min_drop_pct: float = 15.0,
    min_size: int = 5,
) -> float:
    """Seconds relative to t0 at which depth starts to fall, or NaN if it does not."""
    keep = (offsets_s >= search_s[0]) & (offsets_s <= search_s[1])
    x = offsets_s[keep]
    y = depth_pct[keep]
    ok = ~np.isnan(y)
    if ok.sum() < 3 * min_size:
        return float("nan")
    x, y = x[ok], rolling_median(y[ok], smooth)
    algo = rpt.Dynp(model="clinear", min_size=min_size, jump=1).fit(y.reshape(-1, 1))
    bkp = algo.predict(n_bkps=1)[0]  # index where the second segment starts
    if bkp <= 0 or bkp >= y.size:
        return float("nan")
    before = float(np.median(y[:bkp]))
    after_end = float(np.median(y[-min_size:]))
    if before - after_end < min_drop_pct:
        return float("nan")
    return float(x[bkp])
