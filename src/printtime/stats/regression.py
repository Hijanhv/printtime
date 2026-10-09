"""Regression helpers: OLS with HC3 errors, by-event bootstrap, Benjamini-Hochberg FDR.

HC3 standard errors stay reliable when residual variance differs across
events (a big surprise day is noisier than a quiet one) and are the safest
of the heteroskedasticity-robust choices in small samples like ours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OlsResult:
    alpha: float
    beta: float
    se: float
    t: float
    p: float
    r2: float
    n: int


def ols_hc3(x: F64, y: F64) -> OlsResult:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    nan = float("nan")
    if n < 3 or np.std(x) == 0:
        return OlsResult(nan, nan, nan, nan, nan, nan, n)
    res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    return OlsResult(
        float(res.params[0]),
        float(res.params[1]),
        float(res.bse[1]),
        float(res.tvalues[1]),
        float(res.pvalues[1]),
        float(res.rsquared),
        n,
    )


def bootstrap_beta(
    x: F64, y: F64, reps: int, seed: int, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval for the slope, resampling events (pairs)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    xs, ys = x[idx], y[idx]
    xm = xs - xs.mean(axis=1, keepdims=True)
    denom = (xm**2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (xm * (ys - ys.mean(axis=1, keepdims=True))).sum(axis=1) / denom
    b = b[np.isfinite(b)]
    if b.size == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    lo, hi = np.quantile(b, [a, 1 - a])
    return float(lo), float(hi)


def bh_fdr(pvalues: F64, alpha: float) -> tuple[npt.NDArray[np.bool_], F64]:
    """Benjamini-Hochberg: which tests survive at FDR alpha, and the adjusted p-values."""
    p = np.asarray(pvalues, dtype=np.float64)
    reject = np.zeros(p.size, dtype=bool)
    adj = np.full(p.size, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        r, q, _, _ = multipletests(p[ok], alpha=alpha, method="fdr_bh")
        reject[ok], adj[ok] = r, q
    return reject, adj
