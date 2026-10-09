"""Analysis B: price reaction and price discovery (spec 6).

1. Jump size: |mid move| at +1 s, +10 s, +1 min, +5 min, +30 min (and +2 h
   when long ohlcv panels exist), in ticks and in volatility-normalised units
   (move / (pre-release 1 s volatility x sqrt(horizon))).
2. First mover: per event, the first time after t0 that each instrument's mid
   moves more than K ticks from its pre-release level, at exact exchange
   timestamps. K per instrument = the `first_move_quantile` of absolute 100 ms
   mid changes on control days, so "a move" means "bigger than almost any
   100 ms move on a normal morning".
3. Lead-lag: correlation of 100 ms mid returns between ZN and each other
   instrument in the first 60 s, at lags of -1 s to +1 s.

Clock caveat: CBOT, CME, COMEX and NYMEX products run on different matching
engines. ts_event is each engine's own timestamp, so millisecond differences
between venues mix real lead-lag with clock and engine differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from printtime import plots
from printtime.config import Settings
from printtime.log import get_logger
from printtime.panel.store import scan, write_manifest
from printtime.stats.bootstrap import curve_ci, scalar_ci

log = get_logger(__name__)
F64 = npt.NDArray[np.float64]
NS = 1_000_000_000
LEAD_LAG_STEPS = 10  # +/- 10 x 100 ms
EVENT_TYPES = ("CPI", "NFP", "FOMC")


def first_move_thresholds(cfg: Settings, fine: pl.DataFrame) -> pl.DataFrame:
    """K per instrument from control days: quantile of |100 ms mid change|, never below one tick."""
    q = cfg.analysis.first_move_quantile
    ctrl = fine.filter(pl.col("event_type") == "CONTROL").sort(
        ["window_id", "stage", "instrument", "offset_ms"]
    )
    if ctrl.height == 0:
        raise ValueError("first-mover thresholds need control days; none in the panels")
    moves = ctrl.with_columns(
        pl.col("mid").diff().over(["window_id", "stage", "instrument"]).abs().alias("abs_move")
    ).filter(pl.col("abs_move").is_not_nan() & pl.col("abs_move").is_not_null())
    return (
        moves.group_by("instrument")
        .agg(pl.col("abs_move").quantile(q).alias("k_ticks"), pl.len().alias("control_steps"))
        .with_columns(pl.max_horizontal(pl.col("k_ticks"), pl.lit(0.5)).alias("k_ticks"))
        .sort("instrument")
    )


def first_moves(moves: pl.DataFrame, k: pl.DataFrame) -> pl.DataFrame:
    """Per event stage and instrument: ms after t0 of the first |mid move| > K (NaN if none)."""
    d = moves.filter((pl.col("event_type") != "CONTROL") & (pl.col("offset_ns") >= 0)).join(
        k, on="instrument"
    )
    hit = d.filter(pl.col("mid_move").abs() > pl.col("k_ticks"))
    first = hit.group_by(["window_id", "event_type", "stage", "instrument"]).agg(
        (pl.col("offset_ns").min() / 1e6).alias("first_move_ms"),
        pl.col("mid_move").sort_by("offset_ns").first().alias("first_move_ticks"),
    )
    keys = d.select("window_id", "event_type", "stage", "instrument").unique()
    return keys.join(first, on=["window_id", "event_type", "stage", "instrument"], how="left").sort(
        ["window_id", "instrument"]
    )


def ranking(first: pl.DataFrame, reps: int, seed: int, reference: str = "ZN") -> pl.DataFrame:
    """Median first-move time per instrument and its median lead over the reference, by event type."""
    rows = []
    for (et, stage), g in first.group_by(["event_type", "stage"], maintain_order=True):
        ref = g.filter(pl.col("instrument") == reference).select(
            "window_id", pl.col("first_move_ms").alias("ref_ms")
        )
        for inst, h in g.group_by("instrument", maintain_order=True):
            v = h["first_move_ms"].to_numpy().astype(np.float64)
            est, lo, hi, n = scalar_ci(v, reps, seed, "median")
            lead = h.join(ref, on="window_id").select(
                (pl.col("first_move_ms") - pl.col("ref_ms")).alias("d")
            )
            dl, dlo, dhi, dn = scalar_ci(
                lead["d"].to_numpy().astype(np.float64), reps, seed, "median"
            )
            rows.append(
                {
                    "event_type": et,
                    "stage": stage,
                    "instrument": inst[0] if isinstance(inst, tuple) else inst,
                    "events": h.height,
                    "events_with_move": n,
                    "median_first_move_ms": est,
                    "lo": lo,
                    "hi": hi,
                    f"median_minus_{reference}_ms": dl,
                    "diff_lo": dlo,
                    "diff_hi": dhi,
                    "paired_events": dn,
                }
            )
    out = pl.DataFrame(rows)
    return out.sort(["event_type", "stage", "median_first_move_ms"], nulls_last=True)


def jump_sizes(cfg: Settings, coarse: pl.DataFrame) -> pl.DataFrame:
    """|mid move| at each horizon, raw (ticks) and normalised by pre-release volatility."""
    max_ms = int(coarse["offset_ms"].to_numpy().max()) if coarse.height else 0
    horizons = [h for h in cfg.analysis.reaction_horizons_s if h * 1000 <= max_ms]
    lo, hi = cfg.panel.baseline_window_s
    vol = (
        coarse.filter(pl.col("offset_ms").is_between(lo * 1000, hi * 1000 - 1000))
        .sort(["window_id", "stage", "instrument", "offset_ms"])
        .with_columns(pl.col("mid").diff().over(["window_id", "stage", "instrument"]).alias("d1"))
        .group_by(["window_id", "stage", "instrument"])
        .agg(pl.col("d1").fill_nan(None).std().alias("vol_1s"))
    )
    at = coarse.filter(pl.col("offset_ms").is_in([h * 1000 for h in horizons])).join(
        vol, on=["window_id", "stage", "instrument"], how="left"
    )
    return at.with_columns(
        (pl.col("offset_ms") // 1000).alias("horizon_s"),
        pl.col("mid_move").abs().alias("abs_move_ticks"),
        (pl.col("mid_move").abs() / (pl.col("vol_1s") * (pl.col("offset_ms") / 1000).sqrt())).alias(
            "abs_move_vol"
        ),
    ).select(
        "window_id",
        "event_type",
        "stage",
        "instrument",
        "horizon_s",
        "mid_move",
        "abs_move_ticks",
        "vol_1s",
        "abs_move_vol",
    )


def jump_summary(jumps: pl.DataFrame) -> pl.DataFrame:
    return (
        jumps.group_by(["event_type", "stage", "instrument", "horizon_s"])
        .agg(
            pl.len().alias("events"),
            pl.col("abs_move_ticks").fill_nan(None).median().alias("median_abs_move_ticks"),
            pl.col("abs_move_vol").fill_nan(None).median().alias("median_abs_move_vol_units"),
        )
        .sort(["event_type", "stage", "instrument", "horizon_s"])
    )


def lead_lag(
    fine: pl.DataFrame, reps: int, seed: int, reference: str = "ZN", window_s: int = 60
) -> pl.DataFrame:
    """Correlation of 100 ms returns of `reference` at t with each instrument at t + lag, averaged across events.

    A peak at a positive lag means the other instrument follows the reference.
    """
    # Returns are computed on the full series *before* cutting to the window,
    # so the return into t0 (which carries a jump at exactly t0) is kept.
    d = (
        fine.filter(pl.col("event_type") != "CONTROL")
        .sort(["window_id", "stage", "instrument", "offset_ms"])
        .with_columns(pl.col("mid").diff().over(["window_id", "stage", "instrument"]).alias("ret"))
        .filter(pl.col("offset_ms").is_between(0, window_s * 1000))
    )
    wide = d.pivot(
        on="instrument",
        index=["window_id", "stage", "event_type", "offset_ms"],
        values="ret",
        aggregate_function="first",
    ).sort(["window_id", "stage", "offset_ms"])
    if reference not in wide.columns:
        return pl.DataFrame()
    others = [
        c
        for c in wide.columns
        if c not in ("window_id", "stage", "event_type", "offset_ms", reference)
    ]
    lags = np.arange(-LEAD_LAG_STEPS, LEAD_LAG_STEPS + 1)
    rows = []
    for et in EVENT_TYPES:
        sub = wide.filter(pl.col("event_type") == et)
        if not sub.height:
            continue
        for inst in others:
            per_event = []
            for _, g in sub.group_by(["window_id", "stage"], maintain_order=True):
                x = g[reference].to_numpy().astype(np.float64)
                y = g[inst].to_numpy().astype(np.float64)
                per_event.append([_lagged_corr(x, y, int(lag)) for lag in lags])
            ci = curve_ci(np.array(per_event), reps, seed)
            for i, lag in enumerate(lags):
                rows.append(
                    {
                        "event_type": et,
                        "instrument": inst,
                        "lag_ms": int(lag) * 100,
                        "corr": ci.mean[i],
                        "lo": ci.lo[i],
                        "hi": ci.hi[i],
                        "events": ci.n_events,
                    }
                )
    return pl.DataFrame(rows)


def _lagged_corr(x: F64, y: F64, lag: int) -> float:
    """corr(x[t], y[t + lag]) over overlapping finite points."""
    if lag > 0:
        a, b = x[:-lag], y[lag:]
    elif lag < 0:
        a, b = x[-lag:], y[:lag]
    else:
        a, b = x, y
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


@dataclass
class ReactionResults:
    thresholds: pl.DataFrame
    first: pl.DataFrame
    ranking: pl.DataFrame
    jumps: pl.DataFrame
    jump_summary: pl.DataFrame
    lead_lag: pl.DataFrame
    headline: list[str]


def headlines(rank: pl.DataFrame, reference: str = "ZN") -> list[str]:
    out = []
    col = f"median_minus_{reference}_ms"
    for (et, stage), g in rank.group_by(["event_type", "stage"], maintain_order=True):
        g = g.filter(pl.col("median_first_move_ms").is_not_nan())
        if g.height < 2:
            continue
        first = g.row(0, named=True)
        second = g.row(1, named=True)
        out.append(
            f"{et} {stage}: {first['instrument']} reacts first (median {first['median_first_move_ms']:.0f} ms after "
            f"t0, {first['events_with_move']} events), {second['median_first_move_ms'] - first['median_first_move_ms']:.0f} ms "
            f"ahead of {second['instrument']}."
        )
        for r in g.iter_rows(named=True):
            if r["instrument"] != reference and not np.isnan(r[col]):
                out.append(
                    f"  {r['instrument']} vs {reference}: median difference {r[col]:+.0f} ms "
                    f"(95% CI {r['diff_lo']:+.0f} to {r['diff_hi']:+.0f}, {r['paired_events']} paired events)"
                )
    return out


def plot_first_movers(rank: pl.DataFrame, out: Path) -> Path | None:
    groups = rank.select("event_type", "stage").unique(maintain_order=True).rows()
    if not groups:
        return None
    fig, axes = plots.plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 3.8), squeeze=False)
    for ax, (et, stage) in zip(axes[0], groups, strict=True):
        g = rank.filter(
            (pl.col("event_type") == et)
            & (pl.col("stage") == stage)
            & pl.col("median_first_move_ms").is_not_nan()
        ).sort("median_first_move_ms")
        for k, r in enumerate(g.iter_rows(named=True)):
            ax.errorbar(
                r["median_first_move_ms"],
                k,
                xerr=[[r["median_first_move_ms"] - r["lo"]], [r["hi"] - r["median_first_move_ms"]]]
                if not np.isnan(r["lo"])
                else None,
                fmt="o",
                color=plots.color(r["instrument"]),
                ms=7,
                capsize=3,
            )
            ax.text(
                r["median_first_move_ms"],
                k + 0.25,
                f"n={r['events_with_move']}",
                fontsize=7,
                ha="center",
                color=plots.TEXT_2,
            )
        ax.set_yticks(range(g.height), g["instrument"].to_list())
        ax.invert_yaxis()
        ax.set_xlabel("ms after release (median, 95% CI)")
        ax.set_title(f"{et} {stage.replace('_', ' ')}")
    fig.suptitle("Which market moves first", fontsize=11)
    fig.tight_layout()
    return plots.save(fig, out)


def run(cfg: Settings) -> ReactionResults:
    reps, seed = cfg.analysis.bootstrap_reps, 20241002
    fine = scan(cfg, "fine").collect()
    coarse = scan(cfg, "coarse").collect()
    moves = scan(cfg, "moves").collect()
    k = first_move_thresholds(cfg, fine)
    first = first_moves(moves, k)
    rank = ranking(first, reps, seed)
    jumps = jump_sizes(cfg, coarse.filter(pl.col("event_type") != "CONTROL"))
    jsum = jump_summary(jumps)
    ll = lead_lag(fine, reps, seed, window_s=cfg.analysis.lead_lag_window_s)
    res = ReactionResults(k, first, rank, jumps, jsum, ll, headlines(rank))
    t = cfg.paths.tables
    t.mkdir(parents=True, exist_ok=True)
    for name, df in (
        ("first_move_thresholds", k),
        ("first_moves", first),
        ("first_mover_ranking", rank),
        ("jumps", jumps),
        ("jump_summary", jsum),
        ("lead_lag", ll),
    ):
        df.write_parquet(t / f"{name}.parquet")
    md = [
        "# Price reaction and first mover",
        "",
        "Clock caveat: venues run separate matching engines; "
        "cross-venue millisecond differences mix real lead-lag with clock differences.",
        "",
    ]
    md += [f"- {h}" for h in res.headline]
    (t / "reaction_summary.md").write_text("\n".join(md) + "\n")
    plot_first_movers(rank, cfg.paths.figures / "first_movers.png")
    write_manifest(cfg, t, "analyze reaction")
    log.info("reaction_done", events=first["window_id"].n_unique())
    return res
