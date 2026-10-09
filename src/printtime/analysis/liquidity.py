"""Analysis A: liquidity around releases, always against control days (spec 5).

Per group (CPI release, NFP release, FOMC statement, FOMC press conference)
and instrument, compared with control windows at the same clock time:

1. depth withdrawal: top-of-book depth as % of baseline over event time, and
   when the withdrawal starts (change point per event);
2. spreads: share of events with a spread wider than one tick at each second,
   and the widest spread in the first minute;
3. activity: book updates and trades per second;
4. recovery: seconds after t0 until depth is back to 50% and 90% of baseline,
   and until the spread is back to its baseline;
5. every summary carries its number of events and a by-event bootstrap interval.
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
from printtime.panel.store import read_baselines, scan, write_manifest
from printtime.stats.bootstrap import curve_ci, scalar_ci
from printtime.stats.changepoint import rolling_median, withdrawal_start

log = get_logger(__name__)
F64 = npt.NDArray[np.float64]
CURVE_RANGE_S = (-300, 900)
SMOOTH_S = 5
HOLD_S = 10  # seconds a recovered level must be held


def group_label(event_type: str, stage: str, time_et: str) -> str:
    if event_type == "CONTROL":
        return f"CONTROL {time_et}"
    return f"{event_type} {stage.replace('_', ' ')}"


def load(cfg: Settings) -> pl.DataFrame:
    """Coarse panel with baselines joined and depth expressed as % of baseline."""
    base = read_baselines(cfg)
    lo, hi = cfg.panel.coarse_grid.start_s * 1000, cfg.panel.coarse_grid.end_s * 1000
    df = (
        scan(cfg, "coarse")
        .filter(pl.col("offset_ms").is_between(lo, hi))
        .select(
            "window_id",
            "event_type",
            "stage",
            "time_et",
            "instrument",
            "offset_ms",
            "depth_1",
            "spread",
            "updates",
            "trades",
        )
        .collect()
        .join(
            base.select(
                "window_id", "event_type", "stage", "instrument", "base_depth_1", "base_spread"
            ),
            on=["window_id", "event_type", "stage", "instrument"],
            how="left",
        )
    )
    return df.with_columns(
        (100 * pl.col("depth_1") / pl.col("base_depth_1")).alias("depth_pct"),
        pl.struct("event_type", "stage", "time_et")
        .map_elements(
            lambda r: group_label(r["event_type"], r["stage"], r["time_et"]), return_dtype=pl.String
        )
        .alias("group"),
    )


def control_for(group: str) -> str | None:
    """The control group at the same clock time as an event group."""
    times = {
        "CPI release": "08:30",
        "NFP release": "08:30",
        "FOMC statement": "14:00",
        "FOMC press conference": "14:30",
    }
    t = times.get(group)
    return f"CONTROL {t}" if t else None


def matrix(df: pl.DataFrame, value: str) -> tuple[F64, F64, list[str]]:
    """events x seconds matrix for one group and instrument: (offsets_s, matrix, window_ids)."""
    w = df.pivot(on="offset_ms", index="window_id", values=value, aggregate_function="first").sort(
        "window_id"
    )
    ids = w["window_id"].to_list()
    cols = [c for c in w.columns if c != "window_id"]
    offs = np.array([int(c) for c in cols], dtype=np.float64) / 1000
    order = np.argsort(offs)
    m = w.select(cols).to_numpy().astype(np.float64)[:, order]
    return offs[order], m, ids


@dataclass
class EventMetrics:
    window_id: str
    group: str
    instrument: str
    withdrawal_start_s: float
    min_depth_pct: float
    max_spread_first_min: float
    recovery_50_s: float
    recovery_90_s: float
    spread_recovery_s: float


def _first_at_or_after(x: F64, cond: npt.NDArray[np.bool_], start: float = 0.0) -> float:
    idx = np.flatnonzero(cond & (x >= start))
    return float(x[idx[0]]) if idx.size else float("nan")


def event_metrics(
    offs: F64, depth_pct: F64, spread: F64, base_spread: float, levels: list[float]
) -> dict[str, float]:
    smooth = rolling_median(depth_pct, SMOOTH_S)
    around = (offs >= -10) & (offs <= 10)
    first_min = (offs >= 0) & (offs <= 60)
    out = {
        "withdrawal_start_s": withdrawal_start(offs, depth_pct),
        "min_depth_pct": float(np.nanmin(smooth[around]))
        if np.any(~np.isnan(smooth[around]))
        else float("nan"),
        "max_spread_first_min": float(np.nanmax(spread[first_min]))
        if np.any(first_min)
        else float("nan"),
    }
    # "Recovered" means the level is reached and *held* for HOLD_S seconds: depth is
    # noisy, and a single brief touch of 90% is not a recovery.
    for lvl in levels:
        out[f"recovery_{round(lvl * 100)}_s"] = _first_at_or_after(
            offs, _held(smooth >= 100 * lvl, HOLD_S)
        )
    ok = np.nan_to_num(spread, nan=np.inf) <= base_spread + 1e-9
    out["spread_recovery_s"] = _first_at_or_after(offs, _held(ok, HOLD_S))
    return out


def _held(cond: npt.NDArray[np.bool_], n: int) -> npt.NDArray[np.bool_]:
    """True at i when cond holds at i and the following n - 1 points."""
    run = np.convolve(cond.astype(np.int64), np.ones(n, dtype=np.int64), mode="full")[n - 1 :]
    out: npt.NDArray[np.bool_] = run >= n
    return out


@dataclass
class LiquidityResults:
    curves: pl.DataFrame
    events: pl.DataFrame
    summary: pl.DataFrame
    headline: list[str]


def analyse(cfg: Settings, df: pl.DataFrame) -> LiquidityResults:
    reps, seed = cfg.analysis.bootstrap_reps, 20241001
    levels = cfg.analysis.recovery_levels
    curve_rows, event_rows = [], []
    for (group, inst), g in df.group_by(["group", "instrument"], maintain_order=True):
        offs, depth, ids = matrix(g, "depth_pct")
        _, spread, _ = matrix(g, "spread")
        _, upd, _ = matrix(g, "updates")
        _, trd, _ = matrix(g, "trades")
        base_spread = (
            g.group_by("window_id")
            .agg(pl.col("base_spread").first())
            .sort("window_id")["base_spread"]
            .to_numpy()
        )
        sel = (offs >= CURVE_RANGE_S[0]) & (offs <= CURVE_RANGE_S[1])
        for name, m in (
            ("depth_pct", depth),
            ("wide_spread", (spread > 1).astype(np.float64)),
            ("updates_per_s", upd),
            ("trades_per_s", trd),
        ):
            ci = curve_ci(m[:, sel], reps, seed)
            curve_rows.append(
                pl.DataFrame(
                    {
                        "group": group,
                        "instrument": inst,
                        "measure": name,
                        "offset_s": offs[sel],
                        "mean": ci.mean,
                        "lo": ci.lo,
                        "hi": ci.hi,
                        "n_events": ci.n_events,
                    }
                )
            )
        for k, wid in enumerate(ids):
            met = event_metrics(offs, depth[k], spread[k], float(base_spread[k]), levels)
            event_rows.append({"window_id": wid, "group": group, "instrument": inst, **met})
    curves = pl.concat(curve_rows) if curve_rows else pl.DataFrame()
    events = pl.DataFrame(event_rows)
    summary = summarise(events, reps, seed)
    return LiquidityResults(curves, events, summary, headlines(summary))


def summarise(events: pl.DataFrame, reps: int, seed: int) -> pl.DataFrame:
    metrics = [
        "withdrawal_start_s",
        "min_depth_pct",
        "max_spread_first_min",
        "recovery_50_s",
        "recovery_90_s",
        "spread_recovery_s",
    ]
    rows = []
    for (group, inst), g in events.group_by(["group", "instrument"], maintain_order=True):
        row: dict[str, object] = {"group": group, "instrument": inst, "n_events": g.height}
        for m in metrics:
            v = g[m].to_numpy().astype(np.float64)
            est, lo, hi, n = scalar_ci(v, reps, seed, "median")
            row[f"{m}_median"], row[f"{m}_lo"], row[f"{m}_hi"], row[f"{m}_n"] = est, lo, hi, n
        rows.append(row)
    return pl.DataFrame(rows).sort(["group", "instrument"])


def headlines(summary: pl.DataFrame, instrument: str = "ZN") -> list[str]:
    """Spec headline 1, for every release group with data, with event counts and the control comparison."""
    out = []
    for r in summary.filter(pl.col("instrument") == instrument).iter_rows(named=True):
        group = str(r["group"])
        ctrl = control_for(group)
        if ctrl is None:
            continue
        c = summary.filter((pl.col("group") == ctrl) & (pl.col("instrument") == instrument))
        drop = 100 - r["min_depth_pct_median"]
        ctrl_drop = 100 - c["min_depth_pct_median"][0] if c.height else float("nan")
        start = r["withdrawal_start_s_median"]
        start_txt = (
            (
                f"starting about {abs(start):.0f} s before the release (median over "
                f"{r['withdrawal_start_s_n']} events with a detectable start)"
            )
            if not np.isnan(start)
            else "with no detectable start before the release"
        )
        out.append(
            f"{instrument}, {group}: top-of-book depth falls by {drop:.0f}% (median over {r['n_events']} events; "
            f"{ctrl_drop:.0f}% on {c['n_events'][0] if c.height else 0} control days), {start_txt}, "
            f"and takes {r['recovery_50_s_median']:.0f} s to recover to half its normal level "
            f"(95% CI {r['recovery_50_s_lo']:.0f} to {r['recovery_50_s_hi']:.0f} s)."
        )
    return out


def plot_depth(
    cfg: Settings,
    curves: pl.DataFrame,
    group: str,
    out: Path,
    zoom_s: tuple[int, int] = (-120, 300),
) -> Path | None:
    ctrl = control_for(group)
    d = curves.filter(pl.col("measure") == "depth_pct")
    insts = sorted(d.filter(pl.col("group") == group)["instrument"].unique().to_list())
    if not insts:
        return None
    cols = min(3, len(insts))
    rows = int(np.ceil(len(insts) / cols))
    fig, axes = plots.plt.subplots(
        rows, cols, figsize=(4.2 * cols, 3.0 * rows), sharex=True, sharey=True, squeeze=False
    )
    for k, inst in enumerate(insts):
        ax = axes[k // cols][k % cols]
        for g, key in ((group, "release"), (ctrl, "control")):
            s = d.filter(
                (pl.col("group") == g)
                & (pl.col("instrument") == inst)
                & pl.col("offset_s").is_between(*zoom_s)
            ).sort("offset_s")
            if not s.height:
                continue
            x = s["offset_s"].to_numpy()
            ax.fill_between(
                x, s["lo"].to_numpy(), s["hi"].to_numpy(), color=plots.color(key), alpha=0.18, lw=0
            )
            ax.plot(
                x,
                s["mean"].to_numpy(),
                color=plots.color(key),
                lw=1.8,
                label=f"{'release' if key == 'release' else 'control'} (n={s['n_events'][0]})",
            )
        ax.axvline(0, color=plots.TEXT_2, lw=1, ls=(0, (4, 3)))
        ax.axhline(100, color=plots.GRID, lw=1)
        ax.set_title(inst, fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    for k in range(len(insts), rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    fig.supxlabel("seconds from release", fontsize=10, color=plots.TEXT_2)
    fig.supylabel("top-of-book depth, % of baseline", fontsize=10, color=plots.TEXT_2)
    fig.suptitle(
        f"{group}: depth around the release vs control days (95% by-event bootstrap bands)",
        fontsize=11,
    )
    fig.tight_layout()
    return plots.save(fig, out)


def plot_spreads(
    cfg: Settings, curves: pl.DataFrame, group: str, out: Path, zoom_s: tuple[int, int] = (-30, 120)
) -> Path | None:
    d = curves.filter(
        (pl.col("measure") == "wide_spread")
        & (pl.col("group") == group)
        & pl.col("offset_s").is_between(*zoom_s)
    )
    if not d.height:
        return None
    fig, ax = plots.plt.subplots(figsize=(8, 4))
    for inst in sorted(d["instrument"].unique().to_list()):
        s = d.filter(pl.col("instrument") == inst).sort("offset_s")
        ax.plot(
            s["offset_s"].to_numpy(),
            100 * s["mean"].to_numpy(),
            color=plots.color(inst),
            lw=1.6,
            label=f"{inst} (n={s['n_events'][0]})",
        )
    ax.axvline(0, color=plots.TEXT_2, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("seconds from release")
    ax.set_ylabel("events with spread > 1 tick (%)")
    ax.set_title(f"{group}: how often the spread is wider than one tick")
    ax.legend(fontsize=8, ncol=3)
    return plots.save(fig, out)


def markdown_table(summary: pl.DataFrame) -> str:
    lines = [
        "| group | instrument | events | withdrawal starts (s) | depth at t0 (% of base) | widest spread, 1st min | recovery to 50% (s) | to 90% (s) | spread back (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0f}"  # noqa: E731
    for r in summary.iter_rows(named=True):
        lines.append(
            f"| {r['group']} | {r['instrument']} | {r['n_events']} | {f(r['withdrawal_start_s_median'])} "
            f"(n={r['withdrawal_start_s_n']}) | {f(r['min_depth_pct_median'])} | {f(r['max_spread_first_min_median'])} "
            f"| {f(r['recovery_50_s_median'])} [{f(r['recovery_50_s_lo'])}, {f(r['recovery_50_s_hi'])}] "
            f"| {f(r['recovery_90_s_median'])} | {f(r['spread_recovery_s_median'])} |"
        )
    return (
        "\n".join(lines)
        + "\n\nMedians across events; brackets are 95% by-event bootstrap intervals.\n"
    )


def run(cfg: Settings) -> LiquidityResults:
    res = analyse(cfg, load(cfg))
    t, fdir = cfg.paths.tables, cfg.paths.figures
    t.mkdir(parents=True, exist_ok=True)
    res.curves.write_parquet(t / "liquidity_curves.parquet")
    res.events.write_parquet(t / "liquidity_events.parquet")
    res.summary.write_parquet(t / "liquidity_summary.parquet")
    (t / "liquidity_summary.md").write_text(
        markdown_table(res.summary) + "\n" + "\n".join(f"- {h}" for h in res.headline) + "\n"
    )
    for group in res.summary["group"].unique().to_list():
        if str(group).startswith("CONTROL"):
            continue
        slug = str(group).lower().replace(" ", "_")
        plot_depth(cfg, res.curves, str(group), fdir / f"liquidity_depth_{slug}.png")
        plot_spreads(cfg, res.curves, str(group), fdir / f"liquidity_spread_{slug}.png")
    write_manifest(cfg, t, "analyze liquidity", {"groups": res.summary["group"].unique().to_list()})
    log.info("liquidity_done", groups=res.summary["group"].n_unique(), events=res.events.height)
    return res
