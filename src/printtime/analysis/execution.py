"""Analysis D1: what does it cost to trade N contracts around a release? (spec 8.1)

For ZT, ZN and ES (the instruments with 10-level data), at every second from
-5 min to +15 min, the cost of an immediate market order of Q = 1, 5, 10, 25
contracts is found by walking the actual book at that second:

    cost (ticks per contract) = volume-weighted fill price - mid   (buy side)
                                mid - volume-weighted fill price   (sell side)

reported as the average of buying and selling, plus the fee in ticks when the
contract multiplier is known. If Q is larger than the displayed 10 levels, the
cost is left undefined and the case is counted, never extrapolated.

Headline: how many seconds after t0 until the cost is back within 1.25x of
normal, where normal is the median cost on control days at the same clock time.
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
from printtime.panel.build import Book, FrameProvider, Stage
from printtime.panel.grid import last_index, offsets_ns, sample
from printtime.panel.providers import provider_for, stages_from
from printtime.panel.store import write_manifest
from printtime.schema import NO_PRICE
from printtime.stats.bootstrap import curve_ci

log = get_logger(__name__)
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
NS = 1_000_000_000
HOLD_S = 10


def walk_book(prices: F64, sizes: F64, qty: int) -> tuple[F64, npt.NDArray[np.bool_]]:
    """Volume-weighted price to fill `qty` from levels (rows x levels), best level first.

    Returns (vwap, insufficient) where insufficient marks rows whose displayed depth
    is smaller than qty; their vwap is NaN.
    """
    sz = np.where(np.isfinite(prices) & np.isfinite(sizes), np.maximum(sizes, 0.0), 0.0)
    cum = np.cumsum(sz, axis=1)
    prev = np.concatenate([np.zeros((sz.shape[0], 1)), cum[:, :-1]], axis=1)
    take = np.clip(qty - prev, 0.0, sz)
    filled = take.sum(axis=1)
    px = np.where(take > 0, prices, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = (take * px).sum(axis=1) / filled
    insufficient = filled < qty - 1e-9
    out: F64 = np.where(insufficient, np.nan, vwap)
    return out, insufficient


def costs_on_grid(
    book: Book, t0_ns: int, window_s: tuple[int, int], sizes: list[int]
) -> pl.DataFrame:
    """Per second and order size: cost in ticks (buy, sell, average) and whether depth ran out."""
    f = book.frame
    levels = sum(1 for c in f.columns if c.startswith("bid_px_"))
    grid = t0_ns + offsets_ns(window_s[0], window_s[1], 1000)
    idx = last_index(f["ts_event"].to_numpy(), grid)

    def side(prefix: str) -> tuple[F64, F64]:
        px = np.column_stack(
            [sample(_clean(f[f"{prefix}_px_{i}"].to_numpy()), idx) for i in range(levels)]
        )
        sz = np.column_stack([sample(f[f"{prefix}_sz_{i}"].to_numpy(), idx) for i in range(levels)])
        return px, sz

    bpx, bsz = side("bid")
    apx, asz = side("ask")
    mid = (bpx[:, 0] + apx[:, 0]) / 2
    rows = []
    offs = (np.arange(grid.size) + window_s[0]).astype(np.int64)
    for q in sizes:
        buy, short_a = walk_book(apx, asz, q)
        sell, short_b = walk_book(bpx, bsz, q)
        buy_cost, sell_cost = buy - mid, mid - sell
        rows.append(
            pl.DataFrame(
                {
                    "offset_s": offs,
                    "qty": q,
                    "buy_ticks": buy_cost,
                    "sell_ticks": sell_cost,
                    "cost_ticks": (buy_cost + sell_cost) / 2,
                    "insufficient_depth": short_a | short_b,
                }
            )
        )
    return pl.concat(rows)


def _clean(px: I64) -> F64:
    out: F64 = np.where(px == NO_PRICE, np.nan, px.astype(np.float64))
    return out


@dataclass
class ExecutionResults:
    per_event: pl.DataFrame
    curves: pl.DataFrame
    summary: pl.DataFrame
    headline: list[str]


def group_of(event_type: str, stage: str, time_et: str) -> str:
    return (
        f"CONTROL {time_et}"
        if event_type == "CONTROL"
        else f"{event_type} {stage.replace('_', ' ')}"
    )


CONTROL_FOR = {
    "CPI release": "CONTROL 08:30",
    "NFP release": "CONTROL 08:30",
    "FOMC statement": "CONTROL 14:00",
    "FOMC press conference": "CONTROL 14:30",
}


def per_event_costs(
    cfg: Settings, provider: FrameProvider, stages: list[Stage], instruments: list[str]
) -> pl.DataFrame:
    window = cfg.databento.windows.mbp_10
    fee = cfg.analysis.execution.fee_per_contract_side_usd
    parts = []
    for st in stages:
        for inst in instruments:
            book = provider.book(st.window_id, inst)
            if book is None or book.frame.height == 0:
                continue
            c = costs_on_grid(book, st.t0_ns, window, cfg.analysis.execution.order_sizes)
            tick_usd = book.tick_size * book.multiplier
            parts.append(
                c.with_columns(
                    pl.lit(st.window_id).alias("window_id"),
                    pl.lit(group_of(st.event_type, st.stage, st.time_et)).alias("group"),
                    pl.lit(inst).alias("instrument"),
                    pl.lit(
                        fee / tick_usd if np.isfinite(tick_usd) and tick_usd > 0 else None
                    ).alias("fee_ticks"),
                    pl.lit(tick_usd if np.isfinite(tick_usd) else None).alias("tick_usd"),
                )
            )
    return pl.concat(parts) if parts else pl.DataFrame()


def curves_and_summary(
    cfg: Settings, per_event: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    reps = cfg.analysis.bootstrap_reps
    mult = cfg.analysis.execution.normal_cost_multiple
    curve_rows, summ_rows, heads = [], [], []
    for (group, inst, q), g in per_event.group_by(
        ["group", "instrument", "qty"], maintain_order=True
    ):
        w = g.pivot(
            on="offset_s", index="window_id", values="cost_ticks", aggregate_function="first"
        )
        cols = sorted((c for c in w.columns if c != "window_id"), key=int)
        ci = curve_ci(w.select(cols).to_numpy().astype(np.float64), reps, 20241004)
        offs = np.array([int(c) for c in cols])
        curve_rows.append(
            pl.DataFrame(
                {
                    "group": group,
                    "instrument": inst,
                    "qty": q,
                    "offset_s": offs,
                    "mean_ticks": ci.mean,
                    "lo": ci.lo,
                    "hi": ci.hi,
                    "n_events": ci.n_events,
                    "share_insufficient": g.group_by("offset_s")
                    .agg(pl.col("insufficient_depth").mean())
                    .sort("offset_s")["insufficient_depth"],
                }
            )
        )
    curves = pl.concat(curve_rows) if curve_rows else pl.DataFrame()
    for (group, inst, q), g in curves.group_by(["group", "instrument", "qty"], maintain_order=True):
        ctrl_name = CONTROL_FOR.get(str(group))
        if ctrl_name is None:
            continue
        ctrl = curves.filter(
            (pl.col("group") == ctrl_name) & (pl.col("instrument") == inst) & (pl.col("qty") == q)
        )
        if not ctrl.height:
            continue
        normal = float(np.nanmedian(ctrl["mean_ticks"].to_numpy()))
        x = g.sort("offset_s")
        offs = x["offset_s"].to_numpy()
        m = x["mean_ticks"].to_numpy()
        ok = np.nan_to_num(m, nan=np.inf) <= mult * normal
        run = np.convolve(ok.astype(np.int64), np.ones(HOLD_S, dtype=np.int64), mode="full")[
            HOLD_S - 1 :
        ]
        back = np.flatnonzero((run >= HOLD_S) & (offs >= 0))
        first5 = float(np.nanmean(m[(offs >= 0) & (offs < 5)]))
        summ_rows.append(
            {
                "group": group,
                "instrument": inst,
                "qty": q,
                "n_events": int(x["n_events"][0]),
                "n_control": int(ctrl["n_events"][0]),
                "normal_ticks": normal,
                "first_5s_ticks": first5,
                "extra_first_5s_ticks": first5 - normal,
                "seconds_to_normal": float(offs[back[0]]) if back.size else float("nan"),
                "max_share_insufficient": float(np.nanmax(x["share_insufficient"].to_numpy())),
            }
        )
    summary = (
        pl.DataFrame(summ_rows).sort(["group", "instrument", "qty"])
        if summ_rows
        else pl.DataFrame()
    )
    for r in summary.filter((pl.col("instrument") == "ZN") & (pl.col("qty") == 10)).iter_rows(
        named=True
    ):
        heads.append(
            f"{r['group']}: executing 10 ZN contracts in the first 5 seconds costs {r['extra_first_5s_ticks']:+.2f} "
            f"ticks per contract more than normal ({r['normal_ticks']:.2f} ticks on {r['n_control']} control days; "
            f"{r['n_events']} events); costs are back within {mult:g}x of normal after {r['seconds_to_normal']:.0f} s."
        )
    return curves, summary, heads


def plot_costs(
    curves: pl.DataFrame, group: str, out: Path, zoom_s: tuple[int, int] = (-60, 300)
) -> Path | None:
    ctrl = CONTROL_FOR.get(group)
    d = curves.filter(pl.col("offset_s").is_between(*zoom_s))
    insts = sorted(d.filter(pl.col("group") == group)["instrument"].unique().to_list())
    if not insts:
        return None
    sizes = sorted(d["qty"].unique().to_list())
    fig, axes = plots.plt.subplots(
        len(insts),
        len(sizes),
        figsize=(3.2 * len(sizes), 2.6 * len(insts)),
        sharex=True,
        squeeze=False,
    )
    for i, inst in enumerate(insts):
        for j, q in enumerate(sizes):
            ax = axes[i][j]
            for g, key in ((group, "release"), (ctrl, "control")):
                s = d.filter(
                    (pl.col("group") == g) & (pl.col("instrument") == inst) & (pl.col("qty") == q)
                ).sort("offset_s")
                if s.height:
                    x = s["offset_s"].to_numpy()
                    ax.fill_between(
                        x,
                        s["lo"].to_numpy(),
                        s["hi"].to_numpy(),
                        color=plots.color(key),
                        alpha=0.18,
                        lw=0,
                    )
                    ax.plot(
                        x,
                        s["mean_ticks"].to_numpy(),
                        color=plots.color(key),
                        lw=1.6,
                        label=f"{key} (n={s['n_events'][0]})",
                    )
            ax.axvline(0, color=plots.TEXT_2, lw=1, ls=(0, (4, 3)))
            ax.set_title(f"{inst}, {q} lot{'s' if q > 1 else ''}", fontsize=9)
            if j == 0:
                ax.set_ylabel("cost, ticks/contract")
            ax.legend(fontsize=6)
    fig.supxlabel("seconds from release", fontsize=10, color=plots.TEXT_2)
    fig.suptitle(
        f"{group}: cost of an immediate market order (walking the 10-level book), excl. fees",
        fontsize=11,
    )
    fig.tight_layout()
    return plots.save(fig, out)


def run(cfg: Settings) -> ExecutionResults:
    provider = provider_for(cfg, "mbp-10")
    instruments = [k for k, v in cfg.instruments.items() if v.mbp10]
    if hasattr(provider, "scenarios"):
        from printtime.panel.pipeline import stages_for_study
        from printtime.synthetic.generator import SyntheticStudy

        stages = stages_for_study(cfg, SyntheticStudy(provider.scenarios, pl.DataFrame()))
        instruments = [i for i in instruments if i in cfg.synthetic.instruments]
    else:
        from printtime.calendar.events import load_calendar

        ctrl_path = cfg.paths.calendar / "control_days.csv"
        ctrl = pl.read_csv(ctrl_path, try_parse_dates=True) if ctrl_path.exists() else None
        stages = stages_from(load_calendar(cfg), ctrl, cfg.tier(1))
    per_event = per_event_costs(cfg, provider, stages, instruments)
    curves, summary, heads = curves_and_summary(cfg, per_event)
    t = cfg.paths.tables
    t.mkdir(parents=True, exist_ok=True)
    per_event.write_parquet(t / "execution_per_event.parquet")
    curves.write_parquet(t / "execution_curves.parquet")
    summary.write_parquet(t / "execution_summary.parquet")
    (t / "execution_summary.md").write_text(
        "# Execution cost\n\n" + "\n".join(f"- {h}" for h in heads) + "\n"
    )
    for group in CONTROL_FOR:
        plot_costs(
            curves, group, cfg.paths.figures / f"execution_{group.lower().replace(' ', '_')}.png"
        )
    write_manifest(cfg, t, "analyze execution")
    return ExecutionResults(per_event, curves, summary, heads)
