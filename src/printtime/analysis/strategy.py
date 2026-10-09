"""Analysis D2: does trading the surprise make money after costs? (spec 8.2)

A deliberately simple rule, tested strictly forward in time:

* at t0 + delay (1, 5, 30, 60 s), trade one contract in the direction implied
  by the sign of the standardised surprise, if |surprise| is above a threshold;
* exit at +5 min, +30 min (or +2 h where the data reach);
* enter and exit by crossing the spread at the book prevailing at that second,
  and pay the fee on both sides.

For each event, everything the rule uses comes from EARLIER events only:
the surprise scale (expanding standard deviation), the sign mapping (sign of
the slope of past returns on past surprises), and the threshold (the
candidate with the best past net P&L). The first `min_train_events` events of
each type are training only and are never traded.

Two comparisons travel with every result:
* a PERFECT-SIGN UPPER BOUND, which knows the realised direction of each move
  (impossible in practice; it shows what the costs leave in the best case);
* a RANDOM-DIRECTION baseline, averaged over many random draws.

FOMC is excluded: its surprise is market-implied from ZT up to +20 minutes after
the statement, which a trade at +1 s cannot know. Using it would be look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from printtime import plots
from printtime.config import Settings
from printtime.log import get_logger
from printtime.panel.store import scan, write_manifest
from printtime.stats.bootstrap import scalar_ci

log = get_logger(__name__)
F64 = npt.NDArray[np.float64]
THRESHOLDS = (0.0, 0.5, 1.0)
EVENT_TYPES = ("CPI", "NFP")
RANDOM_DRAWS = 1000


def raw_surprises(cfg: Settings) -> pl.DataFrame:
    """event_id, event_type, release_date, surprise (actual - consensus, unscaled)."""
    synth = cfg.paths.calendar / "synthetic_surprises.parquet"
    if synth.exists():
        s = pl.read_parquet(synth).filter(pl.col("event_type").is_in(EVENT_TYPES))
        return s.select("event_id", "event_type", "release_date", pl.col("z").alias("surprise"))
    from printtime.calendar.events import load_calendar
    from printtime.calendar.surprises import validate_surprises

    df, rep = validate_surprises(cfg, cfg.paths.calendar / "surprises.csv", load_calendar(cfg))
    rep.raise_if_errors()
    prim = [cfg.primary_variable(et) for et in EVENT_TYPES]
    return df.filter(pl.col("variable").is_in(prim)).select(
        "event_id",
        "event_type",
        "release_date",
        (pl.col("actual") - pl.col("consensus")).round(10).alias("surprise"),
    )


def book_at(coarse: pl.DataFrame, offsets_s: list[int]) -> pl.DataFrame:
    """bid/ask (ticks) at the given offsets, per event and instrument."""
    d = coarse.filter(pl.col("offset_ms").is_in([o * 1000 for o in offsets_s]))
    return d.select(
        "window_id",
        "instrument",
        (pl.col("offset_ms") // 1000).alias("offset_s"),
        "bid_px",
        "ask_px",
        "tick_size",
        "multiplier",
    )


def trade_pnl(direction: F64, entry_bid: F64, entry_ask: F64, exit_bid: F64, exit_ask: F64) -> F64:
    """Gross P&L in ticks for +1 (long) / -1 (short) / 0 (no trade), crossing the spread both ways."""
    long = exit_bid - entry_ask
    short = entry_bid - exit_ask
    out: F64 = np.where(direction > 0, long, np.where(direction < 0, short, 0.0))
    return out


def walk_forward(surprise: F64, ret: F64, min_train: int) -> tuple[F64, F64, F64, F64]:
    """For each event in time order, using only earlier events:
    z (expanding scale), direction (+1/-1/0), the threshold used, and the past slope sign."""
    n = surprise.size
    z = np.full(n, np.nan)
    direction = np.zeros(n)
    thr_used = np.full(n, np.nan)
    sign_used = np.zeros(n)
    for k in range(n):
        past_s = surprise[:k]
        if k >= 2:
            sd = float(np.std(past_s, ddof=1))
            z[k] = surprise[k] / sd if sd > 1e-9 else np.nan
        if k < min_train or not np.isfinite(z[k]):
            continue
        sd = float(np.std(past_s, ddof=1))
        past_z = past_s / sd
        past_r = ret[:k]
        ok = np.isfinite(past_z) & np.isfinite(past_r)
        if ok.sum() < 3 or np.std(past_z[ok]) == 0:
            continue
        slope = float(np.polyfit(past_z[ok], past_r[ok], 1)[0])
        sgn = float(np.sign(slope))
        if sgn == 0:
            continue
        best_thr, best_pnl = None, -np.inf
        for thr in THRESHOLDS:  # chosen on past events only
            dirs = np.where(np.abs(past_z[ok]) > thr, sgn * np.sign(past_z[ok]), 0.0)
            if not np.any(dirs):
                continue
            pnl = float(np.mean((dirs * past_r[ok])[dirs != 0]))
            if pnl > best_pnl + 1e-12:
                best_thr, best_pnl = thr, pnl
        if best_thr is None:
            continue
        thr_used[k], sign_used[k] = best_thr, sgn
        if abs(z[k]) > best_thr:
            direction[k] = sgn * np.sign(z[k])
    return z, direction, thr_used, sign_used


@dataclass
class StrategyResults:
    trades: pl.DataFrame
    summary: pl.DataFrame
    headline: list[str]


def run_backtest(cfg: Settings, coarse: pl.DataFrame, surprises: pl.DataFrame) -> pl.DataFrame:
    st = cfg.analysis.strategy
    fee_usd = cfg.analysis.execution.fee_per_contract_side_usd
    max_s = int(coarse["offset_ms"].to_numpy().max()) // 1000 if coarse.height else 0
    exits = [e for e in st.exits_s if e <= max_s]
    book = book_at(coarse.filter(pl.col("event_type").is_in(EVENT_TYPES)), [*st.delays_s, *exits])
    rows = []
    rng = np.random.default_rng(20241005)
    for et in EVENT_TYPES:
        ev = surprises.filter(pl.col("event_type") == et).sort("release_date")
        if not ev.height:
            continue
        for inst in sorted(book["instrument"].unique().to_list()):
            b = book.filter(pl.col("instrument") == inst)
            for delay in st.delays_s:
                for exit_s in exits:
                    if exit_s <= delay:
                        continue
                    en = b.filter(pl.col("offset_s") == delay).select(
                        "window_id",
                        pl.col("bid_px").alias("eb"),
                        pl.col("ask_px").alias("ea"),
                        "tick_size",
                        "multiplier",
                    )
                    ex = b.filter(pl.col("offset_s") == exit_s).select(
                        "window_id", pl.col("bid_px").alias("xb"), pl.col("ask_px").alias("xa")
                    )
                    d = (
                        ev.join(en, left_on="event_id", right_on="window_id")
                        .join(ex, left_on="event_id", right_on="window_id")
                        .sort("release_date")
                    )
                    if d.height < st.min_train_events + 1:
                        continue
                    eb, ea, xb, xa = (
                        d[c].to_numpy().astype(np.float64) for c in ("eb", "ea", "xb", "xa")
                    )
                    mid_ret = (xb + xa) / 2 - (eb + ea) / 2
                    z, direction, thr, sgn = walk_forward(
                        d["surprise"].to_numpy().astype(np.float64), mid_ret, st.min_train_events
                    )
                    gross = trade_pnl(direction, eb, ea, xb, xa)
                    tick_usd = d["tick_size"].to_numpy() * d["multiplier"].to_numpy()
                    fee_ticks = np.where(tick_usd > 0, 2 * fee_usd / tick_usd, np.nan)
                    eligible = np.arange(d.height) >= st.min_train_events
                    perfect = trade_pnl(np.where(eligible, np.sign(mid_ret), 0.0), eb, ea, xb, xa)
                    rand_dirs = rng.choice([-1.0, 1.0], size=(RANDOM_DRAWS, d.height))
                    rand = np.mean([trade_pnl(r, eb, ea, xb, xa) for r in rand_dirs], axis=0)
                    for k in range(d.height):
                        traded = direction[k] != 0
                        rows.append(
                            {
                                "event_type": et,
                                "instrument": inst,
                                "delay_s": delay,
                                "exit_s": exit_s,
                                "event_id": d["event_id"][k],
                                "release_date": d["release_date"][k],
                                "training_only": k < st.min_train_events,
                                "z": z[k],
                                "threshold": thr[k],
                                "sign_mapping": sgn[k],
                                "direction": direction[k],
                                "gross_ticks": gross[k] if traded else np.nan,
                                "net_ticks": gross[k] - fee_ticks[k] if traded else np.nan,
                                "net_usd": (gross[k] - fee_ticks[k]) * tick_usd[k]
                                if traded
                                else np.nan,
                                "upper_bound_net_ticks": perfect[k] - fee_ticks[k]
                                if eligible[k] and np.sign(mid_ret[k]) != 0
                                else np.nan,
                                "random_net_ticks": rand[k] - fee_ticks[k]
                                if eligible[k]
                                else np.nan,
                            }
                        )
    # "No trade" is a missing value, not a number: store nulls so aggregates skip them.
    return pl.DataFrame(rows).with_columns(pl.col(pl.Float64).fill_nan(None))


def summarise(cfg: Settings, trades: pl.DataFrame) -> pl.DataFrame:
    reps = cfg.analysis.bootstrap_reps
    out = []
    for keys, g in trades.group_by(
        ["event_type", "instrument", "delay_s", "exit_s"], maintain_order=True
    ):
        net = g["net_ticks"].drop_nulls().drop_nans().to_numpy()
        est, lo, hi, n = scalar_ci(net, reps, 7, "mean")
        ub = g["upper_bound_net_ticks"].drop_nulls().drop_nans().to_numpy()
        rnd = g["random_net_ticks"].drop_nulls().drop_nans().to_numpy()
        out.append(
            {
                "event_type": keys[0],
                "instrument": keys[1],
                "delay_s": keys[2],
                "exit_s": keys[3],
                "events": g.height,
                "training_events": int(g["training_only"].sum()),
                "trades": n,
                "mean_net_ticks": est,
                "ci_lo": lo,
                "ci_hi": hi,
                "hit_rate": float(np.mean(net > 0)) if n else float("nan"),
                "worst_net_ticks": float(net.min()) if n else float("nan"),
                "mean_net_usd": float(np.nanmean(g["net_usd"].drop_nulls().to_numpy()))
                if n
                else float("nan"),
                "upper_bound_mean_net_ticks": float(ub.mean()) if ub.size else float("nan"),
                "random_mean_net_ticks": float(rnd.mean()) if rnd.size else float("nan"),
            }
        )
    return (
        pl.DataFrame(out).sort(["event_type", "instrument", "delay_s", "exit_s"])
        if out
        else pl.DataFrame()
    )


def headlines(summary: pl.DataFrame) -> list[str]:
    heads = []
    for et in EVENT_TYPES:
        s = summary.filter((pl.col("event_type") == et) & (pl.col("trades") > 0))
        if not s.height:
            heads.append(
                f"{et}: no out-of-sample trades yet (needs more than the training events)."
            )
            continue
        pos = s.filter(pl.col("ci_lo") > 0)
        best = s.sort("mean_net_ticks", descending=True).row(0, named=True)
        verdict = (
            f"makes money after costs in {pos.height} of {s.height} settings with a 95% interval above zero"
            if pos.height
            else "does not make money after costs in any setting with confidence"
        )
        heads.append(
            f"{et}: the surprise-direction rule {verdict}. Best setting {best['instrument']} enter +{best['delay_s']} s, "
            f"exit +{best['exit_s'] // 60} min: {best['mean_net_ticks']:+.2f} ticks per trade over {best['trades']} trades "
            f"(95% CI {best['ci_lo']:+.2f} to {best['ci_hi']:+.2f}); perfect-sign upper bound "
            f"{best['upper_bound_mean_net_ticks']:+.2f}, random direction {best['random_mean_net_ticks']:+.2f}. "
            "With this few events the result is noisy."
        )
    return heads


def plot_pnl(trades: pl.DataFrame, summary: pl.DataFrame, out) -> object:  # type: ignore[no-untyped-def]
    s = summary.filter(pl.col("trades") > 0)
    if not s.height:
        return None
    fig, axes = plots.plt.subplots(1, len(EVENT_TYPES), figsize=(11, 4), sharey=True, squeeze=False)
    for ax, et in zip(axes[0], EVENT_TYPES, strict=True):
        g = s.filter(pl.col("event_type") == et).sort("mean_net_ticks", descending=True).head(1)
        if not g.height:
            ax.set_title(f"{et}: no trades")
            continue
        r = g.row(0, named=True)
        t = trades.filter(
            (pl.col("event_type") == et)
            & (pl.col("instrument") == r["instrument"])
            & (pl.col("delay_s") == r["delay_s"])
            & (pl.col("exit_s") == r["exit_s"])
            & ~pl.col("training_only")
        ).sort("release_date")
        x = np.arange(t.height)
        ax.plot(
            x,
            np.nancumsum(t["net_ticks"].fill_null(0).to_numpy()),
            color=plots.color("release"),
            lw=2,
            label="strategy",
        )
        ax.plot(
            x,
            np.nancumsum(t["upper_bound_net_ticks"].fill_null(0).to_numpy()),
            color=plots.color("control"),
            lw=1.4,
            ls=(0, (4, 3)),
            label="perfect-sign upper bound",
        )
        ax.plot(
            x,
            np.nancumsum(t["random_net_ticks"].fill_null(0).to_numpy()),
            color=plots.TEXT_2,
            lw=1.2,
            label="random direction (mean)",
        )
        ax.axhline(0, color=plots.GRID, lw=1)
        ax.set_title(
            f"{et}: {r['instrument']} +{r['delay_s']} s to +{r['exit_s'] // 60} min (n={t.height})",
            fontsize=10,
        )
        ax.set_xlabel("out-of-sample events, in time order")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("cumulative net P&L (ticks)")
    fig.suptitle(
        "Surprise-direction strategy after costs (best setting per type; a selection, so optimistic)",
        fontsize=11,
    )
    fig.tight_layout()
    return plots.save(fig, out)


def run(cfg: Settings) -> StrategyResults:
    coarse = scan(cfg, "coarse").collect()
    trades = run_backtest(cfg, coarse, raw_surprises(cfg))
    summary = summarise(cfg, trades)
    heads = headlines(summary)
    t = cfg.paths.tables
    t.mkdir(parents=True, exist_ok=True)
    trades.write_parquet(t / "strategy_trades.parquet")
    summary.write_parquet(t / "strategy_summary.parquet")
    (t / "strategy_summary.md").write_text(
        "# Surprise-direction strategy\n\nStrictly chronological; parameters from earlier events only; "
        "FOMC excluded (its market-implied surprise is only known 20 minutes after the statement).\n\n"
        + "\n".join(f"- {h}" for h in heads)
        + "\n"
    )
    plot_pnl(trades, summary, cfg.paths.figures / "strategy_pnl.png")
    write_manifest(cfg, t, "analyze strategy")
    return StrategyResults(trades, summary, heads)
