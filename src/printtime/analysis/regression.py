"""Analysis C: the reaction function (spec 7).

For each event type, instrument and horizon h:

    return(t0 -> t0 + h) = alpha + beta * standardised surprise + e

* returns in ticks and in basis points of price; for Treasury futures also an
  approximate yield change, dy(bp) ~= -(dP/P) / D * 1e4, with a rough duration;
* HC3 robust errors, plus a by-event bootstrap interval for beta as a check;
* asymmetry: separate betas for positive and negative surprises;
* continuation vs reversal: does the 0 -> 1 min move predict 1 -> 30 min?
* Benjamini-Hochberg FDR across every (event type, instrument, horizon) test;
* sign check against textbook expectations, flagged for investigation.

Surprises: CPI and NFP from your surprises.csv (primary variable, full-sample
standardisation). FOMC: market-implied from the ZT move from -10 min to
+20 min around the statement, standardised and signed so positive = hawkish.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from printtime import plots
from printtime.config import Settings
from printtime.log import get_logger
from printtime.panel.store import scan, write_manifest
from printtime.stats.regression import bh_fdr, bootstrap_beta, ols_hc3

log = get_logger(__name__)
NS = 1_000_000_000
STAGE_FOR = {"CPI": "release", "NFP": "release", "FOMC": "statement"}


def fomc_surprises(cfg: Settings, coarse: pl.DataFrame) -> pl.DataFrame:
    """Market-implied FOMC surprise from the ZT price change around the statement."""
    inst = cfg.regression.fomc_surprise_instrument
    lo, hi = cfg.analysis.fomc_surprise_window_s
    d = coarse.filter(
        (pl.col("event_type") == "FOMC")
        & (pl.col("stage") == "statement")
        & (pl.col("instrument") == inst)
    )
    w = d.filter(pl.col("offset_ms").is_in([lo * 1000, hi * 1000])).pivot(
        on="offset_ms", index="window_id", values="mid", aggregate_function="first"
    )
    if w.height == 0 or str(lo * 1000) not in w.columns or str(hi * 1000) not in w.columns:
        return pl.DataFrame(
            schema={"event_id": pl.String, "event_type": pl.String, "z": pl.Float64}
        )
    move = (w[str(hi * 1000)] - w[str(lo * 1000)]).to_numpy().astype(np.float64)
    sd = np.nanstd(move, ddof=1) if np.isfinite(move).sum() > 1 else np.nan
    z = (
        -move / sd if sd and sd > 0 else np.full(move.size, np.nan)
    )  # ZT price down = hawkish = positive
    return pl.DataFrame(
        {"event_id": w["window_id"], "event_type": "FOMC", "z": z, "zt_move_ticks": move}
    )


def returns_at_horizons(cfg: Settings, coarse: pl.DataFrame) -> pl.DataFrame:
    """Mid move (ticks) from t0 to each horizon, per event and instrument, plus bps and yield conversions."""
    max_ms = int(coarse["offset_ms"].to_numpy().max()) if coarse.height else 0
    horizons = [h for h in cfg.analysis.reaction_horizons_s if h * 1000 <= max_ms]
    pre = coarse.filter(pl.col("offset_ms") == -1000).select(
        "window_id", "stage", "instrument", pl.col("mid").alias("mid_pre")
    )
    d = (
        coarse.filter(
            (pl.col("event_type") != "CONTROL")
            & pl.col("offset_ms").is_in([h * 1000 for h in horizons])
        )
        .join(pre, on=["window_id", "stage", "instrument"], how="left")
        .with_columns((pl.col("offset_ms") // 1000).alias("horizon_s"))
    )
    dur = cfg.regression.approx_modified_duration
    return (
        d.with_columns(
            pl.col("mid_move").alias("ret_ticks"),
            (1e4 * pl.col("mid_move") / pl.col("mid_pre")).alias("ret_bps"),
            pl.col("instrument")
            .replace_strict(dur, default=None, return_dtype=pl.Float64)
            .alias("duration"),
        )
        .with_columns((-pl.col("ret_bps") / pl.col("duration")).alias("approx_dyield_bp"))
        .select(
            "window_id",
            "event_type",
            "stage",
            "instrument",
            "horizon_s",
            "ret_ticks",
            "ret_bps",
            "approx_dyield_bp",
            "tick_size",
        )
    )


@dataclass
class RegressionResults:
    table: pl.DataFrame
    continuation: pl.DataFrame
    surprises: pl.DataFrame
    headline: list[str]


def regressions(
    cfg: Settings, rets: pl.DataFrame, surprises: pl.DataFrame, reps: int, seed: int
) -> pl.DataFrame:
    rows = []
    d = rets.join(
        surprises.select("event_id", "z").rename({"event_id": "window_id"}),
        on="window_id",
        how="inner",
    )
    for (et, stage, inst, h), g in d.group_by(
        ["event_type", "stage", "instrument", "horizon_s"], maintain_order=True
    ):
        if stage != STAGE_FOR.get(str(et)):
            continue
        z = g["z"].to_numpy().astype(np.float64)
        for unit in ("ret_ticks", "ret_bps"):
            y = g[unit].to_numpy().astype(np.float64)
            r = ols_hc3(z, y)
            blo, bhi = bootstrap_beta(z, y, reps, seed)
            pos, neg = z > 0, z < 0
            rp = ols_hc3(np.where(pos, z, 0.0), y) if pos.sum() >= 3 else None
            rn = ols_hc3(np.where(neg, z, 0.0), y) if neg.sum() >= 3 else None
            exp_sign = cfg.regression.expected_sign.get(str(et), {}).get(str(inst), 0)
            rows.append(
                {
                    "event_type": et,
                    "stage": stage,
                    "instrument": inst,
                    "horizon_s": h,
                    "unit": unit,
                    "beta": r.beta,
                    "se_hc3": r.se,
                    "t": r.t,
                    "p": r.p,
                    "r2": r.r2,
                    "n_events": r.n,
                    "boot_lo": blo,
                    "boot_hi": bhi,
                    "beta_pos": rp.beta if rp else float("nan"),
                    "n_pos": int(pos.sum()),
                    "beta_neg": rn.beta if rn else float("nan"),
                    "n_neg": int(neg.sum()),
                    "expected_sign": exp_sign,
                    "mechanical": str(et) == "FOMC"
                    and str(inst) == cfg.regression.fomc_surprise_instrument,
                }
            )
    out = pl.DataFrame(rows)
    if not out.height:
        return out
    # FDR across every non-mechanical test of the headline unit (ticks).
    main = (out["unit"] == "ret_ticks") & ~out["mechanical"]
    reject, adj = bh_fdr(out.filter(main)["p"].to_numpy(), cfg.analysis.fdr_alpha)
    q = np.full(out.height, np.nan)
    surv = np.zeros(out.height, dtype=bool)
    q[np.flatnonzero(main.to_numpy())] = adj
    surv[np.flatnonzero(main.to_numpy())] = reject
    out = out.with_columns(pl.Series("q_bh", q), pl.Series("survives_fdr", surv))
    return out.with_columns(
        pl.when(pl.col("expected_sign") == 0)
        .then(pl.lit("no expectation"))
        .when(pl.col("beta").sign() == pl.col("expected_sign"))
        .then(pl.lit("as expected"))
        .when(pl.col("survives_fdr"))
        .then(pl.lit("WRONG SIGN: investigate"))
        .otherwise(pl.lit("unexpected but not significant"))
        .alias("sign_check")
    )


def continuation(cfg: Settings, coarse: pl.DataFrame) -> pl.DataFrame:
    """Does the first-minute move predict the move from +1 min to +30 min (and +60 min)?"""
    d = coarse.filter(pl.col("event_type") != "CONTROL")
    piv = d.filter(pl.col("offset_ms").is_in([60_000, 1_800_000, 3_600_000])).pivot(
        on="offset_ms",
        index=["window_id", "event_type", "stage", "instrument"],
        values="mid_move",
        aggregate_function="first",
    )
    rows = []
    for (et, stage, inst), g in piv.group_by(
        ["event_type", "stage", "instrument"], maintain_order=True
    ):
        if stage != STAGE_FOR.get(str(et)) or "60000" not in g.columns:
            continue
        first = g["60000"].to_numpy().astype(np.float64)
        for later in ("1800000", "3600000"):
            if later not in g.columns:
                continue
            after = g[later].to_numpy().astype(np.float64) - first
            r = ols_hc3(first, after)
            rows.append(
                {
                    "event_type": et,
                    "instrument": inst,
                    "window": f"+1 min to +{int(later) // 60000} min",
                    "beta": r.beta,
                    "t": r.t,
                    "p": r.p,
                    "n_events": r.n,
                    "reading": "continuation"
                    if r.beta > 0
                    else "reversal"
                    if r.beta < 0
                    else "none",
                }
            )
    schema = {
        "event_type": pl.String,
        "instrument": pl.String,
        "window": pl.String,
        "beta": pl.Float64,
        "t": pl.Float64,
        "p": pl.Float64,
        "n_events": pl.Int64,
        "reading": pl.String,
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def headlines(table: pl.DataFrame) -> list[str]:
    out = []
    t = table.filter((pl.col("unit") == "ret_ticks") & (pl.col("horizon_s") == 60))
    for et in ("CPI", "NFP", "FOMC"):
        parts = []
        for inst in ("ZN", "ES"):
            r = t.filter((pl.col("event_type") == et) & (pl.col("instrument") == inst))
            if r.height:
                x = r.row(0, named=True)
                star = "survives FDR" if x["survives_fdr"] else "does not survive FDR"
                parts.append(
                    f"{inst} by {x['beta']:+.2f} ticks (n={x['n_events']}, R2={x['r2']:.2f}, {star})"
                )
        if parts:
            out.append(
                f"A one-standard-deviation {et} surprise moves "
                + " and ".join(parts)
                + " in the first minute."
            )
    return out


def plot_heatmap(table: pl.DataFrame, event_type: str, out: Path) -> Path | None:
    t = table.filter((pl.col("event_type") == event_type) & (pl.col("unit") == "ret_bps"))
    if not t.height:
        return None
    insts = sorted(t["instrument"].unique().to_list())
    hs = sorted(t["horizon_s"].unique().to_list())
    m = np.full((len(insts), len(hs)), np.nan)
    star = np.zeros_like(m, dtype=bool)
    ticks_t = table.filter((pl.col("event_type") == event_type) & (pl.col("unit") == "ret_ticks"))
    for r in t.iter_rows(named=True):
        m[insts.index(r["instrument"]), hs.index(r["horizon_s"])] = r["beta"]
    for r in ticks_t.iter_rows(named=True):
        star[insts.index(r["instrument"]), hs.index(r["horizon_s"])] = bool(r["survives_fdr"])
    vmax = float(np.nanmax(np.abs(m))) or 1.0
    fig, ax = plots.plt.subplots(figsize=(1.3 * len(hs) + 2.5, 0.55 * len(insts) + 1.8))
    ax.imshow(m, cmap=plots.DIVERGING.reversed(), vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(insts)):
        for j in range(len(hs)):
            if np.isfinite(m[i, j]):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:+.1f}{'*' if star[i, j] else ''}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=plots.TEXT,
                )
    ax.grid(False)
    ax.set_xticks(range(len(hs)), [f"{h}s" if h < 60 else f"{h // 60}m" for h in hs])
    ax.set_yticks(range(len(insts)), insts)
    n = int(t["n_events"].to_numpy().max())
    ax.set_title(
        f"{event_type}: price response to a 1 SD surprise (bp of price; * survives FDR; n={n})",
        fontsize=10,
    )
    return plots.save(fig, out)


def load_surprises(cfg: Settings, coarse: pl.DataFrame) -> pl.DataFrame:
    """Real run: validated surprises.csv (primary variable) plus market-implied FOMC.
    Synthetic run: the generator's known surprises."""
    synth = cfg.paths.calendar / "synthetic_surprises.parquet"
    if synth.exists():
        s = (
            pl.read_parquet(synth)
            .filter(pl.col("event_type") != "FOMC")
            .select("event_id", "event_type", "z")
        )
    else:
        from printtime.calendar.events import load_calendar
        from printtime.calendar.surprises import standardise, validate_surprises

        df, rep = validate_surprises(cfg, cfg.paths.calendar / "surprises.csv", load_calendar(cfg))
        rep.raise_if_errors()
        primaries = [cfg.primary_variable(et) for et in ("CPI", "NFP")]
        s = standardise(df.filter(pl.col("variable").is_in(primaries))).select(
            "event_id", "event_type", "z"
        )
    fomc = fomc_surprises(cfg, coarse).select("event_id", "event_type", "z")
    return pl.concat([s, fomc]).filter(pl.col("z").is_not_null() & pl.col("z").is_not_nan())


def run(cfg: Settings) -> RegressionResults:
    coarse = scan(cfg, "coarse").collect()
    surprises = load_surprises(cfg, coarse)
    rets = returns_at_horizons(cfg, coarse)
    table = regressions(cfg, rets, surprises, cfg.analysis.bootstrap_reps, 20241003)
    cont = continuation(cfg, coarse)
    res = RegressionResults(table, cont, surprises, headlines(table))
    t = cfg.paths.tables
    t.mkdir(parents=True, exist_ok=True)
    table.write_parquet(t / "reaction_regressions.parquet")
    cont.write_parquet(t / "continuation.parquet")
    surprises.write_parquet(t / "surprises_used.parquet")
    wrong = table.filter(pl.col("sign_check") == "WRONG SIGN: investigate")
    md = [
        "# Reaction-function regressions",
        "",
        f"{table.filter(pl.col('unit') == 'ret_ticks').height} tests; "
        f"{int(table['survives_fdr'].sum())} survive Benjamini-Hochberg FDR at {cfg.analysis.fdr_alpha}.",
        f"Sign check: {wrong.height} significant result(s) with an unexpected sign.",
        "",
    ]
    md += [f"- {h}" for h in res.headline]
    if wrong.height:
        md += ["", "Investigate before reporting:"] + [
            f"- {r['event_type']} {r['instrument']} {r['horizon_s']}s beta {r['beta']:+.2f}"
            for r in wrong.iter_rows(named=True)
        ]
    (t / "regression_summary.md").write_text("\n".join(md) + "\n")
    for et in ("CPI", "NFP", "FOMC"):
        plot_heatmap(table, et, cfg.paths.figures / f"reaction_heatmap_{et.lower()}.png")
    write_manifest(cfg, t, "analyze regression")
    log.info("regression_done", tests=table.height, wrong_signs=wrong.height)
    return res
