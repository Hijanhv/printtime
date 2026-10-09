"""Sanity plots (spec Phase 3): one event per type, every instrument, around t0.

Three rows: mid move in ticks, top-of-book depth as % of baseline, and spread
in ticks. These are for eyeballing alignment before any statistics: the jump
should sit at t0, depth should dip around it, spreads should blow out and
come back.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from printtime import plots

WINDOW_MS = (-300_000, 900_000)


def sanity_plot(
    coarse: pl.DataFrame, baselines: pl.DataFrame, window_id: str, stage: str, out: Path
) -> Path:
    d = coarse.filter(
        (pl.col("window_id") == window_id)
        & (pl.col("stage") == stage)
        & pl.col("offset_ms").is_between(*WINDOW_MS)
    ).join(baselines, on=["window_id", "event_type", "stage", "instrument"], how="left")
    fig, axes = plots.plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    for inst in sorted(d["instrument"].unique().to_list()):
        g = d.filter(pl.col("instrument") == inst).sort("offset_ms")
        x = g["offset_ms"].to_numpy() / 1000
        c = plots.color(inst)
        axes[0].plot(x, g["mid_move"].to_numpy(), color=c, lw=1.4, label=inst)
        axes[1].plot(
            x, 100 * g["depth_1"].to_numpy() / g["base_depth_1"].to_numpy(), color=c, lw=1.4
        )
        axes[2].plot(x, g["spread"].to_numpy(), color=c, lw=1.4)
    for ax in axes:
        ax.axvline(0, color=plots.TEXT_2, lw=1, ls=(0, (4, 3)))
    axes[0].set_ylabel("mid move (ticks)")
    axes[1].set_ylabel("top-of-book depth\n(% of baseline)")
    axes[1].axhline(100, color=plots.GRID, lw=1)
    axes[2].set_ylabel("spread (ticks)")
    axes[2].set_xlabel("seconds from release")
    event_type = d["event_type"][0] if d.height else ""
    axes[0].set_title(f"{window_id} ({event_type}, {stage}): sanity check")
    axes[0].legend(ncol=5, loc="upper left", fontsize=8)
    fig.tight_layout()
    return plots.save(fig, out)
