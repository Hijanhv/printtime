"""`panel build`: per-stage panels on disk, baselines, release validation, sanity plots.

The same function runs on real data (DatabentoProvider) and on a synthetic
study (SyntheticProvider). Synthetic runs write under data/synthetic_run/
and never into reports/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from printtime.config import Settings
from printtime.data.validation import spike_ratio, validate_events
from printtime.log import get_logger
from printtime.panel.build import FrameProvider, Stage, build_panels
from printtime.panel.sanity import sanity_plot
from printtime.panel.store import panel_dir, write_manifest, write_stage
from printtime.synthetic.generator import SyntheticStudy

log = get_logger(__name__)

STAGE_FOR = {"CPI": "release", "NFP": "release", "FOMC": "statement"}


def synthetic_settings(cfg: Settings, root: Path = Path("data/synthetic_run")) -> Settings:
    """Settings with every output under `root`: synthetic runs never touch real outputs."""
    p = cfg.paths
    paths = p.model_copy(
        update={
            "processed": root / "processed",
            "calendar": root / "calendar",
            "reports": root / "reports",
            "figures": root / "reports" / "figures",
            "tables": root / "reports" / "tables",
            "data_quality": root / "reports" / "data_quality",
        }
    )
    return cfg.model_copy(update={"paths": paths})


def stages_for_study(cfg: Settings, study: SyntheticStudy) -> list[Stage]:
    from printtime.panel.providers import control_stages

    out = []
    for wid, s in study.scenarios.items():
        if s.control:
            out += control_stages(wid, s.time_et, s.t0_ns(cfg.timezone))
        else:
            out.append(
                Stage(wid, s.event_type, STAGE_FOR[s.event_type], s.time_et, s.t0_ns(cfg.timezone))
            )
    return sorted(out, key=lambda x: (x.t0_ns, x.window_id))


@dataclass
class PanelBuildResult:
    stages_built: int
    baselines: pl.DataFrame
    validation: pl.DataFrame
    sanity_plots: list[Path]


def run_panel_build(
    cfg: Settings, stages: list[Stage], provider: FrameProvider, instruments: list[str]
) -> PanelBuildResult:
    for grid in ("fine", "coarse", "moves"):
        for old in panel_dir(cfg, grid).glob("*.parquet"):
            old.unlink()
    base_parts = []
    ratio_rows = []
    built = 0
    for st in stages:
        panels = build_panels(cfg, [st], provider, instruments)
        if panels.fine.height == 0:
            log.warning("stage_without_data", window=st.window_id, stage=st.stage)
            continue
        write_stage(cfg, "fine", st.window_id, st.stage, panels.fine)
        write_stage(cfg, "coarse", st.window_id, st.stage, panels.coarse)
        write_stage(cfg, "moves", st.window_id, st.stage, panels.moves)
        base_parts.append(panels.baselines)
        built += 1
        if st.event_type != "CONTROL":
            for inst in cfg.validation.reference_instruments:
                book = provider.book(st.window_id, inst)
                if book is not None and book.frame.height:
                    ratio_rows.append(
                        {
                            "event_id": st.window_id,
                            "stage": st.stage,
                            "instrument": inst,
                            "ratio": spike_ratio(cfg, book.frame, st.t0_ns),
                        }
                    )
    baselines = pl.concat(base_parts) if base_parts else pl.DataFrame()
    (cfg.paths.processed / "panels").mkdir(parents=True, exist_ok=True)
    if baselines.height:
        baselines.write_parquet(cfg.paths.processed / "panels" / "baselines.parquet")
    ratios = pl.DataFrame(ratio_rows)
    validation = validate_events(cfg, ratios) if ratios.height else pl.DataFrame()
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    if validation.height:
        validation.write_csv(cfg.paths.tables / "event_validation.csv")
        unconfirmed = validation.filter(~pl.col("confirmed"))
        if unconfirmed.height:
            log.warning("events_without_volatility_spike", events=unconfirmed["event_id"].to_list())

    plots = []
    for event_type in ("CPI", "NFP", "FOMC"):
        first = next((s for s in stages if s.event_type == event_type), None)
        if (
            first is None
            or not (panel_dir(cfg, "coarse") / f"{first.window_id}__{first.stage}.parquet").exists()
        ):
            continue
        coarse = pl.read_parquet(
            panel_dir(cfg, "coarse") / f"{first.window_id}__{first.stage}.parquet"
        )
        plots.append(
            sanity_plot(
                coarse,
                baselines,
                first.window_id,
                first.stage,
                cfg.paths.figures / f"sanity_{event_type}.png",
            )
        )
    write_manifest(
        cfg,
        cfg.paths.processed / "panels",
        "panel build",
        {"stages": built, "instruments": instruments},
    )
    return PanelBuildResult(built, baselines, validation, plots)
