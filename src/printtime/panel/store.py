"""Reading and writing panels, and the run manifest saved next to outputs.

Panels are written one file per alignment point (window and stage) so a full
study never has to sit in memory; readers scan them lazily with Polars.

    <processed>/panels/fine/<window_id>__<stage>.parquet
    <processed>/panels/coarse/<window_id>__<stage>.parquet
    <processed>/panels/baselines.parquet
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import polars as pl

from printtime.config import Settings


def panel_dir(cfg: Settings, grid: str) -> Path:
    return cfg.paths.processed / "panels" / grid


def write_stage(cfg: Settings, grid: str, window_id: str, stage: str, df: pl.DataFrame) -> Path:
    d = panel_dir(cfg, grid)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{window_id}__{stage}.parquet"
    df.write_parquet(path)
    return path


def scan(cfg: Settings, grid: str) -> pl.LazyFrame:
    d = panel_dir(cfg, grid)
    if not any(d.glob("*.parquet")):
        raise FileNotFoundError(f"no {grid} panels in {d}; run `printtime panel build`")
    return pl.scan_parquet(str(d / "*.parquet"))


def read_baselines(cfg: Settings) -> pl.DataFrame:
    return pl.read_parquet(cfg.paths.processed / "panels" / "baselines.parquet")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_manifest(
    cfg: Settings, out_dir: Path, command: str, extra: dict[str, object] | None = None
) -> Path:
    """Spec 9.1: every run saves its config and git commit hash next to its outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "command": command,
        "when_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": git_commit(),
        "config": json.loads(cfg.model_dump_json()),
        **(extra or {}),
    }
    path = out_dir / f"run_{command.replace(' ', '_')}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
