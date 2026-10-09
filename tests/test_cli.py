"""CLI wiring: every spec command exists; unbuilt ones say so clearly."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from printtime.cli import app
from tests.conftest import ROOT

BASE = ["--config", str(ROOT / "config.yaml"), "--set", "synthetic.window_s=[-120, 240]"]


@pytest.mark.parametrize(
    "command",
    [
        ["calendar", "build"],
        ["calendar", "validate"],
        ["data", "plan-costs"],
        ["data", "download"],
        ["panel", "build"],
        ["analyze", "liquidity"],
        ["analyze", "reaction"],
        ["analyze", "regression"],
        ["analyze", "execution"],
        ["analyze", "strategy"],
        ["report"],
        ["replay"],
    ],
)
def test_every_spec_command_exists(command: list[str]) -> None:
    r = CliRunner().invoke(app, [*BASE, *command, "--help"])
    assert r.exit_code == 0, r.output


def test_synth_writes_one_file_per_instrument(tmp_path: Path) -> None:
    r = CliRunner().invoke(
        app, [*BASE, "synth", "--event", "NFP", "--date", "2025-04-04", "--out", str(tmp_path)]
    )
    assert r.exit_code == 0, r.output
    files = sorted(p.stem for p in tmp_path.glob("NFP_2025-04-04_0830/*.parquet"))
    assert files == ["ES", "GC", "ZN", "ZT"]
    assert pl.read_parquet(tmp_path / "NFP_2025-04-04_0830" / "ZN.parquet").height > 0
