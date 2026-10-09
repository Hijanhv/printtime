"""Phase 4: the liquidity analysis must recover the effects planted in synthetic data.

Generator settings (config `synthetic.event`): withdrawal starts 20 s before t0,
depth falls by 80% at t0 and recovers with a 30 s half-life, spreads blow out
by 6 ticks with a 4 s half-life. So the expected answers are:

    withdrawal start   about -20 s
    depth at t0        about 20% of baseline
    recovery to 50%    30 * log2(1 / 0.625) = 20.3 s
    recovery to 90%    30 * log2(1 / 0.125) = 90 s
    spread back        about 11 s
    control days       none of the above
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from printtime.analysis import liquidity
from printtime.config import Settings
from printtime.panel.pipeline import run_panel_build, stages_for_study, synthetic_settings
from printtime.panel.providers import SyntheticProvider
from printtime.synthetic.generator import synthetic_study
from tests.conftest import test_config


@pytest.fixture(scope="module")
def study_cfg(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    tmp = tmp_path_factory.mktemp("liq")
    cfg = synthetic_settings(
        test_config(
            tmp,
            [
                "panel.coarse_grid.start_s=-300",
                "panel.coarse_grid.end_s=600",
                "panel.baseline_window_s=[-300,-120]",
                "analysis.bootstrap_reps=300",
            ],
        ),
        root=tmp / "synthetic_run",
    )
    study = synthetic_study(cfg, n_per_type=6, n_controls=4)
    run_panel_build(
        cfg, stages_for_study(cfg, study), SyntheticProvider(cfg, study.scenarios), ["ZN", "ES"]
    )
    return cfg


@pytest.fixture(scope="module")
def results(study_cfg: Settings) -> liquidity.LiquidityResults:
    return liquidity.run(study_cfg)


def row(res: liquidity.LiquidityResults, group: str, inst: str) -> dict:  # type: ignore[type-arg]
    return res.summary.filter((pl.col("group") == group) & (pl.col("instrument") == inst)).row(
        0, named=True
    )


def test_recovers_planted_withdrawal_and_recovery(results: liquidity.LiquidityResults) -> None:
    r = row(results, "CPI release", "ZN")
    assert r["n_events"] == 6
    assert r["withdrawal_start_s_median"] == pytest.approx(-20, abs=6)
    assert r["min_depth_pct_median"] == pytest.approx(20, abs=8)
    assert r["recovery_50_s_median"] == pytest.approx(30 * math.log2(1 / 0.625), abs=6)
    assert r["recovery_90_s_median"] == pytest.approx(90, abs=20)
    assert r["spread_recovery_s_median"] == pytest.approx(11, abs=4)
    assert r["max_spread_first_min_median"] == 7


def test_control_days_show_no_effect(results: liquidity.LiquidityResults) -> None:
    c = row(results, "CONTROL 08:30", "ZN")
    assert c["n_events"] == 4
    assert c["min_depth_pct_median"] > 60
    assert np.isnan(c["withdrawal_start_s_median"]) or c["withdrawal_start_s_n"] < c["n_events"]
    assert c["max_spread_first_min_median"] == 1
    assert c["recovery_50_s_median"] == 0


def test_press_conference_gets_a_matching_1430_control(results: liquidity.LiquidityResults) -> None:
    groups = set(results.summary["group"].to_list())
    assert {"FOMC statement", "CONTROL 14:00", "CONTROL 14:30"} <= groups
    assert liquidity.control_for("FOMC press conference") == "CONTROL 14:30"


def test_curves_have_by_event_bands_and_counts(results: liquidity.LiquidityResults) -> None:
    d = results.curves.filter(
        (pl.col("group") == "CPI release")
        & (pl.col("instrument") == "ZN")
        & (pl.col("measure") == "depth_pct")
    )
    at0 = d.filter(pl.col("offset_s") == 0).row(0, named=True)
    assert at0["lo"] <= at0["mean"] <= at0["hi"] and at0["n_events"] == 6
    before = d.filter(pl.col("offset_s") == -200)["mean"][0]
    assert before == pytest.approx(100, abs=15)


def test_headline_reports_events_and_controls(results: liquidity.LiquidityResults) -> None:
    text = " ".join(results.headline)
    assert "ZN, CPI release: top-of-book depth falls by" in text
    assert "6 events" in text and "control days" in text


def test_outputs_written(study_cfg: Settings, results: liquidity.LiquidityResults) -> None:
    t, f = study_cfg.paths.tables, study_cfg.paths.figures
    for name in ("liquidity_curves.parquet", "liquidity_summary.md", "run_analyze_liquidity.json"):
        assert (t / name).exists(), name
    assert (f / "liquidity_depth_cpi_release.png").exists()
    assert (f / "liquidity_spread_fomc_statement.png").exists()
    assert "Medians across events" in (t / "liquidity_summary.md").read_text()


def test_bootstrap_resamples_events_not_seconds() -> None:
    from printtime.stats.bootstrap import curve_ci, scalar_ci

    one = np.array([[1.0, 2.0, 3.0]])
    ci = curve_ci(one, 100, 0)
    assert (
        ci.n_events == 1 and np.isnan(ci.lo).all()
    )  # one event: no interval, however many seconds
    many_seconds = np.tile(np.array([[5.0], [7.0]]), (1, 5000))
    ci = curve_ci(many_seconds, 500, 0)
    assert ci.n_events == 2 and ci.lo.min() >= 5.0 and ci.hi.max() <= 7.0
    assert ci.hi[0] - ci.lo[0] == pytest.approx(
        2.0
    )  # the band reflects two events, not 5000 seconds
    est, lo, hi, n = scalar_ci(np.array([1.0, 2.0, 3.0, np.nan]), 200, 0)
    assert n == 3 and est == 2.0 and lo <= est <= hi
