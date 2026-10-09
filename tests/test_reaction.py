"""Phase 5: price reaction, first mover and lead-lag against planted answers.

Synthetic reaction delays (config `synthetic.instruments.*.lag_ms`):
ZN 0 ms, ZT 5 ms, ES 12 ms, GC 40 ms. Jumps are coefficient x surprise ticks.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from printtime.analysis import reaction
from printtime.panel.build import mid_changes
from printtime.panel.pipeline import run_panel_build, stages_for_study, synthetic_settings
from printtime.panel.providers import SyntheticProvider
from printtime.synthetic.generator import synthetic_study
from tests.conftest import test_config

INSTRUMENTS = ["ZN", "ZT", "ES", "GC"]


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("react")
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
    st = synthetic_study(cfg, n_per_type=8, n_controls=4)
    run_panel_build(
        cfg, stages_for_study(cfg, st), SyntheticProvider(cfg, st.scenarios), INSTRUMENTS
    )
    return cfg, st, reaction.run(cfg)


def test_mid_changes_have_exact_timestamps() -> None:
    t0 = 10**18
    f = pl.DataFrame(
        {
            "ts_event": [t0 - 5, t0 + 3_000_001, t0 + 3_000_002],
            "bid_px_0": [100, 96, 96],
            "ask_px_0": [101, 97, 98],
        }
    )
    m = mid_changes(f, t0)
    assert m["offset_ns"].to_list() == [-5, 3_000_001, 3_000_002]
    assert m["mid_move"].to_list() == [0.0, -4.0, -3.5]


def test_thresholds_come_from_control_days(study) -> None:  # type: ignore[no-untyped-def]
    _, _, res = study
    k = dict(zip(res.thresholds["instrument"], res.thresholds["k_ticks"], strict=True))
    assert set(k) == set(INSTRUMENTS) and all(v >= 0.5 for v in k.values())


def test_first_mover_order_and_gaps_match_the_planted_lags(study) -> None:  # type: ignore[no-untyped-def]
    cfg, _, res = study
    cpi = res.ranking.filter(pl.col("event_type") == "CPI").sort("median_first_move_ms")
    order = [i for i in cpi["instrument"].to_list() if i in INSTRUMENTS]
    assert order == ["ZN", "ZT", "ES", "GC"]
    for r in cpi.iter_rows(named=True):
        lag = cfg.synthetic.instruments[r["instrument"]].lag_ms
        assert r["median_first_move_ms"] == pytest.approx(
            lag, abs=0.001
        )  # exact timestamps, not the 100 ms grid
        if r["instrument"] != "ZN":
            assert r["median_minus_ZN_ms"] == pytest.approx(lag, abs=0.001)
    assert any("ZN reacts first" in h for h in res.headline)


def test_jump_sizes_scale_with_the_surprise(study) -> None:  # type: ignore[no-untyped-def]
    cfg, st, res = study
    j = res.jumps.filter(
        (pl.col("instrument") == "ZN")
        & (pl.col("horizon_s") == 1)
        & (pl.col("event_type") == "CPI")
    )
    z = st.surprises.filter(pl.col("event_type") == "CPI").select(
        pl.col("event_id").alias("window_id"), "z"
    )
    d = j.join(z, on="window_id")
    expected = np.rint(cfg.synthetic.instruments["ZN"].jump_ticks_per_sd * d["z"].to_numpy())
    # one second later the move is the jump plus at most a couple of ordinary ticks
    assert np.abs(d["mid_move"].to_numpy() - expected).max() <= 3
    assert np.corrcoef(d["mid_move"].to_numpy(), d["z"].to_numpy())[0, 1] < -0.9
    assert (res.jump_summary["median_abs_move_vol_units"].drop_nulls() > 0).all()


def test_lead_lag_peak_matches_the_100ms_bins(study) -> None:  # type: ignore[no-untyped-def]
    # ZN jumps at exactly t0, so its move is in the return into the 0 ms grid point;
    # ES jumps at +12 ms, so its move is in the return into +100 ms. At 100 ms
    # resolution ES therefore follows ZN by one step: the peak is at +100 ms.
    _, _, res = study
    es = res.lead_lag.filter((pl.col("event_type") == "CPI") & (pl.col("instrument") == "ES")).sort(
        "lag_ms"
    )
    best = es.sort("corr", descending=True, nulls_last=True).row(0, named=True)
    assert best["lag_ms"] == 100 and best["corr"] > 0  # all synthetic jumps share a sign
    assert set(es["events"]) == {8}


def test_outputs_written(study) -> None:  # type: ignore[no-untyped-def]
    cfg, _, _ = study
    t = cfg.paths.tables
    for name in (
        "first_mover_ranking.parquet",
        "lead_lag.parquet",
        "reaction_summary.md",
        "run_analyze_reaction.json",
    ):
        assert (t / name).exists(), name
    assert (cfg.paths.figures / "first_movers.png").exists()
    assert "Clock caveat" in (t / "reaction_summary.md").read_text()
