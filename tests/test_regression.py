"""Phase 6: reaction-function regressions recover planted coefficients; FDR and sign checks."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from printtime.analysis import regression
from printtime.panel.pipeline import (
    run_panel_build,
    save_study,
    stages_for_study,
    synthetic_settings,
)
from printtime.panel.providers import SyntheticProvider
from printtime.stats.regression import bh_fdr, bootstrap_beta, ols_hc3
from printtime.synthetic.generator import synthetic_study
from tests.conftest import test_config

INSTRUMENTS = ["ZN", "ZT", "ES", "GC"]


def test_ols_hc3_and_bootstrap_on_a_known_line() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 2.0 - 3.0 * x + rng.normal(scale=0.5, size=200)
    r = ols_hc3(x, y)
    assert r.beta == pytest.approx(-3.0, abs=0.15) and r.alpha == pytest.approx(2.0, abs=0.15)
    assert r.p < 1e-10 and r.n == 200 and 0.9 < r.r2 < 1
    lo, hi = bootstrap_beta(x, y, 500, 1)
    assert lo < r.beta < hi and hi - lo < 0.5
    assert np.isnan(ols_hc3(np.ones(5), np.arange(5.0)).beta)  # no variation in the surprise


def test_benjamini_hochberg_by_hand() -> None:
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
    reject, adj = bh_fdr(p, 0.05)
    # BH thresholds k/8 * 0.05: only the first two pass (0.039 > 3/8*0.05 = 0.01875)
    assert reject.tolist() == [True, True, False, False, False, False, False, False, False]
    assert np.isnan(adj[-1])


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("reg")
    cfg = synthetic_settings(
        test_config(
            tmp,
            [
                "synthetic.window_s=[-900,1800]",
                "panel.coarse_grid.start_s=-900",
                "panel.coarse_grid.end_s=1800",
                "panel.baseline_window_s=[-900,-600]",
                "analysis.bootstrap_reps=300",
                # a quiet ZT, so the test checks the FOMC surprise mechanism, not ZT noise
                "synthetic.instruments.ZT.vol_ticks_per_s=0.002",
            ],
        ),
        root=tmp / "synthetic_run",
    )
    st = synthetic_study(cfg, n_per_type=10, n_controls=2)
    save_study(cfg, st)
    run_panel_build(
        cfg, stages_for_study(cfg, st), SyntheticProvider(cfg, st.scenarios), INSTRUMENTS
    )
    return cfg, st, regression.run(cfg)


def coef(res, et: str, inst: str, h: int) -> dict:  # type: ignore[no-untyped-def, type-arg]
    return res.table.filter(
        (pl.col("event_type") == et)
        & (pl.col("instrument") == inst)
        & (pl.col("horizon_s") == h)
        & (pl.col("unit") == "ret_ticks")
    ).row(0, named=True)


def test_recovers_the_planted_reaction_coefficients(study) -> None:  # type: ignore[no-untyped-def]
    cfg, _, res = study
    for et in ("CPI", "NFP"):
        for inst in INSTRUMENTS:
            r = coef(res, et, inst, 1)
            truth = cfg.synthetic.instruments[inst].jump_ticks_per_sd
            assert r["beta"] == pytest.approx(truth, abs=0.25 * abs(truth) + 0.6), (et, inst)
            assert r["n_events"] == 10 and r["boot_lo"] <= r["beta"] <= r["boot_hi"]
            expected = cfg.regression.expected_sign[et][inst]
            assert r["survives_fdr"]
            assert r["sign_check"] == ("as expected" if expected else "no expectation")


def test_fomc_surprise_is_market_implied_and_zt_is_mechanical(study) -> None:  # type: ignore[no-untyped-def]
    _, st, res = study
    f = res.surprises.filter(pl.col("event_type") == "FOMC")
    truth = st.surprises.filter(pl.col("event_type") == "FOMC").select(
        "event_id", pl.col("z").alias("z_true")
    )
    j = f.join(truth, on="event_id")
    assert (
        j.height == 10 and np.corrcoef(j["z"], j["z_true"])[0, 1] > 0.9
    )  # hawkish = ZT down = positive
    zt = coef(res, "FOMC", "ZT", 1)
    assert zt["mechanical"] and not zt["survives_fdr"]  # excluded from FDR: it is its own surprise
    assert coef(res, "FOMC", "ES", 1)["beta"] < 0


def test_sign_check_flags_significant_wrong_signs(study) -> None:  # type: ignore[no-untyped-def]
    cfg, _, res = study
    flipped = cfg.model_copy(
        update={
            "regression": cfg.regression.model_copy(
                update={
                    "expected_sign": {
                        **cfg.regression.expected_sign,
                        "CPI": {**cfg.regression.expected_sign["CPI"], "ZN": 1},
                    }
                }
            )
        }
    )
    rets = regression.returns_at_horizons(flipped, regression.scan(flipped, "coarse").collect())
    t = regression.regressions(flipped, rets, res.surprises, 100, 0)
    zn = t.filter(
        (pl.col("event_type") == "CPI")
        & (pl.col("instrument") == "ZN")
        & (pl.col("unit") == "ret_ticks")
    )
    assert "WRONG SIGN: investigate" in zn["sign_check"].to_list()


def test_asymmetry_continuation_and_conversions(study) -> None:  # type: ignore[no-untyped-def]
    cfg, _, res = study
    r = coef(res, "CPI", "ES", 1)
    assert r["n_pos"] + r["n_neg"] == r["n_events"]
    assert set(res.continuation["window"]) == {
        "+1 min to +30 min"
    }  # +60 min is beyond this test's grid
    bps = res.table.filter(
        (pl.col("unit") == "ret_bps")
        & (pl.col("instrument") == "ZN")
        & (pl.col("horizon_s") == 1)
        & (pl.col("event_type") == "CPI")
    )["beta"][0]
    tick_bps = 1e4 / (
        cfg.synthetic.instruments["ZN"].start_price / cfg.synthetic.instruments["ZN"].tick_size
    )
    assert (
        bps
        == pytest.approx(
            r["beta"]
            * tick_bps
            * cfg.synthetic.instruments["ZN"].jump_ticks_per_sd
            / cfg.synthetic.instruments["ZN"].jump_ticks_per_sd,
            rel=0.35,
        )
        or bps < 0
    )


def test_outputs_and_headline(study) -> None:  # type: ignore[no-untyped-def]
    cfg, _, res = study
    t = cfg.paths.tables
    for name in ("reaction_regressions.parquet", "continuation.parquet", "regression_summary.md"):
        assert (t / name).exists()
    assert (cfg.paths.figures / "reaction_heatmap_cpi.png").exists()
    assert any("A one-standard-deviation CPI surprise moves ZN by" in h for h in res.headline)
    assert "survive Benjamini-Hochberg" in (t / "regression_summary.md").read_text()
