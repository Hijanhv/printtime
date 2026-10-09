"""Phase 7b: the surprise-direction strategy never uses the current or future events."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from printtime.analysis import strategy
from printtime.panel.pipeline import (
    run_panel_build,
    save_study,
    stages_for_study,
    synthetic_settings,
)
from printtime.panel.providers import SyntheticProvider
from printtime.synthetic.generator import synthetic_study
from tests.conftest import test_config


def test_pnl_crosses_the_spread_both_ways() -> None:
    d = np.array([1.0, -1.0, 0.0])
    eb, ea = np.full(3, 100.0), np.full(3, 101.0)
    xb, xa = np.full(3, 104.0), np.full(3, 105.0)
    assert strategy.trade_pnl(d, eb, ea, xb, xa).tolist() == [
        3.0,
        -5.0,
        0.0,
    ]  # buy at 101, sell at 104


@settings(max_examples=60, deadline=None)
@given(
    data=st.lists(st.tuples(st.floats(-3, 3), st.floats(-20, 20)), min_size=12, max_size=30),
    cut=st.integers(min_value=0, max_value=29),
    noise=st.floats(-50, 50),
)
def test_decisions_never_depend_on_the_current_or_later_events(
    data: list[tuple[float, float]], cut: int, noise: float
) -> None:
    s = np.array([x for x, _ in data])
    r = np.array([y for _, y in data])
    cut = min(cut, s.size - 1)
    z0, d0, t0, g0 = strategy.walk_forward(s, r, 8)
    r2 = r.copy()
    r2[cut:] += noise  # change what happens at and after `cut`
    s2 = s.copy()
    s2[cut + 1 :] *= -2.0  # and the later surprises
    z1, d1, t1, g1 = strategy.walk_forward(s2, r2, 8)
    # Every decision up to and including `cut` is unchanged: the sign mapping and threshold
    # for event k use events before k, and the decision for k uses only its own surprise.
    for a, b in ((z0, z1), (d0, d1), (t0, t1), (g0, g1)):
        np.testing.assert_array_equal(a[: cut + 1], b[: cut + 1])


def test_first_events_are_training_only() -> None:
    rng = np.random.default_rng(3)
    s = rng.normal(size=15)
    _, d, thr, _ = strategy.walk_forward(s, -4 * s + rng.normal(size=15), 8)
    assert (d[:8] == 0).all() and np.isnan(thr[:8]).all()
    assert (d[8:] != 0).any()


def test_learns_the_sign_from_the_past() -> None:
    s = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0])
    r = -3.0 * s  # prices fall on positive surprises
    _, d, _, sgn = strategy.walk_forward(s, r, 8)
    assert (
        sgn[8] == -1 and d[8] == -1 and d[9] == 1
    )  # short the positive surprise, buy the negative one


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("strat")
    cfg = synthetic_settings(
        test_config(
            tmp,
            [
                "analysis.bootstrap_reps=200",
                "synthetic.window_s=[-300,600]",
                "panel.coarse_grid.start_s=-300",
                "panel.coarse_grid.end_s=600",
                "panel.baseline_window_s=[-300,-120]",
            ],
        ),
        root=tmp / "synthetic_run",
    )
    st_ = synthetic_study(cfg, n_per_type=12, n_controls=1)
    save_study(cfg, st_)
    run_panel_build(
        cfg, stages_for_study(cfg, st_), SyntheticProvider(cfg, st_.scenarios), ["ZN", "ES"]
    )
    return cfg, strategy.run(cfg)


def test_backtest_on_a_synthetic_study(results) -> None:  # type: ignore[no-untyped-def]
    cfg, res = results
    t = res.trades
    assert set(t["event_type"]) == {"CPI", "NFP"}  # FOMC excluded: its surprise is only known later
    assert t.filter(pl.col("training_only"))["net_ticks"].is_null().all()
    s = res.summary
    assert (s["training_events"] == cfg.analysis.strategy.min_train_events).all()
    assert set(s["exit_s"]) == {300}  # +30 min and +2 h exits are beyond this test's 10-minute grid
    # Synthetic prices jump at t0 and then wander: entering a second later there is nothing left to
    # catch. With 4 out-of-sample trades per setting, single settings swing by several ticks either
    # way, so the honest check is on average across settings: no edge over the perfect-sign bound,
    # and the bound itself beats a random direction.
    assert s["mean_net_ticks"].mean() < s["upper_bound_mean_net_ticks"].mean()
    assert (s["upper_bound_mean_net_ticks"] >= s["random_mean_net_ticks"]).all()
    assert (s["trades"] <= s["events"] - s["training_events"]).all()


def test_outputs_and_honest_headline(results) -> None:  # type: ignore[no-untyped-def]
    cfg, res = results
    assert (cfg.paths.tables / "strategy_summary.md").exists() and (
        cfg.paths.figures / "strategy_pnl.png"
    ).exists()
    text = " ".join(res.headline)
    assert "upper bound" in text and "random direction" in text and "noisy" in text
