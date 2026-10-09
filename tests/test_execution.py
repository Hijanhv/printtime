"""Phase 7a: execution cost by walking the book; release vs control on synthetic data."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from printtime.analysis import execution
from printtime.panel.build import Book
from printtime.panel.pipeline import (
    run_panel_build,
    save_study,
    stages_for_study,
    synthetic_settings,
)
from printtime.panel.providers import SyntheticProvider
from printtime.synthetic.generator import synthetic_study
from tests.conftest import test_config

NAN = np.nan


def test_walk_book_by_hand() -> None:
    prices = np.array([[101.0, 102.0, 103.0], [101.0, 102.0, NAN]])
    sizes = np.array([[5.0, 10.0, 20.0], [5.0, 10.0, NAN]])
    vwap, short = execution.walk_book(prices, sizes, 1)
    assert vwap.tolist() == [101.0, 101.0] and not short.any()
    vwap, short = execution.walk_book(prices, sizes, 10)  # 5 @ 101 + 5 @ 102
    assert vwap.tolist() == [101.5, 101.5]
    vwap, short = execution.walk_book(prices, sizes, 25)  # second book only shows 15 lots
    assert vwap[0] == pytest.approx((5 * 101 + 10 * 102 + 10 * 103) / 25)
    assert np.isnan(vwap[1]) and short.tolist() == [False, True]


@settings(max_examples=100, deadline=None)
@given(
    sizes=st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=10),
    qty=st.integers(min_value=1, max_value=200),
)
def test_walk_book_properties(sizes: list[int], qty: int) -> None:
    px = 100.0 + np.arange(len(sizes), dtype=float)
    vwap, short = execution.walk_book(px[None, :], np.array(sizes, dtype=float)[None, :], qty)
    if sum(sizes) < qty:
        assert short[0] and np.isnan(vwap[0])  # never priced beyond displayed depth
    else:
        best = px[np.flatnonzero(np.array(sizes) > 0)[0]]
        assert best - 1e-9 <= vwap[0] <= px[-1] + 1e-9  # between the best and the worst level used


def test_costs_on_grid_includes_half_spread(tmp_path) -> None:  # type: ignore[no-untyped-def]
    t0 = 10**18
    f = pl.DataFrame(
        {
            "ts_event": [t0 - 10**9, t0],
            "action": [2, 2],
            "side": [0, 0],
            "price": [100, 98],
            "size": [0, 0],
            "bid_px_0": [100, 97],
            "ask_px_0": [101, 103],
            "bid_sz_0": [50, 2],
            "ask_sz_0": [50, 2],
            "bid_px_1": [99, 96],
            "ask_px_1": [102, 104],
            "bid_sz_1": [50, 50],
            "ask_sz_1": [50, 50],
        }
    )
    c = execution.costs_on_grid(Book(f, 1 / 64, "test", 1000.0), t0, (-1, 0), [1, 5])
    one = c.filter(pl.col("qty") == 1).sort("offset_s")["cost_ticks"].to_list()
    assert one == [0.5, 3.0]  # half the spread: 1 tick before, 6 ticks at the release
    five = c.filter((pl.col("qty") == 5) & (pl.col("offset_s") == 0))["cost_ticks"][0]
    assert five == pytest.approx(((2 * 103 + 3 * 104) / 5 - 100 + 100 - (2 * 97 + 3 * 96) / 5) / 2)


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("exe")
    cfg = synthetic_settings(
        test_config(
            tmp,
            [
                "analysis.bootstrap_reps=200",
                "panel.coarse_grid.start_s=-300",
                "panel.coarse_grid.end_s=600",
                "panel.baseline_window_s=[-300,-120]",
            ],
        ),
        root=tmp / "synthetic_run",
    )
    st_ = synthetic_study(cfg, n_per_type=4, n_controls=3)
    save_study(cfg, st_)
    run_panel_build(
        cfg, stages_for_study(cfg, st_), SyntheticProvider(cfg, st_.scenarios), ["ZN", "ES"]
    )
    return cfg, execution.run(cfg)


def test_release_costs_spike_and_return_to_normal(results) -> None:  # type: ignore[no-untyped-def]
    _, res = results
    s = res.summary.filter(
        (pl.col("group") == "CPI release") & (pl.col("instrument") == "ZN") & (pl.col("qty") == 1)
    )
    r = s.row(0, named=True)
    assert r["normal_ticks"] == pytest.approx(0.5, abs=0.01)  # one-tick spread on control days
    assert r["extra_first_5s_ticks"] > 0.5  # the spread blowout makes the first seconds expensive
    assert 0 <= r["seconds_to_normal"] <= 30
    assert r["n_events"] == 4 and r["n_control"] == 3


def test_fees_convert_with_the_contract_multiplier(results) -> None:  # type: ignore[no-untyped-def]
    cfg, res = results
    zn = res.per_event.filter(pl.col("instrument") == "ZN")
    tick_usd = (
        cfg.synthetic.instruments["ZN"].tick_size * cfg.synthetic.instruments["ZN"].multiplier
    )
    assert zn["tick_usd"][0] == pytest.approx(tick_usd)  # 15.625 USD per ZN tick
    assert zn["fee_ticks"][0] == pytest.approx(
        cfg.analysis.execution.fee_per_contract_side_usd / tick_usd
    )


def test_outputs_written(results) -> None:  # type: ignore[no-untyped-def]
    cfg, res = results
    assert (cfg.paths.tables / "execution_summary.md").exists()
    assert (cfg.paths.figures / "execution_cpi_release.png").exists()
    assert any("executing 10 ZN contracts" in h for h in res.headline)
