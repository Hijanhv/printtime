"""Phase 3: event-time alignment, measures, baselines and no-look-ahead."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from printtime.config import Settings
from printtime.panel.build import Stage, baselines, build_panels, ohlcv_panel, panel_for
from printtime.panel.providers import SyntheticProvider, stages_from
from printtime.schema import ASK, MODIFY, TRADE
from printtime.synthetic.generator import Scenario, generate_book

NS = 1_000_000_000
MS = 1_000_000
DAY = dt.date(2025, 3, 12)


def hand_frame(rows: list[tuple[int, int, int, int, int, int, int, int]]) -> pl.DataFrame:
    """rows: (ts_ns, action, side, size, bid_px, ask_px, bid_sz, ask_sz)."""
    a = np.array(rows, dtype=np.int64)
    return pl.DataFrame(
        {
            "ts_event": a[:, 0],
            "action": a[:, 1].astype(np.int8),
            "side": a[:, 2].astype(np.int8),
            "price": a[:, 4],
            "size": a[:, 3],
            "bid_px_0": a[:, 4],
            "ask_px_0": a[:, 5],
            "bid_sz_0": a[:, 6],
            "ask_sz_0": a[:, 7],
        }
    )


T0 = 1_741_782_600 * NS  # 2025-03-12 12:30 UTC = 08:30 EDT


def test_jump_at_t0_appears_exactly_at_offset_zero() -> None:
    f = hand_frame(
        [
            (T0 - 5 * NS, MODIFY, 0, 0, 100, 101, 50, 50),
            (T0 - 1, MODIFY, 0, 0, 100, 101, 40, 40),  # 1 ns before t0
            (T0, TRADE, ASK, 30, 96, 97, 10, 10),  # the jump, exactly at t0
            (T0 + 150 * MS, MODIFY, 0, 0, 96, 98, 12, 12),
        ]
    )
    p = panel_for(f, T0, -1, 1, 100, [1])
    at = dict(zip(p["offset_ms"].to_list(), p["mid_move"].to_list(), strict=True))
    assert at[-100] == 0.0 and at[0] == -4.0 and at[100] == -4.0 and at[200] == -3.5
    sp = dict(zip(p["offset_ms"].to_list(), p["spread"].to_list(), strict=True))
    assert sp[-100] == 1 and sp[0] == 1 and sp[200] == 2
    tr = dict(zip(p["offset_ms"].to_list(), p["trades"].to_list(), strict=True))
    vol = dict(zip(p["offset_ms"].to_list(), p["signed_volume"].to_list(), strict=True))
    assert tr[0] == 1 and vol[0] == -30  # a seller hit the bid in the interval ending at t0
    assert tr[-100] == 0 and np.isnan(tr[-1000])  # the first interval is unknown, not zero


def test_counts_and_depth_on_a_hand_built_series() -> None:
    f = hand_frame([(T0 + k * 250 * MS, MODIFY, 0, 0, 100, 101, 10 + k, 20 + k) for k in range(8)])
    p = panel_for(f, T0, 0, 1, 500, [1, 3])
    assert p["updates"].to_list()[1:] == [2.0, 2.0]  # two updates in each 500 ms interval
    assert p["depth_1"].to_list() == [30.0, 34.0, 38.0]
    assert np.isnan(p["depth_3"].to_list()).all()  # mbp-1 frames have no level-3 depth


def test_baseline_is_the_median_over_its_window() -> None:
    rows = []
    for k, (o, depth) in enumerate([(-1500, 10), (-1200, 30), (-900, 20), (-700, 1000), (-300, 5)]):
        rows.append((T0 + o * NS + k, MODIFY, 0, 0, 100, 101, depth, 0))
    p = panel_for(hand_frame(rows), T0, -1800, 0, 1000, [1]).with_columns(
        pl.lit("w").alias("window_id"),
        pl.lit("CPI").alias("event_type"),
        pl.lit("release").alias("stage"),
        pl.lit("ZN").alias("instrument"),
    )
    base = baselines(p, (-1800, -600))
    # the state is 10 from -1500, 30 from -1200, 20 from -900, 1000 from -700 until -600
    seconds = np.concatenate(
        [np.full(300, 10), np.full(300, 30), np.full(200, 20), np.full(100, 1000)]
    )
    assert base["base_depth_1"][0] == pytest.approx(float(np.median(seconds)))


def test_ohlcv_panel_uses_completed_bars_only() -> None:
    f = pl.DataFrame(
        {"ts_event": [T0 - NS, T0, T0 + NS], "close": [100, 104, 103], "volume": [5, 50, 7]}
    )
    p = ohlcv_panel(f, T0, -1, 2)
    close = dict(zip(p["offset_ms"].to_list(), p["close_move"].to_list(), strict=True))
    # the bar opening at t0 is only complete at t0 + 1 s
    assert close[0] == 0.0 and close[1000] == 4.0 and close[2000] == 3.0


@settings(max_examples=20, deadline=None)
@given(cut_s=st.integers(min_value=-50, max_value=100))
def test_panel_never_looks_ahead(base_cfg: Settings, cut_s: int) -> None:
    book = generate_book(base_cfg, Scenario("CPI", DAY, "08:30", 1.0), "ES")
    t0 = book.t0_ns
    full = panel_for(book.frame, t0, -60, 120, 100, [1, 3])
    cut = book.frame.filter(pl.col("ts_event") <= t0 + cut_s * NS)
    part = panel_for(cut, t0, -60, 120, 100, [1, 3])
    keep = full["offset_ms"] <= cut_s * 1000
    # mid_move is defined against the pre-release mid, so before t0 it may depend
    # on data up to t0 by design (documented in panel/build.py); from t0 on it
    # must be causal like everything else.
    cols = [c for c in full.columns if c != "mid_move" or cut_s >= 0]
    a = full.filter(keep).select(cols).to_numpy()
    b = part.filter(keep).select(cols).to_numpy()
    assert np.array_equal(a, b, equal_nan=True), (
        f"grid points up to {cut_s} s changed when later data was removed"
    )


def test_build_panels_on_a_synthetic_calendar(cfg: Settings) -> None:
    scen = {
        "CPI_2025-03-12": Scenario("CPI", DAY, "08:30", 1.0),
        "CONTROL0830_2025-03-18": Scenario("CPI", dt.date(2025, 3, 18), "08:30", 0.0, control=True),
    }
    stages = [
        Stage(
            "CPI_2025-03-12", "CPI", "release", "08:30", scen["CPI_2025-03-12"].t0_ns(cfg.timezone)
        ),
        Stage(
            "CONTROL0830_2025-03-18",
            "CONTROL",
            "control",
            "08:30",
            scen["CONTROL0830_2025-03-18"].t0_ns(cfg.timezone),
        ),
    ]
    panels = build_panels(
        cfg.model_copy(
            update={
                "panel": cfg.panel.model_copy(
                    update={
                        "coarse_grid": cfg.panel.coarse_grid.model_copy(
                            update={"start_s": -300, "end_s": 600}
                        ),
                        "baseline_window_s": (-300, -120),
                    }
                )
            }
        ),
        stages,
        SyntheticProvider(cfg, scen),
        ["ZN", "ES"],
    )
    assert set(panels.fine["instrument"]) == {"ZN", "ES"}
    assert panels.fine.filter(pl.col("window_id") == "CPI_2025-03-12").height == 2 * 1801
    zn = panels.fine.filter(
        (pl.col("window_id") == "CPI_2025-03-12") & (pl.col("instrument") == "ZN")
    )
    assert zn.filter(pl.col("offset_ms") == 0)["mid_move"][0] == round(
        cfg.synthetic.instruments["ZN"].jump_ticks_per_sd
    )
    assert panels.baselines.height == 4 and (panels.baselines["base_depth_1"] > 0).all()
    ctl = panels.fine.filter(
        (pl.col("window_id").str.starts_with("CONTROL")) & (pl.col("offset_ms") == 0)
    )
    assert ctl["mid_move"].abs().max() <= 1


def test_stages_cover_both_fomc_stages_and_controls() -> None:
    cal = pl.DataFrame(
        {
            "event_id": ["FOMC_2025-06-18", "FOMC_2025-06-18", "CPI_2025-06-11"],
            "event_type": ["FOMC", "FOMC", "CPI"],
            "stage": ["statement", "press_conference", "release"],
            "time_et": ["14:00", "14:30", "08:30"],
            "t0_utc_ns": [3, 4, 1],
        }
    )
    ctl = pl.DataFrame(
        {"control_id": ["CONTROL1400_2025-06-25"], "time_et": ["14:00"], "t0_utc_ns": [9]}
    )
    st_ = stages_from(cal, ctl, ["CPI", "NFP", "FOMC"])
    assert [s.stage for s in st_] == ["release", "statement", "press_conference", "control"]


def test_panel_build_pipeline_on_a_small_synthetic_study(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from printtime.panel.pipeline import run_panel_build, stages_for_study, synthetic_settings
    from printtime.panel.store import scan
    from printtime.synthetic.generator import synthetic_study
    from tests.conftest import test_config

    cfg = synthetic_settings(
        test_config(
            tmp_path,
            [
                "panel.coarse_grid.start_s=-300",
                "panel.coarse_grid.end_s=600",
                "panel.baseline_window_s=[-300,-120]",
            ],
        ),
        root=tmp_path / "synthetic_run",
    )
    study = synthetic_study(cfg, n_per_type=2, n_controls=1)
    stages = stages_for_study(cfg, study)
    result = run_panel_build(cfg, stages, SyntheticProvider(cfg, study.scenarios), ["ZN", "ES"])
    assert result.stages_built == len(stages) == 8
    fine = scan(cfg, "fine").collect()
    assert fine["window_id"].n_unique() == 8 and set(fine["instrument"]) == {"ZN", "ES"}
    assert set(result.validation["event_id"]) == {
        s.window_id for s in stages if s.event_type != "CONTROL"
    }
    assert len(result.sanity_plots) == 3 and all(p.exists() for p in result.sanity_plots)
    manifest = json.loads((cfg.paths.processed / "panels" / "run_panel_build.json").read_text())
    assert manifest["git_commit"] and manifest["config"]["sample"]["start"] == "2024-10-01"
    assert str(tmp_path) in str(cfg.paths.figures)  # synthetic output never lands in reports/
