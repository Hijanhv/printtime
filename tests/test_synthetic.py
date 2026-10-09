"""The synthetic generator must produce exactly the effects it is configured with."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from printtime.config import Settings
from printtime.synthetic.generator import Scenario, depth_factor, generate_book, synthetic_calendar

DAY = dt.date(2025, 3, 12)


def mid(frame):  # type: ignore[no-untyped-def]
    return (frame["bid_px_0"] + frame["ask_px_0"]).to_numpy() / 2


@settings(max_examples=25, deadline=None)
@given(
    surprise=st.floats(min_value=-3, max_value=3, allow_nan=False),
    instrument=st.sampled_from(["ZN", "ZT", "ES", "GC"]),
)
def test_jump_happens_exactly_at_the_reaction_time(
    base_cfg: Settings, surprise: float, instrument: str
) -> None:
    b = generate_book(base_cfg, Scenario("CPI", DAY, "08:30", surprise), instrument)
    ts = b.frame["ts_event"].to_numpy()
    i = int(np.searchsorted(ts, b.reaction_ns))
    assert ts[i] == b.reaction_ns
    sp = base_cfg.synthetic.instruments[instrument]
    assert b.reaction_ns - b.t0_ns == sp.lag_ms * 1_000_000
    m = mid(b.frame)
    assert m[i] - m[i - 1] == round(sp.jump_ticks_per_sd * surprise)


@settings(max_examples=25, deadline=None)
@given(surprise=st.floats(min_value=-3, max_value=3, allow_nan=False), control=st.booleans())
def test_book_never_crosses_and_levels_are_ordered(
    base_cfg: Settings, surprise: float, control: bool
) -> None:
    f = generate_book(base_cfg, Scenario("NFP", DAY, "08:30", surprise, control), "ES").frame
    assert (f["ask_px_0"] - f["bid_px_0"]).min() >= 1
    for i in range(1, base_cfg.synthetic.levels):
        assert (f[f"bid_px_{i - 1}"] > f[f"bid_px_{i}"]).all()
        assert (f[f"ask_px_{i}"] > f[f"ask_px_{i - 1}"]).all()
        assert (f[f"bid_sz_{i}"] >= 1).all()
    assert (np.diff(f["ts_event"].to_numpy()) >= 0).all()


def test_depth_drop_matches_configuration(cfg: Settings) -> None:
    ev = cfg.synthetic.event
    t = np.array([-1000.0, -ev.withdrawal_lead_s / 2, 0.0, ev.depth_recovery_half_life_s])
    f = depth_factor(t, ev, control=False)
    assert f[0] == 1.0
    assert f[1] == pytest.approx(1 - ev.depth_drop / 2)
    assert f[2] == pytest.approx(1 - ev.depth_drop)
    assert f[3] == pytest.approx(1 - ev.depth_drop / 2)  # one half-life later, half recovered
    assert (depth_factor(t, ev, control=True) == 1.0).all()


def test_depth_at_t0_is_reduced_in_the_generated_book(cfg: Settings) -> None:
    b = generate_book(cfg, Scenario("CPI", DAY, "08:30", 1.0), "ZN")
    rel = (b.frame["ts_event"].to_numpy() - b.t0_ns) / 1e9
    depth = (b.frame["bid_sz_0"] + b.frame["ask_sz_0"]).to_numpy()
    before = np.median(depth[rel < -60])
    at = np.median(depth[(rel >= 0) & (rel < 1)])
    assert at / before == pytest.approx(1 - cfg.synthetic.event.depth_drop, abs=0.1)


def test_control_window_has_no_release_effects(cfg: Settings) -> None:
    b = generate_book(cfg, Scenario("CPI", DAY, "08:30", 2.0, control=True), "ZN")
    f = b.frame
    assert (f["ask_px_0"] - f["bid_px_0"]).max() == 1
    m = mid(f)
    assert np.abs(np.diff(m)).max() <= 1  # only ordinary one-tick moves


def test_activity_bursts_after_the_release(cfg: Settings) -> None:
    b = generate_book(cfg, Scenario("CPI", DAY, "08:30", 1.0), "ES")
    rel = (b.frame["ts_event"].to_numpy() - b.t0_ns) / 1e9
    before = np.sum((rel >= -10) & (rel < 0)) / 10
    after = np.sum((rel >= 0) & (rel < 2)) / 2
    assert after > 3 * before


def test_same_seed_same_book_and_different_events_differ(cfg: Settings) -> None:
    s = Scenario("CPI", DAY, "08:30", 0.7)
    assert generate_book(cfg, s, "ZN").frame.equals(generate_book(cfg, s, "ZN").frame)
    other = Scenario("CPI", DAY + dt.timedelta(days=28), "08:30", 0.7)
    assert not generate_book(cfg, other, "ZN").frame.equals(generate_book(cfg, s, "ZN").frame)


def test_calendar_has_events_then_controls(cfg: Settings) -> None:
    cal = synthetic_calendar(cfg, n_events=5, n_controls=3)
    assert [s.control for s in cal] == [False] * 5 + [True] * 3
    assert len({s.event_id for s in cal}) == 8
    assert all(s.time_et == "08:30" for s in cal)
