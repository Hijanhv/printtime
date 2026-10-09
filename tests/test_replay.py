"""Phase 8: the replay exports what the dashboard and alerts need."""

from __future__ import annotations

import datetime as dt
import time

import pytest
from prometheus_client import generate_latest

from printtime.config import Settings
from printtime.panel.build import Book
from printtime.replay.live import ReplayMetrics, baseline_depth, replay
from printtime.synthetic.generator import Scenario, generate_event


def books(cfg: Settings) -> tuple[dict[str, Book], int]:
    s = Scenario("CPI", dt.date(2025, 3, 12), "08:30", 1.5)
    gen = generate_event(cfg, s, ["ZN", "ES"])
    return {k: Book(b.frame, b.tick_size, "synthetic", 1.0) for k, b in gen.items()}, s.t0_ns(
        cfg.timezone
    )


def test_replay_metrics_track_the_release(cfg: Settings) -> None:
    b, t0 = books(cfg)
    m = ReplayMetrics()
    stats = replay(b, t0, m, start_s=-30, end_s=2, realtime=False)
    assert stats.events > 0
    text = generate_latest(m.registry).decode()
    zn_move = m.mid_move.labels("ZN")._value.get()
    assert zn_move == pytest.approx(
        round(cfg.synthetic.instruments["ZN"].jump_ticks_per_sd * 1.5), abs=2
    )
    assert m.depth_pct.labels("ZN")._value.get() < 60  # still depleted 2 s after the release
    assert m.clock._value.get() == pytest.approx(2, abs=0.5)
    assert m.in_window._value.get() == 0  # replay finished
    assert 'printtime_events_total{instrument="ES"}' in text


def test_baseline_uses_the_pre_release_book(cfg: Settings) -> None:
    b, t0 = books(cfg)
    base = baseline_depth(b["ZN"].frame, t0, t0 - 300 * 1_000_000_000)
    z = cfg.synthetic.instruments["ZN"].base_depth
    assert base == pytest.approx(2 * z, rel=0.2)  # bid + ask at the touch, before any withdrawal


def test_realtime_pacing(cfg: Settings) -> None:
    b, t0 = books(cfg)
    t = time.perf_counter()
    replay(b, t0, ReplayMetrics(), start_s=-2, end_s=0, realtime=True, speed=4.0)
    assert 0.3 <= time.perf_counter() - t < 2.0  # 2 s of market time at 4x is about 0.5 s
    with pytest.raises(ValueError):
        replay(b, t0, ReplayMetrics(), speed=0)
