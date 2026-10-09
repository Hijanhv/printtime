"""Synthetic release-window order books, for tests and CI.

The generator draws a plausible order book for each instrument around a fake
release at t0, with every effect the analysis must detect switched on and set
to a *known* value:

* depth withdrawal: depth starts falling `withdrawal_lead_s` before t0, drops
  by `depth_drop` at t0, then recovers with half-life `depth_recovery_half_life_s`;
* spread blowout: `spread_blowout_ticks` extra ticks at the reaction, decaying
  with half-life `spread_half_life_s`;
* activity burst: update and trade rates jump by `activity_multiplier`;
* price jump: `jump_ticks_per_sd` x the standardised surprise, at the
  instrument's reaction time t0 + lag_ms (so first-mover ranking is known);
* control mode: the same clock window with none of the above.

Because the answers are known, tests can check that the panel builder and the
analyses recover them. None of this is market data and none of it appears in
results; real results come only from Databento data.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from printtime.calendar.timezones import et_to_utc_ns
from printtime.config import Settings, SyntheticEvent, SyntheticInstrument
from printtime.schema import ASK, BID, MODIFY, TRADE

NS = 1_000_000_000
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Scenario:
    """One synthetic event (or control) window."""

    event_type: str
    date: dt.date
    time_et: str
    surprise_sd: float = 0.0
    control: bool = False

    def t0_ns(self, tz: str) -> int:
        return et_to_utc_ns(self.date, self.time_et, tz)

    @property
    def event_id(self) -> str:
        kind = "CONTROL" if self.control else self.event_type
        return f"{kind}_{self.date.isoformat()}_{self.time_et.replace(':', '')}"


@dataclass(frozen=True)
class SyntheticBook:
    instrument: str
    tick_size: float
    t0_ns: int
    reaction_ns: int
    frame: pl.DataFrame


def _seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, (base, *parts))).encode()).hexdigest()
    return int(digest[:8], 16)


def _half_life_decay(t_s: F64, half_life_s: float) -> F64:
    out: F64 = np.where(t_s >= 0, np.exp2(-np.maximum(t_s, 0.0) / half_life_s), 0.0)
    return out


def depth_factor(t_rel_s: F64, ev: SyntheticEvent, control: bool) -> F64:
    """Depth as a fraction of normal, relative to t0 (time in seconds)."""
    f = np.ones_like(t_rel_s)
    if control or ev.depth_drop == 0:
        return f
    lead = ev.withdrawal_lead_s
    if lead > 0:
        ramp = (t_rel_s >= -lead) & (t_rel_s < 0)
        f[ramp] = 1.0 - ev.depth_drop * (t_rel_s[ramp] + lead) / lead
    after = t_rel_s >= 0
    f[after] = 1.0 - ev.depth_drop * np.exp2(-t_rel_s[after] / ev.depth_recovery_half_life_s)
    return f


def _event_times(
    rng: np.random.Generator,
    t_start_s: float,
    t_end_s: float,
    base_rate: float,
    burst_at_s: float | None,
    multiplier: float,
    half_life_s: float,
) -> F64:
    """Update times (s, relative to t0): a constant-rate process plus a decaying burst.

    The burst is a second Poisson process with rate base*(multiplier-1)*decay(t),
    drawn by thinning over a window long enough for the decay to vanish.
    """
    n = rng.poisson(base_rate * (t_end_s - t_start_s))
    times = rng.uniform(t_start_s, t_end_s, n)
    if burst_at_s is not None and multiplier > 1:
        span = min(half_life_s * 12, t_end_s - burst_at_s)
        peak = base_rate * (multiplier - 1)
        m = rng.poisson(peak * span)
        cand = rng.uniform(burst_at_s, burst_at_s + span, m)
        keep = rng.uniform(0, 1, m) < np.exp2(-(cand - burst_at_s) / half_life_s)
        times = np.concatenate([times, cand[keep], [burst_at_s]])
    out: F64 = np.sort(times)
    return out


def generate_book(
    cfg: Settings, scenario: Scenario, instrument: str, *, window_s: tuple[int, int] | None = None
) -> SyntheticBook:
    sp: SyntheticInstrument = cfg.synthetic.instruments[instrument]
    ev = cfg.synthetic.event
    levels = cfg.synthetic.levels
    lo, hi = window_s or cfg.synthetic.window_s
    t0 = scenario.t0_ns(cfg.timezone)
    rng = np.random.default_rng(_seed(cfg.synthetic.seed, scenario.event_id, instrument))
    lag_s = sp.lag_ms / 1000.0
    react_s = None if scenario.control else lag_s

    t = _event_times(
        rng, lo, hi, sp.update_rate, react_s, ev.activity_multiplier, ev.activity_half_life_s
    )
    if react_s is not None and not (lo <= react_s <= hi):
        react_s = None
    n = t.size

    # Bid price: a tick-level random walk plus the release jump at the reaction time.
    p_move = min(sp.vol_ticks_per_s / sp.update_rate, 0.5)
    steps = rng.choice(np.array([-1, 0, 1]), size=n, p=[p_move / 2, 1 - p_move, p_move / 2])
    jump = 0 if scenario.control else round(sp.jump_ticks_per_sd * scenario.surprise_sd)
    if react_s is not None:
        idx = int(np.searchsorted(t, react_s, side="left"))
        steps[idx] = jump  # the update exactly at the reaction time carries the jump
    core = round(sp.start_price / sp.tick_size) + np.cumsum(steps).astype(np.int64)

    # Spread: one tick normally. After the reaction it widens symmetrically
    # around the mid (an even number of extra ticks, half on each side), so the
    # blowout never moves the mid and the configured jump is exactly the mid move.
    half_extra = np.zeros(n, dtype=np.int64)
    if react_s is not None and ev.spread_blowout_ticks > 0:
        extra = ev.spread_blowout_ticks * _half_life_decay(t - react_s, ev.spread_half_life_s)
        half_extra = np.rint(extra / 2).astype(np.int64)
    bid = core - half_extra
    ask = core + 1 + half_extra

    factor = depth_factor(t, ev, scenario.control)
    cols: dict[str, npt.NDArray[np.int64] | npt.NDArray[np.int8]] = {
        # Offsets are rounded to integer ns *before* adding t0: t0 is ~1.7e18,
        # where float64 only resolves ~256 ns, so float addition would shift
        # timestamps and break millisecond first-move timing.
        "ts_event": t0 + np.rint(t * NS).astype(np.int64),
    }
    for i in range(levels):
        shape = sp.base_depth * (1.0 + 0.15 * i)
        noise_b = rng.lognormal(0.0, 0.25, n)
        noise_a = rng.lognormal(0.0, 0.25, n)
        cols[f"bid_px_{i}"] = bid - i
        cols[f"ask_px_{i}"] = ask + i
        cols[f"bid_sz_{i}"] = np.maximum(1, np.rint(shape * factor * noise_b)).astype(np.int64)
        cols[f"ask_sz_{i}"] = np.maximum(1, np.rint(shape * factor * noise_a)).astype(np.int64)

    is_trade = rng.uniform(0, 1, n) < sp.trade_share
    aggressor = np.where(rng.uniform(0, 1, n) < 0.5, BID, ASK).astype(np.int8)
    if react_s is not None and jump != 0:
        idx = int(np.searchsorted(t, react_s, side="left"))
        is_trade[idx] = True
        aggressor[idx] = BID if jump > 0 else ASK
    trade_px = np.where(aggressor == BID, ask, bid)
    cols["action"] = np.where(is_trade, TRADE, MODIFY).astype(np.int8)
    cols["side"] = np.where(is_trade, aggressor, 0).astype(np.int8)
    cols["price"] = np.where(is_trade, trade_px, bid).astype(np.int64)
    cols["size"] = np.where(is_trade, rng.geometric(0.3, n), 0).astype(np.int64)

    frame = pl.DataFrame(cols).select(
        "ts_event",
        "action",
        "side",
        "price",
        "size",
        *[c for c in cols if c[:4] in ("bid_", "ask_")],
    )
    reaction_ns = t0 + round((react_s if react_s is not None else 0.0) * NS)
    return SyntheticBook(instrument, sp.tick_size, t0, reaction_ns, frame)


def generate_event(
    cfg: Settings,
    scenario: Scenario,
    instruments: list[str] | None = None,
    *,
    window_s: tuple[int, int] | None = None,
) -> dict[str, SyntheticBook]:
    names = instruments or list(cfg.synthetic.instruments)
    return {name: generate_book(cfg, scenario, name, window_s=window_s) for name in names}


def synthetic_calendar(
    cfg: Settings,
    n_events: int,
    event_type: str = "CPI",
    n_controls: int = 0,
    start: str = "2025-01-08",
) -> list[Scenario]:
    """A sequence of weekly synthetic events with standard-normal surprises, plus controls."""
    rng = np.random.default_rng(_seed(cfg.synthetic.seed, "calendar", event_type))
    time_et = cfg.events[event_type].times_et[0]
    d = dt.date.fromisoformat(start)
    out: list[Scenario] = []
    for k in range(n_events + n_controls):
        day = d + dt.timedelta(days=7 * k)
        control = k >= n_events
        out.append(
            Scenario(
                event_type, day, time_et, 0.0 if control else float(rng.standard_normal()), control
            )
        )
    return out
