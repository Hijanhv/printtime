"""Phase 2: budget guards, cost planning, control days, quality checks, event validation.

Databento is replaced by a fake client whose prices are proportional to window
length, so the planner's extrapolation can be checked exactly. Nothing here
touches the network.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from printtime.calendar.timezones import et_to_utc_ns
from printtime.config import Settings
from printtime.data import databento_io as dbio
from printtime.data.controls import MissingTier2Error, select_control_days, us_market_holidays
from printtime.data.plan import control_windows, event_windows, markdown, plan_costs, requests_for
from printtime.data.quality import check_window
from printtime.data.validation import spike_ratio, validate_events
from printtime.schema import NO_PRICE
from printtime.synthetic.generator import Scenario, generate_book

NS = 1_000_000_000
# Fake price list: USD per second of window, per schema, before the instrument multiplier.
RATE = {"ohlcv-1s": 1e-6, "mbp-1": 1e-4, "mbp-10": 5e-4, "definition": 0.0}
MULT = {"ZN": 1.0, "ES": 3.0}  # other instruments: 0.5


class FakeDatabento:
    def __init__(self) -> None:
        self.quotes: list[dict[str, Any]] = []
        self.bought: list[str] = []
        self.metadata = SimpleNamespace(get_cost=self._cost, get_billable_size=self._size)
        self.timeseries = SimpleNamespace(get_range=self._get_range)

    @staticmethod
    def _seconds(kw: dict[str, Any]) -> float:
        a = dt.datetime.fromisoformat(kw["start"])
        b = dt.datetime.fromisoformat(kw["end"])
        return (b - a).total_seconds()

    def _cost(self, **kw: Any) -> float:
        self.quotes.append(kw)
        root = kw["symbols"][0].split(".")[0]
        return RATE[kw["schema"]] * MULT.get(root, 0.5) * self._seconds(kw)

    def _size(self, **kw: Any) -> int:
        return int(self._cost(**kw) * 1e9)

    def _get_range(self, **kw: Any) -> None:
        self.bought.append(kw["path"])
        Path(kw["path"]).write_bytes(b"dbn")


def calendar(cfg: Settings, with_tier2: bool = True) -> pl.DataFrame:
    rows = []
    for d in ("2024-10-10", "2024-11-13", "2024-12-11", "2025-01-15"):
        rows.append(("CPI", d, "release", "08:30"))
    for d in ("2024-10-04", "2024-11-01", "2024-12-06"):
        rows.append(("NFP", d, "release", "08:30"))
    rows += [
        ("FOMC", "2024-11-07", "statement", "14:00"),
        ("FOMC", "2024-11-07", "press_conference", "14:30"),
    ]
    if with_tier2:
        thursdays = [
            dt.date(2024, 10, 3) + dt.timedelta(weeks=k) for k in range(105)
        ]  # whole sample
        rows += [("CLAIMS", d.isoformat(), "release", "08:30") for d in thursdays]
    out = []
    for et, d, stage, t in rows:
        day = dt.date.fromisoformat(d)
        out.append(
            {
                "event_id": f"{et}_{d}",
                "event_type": et,
                "tier": cfg.events[et].tier,
                "date": day,
                "stage": stage,
                "time_et": t,
                "t0_utc_ns": et_to_utc_ns(day, t),
                "t0_utc": "",
                "source": "test",
            }
        )
    return pl.DataFrame(out)


# --- budget guards ---------------------------------------------------------------------------


def test_dry_run_quotes_but_buys_nothing(cfg: Settings) -> None:
    api = FakeDatabento()
    r = dbio.make_request(
        cfg,
        "ZN",
        "mbp-1",
        "CPI_2024-10-10",
        et_to_utc_ns(dt.date(2024, 10, 10), "08:30"),
        (-1800, 3600),
    )
    assert dbio.download(cfg, r, api, confirm=False) is None
    assert api.quotes and not api.bought
    assert dbio.SpendLedger(cfg.databento.spend_ledger, 40).spent == 0


def test_purchase_is_cached_and_recorded(cfg: Settings) -> None:
    api = FakeDatabento()
    r = dbio.make_request(
        cfg,
        "ES",
        "mbp-1",
        "CPI_2024-10-10",
        et_to_utc_ns(dt.date(2024, 10, 10), "08:30"),
        (-1800, 3600),
    )
    path = dbio.download(cfg, r, api, confirm=True)
    assert path is not None and path.exists() and path == r.path(cfg.paths.raw)
    spent = dbio.SpendLedger(cfg.databento.spend_ledger, 40).spent
    assert spent == pytest.approx(1e-4 * 3.0 * 5400)
    assert dbio.download(cfg, r, api, confirm=True) == path and len(api.bought) == 1  # cache hit


def test_budget_guard_refuses_overspend(tmp_path: Path) -> None:
    from tests.conftest import test_config

    small = test_config(tmp_path, ["databento.budget_usd=0.5"])
    r = dbio.make_request(
        small, "ES", "mbp-10", "x", et_to_utc_ns(dt.date(2024, 10, 10), "08:30"), (-1800, 3600)
    )
    with pytest.raises(dbio.BudgetExceededError):
        dbio.download(small, r, FakeDatabento(), confirm=True)


def test_request_windows_and_symbols(cfg: Settings) -> None:
    t0 = et_to_utc_ns(dt.date(2025, 3, 12), "08:30")
    r = dbio.make_request(cfg, "6E", "mbp-10", "CPI_2025-03-12", t0, (-300, 900))
    assert r.symbol == "6E.v.0" and r.end_ns - r.start_ns == 1200 * NS
    assert r.start_iso.startswith("2025-03-12T12:25")  # 08:25 EDT


# --- plan --------------------------------------------------------------------------------------


def test_fomc_stages_share_one_window(cfg: Settings) -> None:
    wins = event_windows(calendar(cfg), cfg.tier(1))
    fomc = [w for w in wins if w.kind == "FOMC"]
    assert len(fomc) == 1 and fomc[0].last_t0_ns - fomc[0].first_t0_ns == 30 * 60 * NS
    reqs = requests_for(
        cfg, fomc, __import__("printtime.data.plan", fromlist=["plan_settings"]).plan_settings(cfg)
    )
    mbp1 = [r for r in reqs if r.schema == "mbp-1" and r.instrument == "ZN"]
    assert len(mbp1) == 1 and (mbp1[0].end_ns - mbp1[0].start_ns) / NS == 1800 + 1800 + 3600
    assert sum(1 for r in reqs if r.schema == "mbp-10") == 3  # ZT, ZN, ES only
    assert sum(1 for r in reqs if r.schema == "definition") == len(cfg.instruments)


def test_cost_plan_extrapolates_exactly_when_cost_is_linear(cfg: Settings) -> None:
    cal = calendar(cfg)
    events = event_windows(cal, cfg.tier(1))
    controls = control_windows(select_control_days(cfg, cal))
    api = FakeDatabento()
    est = plan_costs(
        cfg,
        events,
        controls,
        lambda r: dbio.quote(cfg, r, api),
        lambda r: dbio.billable_bytes(cfg, r, api),
    )
    # exact cost of the full plan under the fake price list
    from printtime.data.plan import control_windows_from, plan_settings

    ps = plan_settings(cfg)
    wins = events + control_windows_from(controls, ps.controls)
    exact = 0.0
    for schema, (lo, hi) in ps.windows_s.items():
        insts = ps.mbp10_instruments if schema == "mbp-10" else ps.instruments
        for inst in insts:
            for w in wins:
                exact += (
                    RATE[schema]
                    * MULT.get(inst, 0.5)
                    * ((w.last_t0_ns - w.first_t0_ns) / NS + hi - lo)
                )
    full = est.table.filter(pl.col("option") == "full plan")["cost_usd"][0]
    assert full == pytest.approx(exact, rel=1e-6, abs=0.01)
    opts = dict(zip(est.table["option"], est.table["cost_usd"], strict=True))
    assert opts["core instruments only"] < full and opts["shorter mbp-1 window"] < full
    assert opts["fewer control days"] < full and opts["all three trims"] < min(
        opts["core instruments only"], opts["shorter mbp-1 window"], opts["fewer control days"]
    )
    md = markdown(est, 40.0, 40.0)
    assert "| full plan |" in md and "fits budget" in md
    # it priced the spec's 3 CPI samples for ZN, for every schema
    zn_cpi = est.samples.filter(
        (pl.col("instrument") == "ZN") & pl.col("window_id").str.starts_with("CPI")
    )
    assert zn_cpi.filter(pl.col("schema") == "mbp-1")["window_id"].n_unique() == 3
    assert not api.bought


# --- control days --------------------------------------------------------------------------------


def test_controls_avoid_release_days_holidays_and_need_tier2(cfg: Settings) -> None:
    with pytest.raises(MissingTier2Error):
        select_control_days(cfg, calendar(cfg, with_tier2=False))
    cal = calendar(cfg)
    ctrl = select_control_days(cfg, cal)
    busy = {(r["time_et"], r["date"]) for r in cal.iter_rows(named=True)}
    hol = set().union(*(us_market_holidays(y) for y in (2024, 2025, 2026)))
    for r in ctrl.iter_rows(named=True):
        assert (r["time_et"], r["date"]) not in busy
        assert r["date"] not in hol and r["date"].weekday() < 5
        assert (
            r["date"].weekday() != 3 or r["time_et"] != "08:30"
        )  # never a claims Thursday at 08:30
    counts = ctrl.group_by("time_et").len()
    assert dict(zip(counts["time_et"], counts["len"], strict=True)) == cfg.control_days.per_time
    assert ctrl.equals(select_control_days(cfg, cal))  # reproducible


def test_control_weekdays_follow_the_event_mix(cfg: Settings) -> None:
    ctrl = select_control_days(cfg, calendar(cfg)).filter(pl.col("time_et") == "14:00")
    # the only 14:00 event is a Thursday FOMC, so every 14:00 control is a Thursday
    assert set(ctrl["weekday"]) == {"Thu"}


def test_holiday_rules() -> None:
    h = us_market_holidays(2026)
    assert dt.date(2026, 4, 3) in h  # Good Friday 2026
    assert dt.date(2026, 7, 3) in h  # July 4th on a Saturday is observed Friday
    assert dt.date(2026, 11, 26) in h  # Thanksgiving


# --- quality and validation on synthetic books ---------------------------------------------


def synth(cfg: Settings, control: bool = False):  # type: ignore[no-untyped-def]
    return generate_book(cfg, Scenario("CPI", dt.date(2025, 3, 12), "08:30", 1.5, control), "ZN")


def test_clean_synthetic_window_passes(cfg: Settings) -> None:
    b = synth(cfg)
    lo, hi = cfg.synthetic.window_s
    q = check_window(
        cfg,
        b.frame,
        window_id="CPI_2025-03-12",
        instrument="ZN",
        schema="mbp-10",
        start_ns=b.t0_ns + lo * NS,
        end_ns=b.t0_ns + hi * NS,
        t0s_ns=[b.t0_ns],
        release_date=dt.date(2025, 3, 12),
    )
    assert q.status == "ok", q.reasons
    assert q.crossed_book_rows == 0 and q.max_gap_core_s < 1


def test_quality_rejects_crossed_books_and_gaps_at_the_release(cfg: Settings) -> None:
    b = synth(cfg)
    lo, hi = cfg.synthetic.window_s
    kw = dict(
        window_id="w",
        instrument="ZN",
        schema="mbp-10",
        start_ns=b.t0_ns + lo * NS,
        end_ns=b.t0_ns + hi * NS,
        t0s_ns=[b.t0_ns],
        release_date=dt.date(2025, 3, 12),
    )
    crossed = b.frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 10)
        .then(pl.col("ask_px_0") + 5)
        .otherwise(pl.col("bid_px_0"))
        .alias("bid_px_0")
    )
    assert check_window(cfg, crossed, **kw).status == "reject"
    ts = b.frame["ts_event"]
    gap = b.frame.filter((ts < b.t0_ns - 20 * NS) | (ts > b.t0_ns + 20 * NS))
    q = check_window(cfg, gap, **kw)
    assert q.status == "reject" and "around the release" in q.reasons[0]
    dup = pl.concat([b.frame, b.frame.head(3)]).sort("ts_event")
    q = check_window(cfg, dup, **kw)
    assert q.duplicate_records == 3 and q.status == "warn"
    assert check_window(cfg, b.frame.head(0), **kw).status == "reject"


def test_roll_week_flag_uses_the_definition_expiry(cfg: Settings) -> None:
    b = synth(cfg)
    framed = b.frame.with_columns(pl.lit(42).alias("instrument_id"))
    lo, hi = cfg.synthetic.window_s
    defs = pl.DataFrame(
        {
            "instrument_id": [42],
            "raw_symbol": ["ZNM5"],
            "tick_size": [1 / 64],
            "expiration_ns": [et_to_utc_ns(dt.date(2025, 3, 31), "12:00")],
        }
    )
    q = check_window(
        cfg,
        framed,
        window_id="w",
        instrument="ZN",
        schema="mbp-1",
        start_ns=b.t0_ns + lo * NS,
        end_ns=b.t0_ns + hi * NS,
        t0s_ns=[b.t0_ns],
        release_date=dt.date(2025, 3, 12),
        definitions=defs,
    )
    assert q.roll_week and q.raw_symbols == ["ZNM5"] and q.status == "warn"


def test_spike_check_confirms_releases_and_not_controls(cfg: Settings) -> None:
    ev = spike_ratio(cfg, synth(cfg).frame, synth(cfg).t0_ns)
    ctl_book = synth(cfg, control=True)
    ctl = spike_ratio(cfg, ctl_book.frame, ctl_book.t0_ns)
    assert ev > cfg.validation.min_spike_ratio > ctl
    table = validate_events(
        cfg,
        pl.DataFrame(
            {
                "event_id": ["A", "A", "B", "C"],
                "instrument": ["ZN", "ES", "ZN", "ZN"],
                "ratio": [1.2, 6.0, 1.1, float("nan")],
            }
        ),
    )
    verdict = dict(zip(table["event_id"], table["confirmed"], strict=True))
    assert verdict == {"A": True, "B": False, "C": False}


def test_mbp_conversion_reads_byte_chars_and_ticks() -> None:
    px = lambda p: round(p / 1e-9)  # noqa: E731
    dtype = [
        ("ts_event", "u8"),
        ("sequence", "u4"),
        ("instrument_id", "u4"),
        ("action", "S1"),
        ("side", "S1"),
        ("price", "i8"),
        ("size", "u4"),
        ("bid_px_00", "i8"),
        ("ask_px_00", "i8"),
        ("bid_sz_00", "u4"),
        ("ask_sz_00", "u4"),
    ]
    rec = np.zeros(2, dtype=dtype)
    rec["action"], rec["side"] = [b"A", b"T"], [b"B", b"A"]
    rec["price"] = px(112.0)
    rec["bid_px_00"] = px(112.0)
    rec["ask_px_00"] = [px(112.015625), dbio.UNDEF_PRICE]
    f = dbio.mbp_to_frame(rec, 1 / 64)
    assert f["action"].to_list() == [0, 3] and f["side"].to_list() == [1, -1]
    assert f["bid_px_0"].to_list() == [7168, 7168] and f["ask_px_0"].to_list() == [7169, NO_PRICE]
