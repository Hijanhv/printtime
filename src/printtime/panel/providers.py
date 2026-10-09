"""Where panel input comes from: real Databento files, or the synthetic generator.

Both return the same `Book` (canonical frame + tick size), so every step after
this one runs unchanged on synthetic data in tests and CI and on real data in
research runs.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import polars as pl

from printtime.config import Settings
from printtime.data import databento_io as dbio
from printtime.data.quality import WindowQuality, check_window
from printtime.log import get_logger
from printtime.panel.build import Book, Stage
from printtime.synthetic.generator import Scenario, generate_book

log = get_logger(__name__)
NS = 1_000_000_000


def stages_from(
    calendar: pl.DataFrame, controls: pl.DataFrame | None, tier: list[str]
) -> list[Stage]:
    """Alignment points: every event stage in the calendar plus every control window."""
    out = [
        Stage(r["event_id"], r["event_type"], r["stage"], r["time_et"], int(r["t0_utc_ns"]))
        for r in calendar.filter(pl.col("event_type").is_in(tier)).iter_rows(named=True)
    ]
    if controls is not None:
        for r in controls.iter_rows(named=True):
            out += control_stages(r["control_id"], r["time_et"], int(r["t0_utc_ns"]))
    return sorted(out, key=lambda s: (s.t0_ns, s.window_id))


# Extra alignment points inside a control window, so every event time has a
# matching control: FOMC press conferences are at 14:30, and the 14:00 control
# window (-30 min to +60 min) already covers 14:30 at no extra data cost.
EXTRA_CONTROL_TIMES = {"14:00": [("14:30", 30 * 60)]}


def control_stages(control_id: str, time_et: str, t0_ns: int) -> list[Stage]:
    out = [Stage(control_id, "CONTROL", "control", time_et, t0_ns)]
    for extra_time, shift_s in EXTRA_CONTROL_TIMES.get(time_et, []):
        stage = f"control_{extra_time.replace(':', '')}"
        out.append(Stage(control_id, "CONTROL", stage, extra_time, t0_ns + shift_s * NS))
    return out


@dataclass
class SyntheticProvider:
    """Books from the generator, keyed by window_id. For tests, CI and demos only."""

    cfg: Settings
    scenarios: dict[str, Scenario]
    _cache: dict[tuple[str, str], Book] = field(default_factory=dict)

    def book(self, window_id: str, instrument: str) -> Book | None:
        if instrument not in self.cfg.synthetic.instruments or window_id not in self.scenarios:
            return None
        key = (window_id, instrument)
        if key not in self._cache:
            b = generate_book(self.cfg, self.scenarios[window_id], instrument)
            mult = self.cfg.synthetic.instruments[instrument].multiplier
            self._cache[key] = Book(b.frame, b.tick_size, "synthetic", mult)
        return self._cache[key]


@dataclass
class DatabentoProvider:
    """Books from cached Databento files, with quality checks recorded as it goes."""

    cfg: Settings
    stages: list[Stage]
    schema: str = "mbp-1"
    reports: list[WindowQuality] = field(default_factory=list)

    def _definitions(self, instrument: str, date: dt.date) -> pl.DataFrame | None:
        path = self.cfg.paths.raw / "definition" / instrument / f"{date.isoformat()}.dbn.zst"
        return dbio.definitions_frame(dbio.read_dbn(path)) if path.exists() else None

    def book(self, window_id: str, instrument: str) -> Book | None:
        path = self.cfg.paths.raw / self.schema / instrument / f"{window_id}.dbn.zst"
        if not path.exists():
            return None
        stages = [s for s in self.stages if s.window_id == window_id]
        t0s = [s.t0_ns for s in stages]
        date = dt.datetime.fromtimestamp(min(t0s) / NS, dt.UTC).date()
        defs = self._definitions(instrument, date)
        if defs is None or defs.height == 0:
            log.error("missing_definition", instrument=instrument, date=date.isoformat())
            return None
        ticks = sorted(set(round(t, 12) for t in defs["tick_size"].to_list()))
        if len(ticks) != 1:
            raise ValueError(f"{instrument} {date}: definitions disagree on tick size {ticks}")
        records = dbio.read_dbn(path)
        frame = (
            dbio.mbp_to_frame(records, ticks[0])
            if self.schema.startswith("mbp")
            else dbio.ohlcv_to_frame(records, ticks[0])
        ).sort("ts_event", maintain_order=True)
        w = getattr(self.cfg.databento.windows, self.schema.replace("-", "_"))
        q = check_window(
            self.cfg,
            frame,
            window_id=window_id,
            instrument=instrument,
            schema=self.schema,
            start_ns=min(t0s) + w[0] * NS,
            end_ns=max(t0s) + w[1] * NS,
            t0s_ns=t0s,
            release_date=date,
            definitions=defs,
        )
        q.write(self.cfg.paths.data_quality)
        self.reports.append(q)
        if q.status == "reject":
            return None
        mults = [m for m in defs["multiplier"].to_list() if m == m]
        return Book(frame, ticks[0], self.schema, mults[0] if mults else float("nan"))


def provider_for(cfg: Settings, schema: str = "mbp-1") -> SyntheticProvider | DatabentoProvider:
    """Book source for analyses: the synthetic study in a synthetic run, else Databento files."""
    path = cfg.paths.calendar / "synthetic_scenarios.parquet"
    if path.exists():
        rows = pl.read_parquet(path).iter_rows(named=True)
        scen = {
            r["window_id"]: Scenario(
                r["event_type"], r["date"], r["time_et"], r["surprise_sd"], r["control"]
            )
            for r in rows
        }
        return SyntheticProvider(cfg, scen)
    from printtime.calendar.events import load_calendar

    ctrl_path = cfg.paths.calendar / "control_days.csv"
    ctrl = pl.read_csv(ctrl_path, try_parse_dates=True) if ctrl_path.exists() else None
    return DatabentoProvider(cfg, stages_from(load_calendar(cfg), ctrl, cfg.tier(1)), schema=schema)
