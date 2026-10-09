"""Download plan and cost planning (spec 3.2).

The plan lists every request: event windows and control windows, for each
schema and instrument. Before anything is bought, `plan_costs`:

1. prices `plan.sample_events` events of `plan.sample_event_type` for the
   reference instrument (ZN), for each schema, as the spec requires;
2. prices one sample window for every other instrument, because message
   rates differ a lot between markets and ZN alone would mis-price them;
3. extrapolates to the full plan, scaling by window length, and shows the
   trimming options side by side with the budget.

Every quote is a free metadata call; nothing is bought here.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, replace

import polars as pl

from printtime.config import Settings
from printtime.data.databento_io import Request, make_request
from printtime.log import get_logger

log = get_logger(__name__)
NS = 1_000_000_000
DAY_NS = 86_400 * NS

SCHEMA_WINDOWS = {"ohlcv-1s": "ohlcv_1s", "mbp-1": "mbp_1", "mbp-10": "mbp_10"}


@dataclass(frozen=True)
class Window:
    """One market window: an event (all its stages) or a control day."""

    window_id: str
    kind: str  # event type, or CONTROL
    first_t0_ns: int
    last_t0_ns: int
    date: dt.date


@dataclass(frozen=True)
class PlanSettings:
    instruments: tuple[str, ...]
    mbp10_instruments: tuple[str, ...]
    windows_s: dict[str, tuple[int, int]]
    controls: dict[str, int]


def plan_settings(cfg: Settings) -> PlanSettings:
    w = cfg.databento.windows
    return PlanSettings(
        instruments=tuple(cfg.instruments),
        mbp10_instruments=tuple(k for k, v in cfg.instruments.items() if v.mbp10),
        windows_s={"ohlcv-1s": w.ohlcv_1s, "mbp-1": w.mbp_1, "mbp-10": w.mbp_10},
        controls=dict(cfg.control_days.per_time),
    )


def event_windows(calendar: pl.DataFrame, tier: list[str]) -> list[Window]:
    """Group calendar rows by event: an FOMC statement and press conference share one window."""
    out = []
    df = calendar.filter(pl.col("event_type").is_in(tier))
    for (eid, etype), g in df.group_by(["event_id", "event_type"], maintain_order=True):
        ns = g["t0_utc_ns"].to_list()
        d = g["date"][0]
        out.append(
            Window(
                str(eid),
                str(etype),
                min(ns),
                max(ns),
                d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d)),
            )
        )
    return sorted(out, key=lambda w: w.first_t0_ns)


def control_windows(controls: pl.DataFrame) -> list[Window]:
    rows = controls.sort("t0_utc_ns").rows(named=True)
    return [
        Window(r["control_id"], "CONTROL", r["t0_utc_ns"], r["t0_utc_ns"], r["date"]) for r in rows
    ]


def requests_for(cfg: Settings, windows: list[Window], ps: PlanSettings) -> list[Request]:
    reqs: list[Request] = []
    for w in windows:
        for schema, (lo, hi) in ps.windows_s.items():
            insts = ps.mbp10_instruments if schema == "mbp-10" else ps.instruments
            for inst in insts:
                r = make_request(cfg, inst, schema, w.window_id, w.first_t0_ns, (lo, hi))
                reqs.append(replace(r, end_ns=w.last_t0_ns + hi * NS))
        for inst in ps.instruments:
            day0 = (
                w.first_t0_ns // DAY_NS
            ) * DAY_NS  # definitions are published at the start of each UTC day
            reqs.append(
                Request(
                    inst,
                    "definition",
                    w.date.isoformat(),
                    day0,
                    day0 + DAY_NS,
                    cfg.instruments[inst].databento_symbol,
                )
            )
    # Definitions are per instrument per day; several windows on one day share one file.
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for r in reqs:
        key = (r.instrument, r.schema, r.window_id)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


Quote = Callable[[Request], float]


@dataclass
class CostEstimate:
    table: pl.DataFrame  # one row per option
    detail: pl.DataFrame  # full plan, per schema and instrument
    samples: pl.DataFrame  # every quote actually made


def _duration_s(r: Request) -> float:
    return (r.end_ns - r.start_ns) / NS


def plan_costs(
    cfg: Settings,
    events: list[Window],
    controls: list[Window],
    quote: Quote,
    size: Quote | None = None,
) -> CostEstimate:
    """Price samples, extrapolate the full plan and each trimming option."""
    ps = plan_settings(cfg)
    ref = cfg.plan.reference_instrument
    sample_events = [w for w in events if w.kind == cfg.plan.sample_event_type][
        : cfg.plan.sample_events
    ]
    if not sample_events:
        raise ValueError(f"no {cfg.plan.sample_event_type} events in the calendar to price")

    sample_rows = []

    def price(inst: str, schema: str, w: Window) -> tuple[float, float, float]:
        lo, hi = ps.windows_s[schema]
        r = replace(
            make_request(cfg, inst, schema, w.window_id, w.first_t0_ns, (lo, hi)),
            end_ns=w.last_t0_ns + hi * NS,
        )
        c = quote(r)
        b = size(r) if size else float("nan")
        sample_rows.append(
            {
                "instrument": inst,
                "schema": schema,
                "window_id": w.window_id,
                "seconds": _duration_s(r),
                "cost_usd": c,
                "bytes": b,
            }
        )
        return c, b, _duration_s(r)

    # Reference instrument: cost per second of window, per schema, averaged over the sample events.
    per_sec: dict[tuple[str, str], tuple[float, float]] = {}
    for schema in ps.windows_s:
        vals = [price(ref, schema, w) for w in sample_events]
        secs = sum(v[2] for v in vals)
        per_sec[(ref, schema)] = (sum(v[0] for v in vals) / secs, sum(v[1] for v in vals) / secs)
    # Other instruments: one window each, as a ratio to the reference on the same window.
    first = sample_events[0]
    for schema in ps.windows_s:
        insts = ps.mbp10_instruments if schema == "mbp-10" else ps.instruments
        ref_c, ref_b, _ = price(ref, schema, first)
        for inst in insts:
            if inst == ref:
                continue
            c, b, _ = price(inst, schema, first)
            rc = c / ref_c if ref_c > 0 else 1.0
            rb = b / ref_b if ref_b and ref_b > 0 else 1.0
            per_sec[(inst, schema)] = (
                per_sec[(ref, schema)][0] * rc,
                per_sec[(ref, schema)][1] * rb,
            )
    def_cost = quote(
        Request(
            ref,
            "definition",
            first.date.isoformat(),
            (first.first_t0_ns // DAY_NS) * DAY_NS,
            (first.first_t0_ns // DAY_NS) * DAY_NS + DAY_NS,
            cfg.instruments[ref].databento_symbol,
        )
    )
    sample_rows.append(
        {
            "instrument": ref,
            "schema": "definition",
            "window_id": first.date.isoformat(),
            "seconds": 86400.0,
            "cost_usd": def_cost,
            "bytes": float("nan"),
        }
    )

    def estimate(option: str, ps_opt: PlanSettings) -> pl.DataFrame:
        wins = events + control_windows_from(controls, ps_opt.controls)
        rows = []
        for schema, (lo, hi) in ps_opt.windows_s.items():
            insts = ps_opt.mbp10_instruments if schema == "mbp-10" else ps_opt.instruments
            for inst in insts:
                secs = sum((w.last_t0_ns - w.first_t0_ns) / NS + (hi - lo) for w in wins)
                c, b = per_sec[(inst, schema)]
                rows.append(
                    {
                        "option": option,
                        "schema": schema,
                        "instrument": inst,
                        "windows": len(wins),
                        "cost_usd": c * secs,
                        "gb": b * secs / 1e9,
                    }
                )
        days = {(i, w.date) for w in wins for i in ps_opt.instruments}
        rows.append(
            {
                "option": option,
                "schema": "definition",
                "instrument": "all",
                "windows": len(days),
                "cost_usd": def_cost * len(days),
                "gb": 0.0,
            }
        )
        return pl.DataFrame(rows)

    trim = cfg.plan.trim
    core = tuple(i for i in ps.instruments if i in trim.core_instruments)
    options = {
        "full plan": ps,
        "core instruments only": replace(
            ps,
            instruments=core,
            mbp10_instruments=tuple(i for i in ps.mbp10_instruments if i in core),
        ),
        "shorter mbp-1 window": replace(
            ps, windows_s={**ps.windows_s, "mbp-1": trim.short_mbp_1_window}
        ),
        "fewer control days": replace(ps, controls=dict(trim.fewer_controls)),
    }
    options["all three trims"] = replace(
        options["core instruments only"],
        windows_s=options["shorter mbp-1 window"].windows_s,
        controls=dict(trim.fewer_controls),
    )
    detail = pl.concat([estimate(k, v) for k, v in options.items()])
    budget = cfg.databento.budget_usd
    table = (
        detail.group_by("option", maintain_order=True)
        .agg(pl.col("cost_usd").sum().round(2), pl.col("gb").sum().round(2))
        .with_columns((pl.col("cost_usd") <= budget).alias("fits_budget"))
    )
    return CostEstimate(
        table, detail.filter(pl.col("option") == "full plan"), pl.DataFrame(sample_rows)
    )


def control_windows_from(controls: list[Window], per_time: dict[str, int]) -> list[Window]:
    """Keep the first N control windows per clock time (controls are already randomly chosen)."""
    out: list[Window] = []
    for time_et, n in per_time.items():
        tag = "CONTROL" + time_et.replace(":", "")
        out += [w for w in controls if w.window_id.startswith(tag)][:n]
    return out


def markdown(est: CostEstimate, budget: float, remaining: float) -> str:
    lines = [
        "# Databento cost plan",
        "",
        f"Budget for Print Time: ${budget:.2f} "
        f"(remaining according to the ledger: ${remaining:.2f}).",
        "Estimates extrapolate a few priced sample windows by window length; "
        "sizes are Databento's billable (uncompressed) bytes, so disk use will be smaller.",
        "",
        "| option | estimated cost (USD) | estimated size (GB, uncompressed) | fits budget |",
        "|---|---|---|---|",
    ]
    for r in est.table.iter_rows(named=True):
        lines.append(
            f"| {r['option']} | {r['cost_usd']:.2f} | {r['gb']:.2f} | "
            f"{'yes' if r['fits_budget'] else 'no'} |"
        )
    lines += [
        "",
        "## Full plan by schema and instrument",
        "",
        "| schema | instrument | windows | cost (USD) | GB |",
        "|---|---|---|---|---|",
    ]
    for r in est.detail.sort(["schema", "cost_usd"], descending=[False, True]).iter_rows(
        named=True
    ):
        lines.append(
            f"| {r['schema']} | {r['instrument']} | {r['windows']} | "
            f"{r['cost_usd']:.2f} | {r['gb']:.3f} |"
        )
    lines += ["", f"Sample quotes made: {est.samples.height} (all free metadata calls)."]
    return "\n".join(lines) + "\n"
