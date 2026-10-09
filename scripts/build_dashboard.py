"""Generate monitoring/grafana/dashboards/printtime.json.

Run: uv run python scripts/build_dashboard.py

Instrument colours match the report figures (printtime.plots.ENTITY_COLORS).
"""

from __future__ import annotations

import json
from pathlib import Path

from printtime.plots import ENTITY_COLORS

DS = {"type": "prometheus", "uid": "prometheus"}
INSTRUMENTS = ["ZT", "ZF", "ZN", "ZB", "ES", "GC", "CL", "6E", "6J"]
_ids = iter(range(1, 100))


def overrides() -> list[dict[str, object]]:
    return [
        {
            "matcher": {"id": "byName", "options": i},
            "properties": [
                {"id": "color", "value": {"mode": "fixed", "fixedColor": ENTITY_COLORS[i]}}
            ],
        }
        for i in INSTRUMENTS
    ]


def timeseries(
    title: str,
    expr: str,
    x: int,
    y: int,
    w: int,
    h: int,
    unit: str = "short",
    desc: str = "",
    thresholds: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    defaults: dict[str, object] = {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 0}}
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
        defaults["custom"] = {
            "lineWidth": 2,
            "fillOpacity": 0,
            "thresholdsStyle": {"mode": "dashed"},
        }
    return {
        "id": next(_ids),
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {"refId": "A", "datasource": DS, "expr": expr, "legendFormat": "{{instrument}}"}
        ],
        "fieldConfig": {"defaults": defaults, "overrides": overrides()},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "multi"},
        },
    }


def stat(
    title: str, expr: str, x: int, unit: str = "short", decimals: int | None = None
) -> dict[str, object]:
    d: dict[str, object] = {"unit": unit}
    if decimals is not None:
        d["decimals"] = decimals
    return {
        "id": next(_ids),
        "type": "stat",
        "title": title,
        "datasource": DS,
        "gridPos": {"x": x, "y": 0, "w": 6, "h": 4},
        "targets": [{"refId": "A", "datasource": DS, "expr": expr}],
        "fieldConfig": {"defaults": d, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "colorMode": "value",
            "graphMode": "none",
        },
    }


red = [{"color": "green", "value": None}, {"color": "red", "value": 1}]
panels = [
    stat("Seconds from release", "printtime_seconds_from_release", 0, "s", 0),
    stat("Replay active", "printtime_replay_active", 6),
    stat("Updates per second", "sum(rate(printtime_events_total[5s]))", 12, decimals=0),
    stat("Firing alerts", 'count(ALERTS{alertstate="firing"}) or vector(0)', 18),
    timeseries(
        "Top-of-book depth, % of baseline",
        "printtime_depth_pct_of_baseline",
        0,
        4,
        12,
        9,
        "percent",
        "Watch it fall into the release and recover; alert below 25%",
        [{"color": "red", "value": None}, {"color": "transparent", "value": 25}],
    ),
    timeseries(
        "Mid move from the pre-release level (ticks)", "printtime_mid_move_ticks", 12, 4, 12, 9
    ),
    timeseries(
        "Spread (ticks)",
        "printtime_spread_ticks",
        0,
        13,
        12,
        8,
        desc="Alert above 3 ticks",
        thresholds=[{"color": "transparent", "value": None}, {"color": "red", "value": 3}],
    ),
    timeseries("Trades per second", "printtime_trades_per_second", 12, 13, 12, 8),
]
dashboard = {
    "uid": "printtime-replay",
    "title": "Print Time: release replay",
    "tags": ["printtime"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "1s",
    "time": {"from": "now-10m", "to": "now"},
    "panels": panels,
}
out = Path(__file__).resolve().parents[1] / "monitoring/grafana/dashboards/printtime.json"
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out}")
