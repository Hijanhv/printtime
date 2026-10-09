# Release replay dashboard (`replay/live.py`, `monitoring/`, `docker-compose.yml`)

## What it does

`printtime replay` plays back one release window (by default 5 minutes before to 10 minutes after) for every instrument, merged in exchange-time order, at market speed, and serves Prometheus metrics per instrument:

| Metric | Meaning |
|---|---|
| `printtime_depth_pct_of_baseline` | best bid + best ask size, as % of normal pre-release depth |
| `printtime_spread_ticks` | bid-ask spread |
| `printtime_mid_move_ticks` | mid minus the last mid before the release |
| `printtime_trades_per_second` | trades in the last second |
| `printtime_seconds_from_release` | the replay clock |

Grafana shows all instruments side by side, so you can watch liquidity drain into the release, prices jump, spreads blow out and depth come back.

## Alerts (`monitoring/alerts.yml`)

The spec's three, with thresholds from config: spread above 3 ticks, depth below 25% of baseline, and no data for more than 5 seconds during a replay; plus "metrics endpoint down". During a real or synthetic release the first two fire for real, which is the point of the demo.

## Baseline, and a bug worth remembering

Depth is shown as % of the depth before the release, from -30 min to -10 min when the data reach back that far. The first fallback used "the first minute of the replay" when they did not, but a replay that starts close to the release already contains the withdrawal in that minute, so the baseline came out too low and every percentage too high. A test caught it. The fallback now uses only data from at least a minute before t0, and reports no percentage rather than a contaminated one.

## Running it

```bash
docker compose up --build        # Grafana at http://localhost:3000, Prometheus at :9090
```

With no data the app replays a synthetic CPI release; with real data set `PRINTTIME_ARGS="replay --window CPI_2025-03-12 --metrics-port 8000 --loop"`. Without Docker: `uv run printtime replay --synthetic --metrics-port 8000`.

The image is a two-stage uv build that ships only the virtual environment and runs as a non-root user (same design as Halftick).

## Status

Tested: the replay engine, metrics, pacing and baselines, plus a live scrape of the metrics endpoint during a replay. The compose file validates. The full Docker stack has not been run end to end on the development machine, which lacks the free disk the images need.
