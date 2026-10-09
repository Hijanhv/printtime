# Event-time panels (`panel/`)

## What they are

The raw data are millions of order-book updates at irregular times. To compare events, everything is put on the same clock, measured from the release time t0:

| Grid | Steps | Range | Used for |
|---|---|---|---|
| fine | 100 ms | -60 s to +120 s | which market moves first, jump sizes |
| coarse | 1 s | -30 min to +60 min | liquidity dynamics, recovery times |
| long | 1 s ohlcv bars | -2 h to +4 h | longer price reactions |

At each grid point the panel records the **book in force at that instant** (the last update at or before it): best bid/ask price and size, spread in ticks, mid, and depth summed over 1/3/5/10 levels. It also records what happened **since the previous grid point**: book updates, trades, traded volume, and signed volume (buyer-initiated minus seller-initiated).

**Baselines** are the median of each measure from -30 min to -10 min, so effects can be expressed as "% of normal for this event".

## Why "at or before", and the one exception

A grid point only ever sees updates at or before its own time, which is what makes the panel safe for anything predictive. A property-based test cuts the data at random points and checks that no grid point up to the cut changes.

The exception is `mid_move`, the mid minus the last mid before t0. Before t0 it is measured against a price that lies in the future of those grid points. That is deliberate: it describes the path relative to the pre-release level, as the spec asks. It is never used to predict anything, and from t0 onward it is causal like everything else. The test encodes exactly that rule.

## The frame-provider design

Panels are built from a `Book` (canonical frame plus tick size), which comes from a provider:

* `DatabentoProvider` reads cached DBN files, takes the tick size from the `definition` schema (never hard-coded), runs the quality checks, and skips rejected windows;
* `SyntheticProvider` draws books from the generator.

So the whole pipeline, and every analysis after it, runs end to end on synthetic data in CI, where the right answers are known, and runs unchanged on real data.

`printtime panel build --synthetic 10` builds a full synthetic study (CPI, NFP and FOMC events plus control days, with known surprises) under `data/synthetic_run/`. Synthetic outputs never go into `reports/`.

## Bugs the tests caught

* **NaN in medians:** grid points before the first update have no book (NaN). Polars sorts NaN above every number, so a plain median was biased upward (20 became 30 in the hand-built test). Baselines now treat NaN as missing.
* **Counting intervals:** the interval before the first grid point is unknown, so its count is NaN rather than a misleading zero.

## Storage

One Parquet file per window and stage, per grid, read lazily with `pl.scan_parquet`, so a full study never has to be in memory at once. Every run writes a manifest with its full config and the git commit hash next to its outputs (spec 9.1).

## Sanity plots

For one event of each type: mid move, top-of-book depth as % of baseline, and spread, for every instrument, from -5 min to +15 min. On synthetic data they show the planted effects exactly where they should be: depth falling about 20 s before t0 and bottoming at t0, spreads blowing out at t0, jumps at t0.
