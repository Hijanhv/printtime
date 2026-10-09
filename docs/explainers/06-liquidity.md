# Liquidity dynamics (`analysis/liquidity.py`, `stats/`)

## The question

Before, at and after a release: when does order-book depth disappear, how wide do spreads get, and how long does liquidity take to come back? And is any of it different from an ordinary morning at the same clock time?

## What it measures

For each group (CPI release, NFP release, FOMC statement, FOMC press conference) and instrument, always next to the matching **control group** (control days aligned at the same clock time: 08:30, 14:00, or 14:30 for the press conference):

| Measure | How |
|---|---|
| Depth curve | Top-of-book depth as % of that event's own baseline, averaged across events, second by second |
| When withdrawal starts | Per event, a change-point search on depth from -3 min to t0 (below) |
| Depth at the release | Lowest smoothed depth within 10 s of t0 |
| Spreads | Share of events with a spread wider than one tick at each second; widest spread in the first minute |
| Activity | Book updates and trades per second |
| Recovery | Seconds after t0 until depth is back to 50% and 90% of baseline, and until the spread is back to its baseline |

Every summary is a median across events with its number of events and a 95% bootstrap interval.

## Finding when withdrawal starts

Withdrawal is a ramp: depth is roughly flat, then declines into the release. The change-point search (ruptures, `clinear` model) fits two straight lines that join, flat-then-sloping, and reports where they meet. A simpler change-in-mean model would put the break in the middle of the ramp instead of at its start. The break only counts if depth then falls by at least 15 percentage points; otherwise that event shows no detectable withdrawal, and the result reports how many events had one.

## What "recovered" means

Depth is noisy. The first version declared recovery the first second depth touched the level, and the synthetic test caught it: depth built to recover to 90% in 90 s was declared recovered at 63 s on a lucky spike. Recovery now requires the level to be **reached and held for 10 seconds**, for depth and for spreads.

## Why bootstrap by event

Seconds inside one release are not independent: if depth is low at +3 s it is low at +4 s. Resampling seconds would pretend there are thousands of observations when there are about 20 events, and intervals would be far too narrow. Every interval resamples whole events. A test checks that a band built from two events reflects two events however many seconds each has.

## Checked against known answers

On synthetic data built with a 20 s lead, an 80% drop, a 30 s recovery half-life and a 4 s spread half-life, the analysis recovers: withdrawal starting about 20 s before t0, depth near 20% at t0, recovery to 50% in about 20 s and to 90% in about 90 s, spreads back in about 11 s; and on control days, no withdrawal, a one-tick spread and immediate "recovery".

## Caveats

* With about 20 events per type the medians are honest but the intervals will be wide; the report says which results are robust.
* Depth here is top of book only for most instruments (mbp-1); 10-level depth exists only for ZT, ZN and ES, over a shorter window.
* Liquidity results mean something only relative to control days, so every chart shows both.
