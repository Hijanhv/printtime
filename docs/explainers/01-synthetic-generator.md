# Synthetic event generator (`synthetic/generator.py`)

## What it does

Draws a fake but realistic 10-level order book for each instrument around a fake release, with every effect the real analysis must find switched on and set to a **known** value:

| Effect | Setting (config `synthetic.event`) | What the book does |
|---|---|---|
| Depth withdrawal | `withdrawal_lead_s`, `depth_drop`, `depth_recovery_half_life_s` | Depth starts falling 20 s before t0, is down 80% at t0, and recovers halfway every 30 s |
| Spread blowout | `spread_blowout_ticks`, `spread_half_life_s` | 6 extra ticks right after the reaction, widening symmetrically around the mid, halving every 4 s |
| Activity burst | `activity_multiplier`, `activity_half_life_s` | Updates and trades arrive 8 times faster, decaying back to normal |
| Price jump | per instrument `jump_ticks_per_sd` | The mid moves by (coefficient × surprise) ticks in one update |
| Reaction delay | per instrument `lag_ms` | Each instrument reacts at t0 + its own lag, so "who moves first" has a known answer |
| Control window | `Scenario(control=True)` | The same clock time with none of the above |

## Why it exists

CI must never need API keys or network access, and the analysis code needs inputs where the right answer is known. If the panel builder says the jump happened 3 ms late, or the recovery analysis says depth came back in 60 s when it was set to 30 s, there is a bug. Every later phase is tested against this generator before it touches real data.

**It never produces results.** Its tick sizes and reaction coefficients are illustrative settings, labelled as such in the config. Real results come only from Databento data.

## How it works

* **Update times:** a Poisson process at the instrument's normal rate, plus a second "burst" process after the reaction whose rate decays with a half-life (drawn by *thinning*: draw candidates at the peak rate, keep each with probability equal to the decay at its time). An update is always placed exactly at the reaction time.
* **Price:** a random walk of one-tick steps, with the jump added as a single step at the reaction update.
* **Spread:** one tick normally; after the reaction, extra ticks are added half on each side, so the blowout never moves the mid. (An early version widened only the ask, which dragged the mid and hid the jump; a sanity check caught it.)
* **Depth:** each level's size is normal depth × the withdrawal factor × random noise.
* **Determinism:** each (event, instrument) gets its own seed from a hash, so the same scenario always gives the same book.

## A bug worth remembering

Timestamps are integer nanoseconds since 1970, about 1.7 × 10^18. A float64 can only represent integers that large to within about 256 ns. Adding a float offset to t0 silently moved the reaction 64 ns early, so the jump appeared *before* the reaction time. The fix is to round offsets to integer nanoseconds *before* adding t0. The same rule applies to real data: timestamps stay int64 end to end.

## Tests

The jump happens exactly at the reaction timestamp and has exactly the configured size (property-tested over random surprises and instruments). The book never crosses, levels are ordered, and timestamps never go backwards. The depth drop matches the formula. Control windows have no blowout and no jump. Activity bursts after the release. Same seed gives the same book.

## Alternatives considered

* **Replaying recorded real windows:** impossible in CI without data and keys.
* **A full queue-reactive model** (as in Halftick): more realistic microstructure, but this project's analyses work on book snapshots on a time grid, so a simpler generator with exactly controllable effects is more useful for testing.
