# Price reaction and first mover (`analysis/reaction.py`)

## Questions

How far does each market move after a release, which market moves first, and do the others follow?

## Jump sizes

The absolute mid move from the pre-release level at +1 s, +10 s, +1 min, +5 min and +30 min (and +2 h once long ohlcv panels exist), in ticks and in **volatility units**: the move divided by (that event's pre-release one-second volatility × √horizon). Volatility units make a 3-tick ZN move and a 12-tick ES move comparable: both ask "how many normal minutes of movement happened in one go?"

## First mover

For each event and instrument: the first moment after t0 when the mid is more than **K ticks** from its pre-release level.

* **K** comes from control days: the 99th percentile of absolute 100 ms mid changes on ordinary mornings, never below half a tick. So "a move" means "bigger than almost any 100 ms move on a normal day", per instrument.
* **Exact timestamps.** The fine grid has 100 ms steps, too coarse for "N ms ahead". So panel build also saves every mid change near the release at its exact exchange timestamp, and first moves are measured from those.
* Instruments are ranked by their median first-move time, and each one's median difference from ZN is reported with a bootstrap interval over the events where both moved.

**Clock caveat.** CBOT (Treasuries), CME (ES, FX), COMEX (gold) and NYMEX (oil) products run on different matching engines, each stamping its own time. Millisecond differences across venues mix real lead-lag with engine and clock differences. Within one venue (ZT vs ZN) the comparison is cleaner.

## Lead-lag

The correlation of 100 ms returns of ZN at time t with each other instrument at t + lag, for lags from -1 s to +1 s, in the first 60 s after the release, averaged across events with by-event bands. A peak at a positive lag means the other market follows ZN.

## Checked against known answers

The synthetic generator gives each instrument a known reaction delay (ZN 0 ms, ZT 5 ms, ES 12 ms, GC 40 ms). The analysis recovers the order exactly and the gaps to the millisecond. Jumps one second after the release match the planted coefficient × surprise within a couple of ordinary ticks.

**A bug the tests caught:** returns for lead-lag were first computed after cutting the data to the 60 s window, so the return into t0 (which carries a jump at exactly t0) was empty and ZN's jump vanished. Returns are now computed on the full series first. At 100 ms resolution, ES (jumping at +12 ms) lands in the bin after ZN (jumping at exactly t0), so the lead-lag peak is correctly at +100 ms: a reminder that the grid sets the resolution, which is why first moves use exact timestamps.
