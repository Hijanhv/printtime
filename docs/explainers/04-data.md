# Data: cost plan, downloads, control days, quality, validation (`data/`)

## Protecting the budget

Databento credits are shared with the other project and capped at $40 here. Three guards stand between the code and the credit, adapted from Halftick:

1. **Cache first.** Every (schema, instrument, window) is one `.dbn.zst` file in `data/raw/`; if it exists it is never requested again.
2. **Quote before buying.** `metadata.get_cost` is a free metadata call made before every purchase.
3. **A spend ledger** (`data/raw/spend_ledger.json`) records every purchase; any request that would push total spend past `databento.budget_usd` is refused. Nothing is bought without `--yes`.

## The cost plan (spec 3.2)

`printtime data plan-costs` prices samples and extrapolates before anything is bought:

* the spec's sample: 3 CPI events for ZN, for every schema (ohlcv-1s, mbp-1, mbp-10);
* plus one sample window for every other instrument, because message rates differ a lot between markets (ES carries far more book updates than ZF), so extrapolating from ZN alone would mis-price them;
* then the full plan, assuming cost grows in proportion to window length, and four trimmed alternatives: core instruments only, a shorter mbp-1 window, fewer control days, and all three together.

The table shows cost, billable size (uncompressed, an upper bound on disk use) and whether each option fits the budget. Every quote is free.

**One saving built in:** an FOMC statement (14:00) and press conference (14:30) would need two windows that overlap by 90%. They share one window instead.

## Control days

08:30 and 14:00 ET are busy even without news, so every liquidity effect is measured against control windows at the same clock time. A day qualifies when:

* no Tier 1 or Tier 2 release is scheduled at that time (jobless claims come out every Thursday at 08:30, so the selector refuses to run without Tier 2 dates in the calendar);
* it is not a weekend or a U.S. market holiday (computed by rule, including Good Friday from the Easter date) and not in an optional exclusion file (for FOMC minutes days, for example).

Weekdays follow the mix of the Tier 1 events being studied, so Friday-heavy payrolls get Friday-heavy controls. **A bug worth remembering:** the first version matched the weekday mix of *all* 08:30 releases, which is dominated by weekly claims Thursdays, and every Thursday is busy, so it found 2 controls instead of 30. It now matches Tier 1 events only and refills any weekday that runs short from the other weekdays.

## Quality checks (spec 3.5)

Per window and instrument: timestamps or sequence numbers going backwards, duplicate records, crossed and locked books, impossible sizes, the longest silence anywhere in the window and in the core minutes around the release, late starts and early ends, a change of front contract inside the window, and a roll-period flag (release within N days of the front contract's expiry, a documented heuristic).

A window is **rejected** for no data, a crossed book, or a gap longer than 5 s around the release; **warned** for the rest; every count is written to `reports/data_quality/`.

Sequence gaps are not counted as errors: Databento sequence numbers belong to the exchange channel, which carries many instruments, so a single-symbol stream skips numbers all the time. Only decreasing sequence numbers are a problem.

## Did the release really happen at t0?

Releases are sometimes moved (government shutdowns, for example). If the calendar is wrong, every result for that event is wrong. So each event is checked against the market: price moves in the first minute after t0 must be at least 3 times the per-second movement in the quiet stretch from t0 - 10 min to t0 - 1 min, in ZN or ES. Events that fail are listed for checking by hand, never silently dropped.

## Alternatives considered

* **Pricing every request exactly before planning:** possible (quotes are free) but means about 3,000 API calls; the spec's sample-and-extrapolate approach plus per-instrument ratios is accurate enough to choose between options, and `data download` prices every request exactly before buying.
* **Hard-coded holiday lists:** rules are short, testable and do not expire.
