# Calendar and surprises (`calendar/`)

## What it does

Builds the list of events and their exact release times (t0), and turns your hand-collected forecasts into standardised surprises.

| File | Job |
|---|---|
| `fred.py` | Asks FRED for its list of releases, finds "Consumer Price Index" and "Employment Situation" **by name**, then fetches every date each was published in the sample |
| `events.py` | Builds `releases.csv` (one row per release timestamp, t0 in UTC); creates and validates `fomc_dates.csv` |
| `surprises.py` | Creates and validates `surprises.csv`; computes standardised surprises |
| `alfred.py` | Rebuilds each published figure from ALFRED's archive and flags typed values that do not match |
| `timezones.py` | New York release time to UTC, correct across daylight saving |

## Why release IDs are looked up, not hard-coded

The spec forbids hard-coding. Looking up by name also guards against using the wrong release: an exact name match wins, a unique partial match is accepted, and anything ambiguous (for example "Price Index", which matches several releases) stops with an error listing the candidates.

## FOMC

There is no FRED release for FOMC statements, so you supply the dates from federalreserve.gov. Each meeting gives two timestamps: the 14:00 statement and the 14:30 press conference, treated as two stages of one event because markets often react twice.

## Surprises: why a validator

`surprises.csv` is typed by hand from news reports, about 48 releases with several variables each. Hand-typed data has predictable mistakes, and one bad row can flip a regression coefficient. The validator rejects: missing actual or consensus, numbers that do not parse, wrong units, duplicate rows, a variable that belongs to a different release, FOMC rows (no consensus exists), event IDs that do not match the date, dates that are not real releases in `releases.csv`, and missing source links. It warns on values outside a plausible range (core CPI at +3.0% instead of +0.3%) and on releases that have no row yet.

## The standardised surprise

    surprise = actual - consensus
    z = surprise / standard deviation of surprises for that variable

This is the Balduzzi, Elton & Green (2001) convention. It puts CPI (measured in tenths of a percent) and payrolls (thousands of jobs) on the same scale, so "a one-standard-deviation surprise" means the same thing for both.

Two versions:

* **Full sample** (used by the reaction-function regressions): the scale uses all events. This is standard in the literature, and fine for measuring *how much* prices react.
* **Expanding** (used by the trading strategy): the scale for each event uses only earlier events, so the strategy never benefits from information it could not have had. A test changes later events and checks that earlier z values do not move.

Two details caught by tests: in binary floating point 0.4 - 0.3 and 0.3 - 0.2 are not exactly equal, so surprises are rounded to 10 decimals before use; and if all earlier surprises are identical, the scale is zero and z is reported as undefined rather than as an enormous number.

## The ALFRED cross-check

ALFRED stores every published version ("vintage") of a series. A release on date D creates a vintage dated D. The check takes that vintage, finds the newest month in it (the month being announced), and rebuilds the headline figure from that month and the previous month **as they both stood on D**. That matters because releases usually revise the previous month, and the published month-over-month change uses the revised value. The rebuilt figure is rounded like the official one (0.1% or 1,000 jobs) and compared with what you typed. A mismatch usually means a typo, the wrong month, or a release that moved.

## What could go wrong

* FRED could rename a release; the name lookup would then fail loudly rather than pick the wrong one.
* Consensus figures differ slightly between news sources. Record the source link for every row so choices can be audited.
* Rescheduled releases (for example after a government shutdown) are caught twice: the date will not match FRED's calendar, and in Phase 2 the market will show no volatility spike at the scheduled time.

## Alternatives considered

* **fredapi for release dates:** it does not cover the release-dates endpoint, so the spec uses httpx directly.
* **Comparing to the latest ALFRED values instead of the vintage on the release day:** wrong, because later revisions would make correctly typed figures look like mismatches.
