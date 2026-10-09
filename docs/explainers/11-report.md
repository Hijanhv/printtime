# The report (`report.py`)

`printtime report` writes `REPORT.md` from the tables the analyses produced. No number in it is typed by hand, and it follows the spec's honesty rules mechanically:

* **Event counts everywhere.** Every summary line carries the number of events behind it.
* **Robust vs suggestive.** A reaction-function result is labelled *robust* only if it survives the Benjamini-Hochberg correction **and** rests on at least 15 events; everything else is *suggestive*.
* **Missing is said to be missing.** An analysis that has not run shows "Not run yet" instead of anything else.
* **Limitations are part of the template,** not an afterthought: small samples, hand-compiled consensus, noisy market-implied FOMC surprises, approximate yield conversions, cross-venue clocks, no real fills, fee assumptions.
* **Release validation** (events without a volatility spike at the scheduled time) is listed at the top.

## Synthetic runs can never pass as results

`printtime --synthetic-run <command>` runs any command on the synthetic study under `data/synthetic_run/`. A synthetic report carries a "SYNTHETIC: NOT RESULTS" banner, is written under `data/synthetic_run/`, and the README results block is never updated from it (tested). Only a run on real data writes `reports/REPORT.md` and the README headline numbers.
