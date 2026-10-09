# Reaction function (`analysis/regression.py`, `stats/regression.py`)

## The question

How much does each market move per unit of economic surprise?

    return(t0 -> t0 + h) = alpha + beta * standardised surprise + error

estimated for each event type (CPI, NFP, FOMC), instrument and horizon h (+1 s to +30 min, and +1 h or +2 h where the data reach). **beta** is "ticks (or basis points of price) per one-standard-deviation surprise".

## Surprises

* **CPI and NFP:** your `surprises.csv`, primary variable (core CPI m/m, payrolls change), standardised over the full sample.
* **FOMC:** there is no consensus number, so the surprise is market-implied: the ZT price change from 10 min before to 20 min after the statement, standardised across meetings and signed so that positive means hawkish (ZT down). ZT's own FOMC regression is then mechanical (it is regressed on itself), so it is flagged and left out of the significance count. Over a 30-minute window ZT also picks up ordinary price noise; in synthetic tests with realistic noise the implied surprise correlated only about 0.4 with the true one, which is a real limitation of this measure. The spec's SR3 (SOFR futures) alternative would be cleaner if budget allows.

## Statistics

* **HC3 robust standard errors:** residual variance differs across events (big-surprise days are noisier), and HC3 is the safest heteroskedasticity-robust choice in small samples.
* **Bootstrap by event** as a check on the beta interval.
* **Asymmetry:** separate betas for positive and negative surprises (do markets react more to bad news?).
* **Continuation vs reversal:** regress the move from +1 min to +30 min on the first-minute move. A positive slope means the first move keeps going; negative means it partly reverses.
* **Multiple testing:** about 9 instruments x 6 horizons x 3 event types is over 150 tests; at a 5% level some would look significant by luck. Benjamini-Hochberg false discovery rate control decides which results survive, and the report says so for every number.
* **Sign check:** each result is compared with the textbook direction (a hotter inflation print should lower Treasury futures prices, for example). A significant result with an unexpected sign is flagged for investigation before reporting: usually it means a data or alignment problem, not a discovery. The expected signs are config, not code, and some are deliberately "no expectation" (payrolls and equities, crude oil).

## Units

Returns in ticks and in basis points of price. For Treasury futures, an **approximate** yield change: dy (bp) is about -(dP/P) / D x 10,000, using a rough modified duration per contract from config. It ignores convexity and changes in the cheapest-to-deliver bond, so it is labelled approximate everywhere.

## Checked against known answers

On synthetic data the regressions recover the planted coefficients (ZN -4, ZT -6, ES -8, GC -10 ticks per standard deviation) for CPI and NFP, every one surviving FDR with the expected sign; flipping an expected sign in config makes the check flag it.

## Caveats

* About 16 to 25 events per type: betas will be imprecise; the report gives n and confidence intervals with every number.
* Consensus figures are compiled by hand from news sources; the ALFRED cross-check only verifies the actual figures, not the consensus.
