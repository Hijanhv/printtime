# Practical guidance: execution cost and the surprise strategy (`analysis/execution.py`, `analysis/strategy.py`)

## When is it safe to trade? (spec 8.1)

For ZT, ZN and ES (the instruments with 10-level data), at every second from 5 minutes before to 15 minutes after the release, the cost of an immediate market order of 1, 5, 10 or 25 contracts is found by **walking the actual book** at that second: take the best level, then the next, until the order is filled, and compare the average fill price with the mid.

* Cost is reported per contract in ticks, averaged over buying and selling, plus the fee converted to ticks with the contract's multiplier (read from Databento's definition records: one ZN tick is 1/64 of a point on a $100,000 contract, $15.625).
* If the order is bigger than all 10 displayed levels, the cost is **left undefined and counted**, never extrapolated: we do not know what lies beyond the displayed book.
* "Normal" is the median cost on control days at the same clock time. The headline is the number of seconds after the release until the release-day cost is back within 1.25 times normal and stays there for 10 seconds.

## Does trading the surprise make money? (spec 8.2)

A deliberately simple rule: at +1, +5, +30 or +60 s, buy or sell one contract in the direction the surprise implies, if the surprise is big enough; exit at +5 min, +30 min or +2 h. Entry and exit cross the spread at the book prevailing at that second, and the fee is paid both ways.

**Strictly forward in time.** For each event, everything the rule uses comes from earlier events only:

* the surprise's scale (standard deviation of earlier surprises);
* the sign mapping (the sign of the slope of earlier returns on earlier surprises: does a hot CPI push this market down?);
* the threshold (0, 0.5 or 1 standard deviation, whichever had the best net P&L on earlier events).

The first 8 events of each type are training only. A property-based test changes the current and all later events and checks that every earlier decision stays identical.

**Two comparisons, always shown:**

* a **perfect-sign upper bound**, which knows the realised direction of each move. It is impossible in practice; it shows how much the costs leave in the best case;
* a **random direction** baseline, averaged over 1,000 draws, which shows what pure costs do.

**FOMC is excluded,** and this is a design decision worth explaining in an interview: the spec's FOMC surprise is measured from ZT's move up to 20 minutes after the statement. A trade at +1 s cannot know that number, so using it would be look-ahead.

## Honest expectations

With about 24 CPI and 24 NFP releases, minus 8 training events each, there are about 16 out-of-sample trades per type. A few lucky or unlucky events can swing the average by several ticks, so the report gives the bootstrap interval, the hit rate and the worst event, and says plainly that the result is noisy. In synthetic tests the rule correctly shows no edge: prices jump at t0, so entering a second later there is nothing left to catch, and single settings with 4 trades swung by several ticks either way by chance alone.
