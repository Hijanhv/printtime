# The idea in one page

## The question

At 08:30 New York time on CPI and payrolls days, and at 14:00 on FOMC days, a number is published that changes what every bond, equity, metal, oil and FX trader believes. What happens in the futures order books in the seconds and minutes around that moment?

Market makers know the time of the release but not the number. Quoting through it is dangerous: whoever trades against them right after the print may already know which way the price is going. So, in theory:

1. **Before t0**, liquidity providers pull their orders. The book thins out and the spread may widen.
2. **At t0**, the price jumps by an amount related to how far the number is from what economists expected (the *surprise*). Activity explodes.
3. **After t0**, liquidity flows back, spreads narrow, and the price either keeps going or partly reverses.

Print Time measures each of those steps, precisely, across nine CME futures, always against **control days** (the same clock time on days with no release), because 08:30 and 14:00 are busy times anyway.

## Why it matters to a trader

* **Execution:** if you must trade 10 ZN contracts around CPI, when is it safe? What does it cost at +1 s, +5 s, +30 s compared with a normal morning?
* **Positioning:** how much does a one-standard-deviation inflation surprise move 10-year notes, the S&P and gold, and which market moves first?
* **Discipline:** does a simple "trade the surprise" rule make money after realistic costs, tested strictly forward in time? Either answer is useful if it is honest.

## How it is built

```
FRED release dates + Fed FOMC dates + your surprise sheet  ->  event calendar (t0 for each event)
Databento CME order books (budgeted, cached)               ->  quality-checked raw data
raw data aligned on a grid of seconds around t0            ->  event-time panels
panels                                                     ->  liquidity, price reaction, regressions, execution cost, strategy
results                                                    ->  REPORT.md, figures, live replay dashboard
```

## The rules it follows

* Every number comes from code that ran on real data. Consensus forecasts and actual figures come only from files you fill in, cross-checked against the official first-release values.
* Every result shows how many events it rests on. With 16 to 25 events per type, some results will only be suggestive, and the report says which.
* Liquidity effects only count relative to control days.
* No crypto or Web3 of any kind.
