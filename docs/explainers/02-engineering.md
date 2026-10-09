# Engineering (`config.py`, `secrets.py`, `calendar/timezones.py`, `cli.py`, tests, CI)

## Configuration

Every tunable number is in `config.yaml`: instruments, event times, download windows, panel grids, budget, fees, analysis settings. Pydantic validates it at load time, so a typo, an unknown key, a negative budget or a window that ends before it starts fails immediately with a clear message. Any key can be overridden from the command line: `--set databento.budget_usd=30`.

Tick sizes for real instruments are deliberately **not** in the config: the spec requires reading them from Databento's `definition` schema, so they can never drift from the exchange's values.

## API keys

Keys live in `.env`, which git ignores. `secrets.require_key("FRED_API_KEY")` returns the key or raises an error that names the missing variable but never prints a value.

## Release times and daylight saving

Releases are scheduled in New York time; data timestamps are UTC nanoseconds. New York is UTC-4 in summer and UTC-5 in winter, and the switch dates move every year, so a fixed offset would put t0 an hour wrong for part of the year. `zoneinfo` applies the right offset for each date. Tests check CPI-style 08:30 releases on both sides of the March change and FOMC-style 14:00 releases on both sides of the November change.

## Command line

```
printtime calendar build | validate        Phase 1
printtime data plan-costs | download       Phase 2
printtime panel build                      Phase 3
printtime analyze liquidity | reaction | regression | execution | strategy   Phases 4-7
printtime report                           Phase 9
printtime replay                           Phase 8
printtime synth                            synthetic windows for tests and demos
```

Commands from phases not yet built exit with "built in Phase N" instead of doing nothing silently.

## Tests and CI

pytest with Hypothesis for property-based tests. GitHub Actions runs ruff (lint and format), mypy and the tests on every push, using only the synthetic generator: no API keys, no network data. A scope-guard test checks that the config contains no crypto or Web3 terms (whole words, so "definition" is not mistaken for "defi").
