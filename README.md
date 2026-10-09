# Print Time

Print Time measures how CME futures order books thin out, reprice and recover around CPI, payrolls and FOMC releases, and turns that into a clear answer to when it's safe to trade again and what it costs.

[![CI](https://github.com/Hijanhv/printtime/actions/workflows/ci.yml/badge.svg)](https://github.com/Hijanhv/printtime/actions/workflows/ci.yml)

> **Status: in progress.** Phase 0 (skeleton, config, synthetic event generator, tests, CI) is built. No market data has been downloaded and there are no results yet. Every result will come from real Databento data; the synthetic generator exists only for tests.

## Questions

1. **Liquidity:** when does order-book depth disappear around a release, how wide do spreads get, and how long until liquidity recovers?
2. **Price reaction:** how much does each market move per unit of economic surprise, which moves first, and does the first move continue or reverse?
3. **Practice:** when is it safe to trade again, what does it cost to execute N contracts at each moment, and does a simple surprise-based rule make money after costs?

**Instruments** (CME Globex): ZT, ZF, ZN, ZB (Treasury notes and bonds), ES (S&P 500), GC (gold), CL (crude oil), 6E (euro), 6J (yen).
**Events:** CPI and nonfarm payrolls (08:30 ET) and FOMC (14:00 ET statement, 14:30 ET press conference), October 2024 to September 2026, compared against control days at the same clock time.

The full specification is in [PROJECT_2_SPEC.md](PROJECT_2_SPEC.md); plain-English explainers are in [docs/explainers](docs/explainers/README.md).

## Run it

```bash
uv sync
uv run pytest                                   # tests (synthetic data only)
uv run printtime synth --event CPI --date 2025-03-12 --surprise 1.0
```

Real data needs a free FRED API key and a Databento account, both in `.env` (see `.env.example`).
