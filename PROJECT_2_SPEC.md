# Project 2 Spec: Liquidity and Price Reaction Around U.S. Macro Releases

> **For Claude Code:** This file is the source of truth for the project. Read it fully before writing code. Work one phase at a time, stop at the end of each phase, summarise what you built, and wait for my review before starting the next phase. Never invent results, numbers, dates, or economic data — every number in the report must come from code that ran on real data in this repo. **Never fill in economic consensus forecasts or actual release values from memory**; those come only from files I provide or from the official APIs listed below. **Use only the technologies in §13 (Tech Stack)**; ask before adding anything else.
>
> **Scope guard:** This is a quantitative-finance project on regulated CME futures (U.S. Treasuries, equity index, metals, energy, FX). It must contain **no crypto, Web3, blockchain, DeFi, tokens, or crypto-exchange data or APIs** of any kind — not as an instrument, a data source, a comparison, or an example. Do not suggest them as alternatives either.

---

## 1. Goal

Measure, precisely and honestly, what happens in U.S. futures markets in the minutes around major scheduled U.S. macro releases, and turn those measurements into practical trading and execution guidance.

Three questions:

1. **Liquidity:** How does the order book change before, at, and after a release? When does depth disappear, how wide does the spread get, and how long does liquidity take to recover?
2. **Price reaction:** How much does each instrument move per unit of economic surprise, which instrument moves first, and does the first move continue or reverse?
3. **Practice:** When is it safe to trade again after a release, what does it cost to execute N contracts at each moment, and does a simple surprise-based directional rule make money after realistic costs?

### Instruments (CME Globex, all in Databento dataset `GLBX.MDP3`)

| Symbol | Contract | Venue |
|---|---|---|
| ZT | 2-Year U.S. Treasury Note futures | CBOT |
| ZF | 5-Year U.S. Treasury Note futures | CBOT |
| ZN | 10-Year U.S. Treasury Note futures | CBOT |
| ZB | U.S. Treasury Bond futures | CBOT |
| ES | E-mini S&P 500 futures | CME |
| GC | Gold futures | COMEX |
| CL | WTI Crude Oil futures | NYMEX |
| 6E | Euro FX futures | CME |
| 6J | Japanese Yen futures | CME |

Optional, only if budget allows: SR3 (3-Month SOFR futures, CME) for a cleaner FOMC policy-surprise measure.

### Events

**Tier 1 (required):**
- **CPI** (Consumer Price Index), 08:30 ET.
- **NFP** (Employment Situation / nonfarm payrolls), 08:30 ET.
- **FOMC** rate decision statement, 14:00 ET (press conference at 14:30 ET treated as a second, separate timestamp within the same event).

**Tier 2 (optional, only after Tier 1 is complete):** PPI, Retail Sales, Initial Jobless Claims.

**Sample period:** October 2024 – September 2026 (about 24 months → roughly 24 CPI, 24 NFP, 16 FOMC events).

### Headline results the project should produce
- "Top-of-book depth in ZN falls by X% starting about Y seconds before CPI, and takes Z seconds to recover to half its normal level."
- "A one-standard-deviation core CPI surprise moves ZN by A ticks and ES by B points in the first minute."
- "[Instrument] reacts first, on average N ms ahead of [instrument]."
- "Executing 10 ZN contracts in the first 5 seconds after the release costs C ticks more than in normal conditions; costs are back to normal after T seconds."
- "A simple surprise-direction rule makes / does not make money after realistic costs." (Either answer is fine — it must be honest.)

### 1.1 Why this project (alignment with the target role)

This project is designed for a **Quantitative Developer** application at a global macro research firm with a live, semi-systematic futures trading operation centred on U.S. market hours. Its published research covers U.S. Treasury futures (ZT, ZF, ZN, ZB), Bund futures, gold, copper, crude oil, EUR/USD, USD/JPY and S&P 500 futures, and its team builds systems that study how markets respond to macroeconomic data releases. Every design choice should serve the requirements below.

| What the role asks for | Where this project shows it |
|---|---|
| Signals from market, order-book and time-series data | §4 event-time panels from CME order-book data |
| Whether bid/ask liquidity, volume and order flow carry information about short-horizon price action | §5 liquidity dynamics, §6 first-mover and lead–lag, §7 continuation vs. reversal |
| Tools that inform execution decisions | §8.1 execution-cost curve: "when is it safe to trade after a release?" |
| Statistical, event-driven models for directional positioning around macro releases | §7 reaction function, §8.2 surprise-direction strategy |
| Taking a hypothesis from idea → analysis → testing → diagnostics | Phases 1–9, with control days, FDR correction and honest limitations |
| Diagnostics across market conditions | Breakdowns by event type, instrument, surprise size, release vs. control days |
| Reusable research infrastructure for market and time-series data | Event-time panel builder, budgeted data pipeline, Typer CLI, Parquet/DuckDB store |
| Logging, monitoring, alerting, observability | structlog JSON logs, Prometheus metrics and alert rules, provisioned Grafana dashboard (§9.2) |
| Docker / containerised deployment | Dockerfile + docker-compose (`app`, `prometheus`, `grafana`) |
| Git, testing, careful code in a high-consequence environment | CI, property-based tests, no-look-ahead and leakage tests, type checking |
| Intellectual honesty, evaluating ideas on evidence | §12 honesty rules: control days, event counts on every result, labelled upper bounds |

The instruments in §1 deliberately mirror the firm's cross-asset coverage, restricted to contracts available on CME Globex.

---

## 2. Background literature (implement ideas from these; cite them in the report)

- Fleming & Remolona (1999), *Price Formation and Liquidity in the U.S. Treasury Market: The Response to Public Information.*
- Balduzzi, Elton & Green (2001), *Economic News and Bond Prices: Evidence from the U.S. Treasury Market.*
- Andersen, Bollerslev, Diebold & Vega (2003), *Micro Effects of Macro Announcements: Real-Time Price Discovery in Foreign Exchange.*
- Kuttner (2001), *Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market.*
- Gürkaynak, Sack & Swanson (2005), *Do Actions Speak Louder Than Words? The Response of Asset Prices to Monetary Policy Actions and Statements.*

---

## 3. Data

### 3.1 Market data — Databento
- Dataset `GLBX.MDP3`, Python client `databento`.
- Schemas:
  - `ohlcv-1s` for wide windows (−2 h to +4 h around each event), all instruments. Cheap context data.
  - `mbp-1` (top of book + trades) for −30 min to +60 min around each event, all instruments.
  - `mbp-10` (10 levels of depth) for −5 min to +15 min around each event, **ZT, ZN, ES only** (for execution-cost curves).
  - `definition` for tick size, contract symbol, expiry. **Read tick sizes from `definition` (`min_price_increment`); do not hard-code them.**
- Use `ts_event` (exchange timestamp) for all timing analysis, not `ts_recv`.
- Symbology: `stype_in="continuous"` with `<ROOT>.v.0` (front contract by volume). Record the raw contract symbol and `instrument_id` on every row. Flag events that fall in a roll week.

### 3.2 Budget rules (strict — Databento credits are limited and shared with my other project)
1. Before any download, call `client.metadata.get_cost(...)` and print the cost. Never download if cumulative spend would exceed `budget_usd` in `config.yaml` (default **$40**).
2. **Phase 2 must start with a cost-planning step:** price 3 CPI events for ZN only, extrapolate the cost of the full plan (all events × instruments × schemas × windows, plus control days), and present a table to me with options to trim (fewer instruments, shorter windows, fewer control days) **before** downloading anything else.
3. Cache every download in `data/raw/` as `.dbn.zst`; never re-download a file that exists locally.
4. API keys in `.env` (`DATABENTO_API_KEY`, `FRED_API_KEY`); `.env` in `.gitignore`; never print or commit keys.

### 3.3 Release calendar and economic data

**Release dates and times** — build `data/calendar/releases.csv` programmatically, not from memory:
- CPI and NFP dates: from the **FRED API** `fred/release/dates` endpoint. Look up the correct release IDs with the `fred/releases` endpoint (search by name: "Consumer Price Index", "Employment Situation"); do not hard-code IDs.
- FOMC dates: I will provide `data/calendar/fomc_dates.csv` from the Federal Reserve website. Create the file template with columns `date, statement_time_et, press_conference_time_et, source_url` and wait for me to fill it in.
- Releases are sometimes rescheduled (for example, during government shutdowns). Validate every event by checking that the market shows a clear volatility spike at the scheduled time; flag any event where it does not, and show me the list.

**Actual values and consensus forecasts** — `data/calendar/surprises.csv`:
- **I will fill this file in manually** from public news sources. Create the template and a validator; do not populate values yourself.
- Columns: `event_id, event_type, release_date, variable, actual, consensus, prior, unit, source_url, notes`.
- Primary surprise variable per event type:
  - CPI → core CPI month-over-month % (secondary: headline CPI MoM %).
  - NFP → nonfarm payrolls change, thousands (secondary: unemployment rate, average hourly earnings MoM %).
- Cross-check the `actual` column against first-release values from **ALFRED** (FRED's vintage database) using `fredapi` (`get_series_first_release`) for the underlying series (e.g. `CPILFESL` for core CPI index → compute MoM %, `PAYEMS` → monthly change). Report any mismatch larger than rounding (published figures are rounded to 0.1% / 1k).
- **Standardised surprise** = (actual − consensus) / standard deviation of (actual − consensus) across the sample, per variable (Balduzzi–Elton–Green convention).

**FOMC surprise (no consensus number available):** use a market-implied measure — the change in ZT price (in ticks, and converted to an approximate yield change) from 10 minutes before to 20 minutes after the statement. If SR3 data is available, also compute a Kuttner-style surprise from SR3 and compare.

### 3.4 Control days (essential for honest liquidity results)
08:30 and 14:00 ET are busy times even without a release. For every liquidity result, compare release days with **control days**: the same clock window on days **with no scheduled Tier 1 or Tier 2 release** at that time (matched by weekday where possible). Aim for ~30 control windows for 08:30 and ~15 for 14:00, subject to budget. All liquidity charts must show release days vs. control days.

### 3.5 Data quality checks (fail loudly, log counts)
Out-of-order timestamps, sequence gaps, crossed or locked books, zero/negative sizes, missing data inside an event window, duplicate records, events in roll weeks. Write a per-event data-quality report to `reports/data_quality/`.

---

## 4. Event-time alignment

- Every event has `t0` = official release time converted to UTC (handle DST correctly via `zoneinfo`, America/New_York).
- Build an **event-time panel**: for each event × instrument, resample book and trade data onto a grid relative to `t0`:
  - 100 ms grid from −60 s to +120 s (for the first-mover and jump analysis),
  - 1 s grid from −30 min to +60 min (for liquidity dynamics),
  - 1 s `ohlcv` bars from −2 h to +4 h (for longer reactions).
- Measures per grid point: best bid/ask price and size, spread (ticks), depth at levels 1/3/5/10 (where mbp-10 exists), mid-price, number of book updates, number of trades, traded volume, signed (aggressor) volume.
- For each instrument, also compute a **baseline**: the median of each measure in the window −30 min to −10 min, so effects can be expressed as % of baseline.

---

## 5. Analysis A — Liquidity dynamics

For each event type × instrument, release days vs. control days:

1. **Depth withdrawal:** average top-of-book depth (% of baseline) over event time. Detect **when withdrawal starts** using change-point detection (`ruptures`) on each event, and report the distribution of start times.
2. **Spread:** fraction of time the spread is wider than one tick, and maximum spread, by event time.
3. **Activity:** book-update rate and trade rate by event time.
4. **Recovery:** time for depth to return to 50% and 90% of baseline after `t0` (recovery half-life), and the same for spread.
5. Compare across event types (is CPI worse than NFP? is FOMC different because of the press conference?) and across instruments.

Show uncertainty: bootstrap confidence intervals **by event** (resample events, not seconds).

---

## 6. Analysis B — Price reaction and price discovery

1. **Jump size:** absolute mid-price move in ticks and in volatility-normalised units at +1 s, +10 s, +1 min, +5 min, +30 min, +2 h.
2. **First mover:** for each event, the time (ms after `t0`) of each instrument's first mid-price change larger than K ticks (K configurable; default = the instrument's 99th-percentile 100 ms move on control days). Rank instruments by median first-move time. Report clock-precision caveats (exchange timestamps, matching-engine differences between CBOT, CME, COMEX, NYMEX).
3. **Lead–lag in the first 60 s:** cross-correlation of 100 ms returns between ZN and every other instrument.

---

## 7. Analysis C — Reaction function (how much per unit of surprise?)

For each event type, instrument, and horizon h:

`return(t0 → t0+h) = α + β · standardised_surprise + ε`

- Returns in ticks and in basis points of price; for Treasury futures also provide an approximate yield-change conversion and note it is approximate.
- Robust (HC3) standard errors; bootstrap by event as a check.
- Report **β, R², and number of events** in a cross-asset table (instruments × horizons) and as a heatmap.
- **Asymmetry:** separate β for positive vs. negative surprises.
- **Continuation vs. reversal:** does the first-minute move predict the move from +1 min to +30 min and +2 h?
- **Multiple-testing control:** many instruments × horizons × event types are tested. Apply Benjamini–Hochberg false discovery rate correction and report which results survive.
- **Sign check:** confirm economically sensible signs (e.g. a positive inflation surprise should lower Treasury futures prices). If a sign looks wrong, investigate data and alignment before reporting.

---

## 8. Analysis D — Practical guidance

### 8.1 Execution-cost curve (ZT, ZN, ES; mbp-10 events)
For order sizes Q = 1, 5, 10, 25 contracts, compute the cost of an immediate market order by **walking the actual 10-level book** at each second from −5 min to +15 min:
- cost = (volume-weighted fill price − mid) in ticks per contract, plus fees (configurable, default $2.50 per contract per side),
- flag cases where Q exceeds displayed 10-level depth.

Output: cost vs. time since release, release days vs. control days, with confidence bands. Headline: **"how many seconds until execution cost returns to within 1.25× of normal."**

### 8.2 Surprise-direction strategy (event-driven, directional)
A deliberately simple rule:
- At `t0 + delay` (delay = 1 s, 5 s, 30 s, 60 s), trade in the direction implied by the sign of the standardised surprise (sign mapping learned from **past events only**), only if |surprise| > threshold.
- Exit at +5 min, +30 min, or +2 h.
- **Costs:** enter and exit by crossing the spread at the **actual** prevailing book at that moment (from mbp-1; walk mbp-10 where available), plus fees.
- **Validation:** strictly chronological — for each event, parameters (sign mapping, threshold) come only from earlier events. Expanding window; first 8 events of each type are training-only.
- Report per-event P&L, mean, hit rate, worst event, and bootstrap confidence interval. With ~20–25 events per type, results will be noisy — say so prominently.
- Include a **"perfect-surprise-sign" upper bound** (clearly labelled) and a **"random direction"** baseline.

---

## 9. Infrastructure

### 9.1 Pipeline and CLI
Typer CLI commands: `calendar build`, `calendar validate`, `data plan-costs`, `data download`, `panel build`, `analyze liquidity`, `analyze reaction`, `analyze regression`, `analyze execution`, `analyze strategy`, `report`, `replay`.
Every analysis reads Parquet panels and writes tables (Parquet + Markdown) and figures to `reports/`. Each run saves its config and git commit hash next to the outputs.

### 9.2 Release-day replay dashboard (demo)
- A `replay` command that plays back one release window (e.g. a CPI day, −5 min to +10 min) at 1× real-time speed and exposes Prometheus metrics: spread, top-of-book depth, mid-price (as move in ticks from pre-release), trade rate for each instrument.
- `docker-compose.yml` with `app`, `prometheus`, `grafana`; a provisioned Grafana dashboard showing all instruments' depth and price moves side by side, so you can watch liquidity vanish and return in real time.
- Prometheus alert rules: spread above N ticks, depth below X% of baseline, no data for > 5 s.

### 9.3 Engineering standards
Type hints, `ruff`, `mypy` on core modules, `pytest` + `hypothesis`, GitHub Actions CI using a **synthetic event generator** only (no API keys, no network in CI), structured JSON logging, config via `config.yaml` validated with Pydantic, small clear commits.

### 9.4 Required tests (at minimum)
- Time-zone conversion for releases on both sides of a DST change.
- Event-time panel alignment: a synthetic jump injected at `t0` appears exactly at offset 0.
- Baseline and recovery-time calculations on hand-built series.
- Book-walk execution cost on hand-built books (including insufficient depth).
- Standardised surprise computation and the validator for `surprises.csv` (missing values, bad units, duplicate events).
- Strategy validation never uses information from the current or future events.
- Bootstrap resamples events, not individual seconds.

---

## 10. Repository structure

```
macro-release-microstructure/
├── README.md
├── PROJECT_2_SPEC.md
├── config.yaml
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── docker-compose.yml
├── .pre-commit-config.yaml
├── .env.example                 # DATABENTO_API_KEY=, FRED_API_KEY=
├── .github/workflows/ci.yml
├── data/
│   ├── calendar/                # releases.csv, fomc_dates.csv, surprises.csv (templates committed, data filled by me)
│   ├── raw/                     # .dbn.zst cache (gitignored)
│   └── processed/               # Parquet panels (gitignored)
├── src/macrorel/
│   ├── calendar/                # FRED release dates, FOMC loader, surprises validator, ALFRED cross-check
│   ├── data/                    # cost planning, budgeted downloads, quality checks, control-day selection
│   ├── panel/                   # event-time alignment and resampling
│   ├── analysis/                # liquidity, reaction, regression, execution, strategy
│   ├── stats/                   # bootstrap, HC3 regressions, FDR correction, change-point detection
│   ├── replay/                  # release-day replay + Prometheus metrics
│   ├── synthetic/               # synthetic event generator for tests and CI
│   └── cli.py
├── tests/
├── monitoring/                  # prometheus.yml, alert rules, grafana provisioning + dashboard JSON
├── notebooks/                   # exploration only
├── reports/                     # figures, tables, REPORT.md
└── docs/explainers/             # one plain-English explainer per module (see §12)
```

---

## 11. Phases (stop after each one for my review)

**Phase 0 — Skeleton.** Repo, `pyproject.toml`, config, logging, CI, `.env` handling, synthetic event generator (order book with a configurable liquidity drop, spread blowout and price jump at `t0`). CI green.

**Phase 1 — Calendar and surprises.** FRED release-date fetcher (CPI, NFP), FOMC template, `surprises.csv` template + validator, ALFRED cross-check, standardised surprises. **Stop and ask me to fill in `fomc_dates.csv` and `surprises.csv`.**
*Done when:* `calendar validate` passes on my completed files and prints a summary of events and surprise distributions.

**Phase 2 — Cost plan and data.** Cost-planning table (see §3.2) → my approval → budgeted downloads → data-quality report → control-day selection → event validation (volatility spike check).

**Phase 3 — Event-time panels.** §4 alignment and baselines. Sanity plots for 3 events (one per type) for me to inspect visually.

**Phase 4 — Liquidity dynamics (§5).**

**Phase 5 — Price reaction and first mover (§6).**

**Phase 6 — Reaction-function regressions (§7)**, including FDR correction and sign checks.

**Phase 7 — Practical guidance (§8):** execution-cost curves and the surprise-direction strategy.

**Phase 8 — Replay dashboard (§9.2).** *Done when:* `docker compose up` + `replay` shows a release unfolding live in Grafana.

**Phase 9 — Report.** `reports/REPORT.md` (3–5 pages + figures): questions, data, method, results, where results are weak, limitations, future work. `README.md` with the four or five headline numbers and two key charts at the top (the depth-around-release chart and the cross-asset reaction heatmap).

---

## 12. Learning requirement and honesty rules

**Learning:** I need to explain every part of this project in an interview. For each module, write a plain-English explainer in `docs/explainers/`: what it does, why it is designed this way, key assumptions, what could go wrong. When you make a design choice, tell me the alternatives you considered.

**Honesty:**
- No fabricated numbers, dates, consensus values, charts, or citations.
- Always show the number of events behind every result. With ~16–25 events per type, say clearly which results are robust and which are suggestive.
- Liquidity results are only valid relative to control days.
- Upper-bound strategies (perfect surprise sign) must always be labelled as such.
- Limitations to state prominently: small event sample, manually compiled consensus data, approximate yield conversions, cross-venue timestamp comparability, no real fills, fee assumptions.

---

## 13. Tech Stack (use exactly these; ask before adding anything)

Use the latest stable version of each library, pinned in `pyproject.toml` / `uv.lock`.

### 13.1 Language and environment
| Technology | Used for |
|---|---|
| **Python 3.12** (3.11 minimum) | All code |
| **uv** | Environment, dependencies, lockfile, running commands |
| **Git + GitHub** | Version control, public portfolio repo |
| **WSL2 (Ubuntu)** | Required if developing on Windows |

### 13.2 Data sources and clients
| Technology | Used for |
|---|---|
| **databento** (Python client) | CME futures data: `ohlcv-1s`, `mbp-1`, `mbp-10`, `definition`; `metadata.get_cost` before every download |
| **fredapi** | ALFRED first-release values for cross-checking `actual` figures |
| **httpx** | FRED API `release/dates` and `releases` endpoints (fredapi does not cover release dates) |
| **python-dotenv** | Load API keys from `.env` |

### 13.3 Data processing and storage
| Technology | Used for |
|---|---|
| **Polars** | Main dataframe library: cleaning, resampling onto event-time grids, joins, group-bys |
| **NumPy** | Numeric arrays; book-walk execution cost |
| **Numba** | Sequential per-event loops (book reconstruction, first-move detection) where vectorisation is not possible |
| **PyArrow + Parquet** | Storage of panels, results tables (partitioned by event type / instrument) |
| **DuckDB** | SQL queries across all Parquet panels for analysis and reporting |
| **pandas** | Only at the boundary where statsmodels or ruptures need it |
| **zoneinfo** (stdlib) | America/New_York ↔ UTC with correct DST |

### 13.4 Statistics
| Technology | Used for |
|---|---|
| **statsmodels** | OLS reaction-function regressions with HC3 robust standard errors; Benjamini–Hochberg FDR (`multipletests`) |
| **SciPy** | Bootstrap confidence intervals (by event), distribution tests |
| **ruptures** | Change-point detection for the start of depth withdrawal |
| **scikit-learn** | Only if needed for simple utilities (no ML models are required in this project) |

### 13.5 Visualisation and reporting
| Technology | Used for |
|---|---|
| **Matplotlib** | All report figures (PNG in `reports/figures/`): depth/spread around release vs. control, first-mover timeline, reaction heatmap, execution-cost curves, strategy P&L |
| **JupyterLab** | Exploration only |
| **Markdown** | README, REPORT, explainers |

### 13.6 Application structure
| Technology | Used for |
|---|---|
| **Typer** | CLI |
| **Pydantic v2 + PyYAML** | Typed, validated config |
| **structlog** | Structured JSON logging |

### 13.7 Monitoring and deployment
| Technology | Used for |
|---|---|
| **prometheus-client** | Metrics from the replay command |
| **Prometheus** | Scraping metrics, alert rules |
| **Grafana** | Auto-provisioned release-replay dashboard |
| **Docker + Docker Compose** | `app`, `prometheus`, `grafana` with one command; multi-stage Dockerfile, non-root user |

### 13.8 Code quality and testing
| Technology | Used for |
|---|---|
| **pytest**, **pytest-cov** | Tests and coverage (aim >80% on `panel/`, `analysis/`, `stats/`) |
| **Hypothesis** | Property-based tests (alignment, book walking, surprise validator) |
| **ruff**, **mypy**, **pre-commit** | Lint, format, type checks |
| **GitHub Actions** | CI on every push, synthetic data only |

### 13.9 Explicitly out of scope
**No crypto or Web3 of any kind** (no crypto instruments, exchanges, websockets, on-chain data, or crypto libraries such as ccxt or web3.py). No web scraping of paid or terms-restricted sites, no Kafka/Redis, no database servers, no Kubernetes, no cloud deployment, no web frameworks, no deep learning, no live trading. Mention extensions as future work in the report.

### 13.10 Hardware assumptions
Laptop with 8–16 GB RAM and ~15 GB free disk. Process one event at a time and use Polars lazy scans / DuckDB over Parquet if memory is tight.
