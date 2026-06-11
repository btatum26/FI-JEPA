# Dataset Plan

## Status And Source Of Truth

This document describes the implemented dataset layers and the remaining data
roadmap. It is not a code skeleton.

Use these documents for exact current contracts:

- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) for canonical tables and frozen
  artifact schemas.
- [../docs/data_builder.md](../docs/data_builder.md) for the sparse model-ready
  builder.
- [../README.md](../README.md) for runnable commands.
- `src/dataset_pipeline/` and `tests/` when prose and code disagree.

## Source Research Catalog

This section is intentionally broader than the implemented pipeline. A source
does not need to be implemented, free, or immediately usable to belong here.
Add candidates when they could improve coverage, point-in-time validity,
metadata, feature quality, target quality, or independent validation.

Every source must be re-verified before implementation because access terms,
coverage, APIs, and pricing can change.

| Source | Provides | Free? | API key? | Main limitations | Survivorship or point-in-time help? | Prototype fit | Status |
|---|---|---:|---:|---|---|---|---|
| Stooq | Historical OHLCV for stocks, ETFs, indices, FX, commodities, bonds, and international markets; bulk daily archives | Yes | No | No formal stable API; corporate-action handling, ticker continuity, and survivorship status require validation | Weak; broad prices but not a clean point-in-time universe | Very good for first prototype | Implemented canonical price source |
| FRED | Macro, rates, credit spreads, inflation, unemployment, stress indicators, VIX series, and other economic observations | Yes | Yes for official API | Standard observations are commonly latest-revised rather than historical vintages | Helps market-state coverage, not equity survivorship | Excellent | Implemented daily-series source |
| ALFRED | Vintage and real-time-period macro observations | Yes | Usually through FRED API patterns | More ingestion and validation work; vintage coverage differs by series | Stronger protection against macro revision leakage | Strong future upgrade | Planned research |
| CBOE | VIX and other volatility-index history | Often for public historical files | Usually no for public files | Historical files and terms vary by index; independent validation is still required | Not relevant to stock survivorship | Very good for volatility cross-checks and expansion | Candidate |
| Kenneth French Data Library | Fama-French factors, portfolio returns, breakpoints, and archives | Yes | No | Factors are derived products, not raw security data; release-time assumptions need care | Reduces dependence on the local stock universe for factor context | Very good as factors and benchmarks | Candidate |
| Community-maintained S&P 500 files | Current constituents and historical change records | Yes | No | Not an institutional point-in-time membership database; records may be incomplete | Partial membership history, but current-list backfills remain biased | Useful prototype metadata | Implemented with explicit bias flags |
| yfinance / Yahoo Finance | Convenient single-symbol prices and metadata | Convenience library is free | No official Yahoo key | Unofficial wrapper, unstable semantics, and unsuitable as a canonical source | Weak | Fallback and independent spot checks only | Candidate fallback |
| Norgate | End-of-day equities and delisted-stock coverage depending on product | No | Product access | Cost and product-specific workflow constraints | Stronger delisted and point-in-time support | Strong future upgrade | Candidate paid source |
| Sharadar / Nasdaq Data Link | US equity prices, corporate actions, fundamentals, and active/delisted coverage depending on product | No | Yes | Cost and product-specific history/field semantics | Stronger active and delisted coverage | Strong future upgrade | Candidate paid source |
| CRSP / Compustat through WRDS | Research-grade security returns, delistings, identifiers, fundamentals, and corporate actions depending on subscription | No | Institutional access | Expensive and licensing-constrained; integration is materially more complex | Strong | Best research-grade upgrade if access exists | Candidate institutional source |
| SEC EDGAR / Company Facts | Filings and structured reported fundamentals | Yes | No paid key; request identification and rate rules apply | Filing data is not a clean point-in-time panel without substantial normalization | Helps point-in-time fundamental research, not price survivorship by itself | Useful later | Candidate fundamental source |
| Exchange and symbol directories | Listing metadata, current symbols, exchange identity, and status fields | Often | Varies | Usually current-state rather than complete historical security-master data | Partial metadata improvement | Useful for manifest validation | Research needed |
| Independent benchmark-price source | Cross-checks for major indices, ETFs, splits, and extreme returns | Varies | Varies | A second source can still share upstream errors or adjustment ambiguity | Does not solve survivorship, but catches source-quality failures | Required quality upgrade | Source not selected |

### Source-Specific Analysis

#### Stooq

Stooq is the practical first canonical price source because its bulk archives
provide broad, long daily history without per-symbol request overhead. Preserve
the original ZIP files and treat them as immutable source snapshots.

Before relying on single-stock long-horizon returns, validate:

- Split and dividend adjustment behavior.
- Ticker changes and symbol continuity.
- Delisting interpretation.
- Extreme-return and stale-price behavior.
- Major benchmark returns against an independent source.

#### FRED And ALFRED

FRED is the current macro source. Daily series are aligned by configured
availability assumptions, and standard snapshots are explicitly treated as
potentially revised.

ALFRED is the preferred upgrade for macro series where historical vintage
values materially affect leakage risk. An ALFRED integration must preserve both
observation date and real-time availability interval.

#### CBOE And Kenneth French

CBOE volatility histories can improve or independently verify volatility-state
inputs. Kenneth French datasets can provide factor-state inputs and strong
benchmarks without depending on the local equity panel.

Neither source should enter the encoder until its publication timing,
availability assumptions, raw snapshot policy, and missingness behavior are
documented.

#### Current-Constituent Community Data

Community S&P 500 files are useful for a broad equity representation stress
test. They are not sufficient for historical stock-selection claims. Every
backfilled current-constituent row must retain explicit high-survivorship-bias
and non-point-in-time flags.

#### Paid Point-In-Time Upgrades

Norgate, Sharadar, and CRSP/Compustat are candidate replacements or supplements
for the current equity source and security master. The main value is not merely
more columns; it is reliable delisting, identifier, corporate-action, and
historical-membership information.

The downstream feature and model-ready contracts should remain source-agnostic
so a better source can replace Stooq-derived equity facts without redesigning
the model.

### Source Selection Gates

Before implementing a source, record:

| Gate | Required answer |
|---|---|
| Research purpose | Which missing feature, target, metadata field, or validation check does it enable? |
| Access | Cost, license, API key, rate limits, and redistribution restrictions |
| Coverage | Asset classes, symbols, fields, frequencies, first date, and last date |
| Time semantics | Observation time, publication time, revision policy, and earliest safe model availability |
| Point-in-time quality | Historical membership, delistings, identifier changes, and vintage support |
| Adjustment semantics | Splits, dividends, corporate actions, and return construction |
| Raw snapshot policy | Exact files or responses preserved, hashes recorded, and overwrite behavior |
| Join keys | Stable identifiers and mapping confidence into the canonical manifest |
| Failure modes | Missing periods, stale values, schema drift, and known source warnings |
| Validation plan | Independent comparisons and tests required before publication |

### New Source Intake

Add new candidates to the main catalog and create a short analysis block using
this template:

```markdown
#### Source Name

- Research purpose:
- Candidate fields or series:
- Access, cost, and license:
- Coverage:
- Point-in-time and revision behavior:
- Adjustment or identity semantics:
- Expected canonical grain:
- Raw snapshot format:
- Required validation:
- Open questions:
- Decision: research / candidate / planned / implemented / rejected
```

### Source Research Backlog

| Need | Why it matters | Desired source properties | Candidate status |
|---|---|---|---|
| Point-in-time equity membership and delistings | Removes the largest survivorship-bias weakness | Historical membership intervals, delisting returns, stable identifiers | Paid and institutional candidates listed above |
| Corporate actions and adjusted-price validation | Long-horizon single-stock returns are unreliable without clear semantics | Splits, dividends, symbol changes, independent adjusted and raw prices | Source not selected |
| Historical security master | Improves ticker continuity and cross-source joins | Stable IDs, name changes, exchange history, listing and delisting dates | Source not selected |
| Macro vintages | Prevents revised-data leakage | Observation dates plus real-time availability intervals | ALFRED candidate |
| Volatility-index expansion | Adds richer implied-volatility state | Historical index levels with documented calculation and publication timing | CBOE candidate |
| Factor and portfolio benchmarks | Tests whether embeddings add information beyond standard factors | Archived factor and portfolio returns with publication assumptions | Kenneth French candidate |
| Fundamentals | Enables slow valuation and balance-sheet state research | Filing-based point-in-time values and stable company/security mapping | SEC and paid candidates |
| Independent price validation | Detects adjustment, extreme-return, and symbol-continuity errors | Overlapping benchmark and sample-equity coverage | Source not selected |
| Intraday state data | Supports future higher-frequency research | Stable timestamps, exchange calendars, corporate actions, and manageable storage | Out of current scope |
| Options surface data | Adds forward-looking volatility and skew state | Historical chains or surfaces with timestamps and survivorship-aware contracts | Out of current scope |

## Design Rules

1. Encoder inputs use only information available at or before date `t`.
2. The earliest implied execution is the next trading session.
3. Future targets remain physically separate from encoder and embedding
   artifacts.
4. Feature and target definitions preserve their natural grain.
5. Missingness is represented by explicit masks, never inferred from zero-filled
   model values.
6. Train-only normalization and split protection are required.
7. Generated canonical and model-ready artifacts are immutable build products.

## Implemented Data Layers

### Raw Inputs

| Source | Current use | Important limitation |
|---|---|---|
| Stooq daily US/world bulk archives | Price, volume, symbol history, and derived asset features | Symbol metadata and adjustment semantics are limited |
| FRED JSON snapshots | Daily market-implied, rates, credit, and fed-funds series | Standard snapshots are not full point-in-time vintages |
| Community universe CSVs | Current S&P 500 metadata and change history | Current constituents backfilled over history are survivorship-biased |

The production FRED series list is the only feature-source configuration in
`configs/features.yaml`. Asset and cross-sectional feature definitions are
implemented directly in `src/dataset_pipeline/`.

### Canonical DuckDB

Build:

```bash
uv run build-market-database
```

Output:

```text
data/processed/market_data.duckdb
```

Published tables:

| Table | Grain | Role |
|---|---|---|
| `features` | Date | Past-only market and macro features |
| `ticker_features` | Date and symbol | Past-only asset features |
| `targets` | Date | Future outcomes, physically separate |
| `symbol_manifest` | Symbol | Identity, coverage, and bias metadata |
| `trading_calendar` | Date | Full observed-date spine across selected symbols |
| `build_metadata` | Build | Source and validation metadata |

`market_features`, `macro_features`, `macro_data`, and bad-OHLC findings are
build-time intermediates or metadata, not published feature tables.

### Sparse Model-Ready Dataset

Build:

```bash
uv run build-model-dataset --config configs/model_dataset.yaml
```

The output is an immutable sparse artifact under `data/model_ready/`. It
contains selected normalized facts, masks, manifests, and split permissions.
It excludes future targets, OAS-derived model inputs, windows, full
date-by-asset grids, asset samples, temporal patches, and JEPA masks.

## Time And Calendar Contract

The canonical calendar preserves every date observed across selected symbols.
It is not anchored to SPY or another single reference symbol.

The model-ready builder uses a separate configured sample reference symbol to
identify valid sample endpoints. That does not truncate canonical source
history.

The default model-ready contract uses:

| Setting | Value |
|---|---:|
| Context start | `2000-01-01` |
| Sample start | `2005-02-25` |
| Lookback | 252 trading dates |
| Maximum future target horizon | 126 trading dates |

## Implemented Encoder Features

### Asset Features

Asset features are calculated per `(date, symbol)` and include:

- Log returns over 1, 5, 21, 63, and 126 days.
- Realized volatility over 5, 21, 63, and 126 days.
- Drawdown over 5, 21, 63, and 126 days.
- Moving-average distance and path efficiency over 5, 21, 63, and 126 days.
- Dollar volume.

### Market Features

Date-level cross-sectional features currently include:

- One-day cross-sectional dispersion and interquartile range.
- One-day breadth.
- Fraction above the 63-day moving average.
- Valid asset count.

### Macro Features

The canonical database derives level and 1, 5, 21, and 63-day change features
from enabled FRED series:

- VIX.
- Treasury 3-month, 2-year, 10-year, and 30-year rates.
- Effective fed funds.
- High-yield and corporate OAS.
- 10y-2y, 10y-3m, and 30y-10y yield-curve spreads.
- High-yield minus corporate OAS.

OAS-derived columns remain in the canonical database but are intentionally
excluded from the current model-ready export.

## Implemented Targets

The canonical `targets` table currently contains SPY-based future outcomes for
21, 63, and 126 trading-day horizons:

- Future log return.
- Future annualized realized volatility.
- Future maximum drawdown.
- Future trend score, defined as future return divided by future realized
  volatility.

Dispersion targets, breadth targets, correlation targets, tail-risk targets,
bucket labels, and regime labels are not implemented.

Targets are never exported into model-ready pretraining artifacts or embedding
artifacts. They are exported separately for frozen probes.

## Split Protection

Each named validation window protects three disjoint date regions:

| Flag | Meaning |
|---|---|
| `validation_sample` | Date is a validation sample endpoint |
| `protected_input_lookback` | Date is reserved for reconstructing validation inputs |
| `protected_forward_target` | Date is reserved against future-target leakage |
| `protected_holdout` | Union of all protected regions |
| `train_fact_allowed` | Date may enter train fact files |
| `validation_fact_allowed` | Date may enter validation fact files |

Training and validation fact files are date-disjoint. Runtime training windows
reapply `train_fact_allowed`, while validation windows use the protected
validation-relative fact set.

## Availability And Missingness

- Asset observations retain explicit status and validity fields.
- FRED observations join on configured availability dates rather than blindly
  on observation dates.
- Macro values are forward-filled only after they become available.
- Feature masks are created before normalization and zero filling.
- Model values are zero-filled only after invalid positions are masked.
- The model always receives masks with values.

## Normalization

The current model-ready builder supports train-fold robust z-score
normalization:

1. Fit winsorization bounds, center, and scale only on finite train facts.
2. Apply configured transforms before fitting.
3. Normalize train and validation facts with the same train-fit statistics.
4. Fill invalid normalized values with `0.0` while retaining false masks.
5. Publish normalization statistics in `normalization.parquet`.

## Leakage And Quality Gates

Required implemented checks include:

- No future or target columns in canonical feature tables.
- No future or target columns in model-ready facts or embeddings.
- No duplicate canonical feature keys.
- No duplicate sparse facts, including duplicates within one Parquet batch.
- Train and validation fact dates are disjoint.
- Feature manifest indices are contiguous within each input group.
- Model-ready configured features exist in the canonical database.
- Normalization uses only allowed train facts.
- Embedding and probe-target artifacts come from the same canonical database
  version.

## Current Limitations

- Current-constituent equity history is survivorship-biased.
- Standard FRED snapshots can contain revised historical values.
- Stooq metadata does not provide a complete point-in-time security master.
- Exchange and sector metadata remain incomplete for many instruments.
- Availability assumptions are daily and after-close; intraday timing is not
  modeled.
- The model-ready feature list is deliberately narrower than the canonical
  database.

## Planned Dataset Expansion

These are research plans, not claims about the current database.

### Candidate Dataset Scope

| Group | Candidate coverage | Purpose |
|---|---|---|
| Core equity proxies | SPY, QQQ, IWM, DIA, and broad international proxies | Broad risk-on and regional state |
| Sector ETFs | US sector SPDR family and other stable sector proxies | Breadth, dispersion, and sector rotation |
| Rates and bonds | Short, intermediate, and long Treasury proxies plus aggregate bonds | Duration and curve state |
| Credit | High-yield and investment-grade proxies plus spread series | Credit stress and risk appetite |
| Commodities and currency | Gold, oil, dollar, and other liquid macro proxies | Inflation, growth, and dollar state |
| Broad indices | S&P 500, Nasdaq, Russell, Dow, and international indices where reliable | Independent benchmark and state coverage |
| Equity cross-section | Point-in-time or explicitly biased stock universes | Breadth, dispersion, correlation, and concentration |
| Macro vintages | Rates, inflation, labor, growth, liquidity, and stress series | Slow economic state without revision leakage |
| Factors and portfolios | Fama-French and other archived factor or portfolio returns | Baselines and factor-state context |

### Candidate Encoder Feature Families

| Family | Candidate features | Status and main requirement |
|---|---|---|
| Returns | Multi-horizon market, asset, sector, and factor returns | Partially implemented |
| Volatility | Realized volatility, volatility-of-volatility, VIX term structure, and implied-volatility indices | Realized volatility and VIX level/change implemented; broader implied-volatility data needs a source |
| Trend and path | Moving-average distance, path efficiency, breakout distance, and trend consistency | Moving-average distance and path efficiency implemented |
| Drawdown and recovery | Rolling drawdown, time under water, recovery speed, and distance from peak | Rolling drawdown implemented |
| Breadth | Positive-return breadth, moving-average breadth, new highs/lows, and sector breadth | Basic breadth implemented; richer breadth needs stable universe membership |
| Dispersion | Cross-sectional standard deviation, IQR, sector dispersion, and residual dispersion | Basic cross-sectional dispersion implemented |
| Correlation and crowding | Average pairwise correlation, sector correlation, concentration, and correlation regime | Planned; requires robust panel coverage and efficient computation |
| Liquidity and volume | Dollar volume, volume shocks, turnover, spreads, and funding/liquidity proxies | Dollar volume implemented; broader liquidity needs better source data |
| Rates and credit | Curve slopes, rate changes, OAS levels/changes, and credit-relative spreads | Canonical features partly implemented; OAS excluded from current model-ready export |
| Slow macro | Inflation, labor, production, financial conditions, and liquidity | Planned only after point-in-time or conservative release handling |
| Calendar | Cyclical calendar encodings | Ablation only because date shortcuts are a material risk |
| Data quality and availability | Coverage, missingness, stale-price, and source-confidence indicators | Availability masks implemented; richer quality features remain planned |

### Candidate Future Targets

All future targets must remain outside pretraining artifacts. Bucket thresholds
and rule-based labels must be fit or defined without validation leakage.

| Target family | Candidate definition or purpose | Status |
|---|---|---|
| Future return | Close-to-close log return from `t` to `t+h` | Implemented for 21, 63, and 126 days |
| Future realized volatility | Annualized volatility over returns from `t+1` through `t+h` | Implemented for 21, 63, and 126 days |
| Future maximum drawdown | Worst path drawdown over `t+1` through `t+h` | Implemented for 21, 63, and 126 days |
| Future trend score | Future return normalized by future realized volatility | Implemented for 21, 63, and 126 days |
| Future path efficiency | Directional move divided by total path movement | Planned |
| Future dispersion | Mean cross-sectional return dispersion over the horizon | Planned; needs a defensible point-in-time universe |
| Future breadth | Fraction of assets with positive horizon returns | Planned; needs a defensible point-in-time universe |
| Future average correlation | Mean pairwise or sector correlation over the horizon | Planned |
| Future tail risk | Future VaR, CVaR, or downside-tail statistics | Planned |
| Future regime label | Explicit crash, rebound, trend, chop, or calm-up rules | Planned; definitions and train-only thresholds are unresolved |
| Conditional signal performance | Future performance profile of downstream signals | Later alignment research, not a canonical pretraining target |

### Point-In-Time Upgrade Plan

The preferred long-term equity dataset replaces or supplements current
constituent backfills with:

- Historical membership intervals.
- Active and delisted securities.
- Stable security and entity identifiers.
- Listing, delisting, exchange, name, and ticker-change history.
- Corporate actions and clearly defined adjusted/unadjusted prices.
- Source snapshots that can be reconstructed for each build.

The canonical table grains and model-ready export format should remain stable
when this source upgrade occurs.

## Data Roadmap

1. Add point-in-time and delisted-aware equity membership.
2. Add ALFRED-style macro vintages or independently archived release snapshots.
3. Improve security-master metadata and corporate-action handling.
4. Implement additional future targets only with separate storage, train-only
   thresholds where needed, and leakage tests.
5. Expand model-ready feature families only after availability, missingness,
   and normalization contracts are explicit.
6. Add alternate validation windows and dataset versions without changing
   immutable prior artifacts.

## Validation

Relevant tests:

- `tests/test_dataset_pipeline.py`
- `tests/test_fred_loader.py`
- `tests/test_stooq_archive_loader.py`
- `tests/test_community_universes.py`
- `tests/test_model_dataset_builder.py`
- `tests/test_fi_jepa_dataloader.py`

Run:

```bash
uv run pytest -q
```
