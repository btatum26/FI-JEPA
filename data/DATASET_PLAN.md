# Dataset Plan

Below is the dataset plan I would actually build first. The main idea is to separate three things very aggressively:

- **Encoder inputs:** only information available at or before date t.

- **Downstream targets:** future outcomes from t+1 onward.

- **Metadata/quality:** everything that explains what the dataset is, how biased it is, and what assumptions were made.

For storage, use one canonical logical DuckDB dataset with immutable
`build_id`-scoped artifacts. DuckDB keeps each build's normalized tables in one
artifact, supports columnar analytical queries, and exposes joined views
without creating duplicate year partitions or flattened copies.

## 1. Data source plan

| Source | Provides | Free? | API key? | Main limitations | Survivorship help? | Prototype fit |
|---|---|---|---|---|---|---|
| Stooq | Historical OHLCV for stocks, ETFs, indices, FX, commodities, bonds, international markets. Daily, hourly, and 5-minute archive options exist. | Yes | No official API key | No formal API; treat CSV/web download as convenience. Corporate action handling and survivorship status need validation. | Weak. Good for broad prices, not a clean point-in-time universe. | Very good for first prototype. |
| FRED | Macro, rates, credit spreads, inflation, unemployment, stress indicators, VIX series, etc. | Yes | Yes for official API | Many macro series are revised; normal FRED gives latest-revised values unless vintage handling is used. | Not relevant for stock survivorship, but good for market state. | Excellent. |
| ALFRED | Vintage/realtime macro data. | Yes | Usually via FRED API patterns | More work than normal FRED. Not every series may have perfect vintage coverage. | Helps avoid macro revision leakage. | Use later or for important macro releases. |
| CBOE | VIX and other volatility index history. | Yes for many index CSVs | No for public historical CSVs | CBOE itself says historical VIX data is provided for convenience and accuracy is not guaranteed. | Not relevant for stock survivorship. | Very good. |
| Kenneth French Data Library | Fama-French factor returns, portfolios, breakpoints, archives. | Yes | No | Factor construction depends on CRSP-style data; useful as factors/benchmarks, not raw asset data. | Factors reduce dependence on your stock universe. | Very good. |
| yfinance | Convenient Yahoo Finance wrapper. | Free library | No official Yahoo API key | Not affiliated with Yahoo; intended for research/education/personal use; should not be treated as your canonical source. | Weak. | Fallback only. |
| Norgate | EOD equities with delisted stocks depending on subscription level. | Paid | Product access | Cost; Windows-centric workflows in some cases; still requires careful metadata. | Stronger. | Future upgrade. |
| Sharadar / Nasdaq Data Link | US equity prices, corporate actions, fundamentals, active and delisted coverage depending on product. | Paid | Yes | Cost. | Stronger. | Future upgrade. |

Stooq’s official historical data page exposes free historical market data archives, and search results from Stooq’s own page show daily/hourly/5-minute categories and regional archive downloads. For single-symbol work, Stooq pages expose historical rows with Date/Open/High/Low/Close/Volume and CSV download links. A third-party walkthrough notes that Stooq data is accessed through a web interface rather than a formal API, so your code should cache raw downloads and never assume API stability.

FRED is a strong macro source because the official API can retrieve economic observations and return formats such as JSON, XML, Excel, or zipped CSV; official docs also state that API requests require a registered API key. For vintage macro data, ALFRED is the right upgrade path because it lets you retrieve data releases as they were available on historical dates.

CBOE provides VIX data from 1990 to present and links to other volatility index histories, but its page also includes a warning that the data is provided without a guarantee of accuracy. Kenneth French’s library provides factor data, portfolio sorts, breakpoints, and historical archives, including Fama-French factor files. yfinance is useful as a convenience layer, but its own package page says it is not affiliated, endorsed, or vetted by Yahoo and is intended for research and educational use.

## 2. Dataset scope

### Canonical dataset

Goal: learn slow market-wide state from one canonical logical dataset with
immutable physical builds.

Use daily data only. The default canonical build combines broad indices, ETFs,
sectors, daily market-implied variables, and basic feature-availability masks.
The current S&P 500 cross-section is optional and disabled by default. There are
no separate A/B dataset definitions and no year-partitioned canonical Parquet
copies.

The current S&P 500 cross-section is a survivorship-biased representation
stress test, not the basis for historical stock-selection or equity-alpha
claims. Configs, manifests, and the live build must agree about whether this
optional universe is enabled and which date range is actually present.

### Time-of-decision convention

Every encoder row represents the information state after market close on date
`t`. Close, volume, and derived values using observations through `t` are
allowed. A tradable decision based on that row can execute no earlier than the
next trading session. Targets and probes may use `t+1` onward, but they must
remain physically and logically separate from encoder inputs.

Record this convention once in build metadata. Add per-row decision or
execution timestamps only when supporting intraday data or multiple
time-of-decision conventions.

### Core assets
SPY.US, QQQ.US, IWM.US, DIA.US, TLT.US, IEF.US, SHY.US, AGG.US, GLD.US, USO.US, UUP.US, EFA.US, EEM.US, HYG.US, LQD.US.

### Sector ETFs
XLB.US, XLC.US, XLE.US, XLF.US, XLI.US, XLK.US, XLP.US, XLRE.US, XLU.US, XLV.US, XLY.US.

### Indices where available
^SPX, ^NDX or Nasdaq proxy, ^RUT, ^DJI.

### Macro/FRED
VIX, 3-month Treasury, 2-year Treasury, 10-year Treasury, 30-year Treasury, 10Y–2Y slope, fed funds, high-yield OAS, investment-grade/corporate spread, financial stress index, unemployment, CPI, industrial production.

### French
Daily or monthly Fama-French factors. For daily JEPA inputs, daily factors are easier. For monthly factors, align carefully and only use values after the assumed release date.

### Current S&P 500 cross-section

Use current S&P 500 constituents from a free source, then pull historical Stooq prices. Wikipedia maintains a current S&P 500 component table and a changes table, but it is not an institutional point-in-time membership database. The current list is useful for a survivorship-biased prototype, not for historical claims about stock selection.

### Add
current S&P 500 tickers, current GICS sector labels, Stooq historical prices, current sector labels, equal-weight approximations, cross-sectional dispersion, breadth, rolling correlation.

### Dataset labels

| Field | Value |
|---|---|
| `universe_type` | `current_constituents_backfilled` |
| `survivorship_bias` | `high` |
| `point_in_time_membership` | `false` |

### Future point-in-time upgrade

Goal: reduce survivorship bias and improve point-in-time validity.

### Upgrade sources
Norgate for delisted-aware EOD equity data, or Sharadar/Nasdaq Data Link for active and delisted US equity prices/fundamentals if affordable. Norgate states that US delisted stocks are included at certain subscription levels, and Sharadar’s Nasdaq Data Link page describes more than 20,000 US public companies with adjusted/unadjusted EOD prices and corporate actions.

### Pipeline changes
keep the same schemas, but replace symbol metadata, listing dates, delisting dates, corporate actions, and universe membership with point-in-time-aware tables. The rest of the feature/target code should not care whether the source is Stooq or CRSP-like data.

## 3. Project structure
```text
financial-state-dataset/

data/
  raw/
    stooq/
      single_symbol/
      bulk_archives/
    fred/
    alfred/
    french/
    cboe/
    metadata/
  processed/
    builds/
      <build_id>/
        market_data.duckdb
        dataset_manifest.json
    latest/
  manifests/
    dataset_manifest.json

src/
  data_sources/
    stooq_loader.py
    fred_loader.py
    french_loader.py
    cboe_loader.py
  indexing/
    symbol_index.py
    calendar.py
    alignment.py
  cleaning/
    adjust_prices.py
    missing_values.py
    corporate_actions.py
  features/
    market_features.py
    cross_sectional_features.py
    macro_features.py
    calendar_features.py
  targets/
    volatility_targets.py
    trend_targets.py
    drawdown_targets.py
    dispersion_targets.py
    regime_targets.py
  validation/
    checks.py
    reports.py
    leakage.py
  pipelines/
    build_market_database.py
    import_stooq_archives.py
  utils/
    io.py
    config.py
    dates.py

configs/
  universe.yaml
  sources.yaml
  features.yaml
  targets.yaml
  validation.yaml

notebooks/
  00_source_exploration.ipynb
  01_build_dataset.ipynb
  02_quality_checks.ipynb
  03_feature_diagnostics.ipynb
  04_probe_targets.ipynb

```

Store raw downloaded files exactly as received and never overwrite them silently.
All canonical derived data lives in an immutable build directory:

```text
data/processed/builds/<build_id>/market_data.duckdb
data/processed/builds/<build_id>/dataset_manifest.json
```

Tables hold each logical dataset once; views provide joined access without
physically repeating date-level values for every symbol. An optional
`data/processed/latest/` pointer or copied convenience artifact may move to the
newest completed build. Models and reports must reference the immutable
`build_id`, never only `latest`.

### Reproducibility and model-ready outputs

DuckDB remains the canonical source of truth. Each completed `build_id` is
immutable and must never be overwritten. A build records a configuration hash
and source snapshot records containing the source path or URL, request
parameters where applicable, download timestamp, content hash, byte size, row
count, first date, and last date.

Every build also records independent semantic versions:

- `dataset_schema_version`: physical table and column structure
- `feature_schema_version`: feature names, formulas, and availability rules
- `target_schema_version`: target names, formulas, horizons, and bucket rules
- `decision_time_convention_version`: information and execution timing rules

Changing any governed structure or semantic rule requires incrementing the
corresponding version.

Local credentials are stored in the repository-root `.env` file. The current
local setup defines `FRED_API_KEY` there. `.env` must remain ignored, and secret
values must never be copied into configs, logs, manifests, source snapshots, or
the DuckDB database. Because `FredLoader` reads the process environment, invoke
FRED-backed commands with `.env` loaded into that environment.

Frozen model-ready windows are exported under
`data/model_ready/<build_id>/<fold_id>/`. These files are reproducible build
products, not a second canonical dataset.

## 4. Core schemas

### Raw price table

| Field | Type/value | Notes |
|---|---|---|
| `date` | `date` |  |
| `symbol` | `string` | canonical symbol |
| `source_symbol` | `string` | e.g. spy.us |
| `open` | `float64` |  |
| `high` | `float64` |  |
| `low` | `float64` |  |
| `close` | `float64` |  |
| `volume` | `float64` |  |
| `source` | `string` | stooq, yfinance, etc. |
| `adjusted_flag` | `bool` |  |
| `currency` | `string` |  |
| `exchange` | `string \| null` |  |
| `asset_type` | `string` | index, etf, stock, future_proxy, fx, commodity_proxy |
| `download_timestamp` | `timestamp` |  |
| `raw_file` | `string` |  |
| `quality_flag` | `string` | ok, bad_ohlc, duplicate_date, missing_volume, etc. |

### Price and return policy

Do not treat `adjusted_flag` as proof that a source's adjustment semantics are
understood. Record `return_source` and the known adjustment policy for each
source or instrument. Store separate raw and adjusted price columns only when
both values genuinely exist. Until Stooq split and dividend behavior has been
validated, long-horizon returns must be labeled as source-derived and subject
to adjustment uncertainty.

Required validation includes extreme-return detection, split-like jump
detection, stale-price detection, symbol-continuity review, and return
comparison for major benchmarks against an independent source.

### Symbol metadata

| Field | Type/value | Notes |
|---|---|---|
| `canonical_symbol` | `string` |  |
| `source` | `string` |  |
| `source_symbol` | `string` |  |
| `name` | `string \| null` |  |
| `asset_type` | `string` |  |
| `exchange` | `string \| null` |  |
| `sector` | `string \| null` |  |
| `industry` | `string \| null` |  |
| `country` | `string \| null` |  |
| `currency` | `string \| null` |  |
| `first_available_date` | `date` |  |
| `last_available_date` | `date` |  |
| `survivorship_status` | `string` | unknown, active_current, confirmed_delisted, inactive_unconfirmed |
| `universe_membership_type` | `string` | current, point_in_time, static_proxy, unknown |
| `point_in_time_valid` | `bool` |  |
| `data_source` | `string` |  |
| `notes` | `string` |  |

### Macro table

| Field | Type/value | Notes |
|---|---|---|
| `date` | `date` | observation date |
| `series_id` | `string` |  |
| `value` | `float64` |  |
| `source` | `string` | fred, alfred, cboe |
| `frequency` | `string` |  |
| `release_lag_assumption` | `string` | same_day, one_day, conservative_15bd, alfred_vintage |
| `asof_date` | `date \| null` | when assumed available |
| `point_in_time_available` | `bool` |  |
| `revised_data_flag` | `bool` |  |

### Feature table, long form

| Field | Type/value | Notes |
|---|---|---|
| `date` | `date` |  |
| `feature_name` | `string` |  |
| `value` | `float64` |  |
| `source_columns` | `string` |  |
| `lookback_window` | `int \| null` |  |
| `available_asof` | `date` |  |
| `uses_future_data` | `bool` |  |
| `normalization_scope` | `string` |  |

### Panel table

| Field | Type/value | Notes |
|---|---|---|
| `date` | `date` |  |
| `symbol` | `string` |  |
| `return_1d` | `float64` |  |
| `return_5d` | `float64` |  |
| `return_21d` | `float64` |  |
| `realized_vol_21d` | `float64` |  |
| `realized_vol_63d` | `float64` |  |
| `dollar_volume` | `float64` |  |
| `market_relative_return_1d` | `float64` |  |
| `sector_relative_return_1d` | `float64` |  |
| `ma_distance_63d` | `float64` |  |
| `drawdown_126d` | `float64` |  |
| `valid_observation` | `bool` |  |
| `observation_status` | `string` |  |

### Target table

| Field | Type/value | Notes |
|---|---|---|
| `date` | `date` |  |
| `symbol` | `string \| null` | null for market-wide target |
| `target_horizon` | `int` |  |
| `future_realized_volatility` | `float64` |  |
| `future_volatility_bucket` | `int` |  |
| `future_trend` | `float64` |  |
| `future_trend_bucket` | `int` |  |
| `future_max_drawdown` | `float64` |  |
| `future_dispersion` | `float64` |  |
| `future_breadth` | `float64` |  |
| `future_average_correlation` | `float64` |  |
| `future_tail_risk` | `float64` |  |
| `future_regime_label` | `string` |  |
| `uses_future_data` | `true` |  |

## 5. Stooq indexing strategy

### Identifiers

| Identifier | Description |
|---|---|
| `source_symbol` | the symbol exactly used by Stooq, for example spy.us. |
| `canonical_symbol` | your normalized research symbol, for example ETF_US_SPY. |
| `instrument_id` | stable ID for the instrument, not the ticker. For now this can be a deterministic hash of source + source_symbol + first_available_date. |
| `entity_id` | optional future field for company/entity-level mapping. Leave null until you have better data. |

### Symbol manifest

| Field |
|---|
| `canonical_symbol` |
| `instrument_id` |
| `source` |
| `source_symbol` |
| `asset_type` |
| `exchange` |
| `currency` |
| `country` |
| `sector` |
| `start_date` |
| `end_date` |
| `first_price_date` |
| `last_price_date` |
| `expected_calendar` |
| `survivorship_status` |
| `point_in_time_valid` |
| `universe_name` |
| `universe_bias_flag` |

### Source-symbol mapping

| Field |
|---|
| `source` |
| `source_symbol` |
| `canonical_symbol` |
| `valid_from` |
| `valid_to` |
| `mapping_confidence` |
| `mapping_notes` |

### Trading calendar

For Phase 1, use a NYSE-like calendar inferred from SPY.US or ^SPX: all dates where the reference market has a valid close. This avoids importing another dependency too early. Later, replace this with an exchange calendar package or official calendar.

### Date index

| Field |
|---|
| `date` |
| `is_trading_day` |
| `calendar_name` |
| `year` |
| `month` |
| `week` |
| `quarter` |

### Symbol-date MultiIndex

Create a full grid only for the processed panel, not for raw data:

```python
full_index = pd.MultiIndex.from_product(
    [calendar_dates, symbols],
    names=["date", "symbol"]
)
```

### Observation statuses

- ok
- not_listed_yet
- after_last_observed
- confirmed_delisted
- missing_expected
- bad_download
- holiday_or_market_closed
- suspended_or_no_trade_unknown

### Rules

- A date before first_available_date is not_listed_yet.

- A date after last_available_date is after_last_observed, not automatically confirmed_delisted.

- A date inside [first_available_date, last_available_date] with no row on a reference trading day is missing_expected.

- A failed or empty source file is bad_download.

- A non-reference trading day is holiday_or_market_closed.

- Only paid or verified metadata should create confirmed_delisted.

### Storage

Use one canonical logical dataset with immutable DuckDB build artifacts:

```text
data/processed/builds/<build_id>/market_data.duckdb
```

Canonical physical tables:

- `features`: one row per trading date containing past-only market-wide and
  macro encoder inputs
- `ticker_features`: one row per symbol and trading date containing past-only
  ticker-specific prices, metadata, status, and derived features
- `targets`: one row per target date, physically separate from `features`
- `symbol_manifest`, `trading_calendar`, and provenance tables

No canonical view joins `features` to `targets`. Probe and alignment exports
must perform that join explicitly outside the encoder-input contract.

Model-ready exports use separate commands and output contracts:

- `export-encoder-windows`: encoder features and masks only; valid for pretraining
- `export-probe-dataset`: frozen representations or encoder features joined with probe targets
- `export-alignment-dataset`: explicitly approved encoder-target pairs for later alignment work

Targets must never be exported into the same model-ready file used for
pretraining. Pretraining loaders must also reject target columns by schema,
rather than relying only on users to select the correct columns.

### Formats

Long format is best for raw prices, macro series, factor series, and appendable datasets.

Wide format is best for model matrices, rolling correlations, and diagnostics.

MultiIndex format is best inside pandas when computing cross-sectional features.

## 6. Feature engineering

All encoder features must be known at date t. If you define the representation as “after the close on date t,” then close, volume, and realized features through t are allowed. Targets start at t+1.

### Market-wide features

Let P_t be adjusted or chosen close, and r_t = log(P_t / P_{t-1}).

### Return

ret_h(t) = log(P_t / P_{t-h})
h ∈ {1, 5, 21, 63, 126}

### Realized volatility

rv_h(t) = sqrt(252) * std(r_{t-h+1}, ..., r_t)

### Rolling drawdown

dd_h(t) = P_t / max(P_{t-h+1}, ..., P_t) - 1

### Trend strength

trend_strength_h(t) = ret_h(t) / (rv_h(t) + eps)

### Path efficiency

path_eff_h(t) = abs(log(P_t / P_{t-h})) / sum_{j=t-h+1}^{t} abs(r_j)

### Moving average distance

ma_dist_h(t) = P_t / SMA_h(P)_t - 1

### Volatility-of-volatility

vov_h(t) = rolling_std(rv_21, h)

### Volume shock

volume_shock_h(t) = log(volume_t / rolling_median(volume, h)_t)

### Correlation regime

Use rolling correlations among sector ETF returns or major asset proxies:

avg_corr_h(t) = mean upper-triangle corr(asset_returns over last h days)

### Cross-sectional features

For a stock/ETF universe U_t:

dispersion_1d(t) = std_i(r_{i,t})
iqr_1d(t) = Q75_i(r_{i,t}) - Q25_i(r_{i,t})
breadth_1d(t) = mean_i(1[r_{i,t} > 0])
pct_above_ma_63(t) = mean_i(1[P_{i,t} > SMA_63(P_i)_t])
sector_dispersion(t) = std_s(mean_{i in sector s}(r_{i,t}))

### Average pairwise correlation

avg_pairwise_corr_h(t) = mean corr(r_i, r_j) over rolling h-day window

### Sector concentration

sector_return_concentration(t) =
sum_s abs(sector_return_s(t)) / sum_i abs(stock_return_i(t))

If market cap is unavailable, use equal-weight approximations and label them as such.

### Macro/state features

Use daily market/rate series first:

vix_level
vix_1d_change
vix_21d_change
credit_spread_level
credit_spread_21d_change
yield_curve_10y_2y
yield_curve_10y_3m
fed_funds_level
financial_stress_level

The default first-model feature set excludes slow revised macro series such as
CPI, unemployment, payrolls, GDP, industrial production, and NFCI. Evaluate
them in a separate macro ablation using ALFRED vintages or conservative release
lags. Do not align a monthly value to the period it describes as if it were
known during that period.

### Calendar features ablation

Run three versions.

1. First, no calendar features. This should be your default JEPA pretraining dataset.

2. Second, cyclical calendar only:

sin_month = sin(2π * month / 12)
cos_month = cos(2π * month / 12)
sin_dow = sin(2π * day_of_week / 5)
cos_dow = cos(2π * day_of_week / 5)

3. Third, calendar with dropout. Randomly mask calendar features during training so the model cannot lean too much on seasonality.

Avoid raw year, raw date index, days since 1900, or anything that lets the encoder memorize “2008,” “2020,” or “2022” directly.

### Feature-group availability and missingness

Basic availability masks are required in Phase 1. Track availability by feature group, including rates, credit, sector ETFs,
commodity proxies, international proxies, macro series, and equity
cross-sectional features. Report the first usable date and coverage rate for
each group.

Missingness is information, not zero. Model-ready samples should carry masks
that distinguish unavailable history, not listed yet, after last observed,
confirmed delisted, missing expected, stale price, and source failure. Do not
expose raw ticker identity solely to explain missing values.

## 7. Target engineering

Targets are not encoder inputs.

For horizon h ∈ {21, 63, 126}, future windows always use dates t+1 through t+h.

### Future realized volatility

future_rv_h(t) = sqrt(252) * std(r_{t+1}, ..., r_{t+h})

Continuous and bucketed. Good for linear probing and model selection. Uses future information. Keep in separate target files.

### Future trend

future_return_h(t) = log(P_{t+h} / P_t)
future_trend_score_h(t) = future_return_h(t) / (future_rv_h(t) + eps)

### Bucket
strong_down, down, flat, up, strong_up, using train-fold quantiles.

Good for probing and later alignment.

### Future maximum drawdown

future_mdd_h(t) =
min_{k in 1..h} P_{t+k} / max(P_t, P_{t+1}, ..., P_{t+k}) - 1

Continuous and bucketed. Good for risk-state probing.

### Future cross-sectional dispersion

future_dispersion_h(t) = mean_{u=t+1}^{t+h} std_i(r_{i,u})

Continuous and bucketed. Good for market-state probing.

### Future breadth

future_breadth_h(t) = mean_i(1[log(P_{i,t+h}/P_{i,t}) > 0])

Continuous in [0,1], optionally bucketed. Good for probing.

### Future average correlation

future_avg_corr_h(t) = mean upper-triangle corr(stock or sector returns from t+1 to t+h)

Continuous and bucketed. Good for correlation-regime probing.

### Future downside tail risk

future_var_5_h(t) = 5th percentile of daily market returns over t+1..t+h
future_cvar_5_h(t) = mean returns below that percentile

Continuous. Good for risk probing.

### Later transition targets

After the core targets and return semantics are validated, add targets that
measure changes in state rather than only state levels:

- future downside and upside volatility
- future realized skew
- future volatility change relative to trailing volatility
- future correlation change
- future dispersion change
- volatility-regime transition labels

### Future regime label

### Example rule-based label

crash:    future_return_21 < q10 and future_mdd_21 < q10_mdd
rebound:  past_return_21 < q20 and future_return_21 > q80
trend:    abs(future_trend_score_63) > q80 and path_efficiency_future_63 high
chop:     abs(future_return_63) low and realized_vol_63 medium/high
calm_up:  future_return_63 > median and future_rv_63 < median

Use only train-fold thresholds when creating buckets for model selection. For fixed economic labels, store the thresholds in configs/targets.yaml.

## 8. Leakage and validation rules

Use these rules as hard checks.

No future data in encoder inputs. Every feature row should have:

date
available_asof
max_source_date_used
uses_future_data = false

Then assert:

max_source_date_used <= date
available_asof <= date

No global normalization. Use rolling z-scores or fit scalers only on training folds.

Treat folds and fitted preprocessing artifacts as dataset build products:

- `split_manifest`: fold ID and train, validation, and test date boundaries
- `normalization_artifacts`: feature, fold ID, parameters, and fit boundaries
- `bucket_thresholds`: target, horizon, fold ID, thresholds, and fit boundaries

These artifacts must be fit using training dates only and stored with the
`build_id`.

### Required dataset sanity gates

Every immutable build must run and record pass/fail sanity gates before it can
be used for JEPA training. Thresholds belong in versioned validation config.
Required gates include:

- Stooq SPY returns approximately match an independent SPY source
- inferred SPY calendar approximately matches expected U.S. trading days
- FRED VIXCLS approximately matches CBOE VIX over overlapping dates
- an equal-weight sector ETF basket broadly co-moves with SPY
- realized-volatility and drawdown targets behave plausibly around 2008, 2020, and 2022
- feature matrices contain no unexpected infinities, impossible values, or stale flatlines
- coverage and feature-group availability satisfy configured minimums

A failed required gate makes the build invalid for training unless the failure
is explicitly waived and recorded in the immutable build manifest.

No pretraining on future test dates if the goal is honest walk-forward evaluation. Even self-supervised pretraining can leak future distributional information.

Splits should be rolling or expanding:

fold_1 train: 1998-2009, val: 2010-2011, test: 2012-2013
fold_2 train: 1998-2011, val: 2012-2013, test: 2014-2015
...

Macro features need availability lags. Daily market series can usually be treated as available after the close. Monthly/quarterly macro should use ALFRED or conservative release lags.

Macro rolling and change features preload the largest configured window of
business-day source history before the first published dataset date. Warmup
rows are used only for feature calculation and are removed before publication.
Null-valued source observations do not replace the last valid available value.

No revised macro data unless explicitly accepted. Add:

revised_data_flag: true
point_in_time_available: false

No absolute date leakage. Avoid raw date integers and year embeddings.

No ticker identity leakage during JEPA training if you want anonymous assets. Use instrument_slot within each sample, randomize order, mask identity, and store identity only for evaluation.

No target leakage through scaling. Bucket thresholds and scalers must be computed on train only.

## 9. Survivorship bias strategy

With free data, the realistic path is:

Start with market-wide and sector/factor data. This gives you useful state representation without pretending you have a clean stock universe.

The optional Phase 3 dataset includes current S&P 500 constituents. This is biased because it takes today’s successful surviving large-cap companies and backfills their histories. It excludes companies that were once important but later failed, merged, delisted, or dropped out.

### What you can conclude from a current-constituent dataset

You can study whether the representation captures broad volatility, trend, drawdown, dispersion among current survivors, sector structure, and conditional behavior of today’s large-cap universe.

### What you cannot conclude

You cannot make clean claims about historical stock-selection performance, historical investable universes, delisting risk, small-cap behavior, bankruptcies, or realistic index membership.

### Metadata flags

```yaml
dataset_bias:
  survivorship_bias: high
  point_in_time_universe: false
  delisted_assets_included: false
  current_constituents_backfilled: true
  acceptable_uses:
    - representation smoke tests
    - market-state diagnostics
    - feature engineering development
  unacceptable_uses:
    - publishable stock-selection backtest
    - claims of investable historical S&P 500 performance
    - delisting-risk research
```

Later, if you get Norgate, Sharadar, WRDS, or CRSP, keep the same feature pipeline but swap the symbol manifest, listing/delisting metadata, adjustment fields, and point-in-time universe membership.

## 10. Implementation plan

### Phase 1: market-wide dataset only

| Area | Details |
|---|---|
| Deliverables | immutable build artifact, raw Stooq/CBOE/FRED/French downloads, normalized daily prices, daily market-implied features, basic availability masks, target table, leakage report, sanity-gate report, and separate encoder/probe exports. |
| Scripts | stooq_loader.py, calendar.py, market_features.py, targets.py, build_market_database.py. |
| Validation | required sanity gates, date monotonicity, duplicate rows, OHLC sanity, missingness masks, no future features, target starts at t+1, and target-free pretraining exports. |
| Expected files | `data/processed/builds/<build_id>/market_data.duckdb`, immutable manifest and reports, plus `data/model_ready/<build_id>/<fold_id>/` exports. |
| Minimum success | you can produce a masked daily feature matrix with SPY/QQQ/IWM/sectors/VIX/rates/credit/factors and separate target labels for 21/63/126-day future regimes. |

### Phase 2: add Stooq index/ETF/sector/factor-style instruments

| Area | Details |
|---|---|
| Deliverables | larger symbol manifest, international proxies, commodities, bonds, dollar, sector ETFs, correlation features. |
| Scripts | symbol_index.py, alignment.py, cross_asset_features.py. |
| Validation | coverage by symbol, inception dates, stale prices, missing days, correlation matrix stability. |
| Minimum success | you can train a JEPA-style model on market-state windows without individual stock identity. |

### Phase 3: add current S&P 500 cross-section

This phase remains disabled by default until the market-wide Phase 1 and Phase
2 builds pass their required sanity gates.

| Area | Details |
|---|---|
| Deliverables | current S&P 500 manifest, Stooq prices, sector labels, cross-sectional features, survivorship-bias report. |
| Scripts | sp500_current_universe_loader.py, cross_sectional_features.py, dispersion_targets.py. |
| Validation | per-symbol coverage, missingness heatmaps, sector distribution, equal-weight index sanity check versus SPY. |
| Minimum success | you can compute breadth, dispersion, percent above moving average, sector dispersion, and rolling average correlation. |

### Phase 4: add better point-in-time or delisted-aware data

| Area | Details |
|---|---|
| Deliverables | new source adapter, point-in-time symbol metadata, delisting fields, corporate action validation, proper investable universe snapshots. |
| Scripts | norgate_loader.py or sharadar_loader.py, corporate_actions.py, universe_membership.py. |
| Validation | delisted coverage, split/dividend adjustment checks, membership date checks, comparison against benchmark index returns. |
| Minimum success | you can distinguish “asset did not exist,” “asset existed but did not trade,” “asset delisted,” and “asset missing from source.” |

### Later hardening backlog

These items are required before stronger research claims or production use,
but they should not block the first validated market-state representation
dataset:

- point-in-time and delisted-aware equity membership
- ticker reuse and entity continuity risk flags
- verified split, dividend, and total-return handling
- ALFRED vintages for revision-sensitive macro series
- transition and higher-moment targets
- intraday or multiple time-of-decision conventions

## 11. Example configs

### `configs/universe.yaml`

```yaml
dataset_name: jepa_market_state_v1
dataset_schema_version: "0.1.0"
feature_schema_version: "0.1.0"
target_schema_version: "0.1.0"
decision_time_convention_version: "0.1.0"
frequency: daily
base_calendar_symbol: SPY.US
start_date: "1990-01-01"
end_date: null

market_indices:
  - canonical_symbol: IDX_SPX
    source: stooq
    source_symbol: "^spx"
    asset_type: index
  - canonical_symbol: ETF_SPY
    source: stooq
    source_symbol: "spy.us"
    asset_type: etf
  - canonical_symbol: ETF_QQQ
    source: stooq
    source_symbol: "qqq.us"
    asset_type: etf
  - canonical_symbol: ETF_IWM
    source: stooq
    source_symbol: "iwm.us"
    asset_type: etf

sector_etfs:
  - { canonical_symbol: ETF_XLB, source_symbol: "xlb.us", sector: Materials }
  - { canonical_symbol: ETF_XLC, source_symbol: "xlc.us", sector: Communication Services }
  - { canonical_symbol: ETF_XLE, source_symbol: "xle.us", sector: Energy }
  - { canonical_symbol: ETF_XLF, source_symbol: "xlf.us", sector: Financials }
  - { canonical_symbol: ETF_XLI, source_symbol: "xli.us", sector: Industrials }
  - { canonical_symbol: ETF_XLK, source_symbol: "xlk.us", sector: Information Technology }
  - { canonical_symbol: ETF_XLP, source_symbol: "xlp.us", sector: Consumer Staples }
  - { canonical_symbol: ETF_XLRE, source_symbol: "xlre.us", sector: Real Estate }
  - { canonical_symbol: ETF_XLU, source_symbol: "xlu.us", sector: Utilities }
  - { canonical_symbol: ETF_XLV, source_symbol: "xlv.us", sector: Health Care }
  - { canonical_symbol: ETF_XLY, source_symbol: "xly.us", sector: Consumer Discretionary }

macro_series:
  - { series_id: VIXCLS, name: VIX, source: fred, lag: same_day }
  - { series_id: DGS3MO, name: Treasury 3M, source: fred, lag: same_day }
  - { series_id: DGS2, name: Treasury 2Y, source: fred, lag: same_day }
  - { series_id: DGS10, name: Treasury 10Y, source: fred, lag: same_day }
  - { series_id: BAMLH0A0HYM2, name: High Yield OAS, source: fred, lag: same_day }
  - { series_id: NFCI, name: Chicago Fed NFCI, source: fred, lag: conservative_7bd }
  - { series_id: UNRATE, name: Unemployment Rate, source: fred, lag: conservative_5bd }
  - { series_id: CPIAUCSL, name: CPI, source: fred, lag: conservative_15bd }

factor_series:
  - { dataset: F-F_Research_Data_Factors_daily, source: french }

optional_equity_universe:
  enabled: false
  universe_type: current_sp500_backfilled
  survivorship_bias: high
```

### `configs/sources.yaml`

```yaml
stooq:
  enabled: true
  interval: d
  raw_dir: data/raw/stooq/single_symbol
  adjusted_flag: false
  request_sleep_seconds: 0.5
  retry_count: 3

fred:
  enabled: true
  raw_dir: data/raw/fred
  api_key_env: FRED_API_KEY
  use_alfred_vintages: false

french:
  enabled: true
  raw_dir: data/raw/french

cboe:
  enabled: true
  raw_dir: data/raw/cboe

local_csv:
  enabled: true
  raw_dir: data/raw/metadata
```

### `configs/targets.yaml`

```yaml
horizons: [21, 63, 126]

volatility:
  enabled: true
  annualize: true
  buckets:
    method: train_quantile
    n_bins: 5

trend:
  enabled: true
  normalize_by_future_vol: true
  buckets:
    method: train_quantile
    n_bins: 5

drawdown:
  enabled: true
  buckets:
    method: train_quantile
    n_bins: 5

dispersion:
  enabled: true
  universe: optional_equity_universe
  buckets:
    method: train_quantile
    n_bins: 5

regime_labels:
  enabled: true
  labels:
    - crash
    - rebound
    - trend
    - chop
    - calm_up
```

## 12. Python code skeletons

### Install shape

```powershell
pip install pandas numpy pyarrow requests pydantic duckdb pandas-datareader
```

### `src/data_sources/stooq_loader.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from io import StringIO
import time

import pandas as pd
import requests


@dataclass(frozen=True)
class StooqSymbol:
    canonical_symbol: str
    source_symbol: str
    asset_type: str
    exchange: str | None = None
    currency: str | None = "USD"


class StooqLoader:
    def __init__(self, raw_dir: Path, sleep_seconds: float = 0.5):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = sleep_seconds

    def download_symbol_csv(self, source_symbol: str, interval: str = "d") -> Path:
        """
        Convenience downloader for Stooq single-symbol CSV.

        This is not an official API contract. Cache the raw result.
        """
        safe_name = source_symbol.replace("^", "idx_").replace(".", "_")
        out = self.raw_dir / f"{safe_name}_{interval}.csv"

        if out.exists() and out.stat().st_size > 0:
            return out

        url = "https://stooq.com/q/d/l/"
        params = {"s": source_symbol, "i": interval}

        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()

        text = r.text.strip()
        if not text or "No data" in text or len(text.splitlines()) < 2:
            raise ValueError(f"No usable Stooq data for {source_symbol}")

        out.write_text(text)
        time.sleep(self.sleep_seconds)
        return out

    def load_csv(
        self,
        path: Path,
        symbol: StooqSymbol,
        source: str = "stooq",
        adjusted_flag: bool = False,
    ) -> pd.DataFrame:
        df = pd.read_csv(path)

        df.columns = [c.lower().strip() for c in df.columns]
        expected = {"date", "open", "high", "low", "close", "volume"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["symbol"] = symbol.canonical_symbol
        df["source_symbol"] = symbol.source_symbol
        df["source"] = source
        df["adjusted_flag"] = adjusted_flag
        df["currency"] = symbol.currency
        df["exchange"] = symbol.exchange
        df["asset_type"] = symbol.asset_type
        df["raw_file"] = str(path)
        df["quality_flag"] = "ok"

        cols = [
            "date", "symbol", "source_symbol",
            "open", "high", "low", "close", "volume",
            "source", "adjusted_flag", "currency", "exchange",
            "asset_type", "raw_file", "quality_flag",
        ]
        return df[cols].sort_values(["symbol", "date"])
```

### `src/data_sources/fred_loader.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import pandas as pd
import requests


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    name: str
    release_lag_assumption: str = "same_day"
    point_in_time_available: bool = False


class FredLoader:
    def __init__(self, raw_dir: Path, api_key: str | None = None):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    def download_series(self, series: FredSeries) -> Path:
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY is required for official FRED API use.")

        out = self.raw_dir / f"{series.series_id}.json"
        if out.exists() and out.stat().st_size > 0:
            return out

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series.series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        out.write_text(r.text)
        return out

    def load_series(self, path: Path, series: FredSeries) -> pd.DataFrame:
        raw = pd.read_json(path)
        obs = pd.DataFrame(raw["observations"].tolist())

        obs["date"] = pd.to_datetime(obs["date"]).dt.date
        obs["value"] = pd.to_numeric(obs["value"].replace(".", pd.NA), errors="coerce")

        df = obs[["date", "value"]].copy()
        df["series_id"] = series.series_id
        df["source"] = "fred"
        df["release_lag_assumption"] = series.release_lag_assumption
        df["point_in_time_available"] = series.point_in_time_available
        df["revised_data_flag"] = not series.point_in_time_available

        return df[[
            "date", "series_id", "value", "source",
            "release_lag_assumption",
            "point_in_time_available",
            "revised_data_flag",
        ]]
```

### `src/indexing/symbol_index.py`

```python
from __future__ import annotations

import hashlib
import pandas as pd


def make_instrument_id(source: str, source_symbol: str, first_date: str | None) -> str:
    key = f"{source}|{source_symbol}|{first_date or 'unknown'}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def build_symbol_manifest(price_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        price_df.groupby(["symbol", "source", "source_symbol"], as_index=False)
        .agg(
            first_available_date=("date", "min"),
            last_available_date=("date", "max"),
            n_rows=("date", "size"),
        )
    )

    manifest = coverage.merge(
        metadata,
        on=["symbol", "source", "source_symbol"],
        how="left",
    )

    manifest["instrument_id"] = [
        make_instrument_id(src, ss, str(fd))
        for src, ss, fd in zip(
            manifest["source"],
            manifest["source_symbol"],
            manifest["first_available_date"],
        )
    ]

    manifest["survivorship_status"] = manifest.get(
        "survivorship_status", "unknown"
    ).fillna("unknown")

    manifest["point_in_time_valid"] = manifest.get(
        "point_in_time_valid", False
    ).fillna(False)

    return manifest
```

### `src/indexing/calendar.py`

```python
from __future__ import annotations

import pandas as pd


def infer_trading_calendar(
    price_df: pd.DataFrame,
    reference_symbol: str = "ETF_SPY",
) -> pd.DataFrame:
    ref = price_df.loc[
        (price_df["symbol"] == reference_symbol) & price_df["close"].notna(),
        ["date"],
    ].drop_duplicates()

    cal = ref.sort_values("date").copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["is_trading_day"] = True
    cal["calendar_name"] = f"inferred_from_{reference_symbol}"
    cal["year"] = cal["date"].dt.year
    cal["month"] = cal["date"].dt.month
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["quarter"] = cal["date"].dt.quarter
    cal["date"] = cal["date"].dt.date
    return cal
```

### `src/indexing/alignment.py`

```python
from __future__ import annotations

import pandas as pd


def align_prices_to_calendar(
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    dates = calendar.loc[calendar["is_trading_day"], "date"].sort_values().unique()
    symbols = manifest["symbol"].sort_values().unique()

    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    aligned = prices.set_index(["date", "symbol"]).reindex(idx).reset_index()

    aligned = aligned.merge(
        manifest[["symbol", "first_available_date", "last_available_date", "survivorship_status"]],
        on="symbol",
        how="left",
    )

    aligned["observation_status"] = "ok"
    aligned.loc[aligned["close"].isna(), "observation_status"] = "missing_expected"

    aligned.loc[
        aligned["date"] < aligned["first_available_date"],
        "observation_status",
    ] = "not_listed_yet"

    aligned.loc[
        aligned["date"] > aligned["last_available_date"],
        "observation_status",
    ] = "after_last_observed"

    aligned.loc[
        (aligned["date"] > aligned["last_available_date"])
        & (aligned["survivorship_status"] == "confirmed_delisted"),
        "observation_status",
    ] = "confirmed_delisted"

    aligned["valid_observation"] = aligned["observation_status"].eq("ok")
    return aligned
```

### `src/features/market_features.py`

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def add_asset_features(df: pd.DataFrame, windows=(5, 21, 63, 126)) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)

    df["log_close"] = np.log(df["close"])
    df["return_1d"] = g["log_close"].diff()
    df["dollar_volume"] = df["close"] * df["volume"]

    for h in windows:
        df[f"return_{h}d"] = g["log_close"].diff(h)
        df[f"realized_vol_{h}d"] = (
            g["return_1d"]
            .rolling(h, min_periods=max(5, h // 2))
            .std()
            .reset_index(level=0, drop=True)
            * np.sqrt(252)
        )

        rolling_max = (
            g["close"]
            .rolling(h, min_periods=max(5, h // 2))
            .max()
            .reset_index(level=0, drop=True)
        )
        df[f"drawdown_{h}d"] = df["close"] / rolling_max - 1.0

        sma = (
            g["close"]
            .rolling(h, min_periods=max(5, h // 2))
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"ma_distance_{h}d"] = df["close"] / sma - 1.0

        abs_path = (
            g["return_1d"]
            .rolling(h, min_periods=max(5, h // 2))
            .apply(lambda x: np.nansum(np.abs(x)), raw=True)
            .reset_index(level=0, drop=True)
        )
        df[f"path_efficiency_{h}d"] = df[f"return_{h}d"].abs() / (abs_path + 1e-12)

    return df.drop(columns=["log_close"])
```

### `src/features/cross_sectional_features.py`

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def build_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    valid = panel.loc[panel["valid_observation"]].copy()

    def iqr(x: pd.Series) -> float:
        return x.quantile(0.75) - x.quantile(0.25)

    out = (
        valid.groupby("date")
        .agg(
            xs_dispersion_1d=("return_1d", "std"),
            xs_iqr_1d=("return_1d", iqr),
            breadth_1d=("return_1d", lambda x: float((x > 0).mean())),
            n_assets=("symbol", "nunique"),
        )
        .reset_index()
    )

    if "ma_distance_63d" in valid.columns:
        above = (
            valid.groupby("date")["ma_distance_63d"]
            .apply(lambda x: float((x > 0).mean()))
            .rename("pct_above_ma_63d")
            .reset_index()
        )
        out = out.merge(above, on="date", how="left")

    return out
```

### `src/targets/target_builder.py`

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def future_realized_volatility(market: pd.DataFrame, horizon: int) -> pd.Series:
    # Uses future returns t+1 ... t+h
    return (
        market["return_1d"]
        .shift(-1)
        .rolling(horizon, min_periods=horizon)
        .std()
        .shift(-(horizon - 1))
        * np.sqrt(252)
    )


def future_return(market: pd.DataFrame, horizon: int) -> pd.Series:
    return np.log(market["close"].shift(-horizon) / market["close"])


def future_max_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    values = []
    arr = close.to_numpy(dtype=float)

    for i in range(len(arr)):
        end = i + horizon + 1
        if end > len(arr) or not np.isfinite(arr[i]):
            values.append(np.nan)
            continue

        path = arr[i:end]
        running_max = np.maximum.accumulate(path)
        dd = path / running_max - 1.0
        values.append(np.nanmin(dd[1:]))

    return pd.Series(values, index=close.index)


def build_market_targets(
    panel: pd.DataFrame,
    market_symbol: str = "ETF_SPY",
    horizons=(21, 63, 126),
) -> pd.DataFrame:
    market = (
        panel.loc[(panel["symbol"] == market_symbol) & panel["valid_observation"]]
        .sort_values("date")
        .copy()
        .reset_index(drop=True)
    )

    out = market[["date"]].copy()
    out["symbol"] = market_symbol

    for h in horizons:
        out[f"future_return_{h}d"] = future_return(market, h)
        out[f"future_realized_vol_{h}d"] = future_realized_volatility(market, h)
        out[f"future_max_drawdown_{h}d"] = future_max_drawdown(market["close"], h)
        out[f"future_trend_score_{h}d"] = (
            out[f"future_return_{h}d"] / (out[f"future_realized_vol_{h}d"] + 1e-12)
        )

    out["uses_future_data"] = True
    return out
```

### `src/validation/checks.py`

```python
from __future__ import annotations

import pandas as pd


def check_no_duplicate_price_rows(prices: pd.DataFrame) -> None:
    dup = prices.duplicated(["date", "symbol"]).sum()
    if dup:
        raise AssertionError(f"Duplicate date-symbol price rows: {dup}")


def check_ohlc_sanity(prices: pd.DataFrame) -> pd.DataFrame:
    bad = prices.loc[
        (prices["high"] < prices[["open", "close", "low"]].max(axis=1))
        | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))
        | (prices["close"] <= 0)
    ].copy()

    return bad


def check_feature_leakage(features: pd.DataFrame) -> None:
    if "uses_future_data" in features.columns:
        bad = features["uses_future_data"].fillna(False).sum()
        if bad:
            raise AssertionError(f"Encoder features contain future-data rows: {bad}")

    if {"max_source_date_used", "date"}.issubset(features.columns):
        bad = pd.to_datetime(features["max_source_date_used"]) > pd.to_datetime(features["date"])
        if bad.any():
            raise AssertionError(f"Feature rows use data after t: {bad.sum()}")


def check_train_scaler_dates(train_end, scaler_fit_end) -> None:
    if pd.Timestamp(scaler_fit_end) > pd.Timestamp(train_end):
        raise AssertionError("Scaler was fit beyond train_end.")
```

### `src/utils/io.py`

```python
from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd


def write_market_database(
    database_path: Path,
    tables: dict[str, pd.DataFrame],
    views: dict[str, str],
) -> None:
    with duckdb.connect(str(database_path)) as connection:
        for table_name, frame in tables.items():
            connection.register("source_frame", frame)
            connection.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM source_frame'
            )
            connection.unregister("source_frame")
        for view_name, query in views.items():
            connection.execute(f'CREATE OR REPLACE VIEW "{view_name}" AS {query}')
```

### `src/dataset_pipeline/build_market_database.py`

```python
write_market_database(
    DATA / "processed" / "market_data.duckdb",
    tables={
        "features": features,
        "targets": targets,
        "symbol_manifest": symbol_manifest,
        "trading_calendar": trading_calendar,
    },
    views={
    },
)
```

## 13. Recommended first dataset

Build the canonical dataset with:

```powershell
uv run build-market-database
```

### Minimum viable symbol list

| Category | Symbols |
|---|---|
| Core equity | SPY.US, QQQ.US, IWM.US, DIA.US |
| Sectors | XLB.US, XLC.US, XLE.US, XLF.US, XLI.US, XLK.US, XLP.US, XLRE.US, XLU.US, XLV.US, XLY.US |
| Rates/bonds proxies | TLT.US, IEF.US, SHY.US, AGG.US |
| Credit proxies | HYG.US, LQD.US |
| Other macro proxies | GLD.US, USO.US, UUP.US, EFA.US, EEM.US |
| Indices if available | ^SPX, ^RUT, ^DJI, Nasdaq proxy |

### Minimum viable macro list

Default first-model inputs:

- VIXCLS
- DGS3MO
- DGS2
- DGS10
- DGS30
- T10Y2Y
- BAMLH0A0HYM2
- BAMLC0A0CM
- DFF or FEDFUNDS

Separate slow/revised macro ablation:

- NFCI
- UNRATE
- CPIAUCSL
- INDPRO

### Minimum viable target list

- future_realized_vol_21d, 63d, 126d
- future_return_21d, 63d, 126d
- future_trend_score_21d, 63d, 126d
- future_max_drawdown_21d, 63d, 126d
- future_tail_risk_63d
- future_average_sector_correlation_63d
- future_regime_label_63d

### Checklist before JEPA pretraining

- [ ] Raw files are cached and never overwritten silently.
- [ ] The canonical DuckDB and manifest live under an immutable `build_id`.
- [ ] Dataset, feature, target, and decision-time schema versions are recorded.
- [ ] Every feature has max_source_date_used <= date.
- [ ] Targets are stored separately from encoder inputs.
- [ ] All future-return columns are excluded from pretraining inputs.
- [ ] Encoder-window exports contain no target columns and loaders enforce that schema.
- [ ] No scaler is fit across train/val/test together.
- [ ] Walk-forward splits are defined before probing.
- [ ] Macro data is either lagged or marked revised/non-point-in-time.
- [ ] Current-constituent equity data is marked survivorship-biased.
- [ ] Ticker identity can be masked or randomized.
- [ ] Asset order can be randomized inside training samples.
- [ ] Absolute dates are excluded or ablated.
- [ ] Dataset manifest, source snapshot date, and configs are saved.
- [ ] Live build scope, enabled configs, and manifest date range agree.
- [ ] The after-close decision-time convention is recorded in build metadata.
- [ ] Price adjustment and return-source semantics are documented and validated.
- [ ] Bad OHLC and other failed quality rows are excluded from valid observations.
- [ ] Build ID, config hash, and source content hashes are saved.
- [ ] Dataset sanity baselines pass for calendar, benchmark returns, and coverage.
- [ ] Basic feature-group availability masks are present in model-ready samples.
- [ ] Fold, normalization, and bucket artifacts are fit on training dates only.
- [ ] The current S&P 500 cross-section remains disabled unless explicitly requested.

The biggest ways this dataset could mislead you:

1. First, current S&P 500 backfills can make cross-sectional structure look cleaner than it was historically.

2. Second, Stooq price adjustment behavior and ticker continuity need validation before you trust single-stock return histories.

3. Third, revised FRED macro data can leak information unless you use ALFRED or conservative lags.

4. Fourth, ETF proxies have inception-date problems. Sector ETFs and bond ETFs do not give you a clean pre-inception history.

5. Fifth, a JEPA model can memorize dates, crises, or ticker identities unless you explicitly mask or ablate those channels.

6. Sixth, equal-weight cross-sectional features from surviving stocks can exaggerate breadth, dispersion, and rebound quality.

7. Seventh, a representation that probes well on future volatility may still be useless for trading. For this stage, that is fine: the first goal is to learn a stable market-state representation, not prove a tradable signal.
