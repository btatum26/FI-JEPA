# Current Canonical and Model-Ready Schema

Canonical artifact:

```text
data/processed/market_data.duckdb
```

Canonical build timestamp: `2026-06-10 17:19:03.744982 UTC`

Latest verified model-ready artifact:

```text
data/model_ready/fi_jepa_sparse_v1/20260610T173759Z_fa611756af2c4816/
```

Model-ready build timestamp: `2026-06-10 17:37:59.282024 UTC`

The DuckDB database is the canonical source of truth. Model-ready Parquet
artifacts are immutable, normalized, split-aware build products derived from
the canonical database. They are not a second canonical database.

## Core Contract

The database has two physically separate past-only feature tables:

| Table | Grain | Rows | Columns |
|---|---|---:|---:|
| `features` | One row per trading date | 16,217 | 53 |
| `ticker_features` | One row per trading date and ticker | 21,098,223 | 46 |

Future values exist only in `targets`.

There is no published `macro_data` table, `bad_ohlc` table, or view joining
features to targets. Raw FRED JSON files and bad-OHLC findings are build inputs
and build metadata, not canonical database tables.

Current validation:

| Check | Result |
|---|---:|
| Duplicate `features.date` rows | 0 |
| Duplicate `ticker_features.(date, symbol)` rows | 0 |
| Leakage violations in `features` | 0 |
| Leakage violations in `ticker_features` | 0 |
| Target-like columns in either feature table | 0 |

## Table Inventory

| Table | Rows | Purpose |
|---|---:|---|
| `features` | 16,217 | Date-level market and macro state |
| `ticker_features` | 21,098,223 | Ticker-specific prices, status, and rolling features |
| `targets` | 5,353 | Future SPY outcomes |
| `symbol_manifest` | 531 | Symbol identity and coverage metadata |
| `trading_calendar` | 39,733 | Union of all observed selected-symbol dates |
| `build_metadata` | 1 | Published build summary |
| `community_current_constituents` | 634 | Current-constituent source snapshot |
| `community_changes` | 613 | Community-maintained membership changes |

The current schema has no declared primary keys, foreign keys, unique
constraints, or `NOT NULL` constraints. Keys below are logical contracts
enforced by build validation.

## `features`

Purpose: market-wide, cross-sectional, and macro encoder inputs.

Logical key:

```text
date
```

Date range: `1962-01-02` through `2026-06-08`.

### Market-State Columns

| Column | Type | Meaning |
|---|---|---|
| `date` | `DATE` | Trading date |
| `xs_dispersion_1d` | `DOUBLE` | Cross-sectional stock-return standard deviation |
| `xs_iqr_1d` | `DOUBLE` | Cross-sectional stock-return interquartile range |
| `breadth_1d` | `DOUBLE` | Fraction of valid stocks with positive returns |
| `n_assets` | `BIGINT` | Valid stock count used for the date |
| `pct_above_ma_63d` | `DOUBLE` | Fraction of valid stocks above their 63-day moving average |

### Macro Columns

Each configured macro series has a level and changes over `1`, `5`, `21`, and
`63` trading rows. Macro construction preloads the largest configured window
of business-day history before the first published date, computes the changes,
and then removes the warmup rows. First-date changes are therefore populated
when the source series has sufficient older history.

| Family | Columns |
|---|---|
| VIX | `vix_level`, `vix_change_1d`, `vix_change_5d`, `vix_change_21d`, `vix_change_63d` |
| Treasury 3M | `treasury_3m_level`, `treasury_3m_change_1d`, `treasury_3m_change_5d`, `treasury_3m_change_21d`, `treasury_3m_change_63d` |
| Treasury 2Y | `treasury_2y_level`, `treasury_2y_change_1d`, `treasury_2y_change_5d`, `treasury_2y_change_21d`, `treasury_2y_change_63d` |
| Treasury 10Y | `treasury_10y_level`, `treasury_10y_change_1d`, `treasury_10y_change_5d`, `treasury_10y_change_21d`, `treasury_10y_change_63d` |
| Treasury 30Y | `treasury_30y_level`, `treasury_30y_change_1d`, `treasury_30y_change_5d`, `treasury_30y_change_21d`, `treasury_30y_change_63d` |
| High-yield OAS | `high_yield_oas_level`, `high_yield_oas_change_1d`, `high_yield_oas_change_5d`, `high_yield_oas_change_21d`, `high_yield_oas_change_63d` |
| Corporate OAS | `corporate_oas_level`, `corporate_oas_change_1d`, `corporate_oas_change_5d`, `corporate_oas_change_21d`, `corporate_oas_change_63d` |
| Fed funds | `fed_funds_level`, `fed_funds_change_1d`, `fed_funds_change_5d`, `fed_funds_change_21d`, `fed_funds_change_63d` |
| Derived spreads | `yield_curve_10y_2y`, `yield_curve_10y_3m`, `yield_curve_30y_10y`, `hy_minus_corporate_oas` |

All macro columns are `DOUBLE`.

### Availability Columns

| Column | Type | Meaning |
|---|---|---|
| `max_source_date_used` | `DATE` | Latest market or macro source observation used |
| `available_asof` | `DATE` | Latest availability date across joined date-level inputs |
| `uses_future_data` | `BOOLEAN` | Always `false` |

## `ticker_features`

Purpose: ticker-specific observations and derived encoder inputs.

Logical key:

```text
(date, symbol)
```

Contains 531 symbols across 39,733 observed trading dates.

Aligned date range: `1789-05-01` through `2026-06-08`, covering the union of
all observed dates for the selected Stooq instruments.

### Identity, Price, and Status Columns

| Column | Type | Meaning |
|---|---|---|
| `date` | `DATE` | Trading date |
| `symbol` | `VARCHAR` | Canonical symbol |
| `source_symbol` | `VARCHAR` | Source-native ticker |
| `open` | `DOUBLE` | Daily open |
| `high` | `DOUBLE` | Daily high |
| `low` | `DOUBLE` | Daily low |
| `close` | `DOUBLE` | Daily close |
| `volume` | `DOUBLE` | Daily volume |
| `source` | `VARCHAR` | Price source |
| `adjusted_flag` | `BOOLEAN` | Source adjustment flag |
| `currency` | `VARCHAR` | Instrument currency |
| `exchange` | `INTEGER` | Currently all-null; incorrectly inferred from null values |
| `asset_type` | `VARCHAR` | Instrument category |
| `download_timestamp` | `TIMESTAMP WITH TIME ZONE` | Source snapshot timestamp |
| `raw_file` | `VARCHAR` | Source archive/member identifier |
| `quality_flag` | `VARCHAR` | Price quality classification |
| `first_available_date` | `DATE` | First observed source date |
| `last_available_date` | `DATE` | Last observed source date |
| `survivorship_status` | `VARCHAR` | Known instrument status |
| `observation_status` | `VARCHAR` | Aligned-row availability classification |
| `valid_observation` | `BOOLEAN` | Whether the observation is valid |

Current observation statuses:

| Status | Rows |
|---|---:|
| `ok` | 3,758,086 |
| `not_listed_yet` | 17,337,021 |
| `missing_expected` | 3,094 |
| `after_last_observed` | 22 |

### Derived Ticker Features

All columns below are `DOUBLE`.

| Family | Columns |
|---|---|
| Returns | `return_1d`, `return_5d`, `return_21d`, `return_63d`, `return_126d` |
| Dollar volume | `dollar_volume` |
| Realized volatility | `realized_vol_5d`, `realized_vol_21d`, `realized_vol_63d`, `realized_vol_126d` |
| Drawdown | `drawdown_5d`, `drawdown_21d`, `drawdown_63d`, `drawdown_126d` |
| Moving-average distance | `ma_distance_5d`, `ma_distance_21d`, `ma_distance_63d`, `ma_distance_126d` |
| Path efficiency | `path_efficiency_5d`, `path_efficiency_21d`, `path_efficiency_63d`, `path_efficiency_126d` |

### Availability Columns

| Column | Type | Meaning |
|---|---|---|
| `max_source_date_used` | `DATE` | Latest ticker source observation used |
| `available_asof` | `DATE` | Date the ticker feature row is available |
| `uses_future_data` | `BOOLEAN` | Always `false` |

## `targets`

Purpose: future outcomes, physically separate from both feature tables.

Logical key:

```text
(date, symbol)
```

The current `symbol` is always `ETF_SPY`.

| Column | Type |
|---|---|
| `date` | `DATE` |
| `symbol` | `VARCHAR` |
| `future_return_21d` | `DOUBLE` |
| `future_realized_vol_21d` | `DOUBLE` |
| `future_max_drawdown_21d` | `DOUBLE` |
| `future_trend_score_21d` | `DOUBLE` |
| `future_return_63d` | `DOUBLE` |
| `future_realized_vol_63d` | `DOUBLE` |
| `future_max_drawdown_63d` | `DOUBLE` |
| `future_trend_score_63d` | `DOUBLE` |
| `future_return_126d` | `DOUBLE` |
| `future_realized_vol_126d` | `DOUBLE` |
| `future_max_drawdown_126d` | `DOUBLE` |
| `future_trend_score_126d` | `DOUBLE` |
| `uses_future_data` | `BOOLEAN` |

## Supporting Tables

### `symbol_manifest`

Logical key: `symbol`.

```text
symbol VARCHAR
source VARCHAR
source_symbol VARCHAR
first_available_date DATE
last_available_date DATE
n_rows BIGINT
asset_type VARCHAR
exchange INTEGER
currency VARCHAR
survivorship_status VARCHAR
point_in_time_valid BOOLEAN
universe_name VARCHAR
universe_type VARCHAR
survivorship_bias VARCHAR
ticker VARCHAR
name VARCHAR
point_in_time_membership BOOLEAN
source_file VARCHAR
sector INTEGER
sector_metadata_status VARCHAR
instrument_id VARCHAR
```

`exchange` and `sector` are currently all-null and inferred as `INTEGER`; their
intended semantic type is string metadata.

### `trading_calendar`

Logical key: `date`.

```text
date DATE
is_trading_day BOOLEAN
calendar_name VARCHAR
year INTEGER
month INTEGER
week BIGINT
quarter INTEGER
```

### `build_metadata`

One-row build summary. It records both feature-table row counts and the
build-time-only macro and bad-OHLC counts.

```text
dataset_name VARCHAR
database_path VARCHAR
universe_type VARCHAR
survivorship_bias VARCHAR
point_in_time_membership BOOLEAN
price_source VARCHAR
symbol_count BIGINT
price_row_count BIGINT
ticker_feature_row_count BIGINT
feature_row_count BIGINT
macro_series_count BIGINT
macro_observation_count BIGINT
bad_ohlc_row_count BIGINT
target_row_count BIGINT
first_date VARCHAR
last_date VARCHAR
raw_price_first_date VARCHAR
raw_price_last_date VARCHAR
build_timestamp VARCHAR
source_snapshot_date VARCHAR
unavailable_source_symbols VARCHAR
```

### Community Provenance

`community_current_constituents`:

```text
symbol, name, universe_name, source_file, source_symbol, universe_type,
survivorship_bias, point_in_time_membership
```

`community_changes`:

```text
effective_date, added_symbol, added_name, removed_symbol, removed_name,
universe_name, source_file
```

## Relationships

```text
trading_calendar.date
    1 -> 1    features.date
    1 -> many ticker_features.date
    1 -> 1    targets.date

symbol_manifest.symbol
    1 -> many ticker_features.symbol

raw FRED JSON files
    -> transformed into features macro columns during the build

bad OHLC validation findings
    -> counted in build_metadata, not published as a table
```

## Sparse Model-Ready Artifact

Build command:

```powershell
uv run build-model-dataset
```

Configuration:

```text
configs/model_dataset.yaml
```

Output layout:

```text
data/model_ready/<dataset_name>/<UTC timestamp>_<build_id>/
```

`build_id` is the first 16 hexadecimal characters of a SHA-256 hash over the
source DuckDB hash and model-dataset configuration. Re-running an
unchanged source/configuration pair returns the existing artifact instead of
rewriting it.

The current artifact covers `2000-01-03` through `2026-06-08`. Sample dates
begin on `2005-02-25` and are restricted to dates where `ETF_SPY` has a valid
observation.

### Artifact Inventory

| File | Rows | Purpose |
|---|---:|---|
| `dates.parquet` | 6,647 | Date index, sample eligibility, and split-protection flags |
| `assets.parquet` | 531 | Stable asset IDs and trainability metadata |
| `feature_manifest.parquet` | 60 | Ordered exported-feature definitions |
| `normalization.parquet` | 60 | Train-only normalization parameters |
| `train_asset_features.parquet` | 1,732,951 | Sparse valid asset facts on train-allowed dates |
| `validation_asset_features.parquet` | 1,125,147 | Sparse valid asset facts on protected dates |
| `train_market_features.parquet` | 4,314 | Date-grain market facts on train-allowed dates |
| `validation_market_features.parquet` | 2,333 | Date-grain market facts on protected dates |
| `train_macro_features.parquet` | 4,314 | Date-grain macro facts on train-allowed dates |
| `validation_macro_features.parquet` | 2,333 | Date-grain macro facts on protected dates |
| `config_resolved.yaml` | - | Input configuration plus resolved build identity and date range |
| `manifest.json` | - | Immutable artifact identity, source hash, rules, and quality summary |
| `quality_report.json` | - | Row counts and enforced model-dataset invariants |

The artifact stores sparse facts, not preassembled JEPA windows and not a
complete date-by-asset grid. Window and patch assembly happens downstream.

### `dates.parquet`

Logical key: `date_idx`; `date` is also unique.

```text
date_idx INTEGER
date DATE
sample_eligible BOOLEAN
validation_sample BOOLEAN
protected_input_lookback BOOLEAN
protected_forward_target BOOLEAN
protected_holdout BOOLEAN
train_fact_allowed BOOLEAN
validation_fact_allowed BOOLEAN
validation_window_name VARCHAR
```

Current date classifications:

| Classification | Dates |
|---|---:|
| Total date spine | 6,647 |
| Sample eligible | 5,353 |
| Validation samples | 1,202 |
| Protected validation lookback | 753 |
| Protected forward-target region | 378 |
| Train facts allowed | 4,314 |
| Validation facts allowed | 2,333 |

`validation_sample`, `protected_input_lookback`, and
`protected_forward_target` are disjoint. `protected_holdout` is their union,
`train_fact_allowed` is its inverse, and `validation_fact_allowed` equals the
full protected region. Train and validation fact files therefore have no
overlapping dates.

### `assets.parquet`

Logical key: `asset_id`; `symbol` is also unique.

```text
asset_id INTEGER
symbol VARCHAR
asset_type VARCHAR
first_available_date TIMESTAMP
last_available_date TIMESTAMP
valid_train_observations BIGINT
trainable BOOLEAN
exclusion_reason VARCHAR
```

All 531 selected `stock`, `etf`, and `index` assets currently meet the minimum
of 63 valid train observations and are marked trainable.

### `feature_manifest.parquet`

Logical key: `(input_group, feature_name)`.

```text
feature_name VARCHAR
feature_index BIGINT
input_group VARCHAR
feature_family VARCHAR
series_source VARCHAR
dtype VARCHAR
normalized BOOLEAN
normalization_method VARCHAR
transform VARCHAR
```

`feature_index` starts at zero independently within each input group.

Current exported feature groups:

| Input group | Features | Source canonical table |
|---|---:|---|
| `asset` | 22 | `ticker_features` |
| `market` | 5 | `features` |
| `macro` | 33 | `features` |

The model-ready export intentionally excludes all target-like columns and the
canonical OAS feature families. `targets` is never read into the model-ready
artifact. The excluded OAS patterns are `high_yield_oas_*`,
`corporate_oas_*`, and `hy_minus_corporate_oas`.

### `normalization.parquet`

Logical key: `(input_group, feature_name)`.

```text
feature_name VARCHAR
input_group VARCHAR
transform VARCHAR
normalization_method VARCHAR
fit_count BIGINT
lower_bound DOUBLE
center DOUBLE
scale DOUBLE
upper_bound DOUBLE
```

Normalization uses finite real facts from train-allowed dates only:

1. Apply configured `log1p` transforms to `dollar_volume`,
   `realized_vol_*`, and `vix_level`.
2. Winsorize to the train-fold `0.005` and `0.995` quantiles.
3. Center on the train-fold median.
4. Scale by the train-fold interquartile range; use `1.0` if the IQR is not
   finite or positive.

### Asset Fact Files

Files:

```text
train_asset_features.parquet
validation_asset_features.parquet
```

Logical key:

```text
(date, asset_id)
```

Schema pattern:

```text
date DATE
date_idx INTEGER
asset_id INTEGER
valid_asset BOOLEAN
<asset_feature> FLOAT
<asset_feature>__valid BOOLEAN
...
```

Only trainable assets with `ticker_features.valid_observation = true` are
written, so the current exported `valid_asset` values are all `true`. Missing
or non-finite individual feature values are stored as normalized `0.0` with
the corresponding `__valid` mask set to `false`.

Current asset feature families:

| Family | Columns |
|---|---|
| Returns | `return_1d`, `return_5d`, `return_21d`, `return_63d`, `return_126d` |
| Realized volatility | `realized_vol_5d`, `realized_vol_21d`, `realized_vol_63d`, `realized_vol_126d` |
| Drawdown | `drawdown_5d`, `drawdown_21d`, `drawdown_63d`, `drawdown_126d` |
| Moving-average distance | `ma_distance_5d`, `ma_distance_21d`, `ma_distance_63d`, `ma_distance_126d` |
| Path efficiency | `path_efficiency_5d`, `path_efficiency_21d`, `path_efficiency_63d`, `path_efficiency_126d` |
| Liquidity | `dollar_volume` |

### Market and Macro Fact Files

Files:

```text
train_market_features.parquet
validation_market_features.parquet
train_macro_features.parquet
validation_macro_features.parquet
```

Logical key:

```text
date
```

Schema pattern:

```text
date DATE
date_idx INTEGER
valid_date BOOLEAN
<group_feature> FLOAT
<group_feature>__valid BOOLEAN
...
```

`valid_date` is true when at least one feature in that input group is finite.
Missing or non-finite values use the same normalized `0.0` plus `__valid =
false` representation as asset facts.

The market files contain `xs_dispersion_1d`, `xs_iqr_1d`, `breadth_1d`,
`pct_above_ma_63d`, and `n_assets`.

The macro files contain VIX, Treasury 3M/2Y/10Y/30Y, fed-funds level/change
features, and the three yield-curve spreads. They do not contain OAS features.

### JEPA Target Eligibility

The artifact does not contain future targets. Current model-dataset configs
record downstream JEPA patch-eligibility policy metadata:

| Rule | Current value |
|---|---:|
| Minimum valid dates in patch | 10 |
| Minimum valid asset fraction | 0.25 |
| Training holdout patches allowed as prediction targets | `false` |
| Validation holdout patches allowed as prediction targets | `true` |
| Padded patches allowed as prediction targets | `false` |

Holdout permission is split-relative. Training batches cannot target protected
validation facts. Validation JEPA batches may target validation-relative
holdout patches so validation loss measures those patches. Embedding batches
use the complete unmasked context-valid sequence and do not sample JEPA targets.

The latest verified artifact named at the top of this document predates the
split-relative metadata fields and stores the legacy global
`allow_holdout_patches_as_targets: false` training-protection rule. Runtime
behavior is defined by the split-relative dataloader policy above.

## Review Notes

1. `features` and `ticker_features` preserve their natural grains.
2. `targets` remains physically separate and no canonical or model-ready
   target-joined artifact exists.
3. The model-ready export is sparse, mask-explicit, train-normalized, and
   date-disjoint across train and validation facts.
4. The canonical database still has no declared primary or foreign-key
   constraints.
5. `exchange` and `sector` should be explicitly cast to `VARCHAR`.
6. Latest-revised FRED values are used; ALFRED vintages are not currently used.
7. The two OAS raw snapshots currently begin on `2023-06-12`; they remain in
   the canonical database but are intentionally excluded from the current
   model-ready export.
