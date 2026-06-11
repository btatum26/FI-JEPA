# Sparse Frozen Dataset Builder

## Summary

Build immutable, non-partitioned sparse Parquet datasets from DuckDB.

The builder freezes daily normalized facts, split permissions, validity masks, asset eligibility, and feature metadata. It does not store lookback windows, complete date-by-asset grids, `k_assets`, patches, or JEPA temporal masks.

## Output Layout

```text
data/model_ready/<dataset_name>/<build_id>/
    manifest.json
    config_resolved.yaml
    dates.parquet
    assets.parquet
    feature_manifest.parquet
    normalization.parquet

    train_asset_features.parquet
    train_market_features.parquet
    train_macro_features.parquet

    validation_asset_features.parquet
    validation_market_features.parquet
    validation_macro_features.parquet

    quality_report.json
```

- Asset facts contain one row per real valid `(date, asset_id)` observation.
- Date-level facts contain one row per available date.
- Train and validation fact files are date-disjoint.
- No windows, full grids, targets, raw FRED JSON, or repeated daily facts are stored.

## Date And Split Contract

`dates.parquet` contains the continuous configured date spine and these explicit flags:

```text
date_idx
date
sample_eligible
validation_sample
protected_input_lookback
protected_forward_target
protected_holdout
train_fact_allowed
validation_fact_allowed
validation_window_name
```

Definitions:

- `validation_sample`: date is inside an anchor validation window.
- `protected_input_lookback`: date is before a validation window and reserved for reconstructing validation inputs.
- `protected_forward_target`: date is after a validation window and reserved against future-target leakage.
- `protected_holdout`: union of the three protection flags.
- `train_fact_allowed`: date may be written to train fact files; equivalent to `not protected_holdout`.
- `validation_fact_allowed`: date may be written to validation fact files; equivalent to `protected_holdout`.

The three component protection flags describe disjoint regions around each validation window. Overlapping anchor protections are merged deterministically.

Initial defaults:

```yaml
context_start: "2000-01-01"
sample_start: "2005-02-25"
lookback_days: 252
max_forward_horizon: 126
minimum_train_observations: 63
```

## Feature And Asset Contract

`assets.parquet` contains:

```text
asset_id
symbol
asset_type
first_available_date
last_available_date
valid_train_observations
trainable
exclusion_reason
```

An asset is trainable when it has at least the configured number of valid observations on dates where `train_fact_allowed = true`.

`feature_manifest.parquet` explicitly records:

```text
feature_name
feature_index
input_group
feature_family
series_source
dtype
normalized
normalization_method
```

Supported `input_group` values:

```text
asset
market
macro
```

Initial feature families include:

```text
returns
volatility
trend
drawdown
liquidity
breadth
dispersion
coverage
rates
macro
```

Feature indices are contiguous within each input group. Exact resolved feature dimensions and order are stored in the manifest.

The first pass excludes all OAS-derived features:

```text
high_yield_oas_*
corporate_oas_*
hy_minus_corporate_oas
```

## Normalization And Masks

Apply normalization using this strict sequence:

1. Fit normalization only on train facts from dates where `train_fact_allowed = true`.
2. Create feature, asset, and date masks before normalization.
3. Normalize only real finite values.
4. After normalization, fill invalid, missing, or protected values with `0.0`.
5. Always pass masks to the model.

Zero-filled values are never considered valid observations.

Macro files use the future-proof `macro` name. `series_source` in the feature manifest records whether each feature originated from FRED, market-derived data, French factors, or another future source.

## Dataloader Contract

The model-side loader is implemented under `fi_jepa.dataloader` and configured by
`configs/dataloader.yaml`. It:

- Reindex sparse facts against `dates.parquet` and `assets.parquet`.
- Construct 252-day windows.
- Reconstruct protected training sections as zero-filled values with false masks.
- Select `k_assets`.
- Patch time.
- Generate temporal JEPA masks.

It must enforce configurable target eligibility:

```yaml
jepa_target_rules:
  min_valid_dates_in_patch: 10
  min_valid_asset_fraction: 0.25
  allow_holdout_patches_as_targets: false
  allow_padded_patches_as_targets: false
```

A patch may be used as context while remaining ineligible as a JEPA prediction target.

`FrozenPanelStore` streams the sparse fact files once into dense NumPy arrays.
`FIJepaWindowDataset` reconstructs split-safe windows and supports random
training views, all-asset validation views, and deterministic diagnostic views.
`build_fi_jepa_dataloader(...)` pads variable validation panels within each
batch and exposes raw tensors, patched tensor views, feature masks, patch masks,
and temporal JEPA masks.

## Implementation And Tests

- Add `configs/frozen_dataset.yaml`.
- Add `src/data_retrieval/pipelines/build_frozen_dataset.py`.
- Register `uv run build-frozen-dataset --config configs/frozen_dataset.yaml`.
- Publish builds atomically under a deterministic immutable build ID.

Tests must verify:

- No stored windows, complete asset grids, or duplicated facts.
- Train and validation fact dates are disjoint.
- Protection flags and fact permissions have correct semantics.
- Normalization uses only `train_fact_allowed` dates.
- Masks exist before normalization and zero filling.
- Feature groups, families, sources, indices, and normalization metadata are complete.
- Pre-2005 context is exported from the configured `2000-01-01` start.
- OAS-derived features and targets are absent.
- Representative train and validation windows reconstruct correctly.
- JEPA target eligibility rejects protected, padded, and insufficiently valid patches.
