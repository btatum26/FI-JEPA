# Sparse Model-Ready Dataset

## Purpose

`build-model-dataset` freezes selected canonical DuckDB features into an
immutable sparse Parquet artifact for model runtime.

```bash
uv run build-model-dataset --config configs/model_dataset.yaml
```

The builder is implemented under `src/dataset_pipeline/dataset_builder/`. It is
separate from:

- `build-market-database`, which constructs the canonical DuckDB.
- `fi_jepa.dataloader`, which reconstructs windows and samples model views.
- Probe-target export, which handles future outcomes separately.

## Output Layout

```text
data/model_ready/<dataset_name>/<timestamp>_<build_id>/
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

The timestamp makes builds sortable by creation time. The deterministic build
ID identifies the resolved config and source database version. Existing build
directories are never overwritten.

## Stored And Runtime Data

The artifact stores:

- Sparse normalized facts.
- Per-feature validity masks.
- Date and asset manifests.
- Feature order, family, source, transform, and normalization metadata.
- Train-only normalization statistics.
- Explicit split permissions and protection flags.

The artifact does **not** store:

- Dense lookback windows.
- Complete date-by-asset grids.
- Runtime asset samples.
- Temporal patches or JEPA masks.
- Canonical future targets.
- Raw FRED responses or other raw source files.

Asset facts contain one row per real valid `(date, asset_id)` observation.
Market and macro facts contain one row per available date. Train and validation
fact files are date-disjoint.

## Date And Split Contract

`dates.parquet` contains the configured continuous date spine:

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

| Field | Meaning |
|---|---|
| `validation_sample` | Date is inside a named validation window |
| `protected_input_lookback` | Date is reserved for validation input reconstruction |
| `protected_forward_target` | Date is reserved against future-target leakage |
| `protected_holdout` | Union of the three protected regions |
| `train_fact_allowed` | Date may be written to train fact files |
| `validation_fact_allowed` | Date may be written to validation fact files |

Overlapping validation protections are merged deterministically. Train and
validation fact permissions never overlap.

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

An asset is trainable when it has the configured minimum number of valid
observations on `train_fact_allowed` dates.

`feature_manifest.parquet` contains:

```text
feature_name
feature_index
input_group
feature_family
series_source
dtype
normalized
normalization_method
transform
```

Input groups are `asset`, `market`, and `macro`. Feature indices are contiguous
within each group. The manifest, not model code or documentation, is
authoritative for feature dimensions and order.

The current export rejects:

- Names matching `future_`, `target`, or `label`.
- OAS-derived columns.
- Configured features missing from the canonical source table.

## Normalization And Masks

The builder:

1. Fits robust z-score normalization only on finite train facts.
2. Applies configured transforms before fitting.
3. Creates validity masks before normalization.
4. Normalizes real finite values only.
5. Fills invalid normalized values with `0.0`.
6. Publishes masks with every fact row.

Zero-filled values are never considered valid observations.

## Runtime Dataloader

`DensePanelStore` converts each immutable sparse artifact into one split-specific
dense panel cache:

```text
data/cache/dense_panel/<artifact_build_id>_v<cache_format_version>/
```

The cache stores normalized zero-filled values and their source validity masks
at `[date, asset, feature]` or `[date, feature]`. Train and validation arrays
are separate, so the runtime never applies split permissions or clears
inaccessible values.

On first parent-process load, the store prints whether it is checking, reusing,
rebuilding, or publishing the cache. Reuse requires an exact manifest match for
the cache format, source hashes, sparse-file metadata, array names, shapes, and
dtypes. Cache publication is atomic and writes `manifest.json` last.

Workers receive no dense arrays through pickle. They reopen only the completed
`.npy` files using read-only memory maps; worker deserialization never validates,
builds, repairs, deletes, or publishes a cache.

The runtime request dataset starts from artifact-defined endpoint metadata and
filters structurally invalid JEPA endpoints once in the parent process.
The batch assembler:

- Selects a random fixed-K training view, deterministic fixed-K embedding view,
  or the complete global asset axis.
- Gathers values and stored masks through batch date/asset indices.
- Reshapes gathered daily tensors into zero-copy patch views.
- Aggregates daily validity into patch masks and samples temporal JEPA targets.
- Fails loudly when a fixed-K or selected JEPA asset view is not viable.

No dense windows or patches are cached. `persistent_workers` remains disabled
because the training dataset's epoch is updated before each iterator is created.

Target eligibility is split-relative:

| Runtime use | Holdout patches allowed as JEPA targets |
|---|---:|
| Training JEPA batches | No |
| Validation JEPA batches | Yes |
| Embedding batches | Unmasked sequence; target sampling is not used |

Padded patches are never target-eligible. Targets must also satisfy configured
valid-date and valid-asset coverage. Validation holdout permission allows
validation JEPA loss to evaluate validation-relative patches; it does not expose
those facts to training.

## Config Ownership

`configs/model_dataset.yaml` controls:

- Source database and output root.
- Date range and split protections.
- Included asset types.
- Exported feature families and order.
- Train-only normalization.
- Recorded JEPA target-policy metadata.

`configs/dataloader.yaml` controls the cache root, runtime lookback, asset views,
patch validity thresholds, temporal masking, and PyTorch loader settings.
Runtime settings are not baked into the dense panel cache.

## Implementation

| Responsibility | Module |
|---|---|
| Config validation and feature manifest | `dataset_pipeline.dataset_builder.config` |
| Date and asset manifests | `dataset_pipeline.dataset_builder.manifests` |
| Train-only normalization | `dataset_pipeline.dataset_builder.normalization` |
| Sparse Parquet export and quality checks | `dataset_pipeline.dataset_builder.export` |
| Atomic immutable build orchestration | `dataset_pipeline.dataset_builder.builder` |
| Dense panel cache construction/loading | `fi_jepa.dataloader.panel_store` |
| Runtime requests and asset views | `fi_jepa.dataloader.dataset` |
| Runtime patch and JEPA masks | `fi_jepa.dataloader.masking` |

## Tests

The implemented builder and runtime contracts are covered by:

- `tests/test_model_dataset_builder.py`
- `tests/test_fi_jepa_dataloader.py`
- `tests/test_dataset_pipeline.py`

They verify split protection, train-only normalization, sparse fact layout,
target exclusion, OAS exclusion, duplicate rejection, runtime reconstruction,
asset views, and target eligibility.

```bash
uv run pytest -q tests/test_model_dataset_builder.py tests/test_fi_jepa_dataloader.py
```
