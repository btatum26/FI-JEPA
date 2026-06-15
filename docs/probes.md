# Representation Export And Frozen Probes

## Purpose

Representation evaluation and future-target probes are separate from
pretraining. This prevents target columns from entering model-ready data,
runtime batches, checkpoints, or embedding artifacts.

The workflow uses four immutable artifact types:

| Artifact | Default root | Contains future targets |
|---|---|---:|
| Representation evaluation | `runs/evaluation/` | No |
| Probe-target export | `data/probe_targets/` | Yes |
| Reusable probe dataset | `runs/probe_datasets/` | Joined only for evaluation |
| Frozen probe report | `runs/probes/` | Joined evaluation data and predictions |

## Representation Source

`evaluate-fi-jepa` uses `FIJepaModel.encode_pooled_state()`:

1. Encode the complete unmasked context-valid patch sequence.
2. Require the final endpoint patch to be valid.
3. Concatenate the masked temporal mean and endpoint encoder states.
4. Fit PCA only on train pooled states.
5. Apply the train-fit PCA transform to validation states.

## Export Embeddings

```bash
uv run evaluate-fi-jepa \
  --checkpoint runs/pretraining/<run>/checkpoints/best_validation.pt \
  --device auto \
  --batch-size 1
```

`--batch-size` overrides the checkpoint's validation batch size for every
representation-evaluation loader, including the all-valid train and validation
passes. Lower it when those views exceed GPU memory.

The evaluation artifact includes:

```text
manifest.json
diagnostics.json
pca_exporter.npz
embeddings.parquet
validation_k_view_embeddings.parquet
```

`embeddings.parquet` contains one all-valid representation row per date. It
includes split and validation-window metadata plus `z_*` columns, but no
`future_*` columns.

## Export Probe Targets

```bash
uv run export-probe-targets
```

Optional arguments:

```bash
uv run export-probe-targets \
  --database data/processed/market_data.duckdb \
  --output-root data/probe_targets
```

The command reads the canonical `targets` table and writes:

```text
data/probe_targets/<timestamp>_<artifact_id>/
    manifest.json
    targets.parquet
```

The manifest records the canonical database SHA-256 and target columns.

## Build Probe Dataset

```bash
uv run build-probe-dataset \
  --embeddings runs/evaluation/<evaluation_artifact> \
  --targets data/probe_targets/<target_artifact>
```

The builder:

1. Verifies that embedding and target artifacts came from the same canonical
   database version.
2. Rejects embeddings containing `future_*` columns.
3. Joins embeddings and targets one-to-one by date.
4. Adds an explicit availability mask for every target.
5. Records named validation-window boundaries and train cutoffs in the
   dataset manifest.

Outputs:

```text
runs/probe_datasets/<timestamp>_<artifact_id>/
    manifest.json
    probe_dataset.parquet
```

The joined dataset is an evaluation-only artifact. Future targets remain
physically separate from pretraining data, runtime batches, checkpoints, and
embedding artifacts.

## Run Frozen Ridge Probes

```bash
uv run run-fi-jepa-probes \
  --probe-dataset runs/probe_datasets/<probe_dataset_artifact> \
  --alpha 1.0
```

For each named validation window and target, the probe runner:

1. Fits a standardized ridge model only on train dates before the validation
   window begins.
2. Scores both ridge and the train-mean baseline on the held-out window.
3. Reports distribution, calibration, rank-correlation, invalid-prediction,
   and baseline-relative diagnostics.

Outputs:

```text
runs/probes/<timestamp>_<run_id>/
    probe_dataset.parquet
    predictions.parquet
    report.json
```

`predictions.parquet` is long-form with one row per date, target, validation
window, and predictor. It records invalid-prediction flags and reasons.

`report.json` contains per-window/per-target result rows, aggregate
out-of-fold results, window summaries, and ridge coefficients. Every result
includes RMSE, MAE, R-squared, Pearson and Spearman correlation, prediction
and actual distributions, bias, scale ratio, ratios against the train-mean
baseline, invalid-prediction counts, and diagnostic-only validation
recalibration. Recalibration uses validation labels and is never a final
score.

## Interpret Latent Coordinates

Correlate exported PCA coordinates with selected numeric columns from the live
canonical `features`, `ticker_features`, and `targets` tables. Omitting
`--coordinates` analyzes every exported `z_*` dimension:

```bash
uv run python -m fi_jepa.analysis.analyze_latent_factor \
  --embeddings runs/evaluation/<evaluation_artifact> \
  --features vix_level ticker_features.realized_vol_63d targets.future_realized_vol_63d
```

Select specific coordinates or inspect every selectable feature:

```bash
uv run python -m fi_jepa.analysis.analyze_latent_factor \
  --embeddings runs/evaluation/<evaluation_artifact> \
  --coordinates z_1 z_4 z_8 \
  --features features.vix_level breadth_1d elapsed_trading_rows

uv run python -m fi_jepa.analysis.analyze_latent_factor --list-features
```

The analysis verifies the canonical database hash and writes under
`runs/latent_factor_analysis/`:

```text
analysis_dataset.parquet
correlations.csv
report.json
```

Feature selectors may use a unique column name or an explicit
`table.column` name. `--features all` selects every numeric canonical column.
The default feature set remains the original VIX, volatility, drawdown,
breadth, dispersion, future-volatility, and time-control set.

`correlations.csv` reports every coordinate-feature pair for full-sample,
train, validation, and named validation-window segments. It includes raw
levels, first differences, and linear-time-detrended values because a dominant
PCA axis can carry a time trend that creates misleading level correlations.
Future targets are joined only into this analysis artifact.

## Current Limits

- Probes are continuous-target ridge regressions with a train-mean baseline only.
- Target transforms, alpha selection, stronger baselines, and classification
  heads remain later rebuild phases.
- Logistic, nonlinear, bucket, and regime-label probes are not implemented.
- Probe results measure representation association, not tradability.
- Good future-volatility probes can still indicate a volatility-dominated
  representation; residualized and baseline comparisons remain necessary.

## Tests

`tests/test_fi_jepa_representation_probes.py` covers train-fit PCA,
representation diagnostics, target separation, reusable probe datasets,
database-version matching, Phase 1 report diagnostics, and walk-forward probe
fitting.

```bash
uv run pytest -q tests/test_fi_jepa_representation_probes.py
```
