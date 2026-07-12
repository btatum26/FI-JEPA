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
| Reusable probe dataset | `data/probe_targets/` | Joined only for evaluation |
| Frozen probe report | `runs/probes/` | Joined evaluation data and predictions |

## Representation Source

`evaluate-fi-jepa` uses `FIJepaModel.encode_pooled_state()` as the canonical
pooled contract and exports its two source components separately:

1. Encode the complete unmasked context-valid patch sequence.
2. Require the final endpoint patch to be valid.
3. Concatenate the masked temporal mean and endpoint encoder states.
4. Fit each requested PCA only on its training source states.
5. Apply the fitted PCA unchanged to validation states.

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
embeddings_mean_pca_16.parquet
embeddings_endpoint_pca_16.parquet
embeddings_pooled_pca_16.parquet
embeddings_pooled_pca_32.parquet
embeddings_pooled_pca_64.parquet
embeddings_pooled_raw_256.parquet
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
  --output-root data/probe_targets \
  --name market_data_targets
```

The command reads the canonical `targets` table plus past-only market-state
features for baseline comparisons and writes:

```text
data/probe_targets/market_data_targets/
    manifest.json
    targets.parquet
    baseline_features.parquet
```

The manifest records the canonical database SHA-256, future target columns,
baseline feature columns, and market proxy symbol used for trailing features.

## Build Probe Dataset

```bash
uv run build-probe-dataset \
  --embeddings runs/evaluation/<evaluation_artifact> \
  --targets data/probe_targets/market_data_targets \
  --representation-variant pooled_pca_32
```

The builder:

1. Verifies that embedding and target artifacts came from the same canonical
   source database hash.
2. Rejects embeddings containing `future_*` columns.
3. Joins embeddings and targets one-to-one by date.
4. Adds an explicit availability mask for every target.
5. Records named validation-window boundaries and train cutoffs in the
   dataset manifest.

Outputs:

```text
data/probe_targets/<evaluation_artifact>_probe_dataset/
    manifest.json
    probe_dataset.parquet
```

The joined dataset is an evaluation-only artifact. Future targets remain
physically separate from pretraining data, runtime batches, checkpoints, and
embedding artifacts.

## Run Frozen Probes

```bash
uv run run-fi-jepa-probes \
  --embeddings runs/evaluation/<evaluation_artifact> \
  --targets data/probe_targets/market_data_targets \
  --representation-variant pooled_pca_32
```

The standard `report.json` retains only the selected parameter and its inner-fold scores. Use
`--include-debug-diagnostics` only when full grid-candidate scores and fitted coefficients are needed.

One run evaluates one explicit representation variant. A reusable probe
dataset records its selected variant and can be passed with `--probe-dataset`
without selecting it again.

On-demand evaluation also records the compact architecture checks in
`diagnostics.json`: fixed asset views at 32, 128, and 256 assets against the
all-valid view, plus `all_streams`, `without_assets`, `without_market`, and
`without_macro`. Each comparison includes cosine similarity, relative L2
distance, and effective rank. Only all-valid and fixed-k 128 are retained as
the default asset-count probe variants; the other asset counts remain distance
diagnostics unless those results justify additional probe runs.

For each named validation window and target, the probe runner:

1. Fits only on train dates before the validation window begins.
2. Builds raw/log-volatility/drawdown-magnitude target variants.
3. Chooses ridge alpha from the default grid using an inner walk-forward split,
   unless `--alpha` is supplied. The selected value is also used as the compact
   regularization scale for Huber, ElasticNet, and logistic heads.
4. Scores train-mean and trailing-target proxy baselines.
5. Scores ridge, Huber, and ElasticNet heads over `z_only`,
   `hand_market_features`, `hand_market_pca`, and
   `hand_market_features_plus_z` feature families when baseline features are
   available.
6. Adds two residualized tests when baseline features are available:
   `target_residualized_z_only` fits target residuals after removing
   hand-feature predictions, and `feature_residualized_z_only` probes the part
   of `z_*` left after linearly removing hand-feature content.
7. Builds train-thresholded binary classification labels for high volatility,
   severe drawdown, positive return, strong trend, and weak/choppy regimes, then
   scores logistic heads against a class-prior baseline.
8. Reports distribution, calibration, rank-correlation, invalid-prediction,
   and baseline-relative diagnostics.

Outputs:

```text
runs/probes/<timestamp>_<run_id>/
    probe_dataset.parquet
    predictions.parquet
    report.json
    summary.md
```

`summary.md` is the human-facing review surface. It ranks original-unit
incremental regression results against hand features, summarizes classification
AUC and Brier ratios, includes per-window representation diagnostics when the
embedding artifact provides them, and surfaces boundary, invalid-prediction,
sign-reversal, oracle-only, and missing-window warnings. Full report and
prediction artifacts remain authoritative for detailed analysis.

`predictions.parquet` is long-form with one row per date, target or regime
label, validation window, and predictor. It records `task_type`, `model_name`,
`feature_family`, invalid-prediction flags, and reasons.

`report.json` contains per-window/per-target result rows, aggregate
out-of-fold results, window summaries, ridge coefficients, and selected-alpha
diagnostics. Every result includes RMSE, MAE, R-squared, Pearson and Spearman
correlation, prediction and actual distributions, bias, scale ratio, ratios
against the train-mean baseline, invalid-prediction counts, and diagnostic-only
validation recalibration. Recalibration uses validation labels and is never a
final score.

Classification rows report accuracy, balanced accuracy, ROC-AUC, PR-AUC, Brier
score, log loss, and class prevalence. The classification thresholds are fit
from the outer training period only.

The final summary section reports mean, median, and worst-window R-squared,
correlation stability, windows beating the strongest simple baseline, and a
conservative `promising` / `weak` / `failure` gate. The gates are diagnostics,
not scientific proof.

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

- Regression heads are still simple linear heads; no neural probes are included.
- Classification probes are binary regime labels, not multinomial quantile
  buckets yet.
- Probe results measure representation association, not tradability.
- Good future-volatility probes can still indicate a volatility-dominated
  representation; residualized comparisons are the check for this.

## Tests

`tests/test_fi_jepa_representation_probes.py` covers train-fit PCA,
representation diagnostics, target separation, reusable probe datasets,
source-database hash matching, Phase 2 target transforms, report diagnostics,
Phase 3 baselines/classification heads, Phase 4 residualized probes/gates, and
walk-forward probe fitting.

```bash
uv run pytest -q tests/test_fi_jepa_representation_probes.py
```
