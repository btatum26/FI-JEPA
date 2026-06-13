# Representation Export And Frozen Probes

## Purpose

Representation evaluation and future-target probes are separate from
pretraining. This prevents target columns from entering model-ready data,
runtime batches, checkpoints, or embedding artifacts.

The workflow uses three immutable artifact types:

| Artifact | Default root | Contains future targets |
|---|---|---:|
| Representation evaluation | `runs/evaluation/` | No |
| Probe-target export | `data/probe_targets/` | Yes |
| Frozen probe report | `runs/probes/` | Joined only for evaluation |

## Representation Source

`evaluate-fi-jepa` uses `FIJepaModel.encode_pooled_state()`:

1. Encode the complete unmasked context-valid patch sequence.
2. Require the final endpoint patch to be valid.
3. Concatenate the masked temporal mean and endpoint encoder states.
4. Fit PCA only on train pooled states.
5. Apply the train-fit PCA transform to validation states.

The legacy `state_exporter` and `encode()` path is not used. The JEPA loss does
not train `state_exporter`.

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

## Run Frozen Ridge Probes

```bash
uv run run-fi-jepa-probes \
  --embeddings runs/evaluation/<evaluation_artifact> \
  --targets data/probe_targets/<target_artifact> \
  --alpha 1.0
```

For each named validation window and target, the probe runner:

1. Verifies that embedding and target artifacts came from the same canonical
   database version.
2. Rejects embeddings containing `future_*` columns.
3. Joins embeddings and targets one-to-one by date.
4. Fits a standardized ridge model only on train dates before the validation
   window begins.
5. Scores the validation window against a train-mean baseline.

Outputs:

```text
runs/probes/<timestamp>_<run_id>/
    probe_dataset.parquet
    predictions.parquet
    report.json
```

`report.json` contains per-fold and aggregate out-of-fold RMSE, MAE, R-squared,
Pearson correlation, baseline metrics, and ridge coefficients.

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

- Probes are continuous-target ridge regressions only.
- Logistic, nonlinear, bucket, and regime-label probes are not implemented.
- Probe results measure representation association, not tradability.
- Good future-volatility probes can still indicate a volatility-dominated
  representation; residualized and baseline comparisons remain necessary.

## Tests

`tests/test_fi_jepa_representation_probes.py` covers train-fit PCA,
representation diagnostics, target separation, database-version matching,
future-column rejection, and walk-forward probe fitting.

```bash
uv run pytest -q tests/test_fi_jepa_representation_probes.py
```
