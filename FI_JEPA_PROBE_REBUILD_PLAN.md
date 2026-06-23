# FI-JEPA Probe Rebuild Plan

## Summary

The current probe result should be treated as a failed first probe, not as final proof that the FI-JEPA representation is useless.

The ridge probe failed badly on calibrated regression metrics: most fold-level R² values were negative, aggregate out-of-fold R² was negative for every target, and ridge RMSE was usually worse than the train-mean baseline. However, several folds still showed positive correlation, sometimes high correlation, while having terrible R². That pattern suggests the embedding may contain some ordering or regime information, but the current probe setup is poorly calibrated and too brittle across market regimes.

The next step should be to rebuild the probe system before changing pretraining.

The rebuilt probe system should answer four questions:

1. Does `z_t` contain any out-of-sample information?
2. Is that information only volatility or crisis-state information?
3. Is the probe calibrated, or only rank-correlated with the target?
4. Does `z_t` beat simple hand-built market-state baselines?

## Keep The Existing Leakage Boundary

Do not merge future targets into the pretraining or embedding artifact.

Keep the artifact structure conceptually separate:

```text
runs/evaluation/<artifact>/
    embeddings.parquet
    diagnostics.json
    pca_exporter.npz

data/probe_targets/<artifact>/
    targets.parquet
    manifest.json

runs/probes/<probe_run>/
    probe_dataset.parquet
    predictions.parquet
    report.json
```

The probe runner should continue to:

- Load frozen embeddings.
- Load physically separate future targets.
- Join one-to-one by date.
- Reject embeddings that contain `future_*` target columns.
- Fit only on dates before the validation window.
- Score only on held-out validation windows.

The main rebuild is not about leakage. The main rebuild is about target handling, baselines, calibration diagnostics, and probe model selection.

---

# Phase 1 — Rebuild The Probe Dataset And Report

## Goal

Create one clean probe dataset and one much more diagnostic report format.

Right now the report tells you the final scores, but it does not tell you enough about why the probe failed. The rebuilt report should expose calibration failure, invalid predictions, target scale mismatch, and regime-specific behavior.

## Probe Dataset Contract

Build a `probe_dataset.parquet` with one row per date and validation target.

Suggested columns:

```text
date
split
validation_window_name
z_1 ... z_k
raw target columns
transformed target columns
baseline feature columns
target availability masks
fold metadata
```

The probe dataset should be built once, then reused by multiple probe heads.

## Required Report Metrics

For every target, horizon, validation window, model, and baseline, report:

```text
rmse
mae
r2
pearson_correlation
spearman_correlation
rmse_ratio_vs_baseline
mae_ratio_vs_baseline
actual_mean
actual_std
prediction_mean
prediction_std
bias = mean(prediction - actual)
std_ratio = prediction_std / actual_std
invalid_prediction_count
invalid_prediction_rate
```

Invalid prediction checks should include:

```text
realized volatility prediction < 0
max drawdown prediction > 0
max drawdown magnitude prediction < 0
probability prediction outside [0, 1]
trend score outside expected range, if the target is bounded
```

Also add a diagnostic-only recalibration metric:

```text
actual_validation_target ~ a + b * prediction
```

This should not be used as the final score because it uses validation labels. It is only a diagnostic that tells you whether the model has ranking information but bad scale/intercept calibration.

## Output

After Phase 1, the report should make it obvious whether the probe failed because:

- The representation has no target association.
- The prediction has correlation but bad calibration.
- The model predicts values outside the legal target range.
- One validation regime dominates the failure.
- One horizon works while others fail.

---

# Phase 2 — Fix Target Handling And Probe Model Selection

## Goal

Make the regression targets compatible with simple probe heads.

The current raw ridge setup is too brittle. It can predict positive drawdowns and negative volatility, which makes RMSE and R² explode even when correlation is not zero.

## Target Transforms

Use train-only parameters for all transforms. Validation data should never determine thresholds, means, standard deviations, winsorization cutoffs, or quantile buckets.

### Realized Volatility

For each horizon:

```text
future_realized_vol_21d
future_realized_vol_63d
future_realized_vol_126d
```

Create:

```text
raw_vol = future_realized_vol_h
log_vol = log(max(raw_vol, eps))
```

Primary regression target should be `log_vol`, not raw vol.

### Max Drawdown

For each horizon:

```text
future_max_drawdown_21d
future_max_drawdown_63d
future_max_drawdown_126d
```

Create:

```text
drawdown_magnitude = clip(-future_max_drawdown_h, 0, inf)
log_drawdown_magnitude = log1p(drawdown_magnitude)
```

Primary regression target should be `log_drawdown_magnitude`, not raw negative drawdown.

When reporting in original units, invert the prediction:

```text
predicted_drawdown = -expm1(predicted_log_drawdown_magnitude)
```

### Future Returns

For each horizon:

```text
future_return_21d
future_return_63d
future_return_126d
```

Create:

```text
raw_return
winsorized_return
return_sign = raw_return > 0
return_quantile_bucket
```

Use raw or winsorized return for regression. Use sign and buckets for classification.

### Trend Score

For each horizon:

```text
future_trend_score_21d
future_trend_score_63d
future_trend_score_126d
```

Create:

```text
raw_trend_score
train_zscored_trend_score
trend_quantile_bucket
```

Trend score is likely the most interesting first target because it is closer to market-state structure than raw return.

## Model Selection

Do not use only `alpha = 1.0`.

Use an alpha grid:

```yaml
alphas: [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
```

For each outer validation window, choose alpha using only the pre-window training period.

Example:

```text
Outer validation window:
2022-06-01 to 2023-12-29

Available training period:
2005-02-25 to 2021-06-02

Inner train:
2005-02-25 to 2018-12-31

Inner validation:
2019-01-01 to 2021-06-02
```

Choose alpha on the inner validation period, then refit on the full available training period, then score the outer validation period.

## Output

After Phase 2, you should know whether the original failure was mostly caused by:

- Bad target scale.
- Physically invalid raw ridge predictions.
- A poor fixed regularization value.
- True lack of useful signal.

---

# Phase 3 — Add Strong Baselines And Multiple Probe Heads

## Goal

Compare FI-JEPA embeddings against realistic simple alternatives.

The train-mean baseline is too weak. A useful representation should beat obvious trailing market-state features, not just a constant prediction.

## Baseline Families

Implement these baselines:

```text
Baseline 0: train mean
Baseline 1: trailing target proxy
Baseline 2: hand-built market-state features
Baseline 3: PCA of hand-built market-state features
Baseline 4: z_t only
Baseline 5: hand-built features + z_t
```

Suggested hand-built features:

```text
trailing_return_21d
trailing_return_63d
trailing_return_126d

trailing_realized_vol_21d
trailing_realized_vol_63d
trailing_realized_vol_126d

trailing_max_drawdown_21d
trailing_max_drawdown_63d
trailing_max_drawdown_126d

current trend score
current market breadth
current dispersion
current macro/rates/credit features, if available
```

The important comparison is:

```text
z_t vs hand-built state features
```

not:

```text
z_t vs train mean
```

## Probe Heads

Start simple.

Regression heads:

```text
ridge regression
Huber regression
ElasticNet
```

Classification heads:

```text
logistic regression
multinomial logistic regression for quantile buckets
```

Do not add neural-network probes yet. The first rebuilt probe should stay interpretable.

## Classification And Bucket Probes

Regression can fail because of calibration even when the representation contains regime information. Add classification probes to test whether `z_t` can identify future regimes.

Suggested labels:

```text
high_vol_21d:
    future_realized_vol_21d in top train quantile

high_vol_63d:
    future_realized_vol_63d in top train quantile

severe_drawdown_63d:
    -future_max_drawdown_63d in top train quantile

positive_return_21d:
    future_return_21d > 0

strong_trend_63d:
    future_trend_score_63d in top train quantile

weak_or_chop_63d:
    abs(future_trend_score_63d) in bottom train quantile, if meaningful
```

Classification metrics:

```text
accuracy
balanced accuracy
ROC-AUC
PR-AUC
Brier score
log loss
class prevalence
```

For rare events like severe drawdowns, PR-AUC and Brier score matter more than raw accuracy.

## Output

After Phase 3, you should know whether FI-JEPA beats:

- A constant train mean.
- A trailing-value proxy.
- Simple hand-built market-state features.
- PCA of hand-built market-state features.

If `z_t` only beats the train mean, it is not enough.

---

# Phase 4 — Add Residualized Probes And Final Pass/Fail Gates

## Goal

Test whether the embedding contains information beyond obvious volatility and trend features.

This is the phase that decides whether FI-JEPA learned a useful representation or just rediscovered basic market state.

## Residualized Probe Types

Run four model families for each target:

```text
A. hand_features_only
B. z_only
C. hand_features + z
D. residualized_z_only
```

Target residualization is easiest to interpret:

```text
1. Fit target ~ hand_features on train only.
2. Compute residual_target on train and validation.
3. Fit residual_target ~ z on train.
4. Score validation residuals.
```

This asks:

```text
Does z_t explain anything after obvious volatility/trend state is removed?
```

Also test feature residualization:

```text
1. Fit each z_j ~ hand_features on train only.
2. Compute residual_z_j on train and validation.
3. Fit target ~ residual_z on train.
4. Score validation.
```

This asks:

```text
Is the useful part of z_t just a linear copy of hand-built market features?
```

## Representation Variants

Run probes on multiple embedding variants:

```text
pca_8
pca_16
pca_32
raw_pooled_state_256
standardized_raw_pooled_state_256
residualized_raw_pooled_state_256
```

The current 16-dimensional PCA export may be too compressed. It may keep only dominant regime axes and discard smaller useful directions.

Use stronger regularization for raw 256-dimensional probes.

## Final Report Summary

The final report should summarize results by:

```text
target family
horizon
validation window
probe model
baseline family
embedding variant
```

For each model, report:

```text
mean R² across windows
median R² across windows
worst-window R²
number of windows beaten
number of targets beaten
RMSE ratio vs strongest baseline
correlation stability across windows
invalid prediction rate
```

## Pass Gates

A result is promising if:

```text
- it beats the strongest simple baseline in at least 2 of 3 validation windows
- it works better on 63d or 126d targets than on 21d targets
- it has stable positive correlation across validation windows
- it survives target transforms
- it has near-zero invalid prediction rates
- residualized z still adds some signal beyond hand-built features
```

A result is weak if:

```text
- it only beats the train mean
- it fails against trailing volatility/trend baselines
- it only works in one validation window
- all useful signal disappears after residualizing volatility/trend
- calibrated metrics fail even when correlation is positive
- classification probes are no better than hand-built baselines
```

A result is a failure if:

```text
- aggregate R² remains strongly negative after transforms and alpha selection
- RMSE is worse than strong baselines for nearly every target
- correlations are unstable or sign-flipping across windows
- invalid prediction rates remain high
- residualized probes all fail
```

---

# Suggested Implementation Order

## Step 1: Report rebuild

Add prediction distribution diagnostics, invalid prediction checks, RMSE ratios, Spearman correlation, and per-window summaries.

This is the fastest and safest change.

## Step 2: Target transforms

Add log volatility and log drawdown magnitude. Rerun ridge only.

This tests whether the original failure was mainly raw target scale.

## Step 3: Alpha sweep

Add nested walk-forward alpha selection. Rerun transformed ridge probes.

This tests whether `alpha = 1.0` was a bad fixed choice.

## Step 4: Strong baselines

Add trailing target proxies and hand-built market-state features.

This makes the evaluation meaningful.

## Step 5: Classification probes

Add binary and bucket labels for high volatility, severe drawdown, positive return, and trend regimes.

This tests whether the representation knows regimes even if exact regression is hard.

## Step 6: Residualized probes

Test whether `z_t` adds information beyond hand-built volatility and trend features.

This is the most important scientific test.

## Step 7: Representation variants

Probe PCA-8, PCA-16, PCA-32, and raw 256-dimensional pooled states.

This tests whether the PCA export is hiding useful information.

---

# Suggested CLI Shape

Split the probe flow into separate commands.

```bash
uv run build-probe-dataset \
  --embeddings runs/evaluation/<artifact> \
  --targets data/probe_targets/<artifact> \
  --output-root data/probe_targets
```

```bash
uv run run-fi-jepa-probes-v2 \
  --probe-dataset data/probe_targets/<artifact> \
  --config configs/probes.yaml
```

Optional helpers:

```bash
uv run summarize-probe-report \
  --report runs/probes/<run>/report.json
```

```bash
uv run plot-probe-predictions \
  --predictions runs/probes/<run>/predictions.parquet
```

---

# Suggested `configs/probes.yaml`

```yaml
probe:
  version: v2

inputs:
  embedding_variants:
    - pca_8
    - pca_16
    - pca_32
    - raw_256

targets:
  transforms:
    future_realized_vol:
      - raw
      - log
    future_max_drawdown:
      - magnitude
      - log_magnitude
    future_return:
      - raw
      - winsorized
      - sign
      - quantile_bucket
    future_trend_score:
      - raw
      - zscore
      - quantile_bucket

models:
  regression:
    - ridge
    - huber
    - elastic_net
  classification:
    - logistic
    - multinomial_logistic

ridge:
  alphas: [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
  alpha_selection: inner_walk_forward

baselines:
  - train_mean
  - trailing_target_proxy
  - hand_market_features
  - hand_market_pca
  - hand_market_features_plus_z
  - residualized_z

diagnostics:
  report_prediction_distribution: true
  report_invalid_predictions: true
  report_spearman: true
  report_rmse_ratio: true
  report_calibration_slope: true
  report_window_summary: true
```

---

# Bottom Line

Do not change pretraining yet.

The first probe result failed, but it failed in a way that suggests the probe system is too primitive. Rebuild the probe stack around transformed targets, stronger baselines, alpha selection, calibration diagnostics, classification probes, and residualized tests.

Only after this rebuilt probe system fails should you conclude that the FI-JEPA representation itself is not useful.
