# FI-JEPA Evaluation Update Plan

## Objective

Improve the FI-JEPA evaluation pipeline without adding unnecessary abstractions or infrastructure.

The work should focus only on changes that materially affect one of these questions:

1. Does the frozen representation contain useful out-of-sample information?
2. Does it add information beyond the existing hand-built features?
3. Is the signal stable across validation windows?
4. Is useful information being lost by the current PCA export?
5. Which representation component or input branch is responsible for the signal?

Use the existing step-11,000 checkpoint while implementing this plan. Do not retrain the model until the probe pipeline is corrected and rerun.

---

# General Implementation Rules

- Prefer modifying existing modules over creating new framework layers.
- Add a new module only when the existing file would become difficult to understand.
- Do not add registries, plugin systems, factories, or generic experiment frameworks.
- Do not introduce new artifact formats unless the existing format cannot support the required data.
- Keep configuration small and explicit.
- Every new report field must answer a concrete evaluation question.
- Remove obsolete probe paths instead of maintaining compatibility aliases.
- Tests should cover mathematical correctness and leakage prevention, not implementation details.
- Do not track Git state, source hashes, or working-tree status in reports.

---

# Phase 0 — Minimal Artifact Identity and Documentation Cleanup

## Goal

Make artifacts understandable without adding repository-state tracking.

## Required changes

Each representation and probe report should record only:

```text
schema_version
checkpoint_id
checkpoint_step
representation_source
representation_variant
resolved_probe_config
resolved_representation_config
created_at_utc
```

The checkpoint ID and step are enough to associate the report with the frozen model artifact.

## Documentation cleanup

Update the README and architecture documentation so they do not state stale configuration values as permanent architecture rules.

The current YAML configuration should remain authoritative.

Where useful, documentation may state the current defaults:

```text
lookback: 252 dates
patch length: 6 dates
number of patches: 42
training asset sample: 128
fixed evaluation asset sample: 256
asset pooling: attention
PCA export dimension: 16
```

Avoid building automatic documentation generators. Directly correct the relevant documentation.

## Likely files

```text
README.md
FI_JEPA_MODEL_ARCHITECTURE_PLAN.md
src/fi_jepa/representation.py
src/fi_jepa/probes/runner.py
```

## Tests

Only add a small test confirming that required artifact identity fields are present.

Do not add Git-state or hashing tests.

## Completion condition

A report clearly identifies its checkpoint, representation variant, and resolved evaluation configuration.

---

# Phase 1 — Correct Probe Model Selection

## Goal

Stop using one ridge-selected alpha for unrelated probe models.

This is the highest-priority correctness fix.

## Required changes

Tune each model using its own implementation and objective:

```text
Ridge:
    select ridge alpha using ridge validation loss

Huber:
    select Huber alpha using Huber validation loss

Elastic Net:
    select alpha and l1_ratio using Elastic Net validation loss

Logistic regression:
    select logistic alpha using classification validation loss
```

Do not pass the selected ridge alpha into Huber, Elastic Net, or logistic regression.

## Keep the search small

Use compact grids. The purpose is to avoid obviously incorrect regularization, not to create a large hyperparameter search system.

Suggested defaults:

```yaml
ridge_alphas:
  - 0.0001
  - 0.001
  - 0.01
  - 0.1
  - 1.0
  - 10.0
  - 100.0

huber_alphas:
  - 0.0001
  - 0.001
  - 0.01
  - 0.1
  - 1.0
  - 10.0

elastic_net_alphas:
  - 0.0001
  - 0.001
  - 0.01
  - 0.1
  - 1.0

elastic_net_l1_ratios:
  - 0.1
  - 0.5
  - 0.9

logistic_alphas:
  - 0.0001
  - 0.001
  - 0.01
  - 0.1
  - 1.0
  - 10.0
```

Keep these in the existing probe configuration if one already exists. Do not create a new configuration hierarchy unless necessary.

## Normalize regularization by sample count

Make the penalty scale consistent across datasets.

For ridge, use the equivalent of:

```text
mean squared error + alpha * squared coefficient norm
```

The closed-form matrix should therefore use:

```text
X.T @ X / n + alpha * I
```

Apply equivalent sample-count normalization to the other probe heads where needed.

## Inner validation

Replace the current single tail split with three expanding time splits.

A simple implementation is sufficient:

```text
Fold 1:
    early training data -> next validation block

Fold 2:
    larger training prefix -> next validation block

Fold 3:
    largest training prefix -> final inner validation block
```

Select parameters using mean validation loss across the three folds.

Do not build a general cross-validation framework. A small time-split helper is enough.

## Horizon purge

For a target with horizon `h`, exclude training rows whose future target interval overlaps the inner validation block.

Use the target horizon directly:

```text
21-day target  -> 21-date purge
63-day target  -> 63-date purge
126-day target -> 126-date purge
```

Apply the same rule before each outer validation window.

## Report additions

Record:

```text
selected_alpha
selected_l1_ratio
inner_validation_score
inner_fold_scores
```

Only include `selected_l1_ratio` for Elastic Net.

## Tests

Add focused tests for:

```text
each model selects its own parameters
time folds preserve chronological order
horizon purge removes overlapping labels
ridge regularization is stable when sample count changes
```

## Completion condition

Elastic Net, Huber, and logistic probes no longer inherit ridge hyperparameters, and all model selection is chronological and purged.

---

# Phase 2 — Correct Residualized Probes

## Goal

Measure whether the learned representation adds information beyond the hand-built features.

## Target residualization

For each outer training and validation window:

```text
1. Fit target from hand features using outer training rows.
2. Predict the target for outer training and validation rows.
3. Compute training residuals:
       residual = actual - hand_prediction
4. Fit residual from z using training residuals only.
5. Predict validation residuals from validation z.
6. Reconstruct the final validation prediction:
       final_prediction =
           hand_validation_prediction
           + predicted_validation_residual
7. Compare the final prediction directly against the hand-only prediction.
```

The primary metric is improvement in original target units.

## Required metrics

For each target and validation window, report:

```text
hand_only_rmse
hand_plus_z_rmse
delta_rmse
rmse_ratio_vs_hand
hand_only_mae
hand_plus_z_mae
delta_mae
hand_only_r2
hand_plus_z_r2
```

Residual-space metrics may remain as diagnostics, but they must not be the main result.

## Feature residualization

Keep this experiment only if it is already implemented and easy to correct.

Correct procedure:

```text
1. Fit each z dimension from hand features using training rows.
2. Compute residualized z for training and validation using those training coefficients.
3. Fit the target from residualized training z.
4. Evaluate on residualized validation z.
```

Do not expand this into a general residualization framework.

## Fix target support checks

Original targets may have physical bounds:

```text
realized volatility >= 0
maximum drawdown <= 0
drawdown magnitude >= 0
classification probability between 0 and 1
```

Residualized targets are unbounded.

Do not infer residual support from target-name substrings.

The simplest acceptable fix is to pass an explicit boolean or score-space value:

```text
score_space = "original"
score_space = "residual"
score_space = "probability"
```

Then apply bounds only in the correct score space.

## Remove misleading validation recalibration

The current validation recalibration fits a slope and intercept using the actual validation labels.

Keep it only as an explicitly named diagnostic:

```text
oracle_validation_recalibration
```

It must not contribute to:

```text
main performance tables
model selection
pass/fail decisions
best-result summaries
```

If it is not actively useful, remove it instead.

## Tests

Add focused tests for:

```text
final prediction equals hand prediction plus residual prediction
residual targets may be positive or negative
validation labels are not used to fit residualization
incremental metrics compare against hand-only predictions
oracle recalibration is excluded from main summaries
```

## Completion condition

The report directly answers whether adding the FI-JEPA representation improves the hand-feature model.

---

# Phase 3 — Export the Representation Variants That Matter

## Goal

Determine whether PCA-16 or the pooled-state construction is hiding useful information.

## Required representation outputs

The current pooled representation concatenates:

```text
temporal mean state
endpoint state
```

Export these three sources:

```text
mean_state
endpoint_state
pooled_state
```

Do not immediately create every possible dimension and transform.

Start with these probe variants:

```text
mean_pca_16
endpoint_pca_16
pooled_pca_16
pooled_pca_32
pooled_pca_64
pooled_raw_256
```

This is enough to answer:

```text
Does endpoint or mean state carry the signal?
Is PCA-16 too aggressive?
Does the raw state perform differently?
```

Only add more variants after these results justify them.

## Storage

Keep the existing Parquet representation format where practical.

For the raw 256-dimensional state, either:

```text
store z_000 through z_255 in Parquet
```

or:

```text
store one compressed NPZ file with dates and split metadata
```

Choose the simpler implementation based on the current export code. Do not build a new artifact management layer.

## PCA rules

For every PCA variant:

```text
fit PCA on training rows only
apply the fitted PCA to validation rows
record explained variance
record representation source and dimension
```

## Probe runner support

Add one explicit field:

```text
representation_variant
```

The runner should evaluate one selected variant per run or a short configured list.

Do not redesign the probe runner around a generic experiment registry.

## Tests

Add tests confirming:

```text
PCA is fit on training rows only
variant dimensions are correct
pooled state equals mean plus endpoint concatenation
exports do not contain future targets
```

## Completion condition

The same corrected probes can be run against mean, endpoint, pooled PCA, and raw pooled representations.

---

# Phase 4 — Add Only the Diagnostics Needed to Interpret Validation Rank

## Goal

Determine whether low validation effective rank is abnormal or merely caused by comparing short validation periods with the full training history.

## Per-window diagnostics

Compute diagnostics separately for each outer validation window.

Report:

```text
sample_count
effective_rank
top_eigenvalue_share
top_5_eigenvalue_share
mean_pairwise_cosine
median_pairwise_cosine
mean_vector_norm
```

Do this for:

```text
raw pooled state
selected PCA representation
```

Do not calculate every metric for every representation variant.

## Matched-length train comparison

For each validation window:

```text
1. Determine its number of dates.
2. Split or sample the training period into contiguous windows of similar length.
3. Compute the same diagnostics for those training windows.
4. Report the validation value relative to the training-window distribution.
```

Required comparison fields:

```text
matched_train_median
matched_train_5th_percentile
matched_train_95th_percentile
validation_value
validation_percentile
```

This is the most important addition to the representation diagnostics.

## Optional subspace metric

Add principal-angle comparison only if it can be implemented compactly with existing linear algebra tools.

Use the top 8 or top 16 dimensions.

Do not add a large collection of covariance-distance measures in the first pass.

## Do not add yet

Do not implement these unless later results require them:

```text
rolling rank dashboards
Mahalanobis drift systems
multiple covariance distance measures
automatic drift alerts
large geometry report frameworks
```

## Tests

Add tests confirming:

```text
validation windows are evaluated separately
matched train windows have comparable lengths
validation percentile is computed correctly
```

## Completion condition

The report can state whether each validation window's effective rank is unusual relative to similarly sized training periods.

---

# Phase 5 — Compact Input and Asset-View Ablations

## Goal

Determine whether the representation is dominated by one input branch and whether asset sampling materially affects it.

## Asset-count test

Evaluate:

```text
k = 32
k = 128
k = 256
all valid assets
```

Do not begin with a large k sweep.

For each k, compare against the all-valid representation using:

```text
cosine similarity
relative L2 distance
```

Cosine alone is insufficient because vectors may preserve direction while changing magnitude.

Run the corrected probes for:

```text
k = 128
all valid assets
```

Only run all k values through probes if the representation distances show meaningful differences.

## Input-branch ablations

Evaluate these four modes:

```text
all streams
without assets
without market
without macro
```

This directly answers which branch is necessary.

Do not add every possible branch combination in the first pass.

Implement branch removal using the simplest safe mechanism already supported by the model:

```text
zero the branch tokens
or
skip the branch and replace it with a neutral tensor of the same shape
```

Do not retrain ablation models yet.

## Metrics

For each ablation, report:

```text
cosine similarity to full representation
relative L2 distance to full representation
effective rank
probe performance change
```

## Tests

Add tests confirming:

```text
ablated inputs preserve expected tensor shapes
the full mode matches normal encoding
each ablation changes only the intended branch
```

## Completion condition

The evaluation identifies whether assets, market, or macro inputs drive the useful representation signal.

---

# Phase 6 — Minimal Statistical Reliability Checks

## Goal

Avoid treating isolated correlations or AUC values as reliable results.

## Block bootstrap

Add paired moving-block bootstrap confidence intervals for:

```text
RMSE difference versus hand features
Pearson correlation
ROC-AUC
Brier-score difference
```

Use one default block length based on the target horizon:

```text
block_length = target_horizon
```

Use a moderate number of bootstrap samples, such as:

```text
500
```

This is enough for evaluation without making runs unnecessarily expensive.

## Stability summary

For each target across the three outer windows, report:

```text
correlation signs
number of windows beating hand features
worst-window metric
median metric
bootstrap interval
```

## Do not add yet

Do not add formal multiple-comparison correction in this implementation pass.

Instead, include a clear count of how many target/model/variant combinations were evaluated so isolated best results are shown in context.

## Tests

Add tests confirming:

```text
paired bootstrap uses aligned predictions
block sampling preserves contiguous sequences
confidence intervals are deterministic with a fixed seed
```

## Completion condition

Promising results include both cross-window stability and uncertainty estimates.

---

# Phase 7 — Replace the Human-Facing JSON Review With a Small Summary

## Goal

Keep full JSON and prediction artifacts, but generate a readable summary.

## Output

Generate:

```text
summary.md
```

A separate CSV is optional and should only be added if the existing analysis workflow needs it.

## Required sections

### Run identity

```text
checkpoint
checkpoint step
representation variant
probe configuration
validation windows
```

### Representation diagnostics

```text
window
effective rank
matched-train percentile
pairwise cosine
mean norm
```

### Incremental regression results

Rank primarily by:

```text
median RMSE ratio versus hand features
number of windows improved
worst-window RMSE ratio
```

### Classification results

Report:

```text
mean AUC
worst-window AUC
mean Brier ratio
number of windows improved
```

### Warnings

Include warnings for:

```text
selected alpha at search-grid boundary
invalid predictions
correlation sign reversals
oracle-only metrics
missing validation windows
```

## Result labels

Use only:

```text
SUPPORTED
PROMISING
INCONCLUSIVE
FAILED
INVALID
```

Suggested interpretation:

```text
SUPPORTED:
    improves the hand-feature baseline in at least two windows,
    does not fail badly in the remaining window,
    and has stable direction

PROMISING:
    useful ranking or classification signal,
    but weak calibration or one unstable window

INCONCLUSIVE:
    mixed window results or confidence interval includes no improvement

FAILED:
    consistently worse than the relevant baseline

INVALID:
    leakage, unsupported score-space checks, or incomplete evaluation
```

Do not encode these labels into a complicated rules engine. A small summary function is sufficient.

## Completion condition

The major conclusions can be understood from `summary.md` without manually reading `report.json`.

---

# Final Rerun Order

Use the existing frozen checkpoint.

```text
1. Run tests.
2. Export the six representation variants.
3. Run corrected model-specific probes.
4. Run hand-feature incremental residual probes.
5. Run per-window and matched-length representation diagnostics.
6. Run compact asset-count tests.
7. Run four input-branch ablations.
8. Run bootstrap reliability checks.
9. Generate summary.md.
```

Do not retrain during this evaluation pass.

---

# Priority Order

## Required before interpreting probe performance

```text
Phase 1 — model-specific selection
Phase 2 — residualized incremental evaluation
Phase 7 — readable summary
```

## Required before deciding whether to change the model

```text
Phase 3 — meaningful representation variants
Phase 4 — matched-length validation diagnostics
```

## Useful for architecture decisions

```text
Phase 5 — compact branch and asset-view ablations
Phase 6 — statistical reliability
```

Phase 0 is a small cleanup task and should not expand into an artifact-tracking system.

---

# Final Decision Criteria

Do not modify the pretraining architecture merely because one validation period performs poorly.

Consider the frozen representation meaningfully useful when at least one representation variant:

```text
improves hand-feature predictions in at least two validation windows
has a stable correlation direction
shows improvement after proper target residualization
is not dependent on a single isolated target
has a bootstrap interval that supports the improvement
```

Consider architecture changes when the corrected evaluation shows that:

```text
all representation variants fail against hand features
incremental residual predictions provide no improvement
signals repeatedly reverse across validation windows
removing an input branch has no measurable effect
validation geometry is abnormal relative to matched-length train windows
```

The immediate objective is to fix evaluation correctness, not to expand the evaluation framework.