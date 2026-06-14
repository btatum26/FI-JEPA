# FI-JEPA System Plan

## Status

This document separates the implemented system from the research backlog.

Implemented behavior is defined by:

- [README.md](README.md) for the runnable workflow.
- [data/DATABASE_SCHEMA.md](data/DATABASE_SCHEMA.md) for canonical and
  model-ready data contracts.
- [FI_JEPA_MODEL_ARCHITECTURE_PLAN.md](FI_JEPA_MODEL_ARCHITECTURE_PLAN.md) for
  the current model contract.
- `src/` and `tests/` when prose and code disagree.

## Research Goal

FI-JEPA tests whether a self-supervised encoder can learn a stable,
low-dimensional market-state representation from past-only asset, market, and
macro observations.

The immediate goal is representation quality, not a trading strategy. A useful
representation should:

- Remain stable across different valid asset samples for the same date.
- Preserve more than a volatility or crisis indicator.
- Improve out-of-sample prediction of slow future market outcomes relative to
  simple baselines.
- Avoid leakage through targets, revised macro assumptions, split boundaries,
  or normalization.

## Implemented System

### Data

The canonical database is built from Stooq bulk archives, configured FRED
series, and community universe metadata:

```bash
uv run build-market-database
```

It publishes:

| Table | Contract |
|---|---|
| `features` | One past-only market and macro row per date |
| `ticker_features` | One past-only asset row per date and symbol |
| `targets` | Separate future outcomes used only after representation export |
| `symbol_manifest` | Identity, source coverage, and survivorship metadata |
| `trading_calendar` | Full selected-symbol observed-date spine |

The model-ready builder freezes selected feature columns into immutable sparse
Parquet artifacts:

```bash
uv run build-model-dataset --config configs/model_dataset.yaml
```

It does not store dense windows, asset samples, temporal patches, JEPA masks, or
future targets. Those boundaries are described in
[docs/data_builder.md](docs/data_builder.md).

### Model

The current model:

1. Reconstructs 252-day windows from sparse facts.
2. Splits each window into twelve 21-day patches.
3. Tokenizes asset, market, and macro streams separately.
4. Pools valid assets inside each patch and fuses the three streams.
5. Encodes visible patches with an online context encoder.
6. Builds and encodes the complete valid patch sequence with a full EMA target
   branch covering tokenization, asset pooling, fusion, position embeddings,
   and temporal encoding.
7. Predicts gathered target-position representations from visible context.
8. Trains with normalized latent prediction loss plus a weak variance and
   covariance guardrail on pooled visible-context states.

The exact model and representation contracts are in
[FI_JEPA_MODEL_ARCHITECTURE_PLAN.md](FI_JEPA_MODEL_ARCHITECTURE_PLAN.md).

### Training And Evaluation

Training, representation evaluation, and probes are separate stages:

```bash
uv run train-fi-jepa --config configs/pretraining.yaml --device auto

uv run evaluate-fi-jepa \
  --checkpoint runs/pretraining/<run>/checkpoints/best_validation.pt \
  --device auto

uv run export-probe-targets

uv run run-fi-jepa-probes \
  --embeddings runs/evaluation/<evaluation_artifact> \
  --targets data/probe_targets/<target_artifact>
```

Evaluation uses `encode_pooled_state()` followed by PCA fit only on train
states. Future targets remain separate until frozen probe evaluation. See
[docs/probes.md](docs/probes.md).

## Current Validation Design

The frozen dataset defines named validation windows and protects:

- The validation dates themselves.
- The preceding input lookback needed to reconstruct validation windows.
- The following maximum target horizon.

Training facts cannot use any protected date. Validation reconstruction can use
the protected region. Validation JEPA loss can target validation-relative
holdout patches; training JEPA loss cannot target protected holdout patches.

Frozen ridge probes are walk-forward. For each validation window, they fit only
on train embeddings dated before that window begins.

## Current Limitations

- Current S&P 500 constituents are backfilled over history. Equity
  cross-sectional analysis is survivorship-biased.
- Standard FRED snapshots are not full ALFRED point-in-time vintages.
- The canonical target table currently contains only future return, realized
  volatility, maximum drawdown, and trend score for 21, 63, and 126 days.
- Frozen probes are continuous-target ridge regressions. Classification probes,
  nonlinear probes, target buckets, and regime labels are not implemented.
- The current anti-collapse regularizer is only a weak batch-level guardrail on
  pooled visible-context states. It does not regularize the EMA target branch
  or guarantee a full-rank representation.
- No conditional IC integration, alignment objective, mixture-of-experts model,
  or trading backtest is implemented.

## Research Backlog

### Near Term

1. Run stable pretraining experiments and inspect validation JEPA loss,
   representation rank, pairwise cosine geometry, and asset-view stability.
2. Compare frozen embeddings against simple baselines:
   - Realized volatility and VIX-like state features.
   - Hand-built canonical features.
   - PCA on input features.
   - A small autoencoder or supervised baseline.
3. Add residualized probes that measure information beyond volatility.
4. Tune or broaden anti-collapse regularization only if training diagnostics
   show that the weak pooled-state guardrail is insufficient.

### Data

1. Replace current-constituent backfills with point-in-time and delisted-aware
   equity data.
2. Add ALFRED-style vintage macro data or stricter release-time snapshots.
3. Add future dispersion, breadth, correlation, tail-risk, and regime targets
   only when their definitions and train-only thresholding are implemented and
   tested.
4. Expand the model-ready feature set only after each family has a documented
   availability and leakage contract.

### Downstream Research

1. Integrate frozen market-state embeddings with conditional IC analysis.
2. Test alignment objectives only after the self-supervised representation
   passes frozen-probe baselines.
3. Test mixture-of-experts or regime-specific heads only after a simpler model
   demonstrates stable out-of-sample structure.
4. Treat backtests as a late-stage evaluation, not evidence that the
   representation itself is valid.

## Decision Gates

Continue from data to model experiments only when:

- Canonical and model-ready leakage checks pass.
- Train and validation facts are date-disjoint.
- Feature masks and train-only normalization are verified.

Continue from pretraining to downstream use only when:

- Validation loss is stable and deterministic.
- Embeddings do not collapse to a near-constant or low-rank output.
- Asset-view stability is acceptable.
- Frozen probes beat simple baselines on multiple validation windows.

Continue to alignment or conditional IC work only when:

- Probe performance survives volatility residualization.
- Results are not driven by one validation window.
- The representation adds information beyond hand-built state features.

## Non-Goals

- Predicting future returns directly during JEPA pretraining.
- Joining future targets into model-ready or embedding artifacts.
- Claiming point-in-time equity validity from current-constituent data.
- Treating a visually appealing latent space as proof of economic usefulness.
