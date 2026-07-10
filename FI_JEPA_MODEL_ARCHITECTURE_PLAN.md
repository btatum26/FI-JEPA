# FI-JEPA Model Architecture

## Status And Source Of Truth

This document describes the implemented v1 model and keeps future work in one
explicit backlog.

The source-of-truth modules are:

- `src/fi_jepa/dataloader/` for sparse-artifact reconstruction, asset views,
  patches, and masks.
- `src/fi_jepa/model.py` and `src/fi_jepa/tokenizer.py` for the model.
- `src/fi_jepa/training.py` for optimization, EMA updates, validation, and
  checkpointing.
- `src/fi_jepa/representation.py` for representation evaluation and export.
- `src/fi_jepa/probes/` for frozen downstream probes.

## Current Configured Defaults

The YAML files under `configs/` are authoritative. The values below describe
the current defaults, not permanent architecture constraints.

| Property | Current value |
|---|---:|
| Lookback | 252 trading dates |
| Patch length | 6 dates |
| Temporal patches | 42 |
| Training asset view | Random endpoint-valid `k=128` panel |
| Validation JEPA view | All endpoint-valid assets |
| Diagnostic embedding view | Deterministic fixed `k=256` or all-valid |
| Fused model width | 128 |
| Context encoder | 2-layer, 4-head Transformer encoder |
| Target encoder | Frozen EMA copy of context encoder |
| Predictor | 2-layer, 4-head Transformer decoder |
| JEPA target count | 5 to 15 eligible patches |
| Representation source | `encode_pooled_state()` |
| Export dimension | Train-fit PCA, default 16 components |

Feature dimensions are not hard-coded in the model. They are derived from the
frozen artifact's `feature_manifest.parquet`.

## Runtime Data Flow

`FrozenPanelStore` streams the six sparse fact files into dense NumPy arrays
once. Every window slice reapplies split permissions, so protected validation
facts cannot enter training windows even though both fact sets share the same
in-memory store.

For one sample, the dataloader returns:

| Tensor group | Shape |
|---|---|
| Asset features and masks | `[B, W, A, F_asset]` |
| Market features and masks | `[B, W, F_market]` |
| Macro features and masks | `[B, W, F_macro]` |
| Patched asset features | `[B, P, L, A, F_asset]` |
| Patched market and macro features | `[B, P, L, F]` |
| Patch asset mask | `[B, P, A]` |
| Patch context and target masks | `[B, P]` |
| Gathered target patch IDs | `[B, T]` |

With the current defaults, `W=252`, `P=42`, and `L=6`. `A` varies for all-asset
validation batches and is padded by collation. Values in invalid feature, date,
or asset slots are zero; masks remain authoritative.

## Tokenization And Fusion

The model tokenizes each stream independently:

1. The asset tokenizer processes each asset patch with observation masks.
2. Valid asset tokens are combined by the configured asset-pooling module
   (currently attention pooling).
3. Market and macro tokenizers process their date-level patch features.
4. The pooled asset, market, and macro tokens are concatenated.
5. A dropout-free fusion projection maps the combined stream to width 128.
6. Learned temporal position embeddings are added after fusion.

The online and EMA target branches own separate tokenizer, asset-pooling,
fusion, positional-embedding, and temporal-encoder parameters. Tokenizers,
pooling, and fusion remain dropout-free in both branches. Branch-specific
stochasticity begins in the online context encoder and predictor.

## JEPA Branches

### Online Context Branch

The online branch packs only patches selected by `jepa_context_mask`. Masked
targets and invalid patches are absent from the context-encoder sequence.

```text
positioned full tokens [B, P, D]
    -> pack visible valid patches
    -> context encoder
    -> context representations [B, C, D]
```

### EMA Target Branch

The target encoder does **not** receive only gathered target tokens. It encodes
the complete valid positioned patch sequence, then target positions are
gathered from that encoded sequence:

```text
raw patched streams
    -> EMA target tokenizers and asset pooling
    -> EMA target fusion and position embeddings
    -> positioned full tokens [B, P, D]
    -> EMA target encoder with invalid-patch padding mask
    -> full target sequence [B, P, D]
    -> gather requested target IDs
    -> target representations [B, T, D]
```

The complete target branch runs under `torch.no_grad()`, remains in evaluation
mode, and is updated only by exponential moving average from the corresponding
online tokenizer, pooler, fusion, position-embedding, and context-encoder state.

### Predictor And Loss

The predictor receives learned target-mask queries with target position
embeddings and attends to online context representations. Prediction and EMA
target vectors are L2-normalized before squared error:

```text
loss = mean(sum((normalize(prediction) - normalize(target)) ** 2))
```

Padded target slots are zeroed and excluded from the loss.

Training adds a weak batch-level regularizer on the masked-mean visible-context
state produced by the online encoder:

```text
train_loss = jepa_loss + lambda_var * variance_loss + lambda_cov * covariance_loss
```

The variance term floors aggregate mean feature standard deviation, not every
dimension independently. The covariance term penalizes off-diagonal covariance
per feature. This is a low-weight collapse guardrail, not a requirement that all
128 dimensions remain equally active. Validation and best-checkpoint selection
continue to use pure JEPA prediction loss.

## Target Eligibility

Patch eligibility is split-relative:

| Use | Holdout patches can be targets | Padded patches can be targets |
|---|---:|---:|
| Training JEPA batches | No | No |
| Validation JEPA batches | Yes, within the validation-relative fact set | No |
| Train embedding export | Not applicable; sequence is unmasked | No |
| Validation embedding export | Not applicable; sequence is unmasked | No |

All targets must also satisfy configured valid-date and valid-asset coverage.
A target-eligible patch must be context-valid. Temporal target sampling retains
at least one visible context patch.

The frozen dataset's holdout flags protect training. They do not mean validation
JEPA loss is forbidden from using validation patches as validation-relative
targets.

## Representation Contract

`encode_pooled_state()` is the only representation source used by evaluation:

1. Encode the complete unmasked context-valid sequence with the online encoder.
2. Require the final patch at sample endpoint `t` to be context-valid.
3. Compute a masked temporal mean state `[B, D]`.
4. Take the endpoint state `[B, D]`.
5. Concatenate them into `[B, 2D]`.

Evaluation fits PCA only on train pooled states, canonicalizes component signs,
and applies the same transform to validation states.

## Training Contract

`train-fi-jepa`:

- Uses AdamW on trainable online-model parameters only.
- Excludes the complete frozen target branch from optimization.
- Applies warmup plus cosine learning-rate scheduling.
- Increases EMA momentum linearly over training.
- Supports mixed precision where available.
- Updates random training views by epoch.
- Keeps validation views and validation temporal masks deterministic.
- Writes resolved configs, JSONL logs, step checkpoints, epoch checkpoints, and
  `best_validation.pt`.
- Resumes from the checkpoint's resolved config and validates runtime
  compatibility.
- Writes full-EMA checkpoint format version 2. Version-1 checkpoints initialize
  missing target preprocessing state from their saved online modules.

Validation JEPA loss is weighted by the number of real target patches rather
than averaging batch means equally.

## Representation Evaluation

`evaluate-fi-jepa` produces an immutable evaluation artifact containing:

- Raw pooled-state diagnostics.
- Train-fit PCA diagnostics.
- All-valid train and validation embeddings.
- Fixed-k validation view embeddings.
- Asset-view cosine-stability reports.
- Version metadata for the model, dataset, checkpoint, PCA, and source database.

Embedding artifacts contain no `future_*` columns. Probe targets and frozen
probe evaluation are documented in [docs/probes.md](docs/probes.md).

## Config Ownership

| Config | Owns |
|---|---|
| `configs/dataloader.yaml` | Runtime artifact path, windows, asset views, patch validity, and temporal masking |
| `configs/model.yaml` | Tokenizer, fusion, encoder, and predictor dimensions |
| `configs/pretraining.yaml` | Optimization, EMA, validation, representation evaluation, checkpointing, and logging |

## Tests

The implemented contracts are covered by:

- `tests/test_fi_jepa_dataloader.py`
- `tests/test_fi_jepa_model.py`
- `tests/test_fi_jepa_training.py`
- `tests/test_fi_jepa_representation_probes.py`

Run them with:

```bash
uv run pytest -q
```

## Future Investigations

- Broader patch-state or target-path anti-collapse regularization if the weak
  pooled visible-context guardrail proves insufficient.
- Alternative target-block sampling strategies.
- Learned or attention-based asset pooling.
- Broader representation baselines and residualized volatility diagnostics.
- Nonlinear and classification probes.
- Conditional IC alignment after frozen representations pass simple baselines.
