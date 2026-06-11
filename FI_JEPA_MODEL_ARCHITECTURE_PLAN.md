# FI-JEPA Model Architecture Planning Document

**Project:** FI-JEPA — Financial Joint-Embedding Predictive Architecture  
**Document purpose:** Working architecture plan for the first real FI-JEPA model implementation.  
**Current focus:** Mean-pooled variable-asset panel tokenizer + simple temporal Transformer JEPA.  
**Output:** One compact market-state embedding `z_t` per sample date.

---

## 1. Core Goal

The model should learn one compact market-state embedding per sample date.

```text
input:  past-only 252-trading-day market window ending at date t
output: z_t, one market-state embedding for date t
```

The model should **not** treat each `(date, asset)` row as a separate sample. The correct model grain is:

```text
one sample = one 252-day lookback window ending at sample date t
one output = one market-state embedding z_t
```

The first real model should answer a narrow research question:

```text
Can a simple JEPA-style temporal encoder learn a nontrivial, stable,
low-dimensional market-state representation from past-only data?
```

It should not initially be judged as a trading model, direct return predictor, or stock-selection system.

---

## 2. Relationship to the Frozen Dataset Builder

The frozen dataset builder stores sparse normalized facts, split permissions, validity masks, feature metadata, and date/asset manifests. It intentionally does **not** store:

```text
lookback windows
complete date-by-asset grids
k_assets selections
patches
JEPA temporal masks
```

Therefore, the dataloader/model side is responsible for:

```text
1. Reconstructing 252-day windows.
2. Reindexing sparse facts against dates and assets.
3. Selecting assets for training.
4. Building dense masked tensors.
5. Patching time.
6. Building patch validity and target eligibility masks.
7. Generating temporal JEPA masks.
8. Passing masks through the model.
```

Invalid, missing, padded, or protected values may be zero-filled, but the model must never infer validity from zeros. Validity always comes from explicit masks.

---

## 3. Fixed v1 Design Constants

```yaml
input:
  lookback_days: 252
  patch_len: 21
  num_patches: 12
  max_forward_horizon: 126

current_feature_dims:
  F_asset: 22
  F_market: 6
  F_macro: 44

asset_sampling:
  train_k_assets: 256
  validation_assets: all_valid_assets
  diagnostic_k_assets: 256
  diagnostic_views_per_date: 3

model_dims:
  d_asset: 128
  d_market: 64
  d_macro: 64
  d_model: 128
  latent_dim: 8

temporal_encoder:
  layers: 2
  heads: 4
  mlp_ratio: 4
  dropout: 0.1

predictor:
  type: transformer_decoder
  layers: 2
  heads: 4
  d_model: 128
  dropout: 0.1

masking:
  mask_ratio: 0.35
  min_masked_patches: 3
  max_masked_patches: 5
```

Shape symbols:

```text
B = batch size
T = 252 trading days
N = number of selected assets
L = patch length = 21 trading days
P = number of patches = 12
F_asset = 22
F_market = 6
F_macro = 44
D = d_model = 128
Z = latent_dim = 8
```

---

## 4. High-Level Model Diagram

```text
Sparse frozen Parquet facts
        |
        |  Dataloader reconstructs dense masked windows
        v

asset_x:        [B, 252, N, 22]
market_x:       [B, 252, 6]
macro_x:        [B, 252, 44]
valid masks:    [B, 252, N], [B, 252]
        |
        |  Patch time into 12 x 21-day blocks
        v

asset_patches:  [B, 12, 21, N, 22]
market_patches: [B, 12, 21, 6]
macro_patches:  [B, 12, 21, 44]
        |
        |  Per-stream patch tokenizers
        v

asset_tokens:   [B, 12, N, 128]
market_tokens:  [B, 12, 64]
macro_tokens:   [B, 12, 64]
        |
        |  Masked mean asset pooling
        v

panel_tokens:   [B, 12, 128]
        |
        |  Fuse panel + market + macro
        v

concat:         [B, 12, 256]
x_tokens:       [B, 12, 128]
        |
        |  Add patch position embeddings
        v

x_tokens:       [B, 12, 128]
        |
        |  Temporal JEPA masking
        v

visible tokens: [B, P_visible, 128]
masked targets: [B, P_masked, 128]
        |
        |----------------------------|
        |                            |
        v                            v
context encoder               EMA target encoder
        |                            |
        v                            v
context reps                  target reps
[B, P_visible, 128]           [B, P_masked, 128]
        |                            ^
        v                            |
Transformer predictor ---------------|
        |
        v

predicted targets: [B, P_masked, 128]
JEPA loss: normalized MSE

Full export path:
x_tokens [B, 12, 128]
    -> context encoder with all patches
    -> state exporter
    -> z_t [B, 8]
```

---

## 5. Dataloader Output Contract

The model should receive already-normalized, zero-filled tensors plus masks.

```python
batch = {
    "sample_date": list[str] or Tensor,

    "asset_x": Tensor[B, 252, N, 22],
    "market_x": Tensor[B, 252, 6],
    "macro_x": Tensor[B, 252, 44],
_
    "valid_asset_mask": Tensor[B, 252, N],
    "valid_date_mask": Tensor[B, 252],
    "holdout_date_mask": Tensor[B, 252],

    # optional but useful
    "asset_ids": Tensor[B, N],
    "split_label": list[str],
}
```

The model should never treat zero values as valid observations. Zeros are just the fill value for invalid, missing, padded, or protected data.

---

## 6. Time Patching

Raw input shapes:

```text
asset_x:  [B, 252, N, 22]
market_x: [B, 252, 6]
macro_x:  [B, 252, 44]
```

Patch the 252-day window into 12 blocks of 21 trading days:

```text
asset_patches:  [B, 12, 21, N, 22]
market_patches: [B, 12, 21, 6]
macro_patches:  [B, 12, 21, 44]
```

The 21-day axis is called `L` in the model code. It should not be hard-coded as `21` inside modules when possible. It should be represented as `patch_len` or `L`.

---

## 7. Asset Patch Tokenizer

### 7.1 Conceptual Operation

For every sample, patch, and asset, the asset patch tokenizer maps:

```text
[L, F_asset] -> [D_asset]
```

With the current first config:

```text
[21, 22] -> [128]
```

This means:

```text
For asset n, inside patch p, inside sample b:
    take the asset's 21-day feature history
    compress it into one 128-dimensional token
```

Output:

```text
asset_tokens: [B, P, N, D_asset]
asset_tokens: [B, 12, N, 128]
```

### 7.2 Why reshape `[B, P, L, N, F] -> [B * P * N, L, F]`?

This reshape is only an implementation/vectorization trick. It is not a modeling decision.

The asset patches start as:

```python
asset_patches.shape = [B, P, L, N, F_asset]
# example: [32, 12, 21, 256, 22]
```

For each `(b, p, n)`, we want to apply the same tokenizer to:

```python
one_asset_patch.shape = [L, F_asset]
# [21, 22]
```

To do this efficiently, we can temporarily reshape:

```python
x = asset_patches.permute(0, 1, 3, 2, 4)
# [B, P, N, L, F]

x = x.reshape(B * P * N, L, F_asset)
# [32 * 12 * 256, 21, 22]
# [98,304, 21, 22]
```

Then the tokenizer maps:

```python
[B * P * N, L, F_asset] -> [B * P * N, D_asset]
```

After tokenization, reshape back:

```python
asset_tokens = asset_tokens.reshape(B, P, N, D_asset)
# [32, 12, 256, 128]
```

So batch, patch, and asset are not permanently combined. They are only flattened temporarily so the same `[L, F] -> [D]` function can be applied to every asset patch efficiently.

### 7.3 v1 Asset Tokenizer

Recommended first version:

```text
Input: [B * P * N, L, F_asset]

Linear(F_asset -> hidden_dim)
GELU
LayerNorm(hidden_dim)
masked mean over L
Linear(hidden_dim -> D_asset)
GELU
LayerNorm(D_asset)

Output: [B * P * N, D_asset]
Reshape: [B, P, N, D_asset]
```

Example dimensions:

```text
Input:  [98,304, 21, 22]
Linear: [98,304, 21, 64]
Mean:   [98,304, 64]
Linear: [98,304, 128]
Output: [32, 12, 256, 128]
```

The `L` axis is used inside the masked mean. It is the temporal dimension being summarized by the tokenizer.

### 7.4 Why GELU?

GELU is not mandatory. It is a reasonable default because:

```text
GELU is standard in Transformer-style MLP blocks.
It is smooth around zero.
The model inputs are normalized continuous features.
It usually behaves well without much tuning.
```

A future comparison could test GELU vs SiLU vs ReLU, but for v1:

```yaml
activation: gelu
```

is a fine default.

---

## 8. Market and Macro Patch Tokenizers

### 8.1 Market Patch Tokenizer

Input:

```text
market_patches: [B, P, L, F_market]
market_patches: [B, 12, 21, 6]
```

Recommended v1 tokenizer:

```text
Linear(6 -> 32)
GELU
LayerNorm(32)
masked mean over L
Linear(32 -> 64)
GELU
LayerNorm(64)
```

Output:

```text
market_tokens: [B, P, D_market]
market_tokens: [B, 12, 64]
```

Purpose:

```text
Represent broad market-level conditions inside each 21-day patch.
```

### 8.2 Macro Patch Tokenizer

Input:

```text
macro_patches: [B, P, L, F_macro]
macro_patches: [B, 12, 21, 44]
```

Recommended v1 tokenizer:

```text
Linear(44 -> 64)
GELU
LayerNorm(64)
masked mean over L
Linear(64 -> 64)
GELU
LayerNorm(64)
```

Output:

```text
macro_tokens: [B, P, D_macro]
macro_tokens: [B, 12, 64]
```

Purpose:

```text
Represent slower rates, macro, credit, and factor context inside each patch.
```

---

## 9. Masked Mean Asset Pooling

After tokenization:

```text
asset_tokens: [B, P, N, D_asset]
asset_tokens: [B, 12, N, 128]
```

The model needs one asset-panel summary token per patch:

```text
panel_tokens: [B, P, D_asset]
panel_tokens: [B, 12, 128]
```

v1 pooling method:

```text
masked mean over assets
```

Formula:

```text
panel_token[b, p] =
    sum_n asset_tokens[b, p, n] * patch_asset_mask[b, p, n]
    /
    clamp(sum_n patch_asset_mask[b, p, n], min=1)
```

This is intentionally simple. It forces the first model to learn from broad panel structure instead of immediately learning to focus on a small number of crisis-sensitive or high-volatility assets.

---

## 10. Stream Fusion

Inputs:

```text
panel_tokens:  [B, 12, 128]
market_tokens: [B, 12, 64]
macro_tokens:  [B, 12, 64]
```

Concatenate per patch:

```text
combined: [B, 12, 256]
```

Fusion projection:

```text
Linear(256 -> 128)
GELU
Dropout(0.1)
LayerNorm(128)
```

Output:

```text
x_tokens: [B, 12, 128]
```

Each `x_tokens[:, p, :]` is one fused 21-day market-state patch token containing asset-panel, market-level, and macro context.

---

## 11. Why Add Position Embeddings After Fusion?

The cleanest place to add temporal position embeddings is after the streams have been fused into the shared model dimension.

Before fusion:

```text
panel_tokens:  [B, 12, 128]
market_tokens: [B, 12, 64]
macro_tokens:  [B, 12, 64]
```

After fusion:

```text
x_tokens: [B, 12, 128]
```

The temporal Transformer expects tokens in `D_model = 128`, so use:

```python
x_tokens = fusion_projection(combined)
x_tokens = x_tokens + patch_pos_embedding
```

where:

```text
patch_pos_embedding: [12, 128]
```

This means:

```text
this token is patch 0
this token is patch 1
...
this token is patch 11
```

Adding position embeddings before fusion would require separate position embeddings for the asset, market, and macro streams. That adds complexity without clear benefit for v1.

---

## 12. Masking Flow in Detail

There are three levels of masking:

```text
1. Raw observation masks
2. Patch validity and target eligibility masks
3. JEPA temporal masks
```

These should be treated separately.

---

### 12.1 Raw Observation Masks

The dataloader provides:

```python
valid_asset_mask.shape  = [B, T, N]
valid_date_mask.shape   = [B, T]
holdout_date_mask.shape = [B, T]
```

Meaning:

```python
valid_asset_mask[b, day, n] = 1
```

means asset `n` has a real valid observation on that day.

```python
valid_date_mask[b, day] = 1
```

means the day is real usable data for this sample.

```python
holdout_date_mask[b, day] = 1
```

means this day belongs to protected validation/holdout context.

For training, protected lookback dates may appear in reconstructed windows as zero-filled masked context, but they must not be JEPA targets.

---

### 12.2 Patch Masks

After time patching:

```python
valid_asset_mask_patched.shape  = [B, P, L, N]
valid_date_mask_patched.shape   = [B, P, L]
holdout_mask_patched.shape      = [B, P, L]
```

Compute asset-level patch validity:

```python
asset_valid_days = valid_asset_mask_patched.sum(dim=2)
# [B, P, N]
```

Then:

```python
patch_asset_mask = asset_valid_days >= min_valid_days_per_asset_patch
# [B, P, N]
```

Recommended first setting:

```yaml
min_valid_days_per_asset_patch: 10
```

This means an asset is considered valid inside a 21-day patch if it has at least 10 valid days in that patch.

Compute date-level patch validity:

```python
patch_valid_date_count = valid_date_mask_patched.sum(dim=2)
# [B, P]

patch_has_enough_dates = patch_valid_date_count >= 10
# [B, P]
```

Compute whether a patch touches protected data:

```python
patch_has_holdout = holdout_mask_patched.any(dim=2)
# [B, P]
```

Compute asset coverage:

```python
valid_asset_fraction = patch_asset_mask.float().mean(dim=2)
# [B, P]
```

Then compute JEPA target eligibility:

```python
patch_target_eligible = (
    patch_has_enough_dates
    & (~patch_has_holdout)
    & (valid_asset_fraction >= 0.25)
)
```

Shape:

```text
patch_target_eligible: [B, 12]
```

Rules:

```text
A patch may be used as context even if it is not target-eligible.
A patch must not be used as a JEPA prediction target if it is padded.
A patch must not be used as a JEPA prediction target if it touches protected holdout data.
A patch must not be used as a JEPA prediction target if it has insufficient valid data.
```

---

### 12.3 Masks Inside Asset Tokenization

The asset tokenizer uses day-level masks when reducing over `L`.

For each flattened asset patch:

```text
x:        [B * P * N, L, F_asset]
day_mask: [B * P * N, L]
```

After daily projection:

```text
x: [B * P * N, L, hidden_dim]
```

Masked mean over `L`:

```text
x_token = sum_l x[:, l, :] * day_mask[:, l]
          /
          clamp(sum_l day_mask[:, l], min=1)
```

Output:

```text
x_token: [B * P * N, D_asset]
```

Then reshape back:

```text
asset_tokens: [B, P, N, D_asset]
```

---

### 12.4 Masks Inside Asset Pooling

Asset pooling uses:

```text
asset_tokens:      [B, P, N, D_asset]
patch_asset_mask:  [B, P, N]
```

Then masked mean over assets:

```text
panel_tokens: [B, P, D_asset]
```

This is where the model reduces the asset dimension.

---

### 12.5 JEPA Temporal Masking

After fusion:

```text
x_tokens: [B, 12, 128]
patch_target_eligible: [B, 12]
```

Sample target patches only from eligible patches.

Example:

```text
patch_target_eligible[b] =
[False, True, True, True, True, True, True, True, True, True, False, False]
```

Allowed target patches:

```text
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Possible masked patches:

```text
masked_ids = [2, 5, 6, 9]
```

Visible patches are all patches not selected as masked targets:

```text
visible_ids = all_patch_ids - masked_ids
```

For v1, target-ineligible patches may still appear as context if they have some useful information. However, fully invalid/padded context patches should be masked from attention using a context key-padding mask.

Shapes:

```text
visible_tokens: [B, P_visible, 128]
target_tokens:  [B, P_masked, 128]
```

---

## 13. Context Encoder

Input:

```text
visible_tokens: [B, P_visible, 128]
visible_positions: patch ids for visible tokens
```

Add position embeddings before encoding:

```python
visible_tokens = x_tokens[:, visible_ids, :] + pos_embed[visible_ids]
```

Recommended v1 encoder:

```yaml
context_encoder:
  type: temporal_transformer
  d_model: 128
  layers: 2
  heads: 4
  mlp_ratio: 4
  dropout: 0.1
  pre_norm: true
```

Output:

```text
context_encoded: [B, P_visible, 128]
```

Purpose:

```text
Encode the visible historical patch sequence into contextual representations.
```

---

## 14. Target Encoder and Target Tokenizer Options

The target encoder creates the representation that the predictor tries to match.

The loss is:

```python
loss = mse(
    normalize(predicted_target),
    stopgrad(normalize(target_encoded))
)
```

The predictor is trained to match the target representation. The target representation is stop-gradient so the model cannot trivially move the target to make the loss easy.

There are three reasonable target-branch options.

---

### 14.1 Option A — Shared Tokenizer, EMA Temporal Target Encoder

This is the recommended v1 option.

Compute fused tokens once using the online tokenizer and fusion stack:

```python
x_tokens = tokenize_and_fuse(asset_x, market_x, macro_x)
# [B, 12, 128]
```

Split into visible and target tokens:

```python
visible_tokens = x_tokens[:, visible_ids, :]
target_tokens  = x_tokens[:, target_ids, :]
```

Context branch:

```python
context_encoded = online_context_encoder(visible_tokens + pos_embed[visible_ids])
```

Target branch:

```python
with torch.no_grad():
    target_encoded = ema_target_encoder(target_tokens + pos_embed[target_ids])
```

The target encoder weights are an EMA copy of the online context encoder.

Pros:

```text
Simple.
Good first real implementation.
Less code duplication.
Fewer moving parts.
```

Cons:

```text
The target tokens still come from the online tokenizer/fusion stack.
The tokenizer is not EMA-smoothed.
```

Recommended v1 config:

```yaml
target_branch:
  type: ema_temporal_encoder
  tokenizer: shared_online_tokenizer
  fusion: shared_online_fusion
  stop_gradient_target: true
```

---

### 14.2 Option B — Full EMA Stack

This is a future improvement.

Online branch:

```text
asset tokenizer
market tokenizer
macro tokenizer
fusion projection
context encoder
predictor
```

Target branch:

```text
EMA asset tokenizer
EMA market tokenizer
EMA macro tokenizer
EMA fusion projection
EMA target encoder
```

The target branch receives the raw masked patch data and produces target representations through its own EMA copy.

Pros:

```text
More stable target representations.
Closer to the full JEPA idea.
Cleaner separation between online and target branches.
```

Cons:

```text
More code.
More memory.
More chances for bugs.
EMA updates must cover more modules.
```

This should not be the first v1 implementation unless the simpler EMA temporal target encoder is unstable.

---

### 14.3 Option C — Same Encoder with Stop-Gradient Only

Simplest debugging version:

```python
target_encoded = online_context_encoder(target_tokens + pos_embed[target_ids]).detach()
```

Pros:

```text
Very easy to implement.
Useful for debugging.
```

Cons:

```text
Higher collapse risk.
Less stable.
Not ideal for the real model.
```

This can be used during development, but it does not need to be part of the formal architecture plan.

---

## 15. Transformer Predictor

The v1 plan should use a full Transformer predictor, not a mean-summary predictor.

The predictor receives:

```text
context_encoded: [B, P_visible, 128]
target_ids:      [B, P_masked]
```

Create target query tokens:

```python
target_queries = learned_mask_token + pos_embed[target_ids]
# [B, P_masked, 128]
```

Then use a Transformer decoder:

```python
predicted_target = predictor_decoder(
    tgt=target_queries,
    memory=context_encoded
)
```

Output:

```text
predicted_target: [B, P_masked, 128]
```

Purpose:

```text
Given the encoded visible patch sequence and the positions of the missing target patches,
predict the target representations for those masked patches.
```

Recommended config:

```yaml
predictor:
  type: transformer_decoder
  d_model: 128
  layers: 2
  heads: 4
  mlp_ratio: 4
  dropout: 0.1
  target_query: learned_mask_token_plus_position
```

Important detail:

```text
The predictor should not receive the raw target tokens.
It receives learned target queries that tell it which positions to predict.
```

---

## 16. JEPA Loss

Use normalized MSE:

```python
pred = F.normalize(predicted_target, dim=-1)
targ = F.normalize(target_encoded.detach(), dim=-1)

loss_jepa = ((pred - targ) ** 2).sum(dim=-1).mean()
```

Initial v1 loss:

```yaml
loss:
  jepa:
    type: normalized_mse
    weight: 1.0
```

Variance and covariance regularizers should be logged as diagnostics first. They can be enabled later if collapse appears.

Possible later config:

```yaml
loss:
  jepa:
    weight: 1.0
  variance:
    enabled: true
    weight: 0.01
  covariance:
    enabled: true
    weight: 0.001
```

---

## 17. Representation Evaluation and Export

`encode_pooled_state()` is the representation source of truth for evaluation.
The original parameterized but untrained `state_exporter` projection is retained only for checkpoint
compatibility. The JEPA loss does not train that projection, so its output must
not be used for diagnostics, embedding exports, or probes.

During pretraining, the model learns by predicting masked patch representations. After training, the main artifact is:

```text
z_t = compact market-state embedding for sample date t
```

For export, do **not** mask patches.

Use the full 252-day window:

```text
x_tokens: [B, 12, 128]
```

Pass all patch tokens through the context encoder:

```python
full_encoded = context_encoder(x_tokens + pos_embed)
# [B, 12, 128]
```

Then pool into one learned encoder state.

Required pooled-state contract:

```python
mean_state = masked_mean(full_encoded, patch_context_mask, dim=1)
# [B, 128]

require patch_context_mask[:, -1] == True
endpoint_state = full_encoded[:, -1, :]
# [B, 128]

pooled_state = torch.cat([mean_state, endpoint_state], dim=-1)
# [B, 256]
```

The endpoint patch is required because the exported state is explicitly the
state at date `t`. Evaluation must fail rather than silently substitute an
earlier valid patch.

Fit an 8-dimensional, non-whitened PCA exporter on all-valid **train pooled
states only**. Apply that frozen checkpoint-specific projection to validation
and deterministic K-asset views:

```text
raw pooled state: [B, 256]  -> collapse source of truth
PCA export:       [B, 8]    -> frozen published representation
```

PCA component signs should be canonicalized, but PCA axes remain
checkpoint-specific and must not be interpreted as stable coordinates across
epochs.

Output:

```text
z_t: [B, 8]
```

Why mean plus last?

```text
mean_state captures broad 252-day regime context.
last_state captures the endpoint condition near date t.
z_t should represent the state at date t, not merely the average condition over the full year.
```

Export format:

```text
date
z_1
z_2
...
z_8
split_label
model_version
dataset_version
pca_version
```

---

## 18. Training Loop v1

For each batch:

```text
1. Load sample dates.
2. Reconstruct 252-day windows.
3. Select k_assets = 256 for training.
4. Zero-fill invalid/protected/missing values.
5. Build raw masks.
6. Patch time into 12 blocks of L=21 days.
7. Build patch validity masks.
8. Build patch target eligibility masks.
9. Tokenize asset, market, and macro patches.
10. Pool asset tokens with masked mean.
11. Fuse panel, market, and macro streams.
12. Add patch position embeddings.
13. Select 3-5 eligible masked target patches.
14. Send visible patches to the context encoder.
15. Send masked target patches to the EMA target encoder.
16. Create target query tokens for the predictor.
17. Predict masked target representations with Transformer predictor.
18. Compute normalized JEPA loss.
19. Apply optimizer step.
20. Update EMA target encoder.
21. Log diagnostics.
```

Recommended optimization config:

```yaml
optimization:
  optimizer: adamw
  lr: 0.0001
  weight_decay: 0.01
  batch_size: 32
  validation_batch_size: 8
  epochs: 100
  warmup_epochs: 5
  grad_clip_norm: 1.0
  mixed_precision: true

ema:
  momentum_start: 0.99
  momentum_end: 0.999
```

---

## 19. Validation Plan

Main validation should be deterministic:

```text
Use validation sample dates only.
Use real validation data.
Use all valid assets when memory allows.
Use frozen temporal masks.
Do not use random K-asset sampling in main validation.
Use normalization fitted only on train-allowed facts.
```

Recommended validation batch size:

```yaml
validation_batch_size: 8
```

because all-valid-assets validation is heavier than training with `k_assets = 256`.

---

## 20. K-Asset View Stability Diagnostics

For each validation sample date:

```python
z_all = encoder(all_valid_assets)
z_k1  = encoder(fixed_k_asset_view_1)
z_k2  = encoder(fixed_k_asset_view_2)
z_k3  = encoder(fixed_k_asset_view_3)
```

Track:

```text
cosine_similarity(z_all, z_k1)
cosine_similarity(z_all, z_k2)
cosine_similarity(z_all, z_k3)

cosine_distance(z_all, z_k1) = 1 - cosine_similarity(z_all, z_k1)
cosine_distance(z_all, z_k2) = 1 - cosine_similarity(z_all, z_k2)
cosine_distance(z_all, z_k3) = 1 - cosine_similarity(z_all, z_k3)
```

Purpose:

```text
Check whether the same date produces similar z_t under different asset subsets.
```

If the model changes wildly across K-asset views, the asset pooling method may be unstable.

---

## 21. All-Validation Cosine Geometry Diagnostics

Also track pairwise distances among all `z_all` validation embeddings:

```text
pairwise cosine similarity among z_all vectors across validation dates
pairwise cosine distance among z_all vectors across validation dates
mean / median / std / min / max of those distances
```

Purpose:

```text
Check whether different validation dates are actually separated in latent space,
or whether all z_all vectors are nearly identical.
```

This gives two different validation checks:

```text
K-view stability:
    Does the same date remain stable under different asset subsets?

All-validation geometry:
    Are different dates separated, or has the representation collapsed?
```

---

## 22. Required Diagnostics

Track these during pretraining for both raw pooled states and PCA exports.
Raw pooled-state diagnostics are the collapse source of truth because train-fit
PCA makes train covariance mostly diagonal by construction:

```text
train_jepa_loss
validation_jepa_loss
target_patch_eligibility_rate
masked_patch_count_mean
padded_context_patch_rate
protected_context_patch_rate

z_mean
z_std_per_dim
z_covariance
z_correlation_matrix
z_effective_rank
mean_pairwise_cosine_similarity_z
mean_pairwise_cosine_distance_z

z_all_vs_k_view_cosine_similarity
z_all_vs_k_view_cosine_distance
pairwise_cosine_similarity_all_validation_z
pairwise_cosine_distance_all_validation_z
```

Fixed-K asset selection must be a pure deterministic function of dataset build
ID, sample date, view index, K, and candidate asset ID. It must not depend on
dataloader workers, batch order, Python hash seed, or hardware.

Collapse warning signs:

```text
z_t std near zero
effective rank close to 1
all z dimensions highly correlated
mean pairwise cosine similarity near 1
mean pairwise cosine distance near 0
validation JEPA loss improves while probes show only volatility
z_t changes heavily across K-asset samples for the same date
```

---

## 23. v1 Config Draft

```yaml
model:
  name: fi_jepa_v1_mean_pool_temporal_transformer

input:
  lookback_days: 252
  patch_len: 21
  num_patches: 12

features:
  asset_dim: 22
  market_dim: 6
  macro_dim: 44

asset_sampling:
  train_mode: random_k
  train_k_assets: 256
  validation_mode: all_valid_assets
  diagnostic_k_assets: 256
  diagnostic_views_per_date: 3

tokenizers:
  asset:
    type: daily_projection_masked_mean
    hidden_dim: 64
    output_dim: 128
    activation: gelu
  market:
    type: daily_projection_masked_mean
    hidden_dim: 32
    output_dim: 64
    activation: gelu
  macro:
    type: daily_projection_masked_mean
    hidden_dim: 64
    output_dim: 64
    activation: gelu

asset_pooling:
  type: masked_mean

fusion:
  input_dim: 256
  output_dim: 128
  add_patch_position_after_projection: true

context_encoder:
  type: temporal_transformer
  d_model: 128
  layers: 2
  heads: 4
  mlp_ratio: 4
  dropout: 0.1
  pre_norm: true

target_branch:
  type: ema_temporal_encoder
  tokenizer: shared_online_tokenizer
  fusion: shared_online_fusion
  stop_gradient_target: true
  ema_momentum_start: 0.99
  ema_momentum_end: 0.999

predictor:
  type: transformer_decoder
  d_model: 128
  layers: 2
  heads: 4
  mlp_ratio: 4
  dropout: 0.1
  target_query: learned_mask_token_plus_position

state_exporter:
  type: legacy_checkpoint_compatibility_only
  input_dim: 256
  hidden_dim: 128
  latent_dim: 8

representation_export:
  source: encode_pooled_state
  projection: train_only_non_whitened_pca
  latent_dim: 8

masking:
  type: temporal_block
  mask_ratio: 0.35
  min_masked_patches: 3
  max_masked_patches: 5
  min_valid_dates_in_patch: 10
  min_valid_asset_fraction: 0.25
  allow_holdout_patches_as_targets: false
  allow_padded_patches_as_targets: false

loss:
  jepa:
    type: normalized_mse
    weight: 1.0
  variance:
    enabled: false
    weight: 0.01
  covariance:
    enabled: false
    weight: 0.001

optimization:
  optimizer: adamw
  lr: 0.0001
  weight_decay: 0.01
  batch_size: 32
  validation_batch_size: 8
  epochs: 100
  warmup_epochs: 5
  grad_clip_norm: 1.0
  mixed_precision: true

diagnostics:
  validation_jepa_loss: true
  z_variance_per_dim: true
  z_covariance: true
  z_effective_rank: true
  z_pairwise_cosine_similarity_all_validation: true
  z_pairwise_cosine_distance_all_validation: true
  z_all_vs_k_view_cosine_similarity: true
  z_all_vs_k_view_cosine_distance: true
  target_eligibility_rate: true
  padded_patch_rate: true
  protected_patch_rate: true
```

---

## 24. Future Investigations

These are not part of the v1 architecture. They should be kept as ideas to revisit after the mean-pooling temporal Transformer baseline is working.

Attention pooling over assets instead of masked mean pooling. This would let the model decide which assets matter most inside each patch, but it may over-focus on crisis-sensitive or high-volatility names.

Multiple learned asset summary tokens. Instead of one pooled panel token per patch, the model could learn several summary tokens that may capture broad market state, stressed assets, dispersion, liquidity, or sector/factor rotation.

Asset-temporal encoding before pooling. Each asset's 12-patch trajectory could be encoded before reducing across assets, giving the model more ability to understand individual asset dynamics before panel aggregation.

Cross-asset temporal Transformers. Eventually test models that explicitly allow assets to interact before pooling, but only after the simpler baseline is strong.

Full EMA target stack. Move from shared tokenizer plus EMA temporal encoder to an EMA copy of the tokenizers, fusion layer, and target encoder.

Mean-summary predictor baseline. Compare the full Transformer predictor against a simpler predictor that uses a mean-pooled context summary.

Volatility dominance diagnostics. Check whether `z_t` is mostly a volatility/crisis embedding by comparing against realized volatility, VIX-like features, nearest-neighbor overlap, and residualized probes.

Volatility-aware regularizers. If attention pooling or `z_t` collapses toward volatility, test attention entropy penalties, attention-volatility decorrelation, latent-volatility decorrelation, or adversarial volatility heads.

Alignment losses. Only after the representation is stable, test small alignment heads for future volatility, drawdown, trend/chop, breadth, dispersion, correlation, and conditional signal-performance profiles.

MoE heads. Potentially useful later, but only after non-MoE baselines are strong and routing can be diagnosed for stability.

---

## 25. Practical Implementation Order

Recommended implementation order:

```text
1. Batch reconstruction from frozen sparse facts.
2. Time patching utilities.
3. Raw mask patching utilities.
4. Patch validity and target eligibility logic.
5. Asset patch tokenizer.
6. Market patch tokenizer.
7. Macro patch tokenizer.
8. Masked mean asset pooling.
9. Fusion projection.
10. Patch position embeddings.
11. Temporal JEPA mask sampler.
12. Context encoder.
13. EMA target encoder.
14. Transformer predictor.
15. JEPA loss.
16. EMA update logic.
17. State exporter.
18. Main validation loop.
19. K-asset diagnostic validation.
20. All-validation cosine geometry diagnostics.
21. Embedding export.
22. Frozen probes.
```

The main principle is to make the v1 mean-pooling temporal JEPA strong and well-diagnosed before adding richer asset interaction mechanisms.

---

## 26. Core Rule Summary

```text
One sample is one market-date window, not one asset-date row.
The model outputs one z_t per sample date.
The dataloader reconstructs windows from sparse facts.
The model must receive masks and must not infer validity from zero-filled values.
Patch time first, then tokenize streams.
Asset patch tokenizer maps [L, F_asset] -> [D_asset] for each asset-patch.
Masked mean pooling reduces [B, P, N, D] -> [B, P, D].
Fusion creates [B, 12, 128] temporal patch tokens.
Position embeddings are added after fusion.
JEPA targets must be sampled only from eligible real patches.
The predictor receives context representations and target position queries, not raw target tokens.
The state exporter uses the full unmasked window to produce z_t.
Validation should check both K-asset stability and all-validation latent geometry.
Future volatility regularizers should be saved for later, not included in v1.
```
