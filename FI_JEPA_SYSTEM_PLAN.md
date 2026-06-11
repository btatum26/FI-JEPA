# FI-JEPA System Planning Document

**Project name:** FI-JEPA — Financial Joint-Embedding Predictive Architecture  
**Document purpose:** Standalone research and engineering plan for building a JEPA-style market-state representation system.  
**Intended use:** Research prototype first; slow market-state prediction and conditional signal-performance analysis later.  
**Current stance:** Build a self-supervised market-state encoder first. Add tradability alignment only after the representation proves that it captures useful, stable, nontrivial market structure.

---

## 1. Executive Summary

FI-JEPA is a proposed self-supervised representation-learning system for financial market state. The goal is to learn a compact latent state vector, `z_t`, from past-only market information. The model should not begin as a direct return predictor. It should first learn broad market structure: volatility, trend, drawdown risk, dispersion, breadth, correlation, liquidity stress, sector/factor rotation, and macro/credit/rates context.

The project is motivated by the Conditional IC Surface idea: signal usefulness is not universal; it depends on state. The earlier Conditional IC framework defines a state-dependent relationship between a signal and forward returns. FI-JEPA generalizes this direction by trying to learn the state representation itself rather than only specifying state variables by hand.

The central research question is:

> Can a self-supervised predictive embedding of historical market structure produce a compact market-state representation that improves slow, out-of-sample prediction of market conditions and conditional signal performance?

The project should be built in stages:

1. Build a clean, low-cost market-state dataset.
2. Train a simple JEPA-style encoder with no future-return inputs.
3. Diagnose whether the latent space captures meaningful structure or collapses into volatility.
4. Probe the frozen embedding against slow future targets.
5. Only then add small alignment losses or downstream heads related to returns, IC, or signal usefulness.
6. Treat MoE heads as a later optional extension, not the starting point.

The system must be designed around leakage control, survivorship-bias labeling, walk-forward evaluation, and harsh baselines.

---

## 2. Core Hypothesis

Financial markets are non-stationary, but not structureless. Some market movements may be better understood as transitions between slowly evolving latent states. Hand-designed indicators partially describe these states, but they are limited by human design and by low-dimensional assumptions.

A JEPA-style model may be useful because it can learn by predicting missing or future latent structure from context, without reconstructing every noisy price movement and without directly optimizing noisy raw returns at the start.

The core hypothesis is not:

> JEPA will directly discover alpha.

The stronger hypothesis is:

> A JEPA-style encoder can learn a compact market-state representation from past-only data, and that representation can later be tested or gently aligned toward slow tradable outcomes.

---

## 3. Relationship to the Conditional IC Surface

The Conditional IC Surface asks whether a signal's predictive relationship with future returns changes as a function of state. It distinguishes the state variable from the signal and from the forward return.

FI-JEPA shifts the question from:

> Given a chosen state vector `X_t`, where does a signal work?

To:

> Can we learn a better state vector `z_t = f(history_t)` from market data itself?

The Conditional IC Surface remains useful for evaluation. It provides a downstream test: does the learned state `z_t` help explain where simple signals work or fail? If not, then the learned representation may be visually or statistically interesting but not financially useful.

Important distinction:

- The encoder should not receive future returns as inputs.
- Future returns may be used later for evaluation, probing, or carefully controlled alignment.
- Tradability must enter the research loop somewhere, but it does not need to enter the first pretraining objective.

---

## 4. What FI-JEPA Is Not

FI-JEPA is not a high-frequency model.  
It is not meant to trade many times per day.  
It is not initially a stock-selection model.  
It is not a direct next-day return predictor.  
It is not a replacement for careful backtesting, IC analysis, or walk-forward validation.  
It is not automatically survivorship-bias-free just because ticker identity is hidden.  
It is not safe from leakage just because masking is used.

The intended first version is a slow-moving market-state representation model.

---

## 5. Design Principles

### 5.1 Past-only inputs

For representation at time `t`, the context encoder may only receive information available at or before `t`.

Allowed:

- Past returns.
- Past volume/liquidity features.
- Past realized volatility.
- Past sector/factor returns.
- Macro data with realistic availability assumptions.
- Past breadth, dispersion, correlation, and drawdown features.

Not allowed as encoder inputs:

- Future returns.
- Forward volatility.
- Forward drawdown.
- Future realized IC.
- Features normalized using future data.
- Macro data before its release date.
- Absolute date encodings that let the model memorize historical episodes.

### 5.2 Separate representation learning from tradability alignment

The first model should learn structure. Later models can align structure toward slow market outcomes.

Recommended order:

1. Self-supervised pretraining.
2. Frozen linear probes.
3. Frozen nonlinear probes.
4. Small supervised heads.
5. Tiny alignment regularizer.
6. Optional partial encoder fine-tuning.

### 5.3 Use harsh baselines

FI-JEPA must beat simple alternatives:

- VIX or realized volatility alone.
- Hand-designed market-state features.
- PCA on features.
- Autoencoder.
- Masked autoencoder.
- Random projection.
- Simple supervised MLP.
- Linear/logistic models using traditional indicators.

If FI-JEPA does not beat these on walk-forward probes, the model is not yet useful.

### 5.4 Do not trust pretty clusters

A latent space can look clean while being useless. Clustering should be treated as visualization, not validation. The validation target is whether `z_t` improves slow out-of-sample prediction or conditional signal-performance estimation.

### 5.5 Use low-dimensional embeddings carefully

A low-dimensional `z_t` helps reduce density problems only if distance in latent space corresponds to similarity in relevant future behavior. Low dimension does not remove the need for effective sample size, confidence intervals, and novelty detection.

---

## 6. Data Strategy

### 6.1 First dataset goal

The first dataset should be market-wide, not stock-selection-heavy.

This is because the project goal is slow market-state representation, and market-wide data avoids some early survivorship-bias problems. A flawed but broad stock universe can be added later.

### 6.2 Minimum viable free/cheap data sources

Recommended initial sources:

- Stooq: historical OHLCV for indices, ETFs, stocks, and other instruments.
- FRED: macro, rates, credit spreads, financial stress, economic series.
- CBOE or FRED: VIX and volatility-related series.
- Kenneth French Data Library: factor and industry portfolio returns.
- yfinance: fallback or convenience source only.
- EODHD / Sharadar / WRDS / CRSP: future upgrades if affordable or available through university access.

### 6.3 Dataset phases

#### Phase A — Market-wide free dataset

Goal: learn broad market-state structure.

Potential instruments/features:

- S&P 500 proxy.
- Nasdaq proxy.
- Russell 2000 proxy.
- Dow proxy.
- Sector ETFs or industry portfolios.
- Treasury yields.
- Yield curve slopes.
- VIX.
- High-yield credit spread.
- Investment-grade credit spread.
- Dollar index proxy.
- Gold.
- Oil.
- Fama-French factors.
- Momentum factor.
- Industry portfolio returns.

This phase is enough to test the basic representation idea.

#### Phase B — Broader cross-sectional proxy dataset

Goal: estimate breadth, dispersion, and correlation more directly.

Possible universe:

- Current S&P 500 constituents from free sources.
- Stooq historical prices.
- Sector labels where available.

This is survivorship-biased. It can be useful for representation experiments, but not for strong historical strategy claims.

#### Phase C — Better equity dataset

Goal: reduce survivorship bias and improve cross-sectional realism.

Possible upgrades:

- Delisted-aware EOD data provider.
- Sharadar-style dataset.
- WRDS/CRSP access through a university.
- Point-in-time index membership data.

This phase is required before making strong claims about stock-level alpha or historical cross-sectional performance.

### 6.4 Survivorship-bias handling

Anonymization is not the same as survivorship-bias correction.

Removing ticker identity prevents direct ticker memorization, but it does not restore missing bankrupt, merged, or delisted companies. Every dataset should include metadata flags:

```yaml
survivorship_status: unknown | biased_current_constituents | delisted_aware | point_in_time
universe_construction: market_wide | current_members | historical_members | vendor_point_in_time
data_quality_grade: prototype | research | production_candidate
```

Allowed claims with biased data:

- The pipeline works.
- The model can learn some market-state structure.
- The latent space has or does not have certain diagnostics.

Not allowed with biased data:

- The strategy is historically profitable in a realistic tradable universe.
- The model solves survivorship bias.
- Cross-sectional alpha results are reliable.

---

## 7. Data Schema

### 7.1 Raw price table

```text
date
symbol
source_symbol
open
high
low
close
volume
source
adjusted_flag
currency
exchange
asset_type
```

### 7.2 Symbol metadata table

```text
canonical_symbol
source_symbol
name
asset_type
exchange
sector
industry
country
first_available_date
last_available_date
survivorship_status
data_source
notes
```

### 7.3 Macro table

```text
date
series_id
value
source
release_lag_assumption
point_in_time_available
revision_policy
```

### 7.4 Feature table

Wide format is preferred for model input:

```text
date
feature_001
feature_002
...
feature_n
```

Long format is preferred for storage and auditability:

```text
date
feature_name
value
feature_family
source
availability_lag
```

### 7.5 Target table

```text
date
horizon
future_realized_vol
future_vol_bucket
future_trend_return
future_trend_bucket
future_path_efficiency
future_max_drawdown
future_drawdown_bucket
future_dispersion
future_breadth
future_avg_correlation
future_tail_risk
future_regime_label
```

Targets are not encoder inputs.

---

## 8. Feature Families

Feature families should be separable because FI-JEPA needs feature-family dropout and masking.

### 8.1 Market return features

- 1-day, 5-day, 21-day, 63-day, 126-day returns.
- Market-relative returns.
- Rolling cumulative returns.
- Rolling downside returns.

### 8.2 Volatility features

- Realized volatility over 21, 63, 126 days.
- Volatility of volatility.
- Absolute return moving averages.
- VIX level.
- VIX change.
- VIX term structure if available.

### 8.3 Trend and path features

- Moving-average distance.
- Moving-average slope.
- Path efficiency.
- Drawdown from rolling peak.
- Rebound from rolling trough.

### 8.4 Breadth features

- Percentage of assets up over 1, 5, 21 days.
- Percentage above moving averages.
- Advance/decline ratio if available.
- Equal-weight minus cap-weight proxy if available.

### 8.5 Dispersion features

- Cross-sectional standard deviation of returns.
- Interquartile range of returns.
- Sector dispersion.
- Factor dispersion.

### 8.6 Correlation and crowding features

- Average pairwise correlation.
- Rolling correlation between sectors.
- First principal component explained variance.
- Correlation between risk assets.

### 8.7 Credit/rates/macro features

- Treasury yield levels.
- Yield curve slope.
- Credit spread level.
- Credit spread change.
- Financial stress index.
- Inflation/unemployment only with release-lag assumptions.

### 8.8 Liquidity and volume features

- Dollar volume.
- Volume shock.
- Amihud-style illiquidity proxy if equity data supports it.
- Turnover proxy.

### 8.9 Calendar features

Calendar features should be ablated.

Experiments:

1. No calendar features.
2. Cyclical month/week features.
3. Calendar features with high dropout.

Absolute date should not be used.

---

## 9. Slow Future Targets for Probing and Alignment

These targets should be used for evaluation, probing, and later alignment. They should not be passed into the encoder as inputs.

Let `r_t` be daily market return and `h` be a horizon such as 21, 63, or 126 trading days.

### 9.1 Future realized volatility

```math
RV_{t,h} = \sqrt{\frac{252}{h} \sum_{i=1}^{h} r_{t+i}^2}
```

Use both continuous value and bucketed labels.

Recommended buckets:

- Low.
- Normal.
- High.
- Extreme.

### 9.2 Future trend

```math
TR_{t,h} = \sum_{i=1}^{h} r_{t+i}
```

Use continuous return and bucketed labels:

- Strong downtrend.
- Weak downtrend.
- Flat/chop.
- Weak uptrend.
- Strong uptrend.

### 9.3 Future path efficiency

```math
PE_{t,h} = \frac{|P_{t+h} - P_t|}{\sum_{i=1}^{h} |P_{t+i} - P_{t+i-1}|}
```

High path efficiency means cleaner trend. Low path efficiency means chop.

### 9.4 Future maximum drawdown

```math
MDD_{t,h} = \max_{0 \le u \le v \le h} \frac{P_{t+u} - P_{t+v}}{P_{t+u}}
```

Use buckets:

- Normal.
- Moderate drawdown.
- Severe drawdown.

### 9.5 Future dispersion

For a universe of assets `i = 1...N`:

```math
DISP_{t,h} = std_i(R_{i,t:t+h})
```

Also use interquartile range for robustness.

### 9.6 Future breadth

```math
BREADTH_{t,h} = \frac{1}{N} \sum_i 1[R_{i,t:t+h} > 0]
```

Alternative: percentage of assets above moving average at `t+h`.

### 9.7 Future average correlation

Compute rolling pairwise correlations across assets/sectors/factors inside the forward window.

Use:

- Average pairwise correlation.
- First principal component explained variance.
- Sector correlation concentration.

### 9.8 Future tail risk

Possible definitions:

- 5th percentile daily return inside the forward window.
- Downside semivariance.
- Count of days below a threshold.
- Expected-shortfall-style average of worst days.

### 9.9 Future conditional signal-performance profile

Later, once simple signals exist, define a vector of future signal behavior:

```text
momentum_IC_h
mean_reversion_IC_h
volatility_signal_IC_h
quality_or_factor_IC_h
trend_signal_IC_h
```

This can become an alignment target, but only after the core encoder is validated.

---

## 10. Model Architecture

### 10.1 High-level architecture

```text
Past-only market window
        |
        v
Feature tokenizer / patcher
        |
        v
Context encoder  -------------------------
        |                                  |
        v                                  |
Context representation c_t                |
        |                                  |
        v                                  |
Predictor predicts target representations |
        |                                  |
        v                                  |
Predicted target embeddings  <---- Target encoder on masked target blocks
        |
        v
JEPA loss + collapse regularizers
        |
        v
Market-state embedding z_t
        |
        +--> Diagnostics
        +--> Frozen probes
        +--> Slow target heads
        +--> Optional alignment
        +--> Optional signal heads
```

### 10.2 Input shape options

```python
# ------------------------------------------------------------
# Raw batch
# ------------------------------------------------------------
asset_x = Tensor[B, T, N, F_asset]       # [32, 252, N, 22]
market_x = Tensor[B, T, F_market]        # [32, 252, 6]
macro_x = Tensor[B, T, F_macro]          # [32, 252, 44]
valid_mask = Tensor[B, T, N]             # [32, 252, N]

# ------------------------------------------------------------
# Patch time
# ------------------------------------------------------------
asset_patches = patch_time(asset_x, patch_len=21)
# [B, P, L, N, F_asset]
# [32, 12, 21, N, 22]

market_patches = patch_time(market_x, patch_len=21)
# [B, P, L, F_market]
# [32, 12, 21, 6]

macro_patches = patch_time(macro_x, patch_len=21)
# [B, P, L, F_macro]
# [32, 12, 21, 44]

# ------------------------------------------------------------
# Embed patches
# ------------------------------------------------------------
asset_tokens = asset_patch_embed(asset_patches)
# [B, P, N, D_asset]
# [32, 12, N, 128]

market_tokens = market_patch_embed(market_patches)
# [B, P, D_market]
# [32, 12, 64]

macro_tokens = macro_patch_embed(macro_patches)
# [B, P, D_macro]
# [32, 12, 64]

# ------------------------------------------------------------
# Pool assets inside each patch
# ------------------------------------------------------------
patch_asset_mask = patch_valid_mask(valid_mask, patch_len=21)
# [B, P, N]

panel_tokens = masked_asset_pool(asset_tokens, patch_asset_mask)
# [B, P, D_asset]
# [32, 12, 128]

# ------------------------------------------------------------
# Fuse asset panel + market + macro per patch
# ------------------------------------------------------------
combined = concat([panel_tokens, market_tokens, macro_tokens], dim=-1)
# [B, P, 256]

x_tokens = fusion_projection(combined)
# [B, P, D]
# [32, 12, 128]

# Add temporal position embeddings
x_tokens = x_tokens + patch_position_embedding
# [32, 12, 128]

# ------------------------------------------------------------
# Mask patches
# ------------------------------------------------------------
visible_ids, masked_ids = sample_temporal_masks(P=12)

visible_tokens = x_tokens[:, visible_ids, :]
# [B, P_visible, D]

target_tokens = x_tokens[:, masked_ids, :]
# [B, P_masked, D]

# ------------------------------------------------------------
# JEPA encoders
# ------------------------------------------------------------
context_encoded = context_encoder(
    visible_tokens,
    visible_positions=visible_ids,
)
# [B, P_visible, D]

with no_grad_or_ema():
    target_encoded = target_encoder(
        target_tokens,
        target_positions=masked_ids,
    )
# [B, P_masked, D]

# ------------------------------------------------------------
# Predictor
# ------------------------------------------------------------
predicted_target = predictor(
    context_encoded=context_encoded,
    visible_positions=visible_ids,
    target_positions=masked_ids,
)
# [B, P_masked, D]

# ------------------------------------------------------------
# JEPA loss
# ------------------------------------------------------------
loss = mse(
    normalize(predicted_target),
    stopgrad(normalize(target_encoded)),
)

# ------------------------------------------------------------
# Final state embedding
# ------------------------------------------------------------
# During training, optional:
z_t = state_pool(context_encoded)
# [B, latent_dim] = [32, 8]

# During embedding export, better:
full_encoded = context_encoder(x_tokens, positions=all_patch_ids)
z_t = state_pool(full_encoded)
# [B, 8]
```

### 10.3 Encoder

Start simple:

- Temporal Transformer encoder, or
- Small 1D Conv + Transformer hybrid, or
- GRU/TCN baseline before Transformer.

Do not start with a large architecture. Financial data has limited independent regime samples.

Suggested first configuration:

```yaml
lookback_days: 252
latent_dim: 8
model_dim: 128
num_layers: 2
num_heads: 4
dropout: 0.1
feature_dropout: 0.1
```

Test latent dimensions:

```text
2, 4, 8, 16
```

If 16 works but 2/4/8 fail, check whether the model is memorizing rather than learning compact state.

### 10.4 Context encoder and target encoder

Use a target encoder with stop-gradient or EMA updates. The predictor should map context representations to target-block representations.

The target encoder should not receive data outside the training period for a fold.

### 10.5 Predictor

Start with a small MLP or shallow Transformer predictor.

Do not start with MoE.

Later optional MoE predictor/head:

```yaml
num_experts: 2-4
gating_input: z_t or context representation
load_balance_regularization: true
entropy_regularization: true
expert_dropout: true
```

MoE should be added only if the plain model passes representation diagnostics.

---

## 11. Masking Strategy

Masking is useful, but it does not automatically prevent leakage.

### 11.1 Mask types

Use multiple mask types:

1. Temporal block masks.
2. Feature-family masks.
3. Asset-group masks, later.
4. Random feature masks.
5. Regime-stress masks, such as masking volatility features and asking the model to infer state from other families.

### 11.2 Dynamic masks

Changing masks every epoch is acceptable and likely useful. It prevents the model from solving one static missing-block puzzle.

### 11.3 Safe masking rule

For representation at time `t`, the context must only include data from `[t - L + 1, t]`.

Target blocks may be inside that same past window for the first version.

Forward latent prediction can be tested later, but only inside the training fold. It must not train on dates after the evaluation date.

### 11.4 Leakage warning

Masking helps with inside-sample leakage. It does not solve train/test leakage.

Unsafe:

```text
Pretrain on 1990-2026, then evaluate 2015-2020.
```

Safe:

```text
For a 2015-2020 evaluation fold, pretrain only on data available before 2015 or within the allowed training window.
```

---

## 12. Loss Design

### 12.1 Base JEPA loss

Let `p_t` be the predicted target representation and `y_t` be the target encoder representation.

```math
L_{JEPA} = \| p_t - stopgrad(y_t) \|_2^2
```

Alternative:

```math
L_{JEPA} = 1 - cos(p_t, stopgrad(y_t))
```

Start with MSE on normalized embeddings.

### 12.2 Collapse prevention

Use collapse diagnostics first. Add regularizers only when needed.

Possible regularizers:

#### Variance regularization

Encourage each latent dimension to maintain nontrivial variance.

```math
L_{var} = \frac{1}{d} \sum_j max(0, \gamma - std(z_j))
```

#### Covariance regularization

Discourage all latent dimensions from encoding the same thing.

```math
L_{cov} = \sum_{i \ne j} Cov(z)_ {ij}^2
```

#### Redundancy reduction

Use Barlow Twins-style cross-correlation pressure if training paired views.

#### Feature-family dropout

Randomly remove whole feature groups so the model cannot rely only on VIX/volatility.

#### Multi-target block prediction

Predict multiple target blocks so the model cannot solve only one easy target.

### 12.3 Total pretraining loss

```math
L_{pretrain} = L_{JEPA} + \lambda_{var}L_{var} + \lambda_{cov}L_{cov}
```

Start small:

```yaml
lambda_var: 0.01
lambda_cov: 0.001
```

Tune only after diagnosing collapse.

### 12.4 Alignment loss

After pretraining, introduce slow-market alignment.

```math
L_{total} = L_{pretrain} + \lambda_{align}(epoch)L_{align}
```

Where `L_align` may include prediction of:

- Future volatility bucket.
- Future trend/chop bucket.
- Future drawdown bucket.
- Future dispersion bucket.
- Future breadth bucket.
- Future correlation bucket.
- Later: conditional signal-performance profile.

`lambda_align` should start at zero and increase slowly.

Example schedule:

```yaml
alignment_start_epoch: 50
lambda_align_initial: 0.0
lambda_align_final: 0.05
alignment_warmup_epochs: 50
```

Do not let alignment dominate the representation objective early.

### 12.5 Return-aware regularizer

A return-aware regularizer should not mean feeding future returns into the encoder. It means adding a small loss term after pretraining that encourages `z_t` to organize slow future market outcomes.

Safer targets than raw returns:

- Future volatility bucket.
- Future drawdown bucket.
- Future trend/chop label.
- Future breadth/dispersion regime.
- Future IC profile of simple signals.

Avoid starting with raw next-day return.

---

## 13. Training Protocol

### 13.1 Walk-forward training

Use rolling or expanding windows.

Example:

```text
Train: 1990-2005
Validation: 2006-2008
Test: 2009-2011

Train: 1990-2008
Validation: 2009-2011
Test: 2012-2014

Train: 1990-2011
Validation: 2012-2014
Test: 2015-2017
```

For each fold:

- Fit scalers only on training data.
- Pretrain encoder only on training data.
- Tune hyperparameters only using validation data.
- Report test results once.

### 13.2 Normalization

Never normalize using the full dataset.

Recommended:

- Rolling z-scores for features where realistic.
- Train-fold scalers for model features.
- Cross-sectional ranks for equity-panel features.
- Volatility-normalized returns.
- Winsorization fitted only on training data.

### 13.3 Macro release lag

Macro features should include realistic availability rules.

If point-in-time data is unavailable, use conservative lag assumptions or exclude the series from the first version.

### 13.4 Reproducibility

Log:

- Dataset version.
- Source manifests.
- Universe version.
- Feature config.
- Target config.
- Split config.
- Model config.
- Random seeds.
- Git commit hash.
- Training logs.
- Probe results.

---

## 14. Diagnostics

These diagnostics are required before any trading interpretation.

### 14.1 Latent dimension diversity

Question:

> Can each latent dimension explain something different?

Tests:

- Correlation matrix of latent dimensions.
- Linear probes per dimension.
- Mutual information estimates.
- Ablate dimensions one at a time.

### 14.2 Residual information after volatility removal

Question:

> Does the embedding still contain information after regressing out VIX or realized volatility?

Test:

1. Regress each latent dimension on VIX and realized volatility.
2. Use residual latent vectors.
3. Probe residual vectors for trend, dispersion, breadth, drawdown, and correlation targets.

If performance vanishes, `z_t` is probably just volatility.

### 14.3 Dispersion independent of volatility

Question:

> Does the embedding separate high-dispersion vs low-dispersion markets independent of volatility?

Test:

- Match dates by realized volatility bucket.
- Compare dispersion separability within each volatility bucket.

### 14.4 Trend/crash/rebound/chop separation

Question:

> Does the embedding separate trend, crash, rebound, and chop environments?

Define labels from future and historical path structure, then test whether clusters/neighborhoods align with these labels out-of-sample.

### 14.5 Linear probes

Question:

> Do linear probes from `z_t` predict multiple known state variables, or only one?

Targets:

- VIX/realized volatility.
- Credit spread change.
- Yield curve slope.
- Dispersion.
- Breadth.
- Drawdown state.
- Correlation regime.
- Trend/chop.

### 14.6 Walk-forward stability

Question:

> Does the embedding remain stable across walk-forward folds?

Tests:

- Procrustes alignment between fold embeddings.
- Similar probe coefficients across folds.
- Similar nearest-neighbor behavior across folds.
- Similar latent-state transition behavior.

### 14.7 VIX-transform test

Question:

> Is `z_t` just a fancy VIX transform?

Tests:

- Predict `z_t` from VIX and realized volatility.
- Compare probes using `z_t` vs VIX-only.
- Compare probes using residualized `z_t`.
- Compare nearest neighbors from `z_t` vs nearest neighbors from volatility features.

### 14.8 Novelty and density test

Low-dimensional embeddings can create false density.

Tests:

- Effective neighbor count in `z_t` space.
- Distance to training distribution.
- Local probe uncertainty.
- Performance degradation in sparse latent regions.

---

## 15. Evaluation Framework

### 15.1 Representation evaluation

Evaluate whether the latent representation captures market structure.

Metrics:

- Linear probe R² for continuous targets.
- Classification AUC / balanced accuracy for buckets.
- Calibration for probabilistic buckets.
- Stability across folds.
- Residual performance after volatility removal.

### 15.2 Slow target evaluation

Evaluate future targets:

- Future volatility bucket.
- Future trend/chop bucket.
- Future drawdown bucket.
- Future dispersion bucket.
- Future breadth bucket.
- Future correlation bucket.
- Future tail-risk bucket.

Compare against:

- VIX-only baseline.
- Realized-vol-only baseline.
- Hand state baseline.
- PCA baseline.
- Autoencoder baseline.
- Random projection baseline.

### 15.3 Conditional IC evaluation

Later, evaluate whether `z_t` improves conditional IC estimation.

Procedure:

1. Define simple signals: momentum, mean reversion, volatility, trend, sector-relative strength.
2. Compute forward cross-sectional IC out-of-sample.
3. Estimate conditional IC using hand state `X_t`.
4. Estimate conditional IC using learned state `z_t`.
5. Estimate conditional IC using combined state `[X_t, z_t]`.
6. Compare calibration and realized future IC.

Success requires:

- Better out-of-sample IC calibration.
- Better separation of high-IC and low-IC states.
- No collapse into volatility-only explanation.
- Stable behavior across folds.

### 15.4 Backtest evaluation

Backtesting is last, not first.

Do not judge the project by early PnL. Early PnL is too easy to overfit.

If a backtest is eventually used, require:

- Walk-forward training.
- Transaction costs.
- Slippage assumptions.
- Liquidity filters.
- Universe-bias labeling.
- No parameter tuning on test periods.
- Comparison against hand-state rules.

---

## 16. MoE Plan

MoE is optional and late-stage.

### 16.1 Why not start with MoE

MoE can easily memorize historical episodes. It may route crisis periods, low-vol periods, inflation periods, and rebound periods into separate experts in a way that looks good in-sample but fails out-of-sample.

### 16.2 Safer MoE use

If used, put MoE in the decoder or downstream signal head first, not in the main encoder.

Possible uses:

- Volatility expert.
- Trend expert.
- Dispersion expert.
- Drawdown expert.
- Conditional signal-performance expert.

### 16.3 MoE continuation criteria

Only keep MoE if:

- Experts specialize in interpretable and stable ways.
- Routing is stable across folds.
- MoE improves validation and test probes.
- MoE does not destroy latent-state interpretability.
- MoE beats a same-parameter non-MoE baseline.

---

## 17. Final System Architecture

### 17.1 Repository structure

```text
fi-jepa/
  README.md
  configs/
    data.yaml
    universe.yaml
    features.yaml
    targets.yaml
    model.yaml
    train.yaml
    splits.yaml
  data/
    raw/
    interim/
    processed/
    manifests/
    quality_reports/
  src/
    data/
      stooq_loader.py
      fred_loader.py
      french_loader.py
      calendar.py
      symbol_manifest.py
    features/
      market_features.py
      cross_sectional_features.py
      macro_features.py
      normalization.py
    targets/
      volatility.py
      trend.py
      drawdown.py
      dispersion.py
      breadth.py
      correlation.py
      tail_risk.py
    models/
      tokenizer.py
      encoder.py
      target_encoder.py
      predictor.py
      fi_jepa.py
      heads.py
    losses/
      jepa_loss.py
      vicreg_loss.py
      alignment_loss.py
    training/
      pretrain.py
      train_heads.py
      walk_forward.py
      checkpoints.py
    evaluation/
      probes.py
      diagnostics.py
      baselines.py
      conditional_ic.py
      reports.py
    utils/
      config.py
      logging.py
      seeds.py
  notebooks/
    00_data_exploration.ipynb
    01_feature_diagnostics.ipynb
    02_pretraining_sanity.ipynb
    03_latent_space_diagnostics.ipynb
    04_probe_results.ipynb
  artifacts/
    models/
    embeddings/
    reports/
```

### 17.2 Core artifacts

Each run should produce:

```text
model_checkpoint.pt
encoder_checkpoint.pt
config_resolved.yaml
scaler.pkl
feature_manifest.parquet
target_manifest.parquet
split_manifest.json
z_embeddings.parquet
probe_results.json
diagnostic_report.md
```

### 17.3 Model output

The core output is:

```text
date
z_1
z_2
...
z_d
fold_id
model_version
dataset_version
```

Optional outputs:

```text
volatility_bucket_probability
trend_bucket_probability
drawdown_bucket_probability
dispersion_bucket_probability
breadth_bucket_probability
correlation_bucket_probability
```

---

## 18. Experiment Roadmap

### Milestone 0 — Dataset pipeline sanity

Deliverables:

- Market-wide dataset.
- Feature table.
- Target table.
- Walk-forward splits.
- Leakage checks.

Pass criteria:

- No future data in features.
- Scalers fit only on train folds.
- Feature and target dates align correctly.
- Dataset can be regenerated from config.

### Milestone 1 — Tiny FI-JEPA smoke test

Deliverables:

- Small encoder.
- JEPA loss training loop.
- Dynamic masking.
- Embedding export.

Pass criteria:

- Loss decreases.
- Embeddings are not constant.
- Latent dimensions have nonzero variance.
- No obvious leakage.

### Milestone 2 — Collapse and volatility tests

Deliverables:

- Latent diagnostics.
- VIX-only comparison.
- Residualized embedding probes.

Pass criteria:

- `z_t` is not fully explained by VIX/realized volatility.
- At least some non-volatility target remains predictable from residualized `z_t`.

### Milestone 3 — Frozen probes

Deliverables:

- Linear probes.
- Logistic probes.
- Baseline comparisons.

Pass criteria:

- FI-JEPA beats simple baselines on at least some slow targets across walk-forward folds.
- Results are stable enough to justify further work.

### Milestone 4 — Alignment experiment

Deliverables:

- Small alignment heads.
- Alignment loss schedule.
- Comparison between frozen and lightly fine-tuned encoder.

Pass criteria:

- Alignment improves slow-target prediction without collapsing representation diversity.
- Alignment does not simply turn `z_t` into a volatility proxy.

### Milestone 5 — Conditional IC integration

Deliverables:

- Simple signal library.
- Conditional IC estimator using hand state.
- Conditional IC estimator using learned state.
- Conditional IC estimator using combined state.

Pass criteria:

- Learned state improves out-of-sample conditional IC calibration.
- High-confidence favorable states produce better realized IC than low-confidence states.

### Milestone 6 — Optional MoE heads

Deliverables:

- Small MoE decoder/head.
- Routing diagnostics.
- Expert specialization report.

Pass criteria:

- MoE beats non-MoE baseline.
- Routing is stable and interpretable.
- No obvious fold-specific memorization.

---

## 19. Example Configuration

```yaml
project:
  name: fi_jepa
  mode: market_state_representation

data:
  dataset_version: v0_market_wide_free
  start_date: 1990-01-01
  end_date: 2026-01-01
  sources:
    - stooq
    - fred
    - french
  survivorship_status: market_wide_low_single_stock_dependence

features:
  lookback_days: 252
  families:
    - market_returns
    - volatility
    - trend
    - breadth
    - dispersion
    - correlation
    - credit_rates_macro
    - liquidity_volume
  calendar_features: none
  normalization:
    method: train_fold_standardize
    winsorize: true
    winsorize_limits: [0.01, 0.99]

masking:
  temporal_block_mask: true
  feature_family_mask: true
  random_feature_mask: true
  mask_ratio: 0.35
  dynamic_masks_per_epoch: true

model:
  encoder_type: temporal_transformer
  latent_dim: 8
  model_dim: 128
  num_layers: 2
  num_heads: 4
  dropout: 0.1
  target_encoder: ema
  predictor_type: mlp

loss:
  jepa_loss: mse_normalized
  lambda_var: 0.01
  lambda_cov: 0.001
  alignment:
    enabled: false
    start_epoch: 50
    lambda_final: 0.05

targets:
  horizons: [21, 63, 126]
  probe_targets:
    - future_realized_vol_bucket
    - future_trend_bucket
    - future_max_drawdown_bucket
    - future_dispersion_bucket
    - future_breadth_bucket
    - future_average_correlation_bucket
    - future_tail_risk_bucket

splits:
  type: walk_forward_expanding
  validation_years: 3
  test_years: 3
  purge_days: 126

evaluation:
  baselines:
    - vix_only
    - realized_vol_only
    - hand_state
    - pca
    - autoencoder
    - random_projection
  diagnostics:
    - latent_dimension_diversity
    - volatility_residual_probe
    - dispersion_independent_of_volatility
    - trend_crash_rebound_chop_separation
    - linear_probe_multi_target
    - fold_stability
    - vix_transform_test
```

---

## 20. Main Caveats

### 20.1 JEPA does not automatically solve non-stationarity

The model may learn stable structure across known regimes, but it cannot guarantee generalization to a new causal market structure.

### 20.2 Masking does not automatically prevent leakage

Masking only hides parts of a sample. It does not prevent the model from learning future distributional information if pretraining uses dates beyond the evaluation period.

### 20.3 No future returns as inputs does not mean no return information

Past returns are valid market data. The forbidden object is future returns or future outcomes entering the encoder input for time `t`.

### 20.4 Anonymization does not fix missing companies

Removing ticker identity helps prevent memorization. It does not correct survivorship bias.

### 20.5 Low-dimensional embedding does not remove uncertainty

A compact latent state may help density, but uncertainty must still be measured locally.

### 20.6 Alignment can destroy representation quality

If the return/IC-aware regularizer becomes too strong, the encoder may become a noisy supervised predictor rather than a general market-state encoder.

### 20.7 MoE can overfit historical episodes

MoE should be added late and tested against non-MoE baselines.

### 20.8 Backtest performance is not the first success criterion

The first success criterion is robust representation quality and out-of-sample probe performance.

---

## 21. Decision Gates

### Continue from dataset to model if:

- The dataset passes leakage checks.
- The feature table has enough history.
- Targets are aligned correctly.
- Survivorship limitations are documented.

### Continue from pretraining to probes if:

- Embeddings do not collapse.
- Latent dimensions have nontrivial variance.
- Embeddings are stable across seeds.

### Continue from probes to alignment if:

- `z_t` beats simple baselines on at least some slow targets.
- `z_t` is not just a volatility transform.
- Probe performance is stable across folds.

### Continue from alignment to conditional IC if:

- Alignment improves target prediction without destroying latent diversity.
- Walk-forward validation remains clean.

### Continue to MoE only if:

- Small non-MoE heads are not expressive enough.
- There is evidence of stable sub-state specialization.
- MoE improves out-of-sample performance after parameter-count controls.

---

## 22. Recommended First Build

Build the smallest useful version:

```text
Dataset: market-wide daily data
Lookback: 252 trading days
Features: market returns, volatility, trend, credit/rates, factor/sector returns
Latent dimension: 8
Model: 2-layer temporal Transformer
Loss: JEPA + small variance/covariance regularization
Targets: future volatility, trend, drawdown, dispersion, breadth, correlation
Evaluation: frozen probes and VIX residual tests
No MoE
No trading backtest
No return-aware alignment yet
```

This version answers the first real question:

> Does a JEPA-style financial encoder learn a nontrivial, stable market-state representation?

Only after that should the project move toward tradability alignment.

---

## 23. References and Background Sources

This planning document is based on the project discussion, the Conditional IC Surface framework from the uploaded planning PDF, and the following background sources:

- I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture — https://arxiv.org/abs/2301.08243
- V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video — https://arxiv.org/abs/2404.08471
- VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning — https://arxiv.org/abs/2105.04906
- Barlow Twins: Self-Supervised Learning via Redundancy Reduction — https://arxiv.org/abs/2103.03230
- Stooq historical data download — https://stooq.com/db/h/
- FRED API documentation — https://fred.stlouisfed.org/docs/api/fred/
- Kenneth French Data Library — https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- EODHD delisted company data documentation — https://eodhd.com/financial-apis/delisted-stock-companies-data

---

## 24. Final Research Statement

FI-JEPA should be treated as a market-state representation project first and a trading project second.

The project succeeds early if it learns a compact, stable, nontrivial representation of market state that cannot be reduced to VIX or realized volatility. It succeeds later if that representation improves slow market outcome prediction and conditional signal-performance estimation out-of-sample.

The strongest version of the system is not a black-box return predictor. It is a disciplined representation-learning pipeline that learns market state from past-only data, diagnoses what the embedding contains, and only then aligns the representation toward tradable structure.
