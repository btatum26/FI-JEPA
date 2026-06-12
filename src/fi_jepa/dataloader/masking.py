from __future__ import annotations

import numpy as np


# ============================================================================
# PATCH VALIDITY MASKING
# ============================================================================


def compute_batched_patch_masks(
    valid_asset_mask: np.ndarray,
    valid_date_mask: np.ndarray,
    target_date_mask: np.ndarray,
    *,
    patch_len: int,
    min_valid_days_per_asset_patch: int,
    min_valid_dates_in_patch: int,
    min_valid_asset_fraction: float,
) -> dict[str, np.ndarray]:
    """Aggregate stored daily validity into model-facing patch masks.

    Args:
        valid_asset_mask: Stored split validity shaped ``[B, W, A]``.
        valid_date_mask: Whether any split stream is usable, shaped ``[B, W]``.
        target_date_mask: Whether each date may participate in a JEPA target,
            shaped ``[B, W]``. Validation passes an all-true mask.

    Returns:
        Asset pooling validity ``[B, P, A]`` and context/target validity
        ``[B, P]``. No feature masks are inferred or modified here.
    """
    valid_assets = np.asarray(valid_asset_mask, dtype=bool)
    valid_dates = np.asarray(valid_date_mask, dtype=bool)
    target_dates = np.asarray(target_date_mask, dtype=bool)
    if valid_assets.ndim != 3:
        raise ValueError("valid_asset_mask must have shape [batch, dates, assets].")
    batch_size, n_dates, n_assets = valid_assets.shape
    if n_dates % patch_len:
        raise ValueError("Window length must be divisible by patch_len.")
    for name, mask in (("valid_date_mask", valid_dates), ("target_date_mask", target_dates)):
        if mask.shape != (batch_size, n_dates):
            raise ValueError(f"{name} must have shape [batch, dates].")

    n_patches = n_dates // patch_len
    asset_patches = valid_assets.reshape(batch_size, n_patches, patch_len, n_assets)
    date_patches = valid_dates.reshape(batch_size, n_patches, patch_len)
    target_date_patches = target_dates.reshape(batch_size, n_patches, patch_len)

    # [B, P, L, A] -> [B, P, A]. Every selected asset is a real global slot,
    # so the selected axis itself is the asset-coverage denominator.
    patch_asset_mask = asset_patches.sum(axis=2) >= min_valid_days_per_asset_patch
    valid_asset_fraction = patch_asset_mask.sum(axis=2) / n_assets
    patch_context_mask = date_patches.any(axis=2)
    patch_target_eligible = (
        (date_patches.sum(axis=2) >= min_valid_dates_in_patch)
        & (valid_asset_fraction >= min_valid_asset_fraction)
        & target_date_patches.all(axis=2)
    )
    return {
        "patch_asset_mask": patch_asset_mask,
        "patch_context_mask": patch_context_mask,
        "patch_target_eligible": patch_target_eligible,
        "valid_asset_fraction": valid_asset_fraction.astype(np.float32),
    }


def compute_patch_masks(
    valid_asset_mask: np.ndarray,
    valid_date_mask: np.ndarray,
    target_date_mask: np.ndarray,
    *,
    patch_len: int,
    min_valid_days_per_asset_patch: int,
    min_valid_dates_in_patch: int,
    min_valid_asset_fraction: float,
) -> dict[str, np.ndarray]:
    """Aggregate one window through the batch-first patch-mask implementation."""
    batched = compute_batched_patch_masks(
        np.asarray(valid_asset_mask)[None, ...],
        np.asarray(valid_date_mask)[None, ...],
        np.asarray(target_date_mask)[None, ...],
        patch_len=patch_len,
        min_valid_days_per_asset_patch=min_valid_days_per_asset_patch,
        min_valid_dates_in_patch=min_valid_dates_in_patch,
        min_valid_asset_fraction=min_valid_asset_fraction,
    )
    return {name: value[0] for name, value in batched.items()}


# ============================================================================
# TEMPORAL JEPA MASKING
# ============================================================================


def sample_jepa_target_mask(
    patch_target_eligible: np.ndarray,
    patch_context_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    mask_ratio: float,
    min_masked_patches: int,
    max_masked_patches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample targets and remove them from visible online-encoder context."""
    eligible = np.asarray(patch_target_eligible, dtype=bool)
    context = np.asarray(patch_context_mask, dtype=bool)
    if eligible.ndim != 1 or context.shape != eligible.shape:
        raise ValueError("Patch target and context masks must be one-dimensional and aligned.")
    if np.any(eligible & ~context):
        raise ValueError("Target-eligible patches must also be context-valid.")

    eligible_ids = np.flatnonzero(eligible)
    max_target_count = min(eligible_ids.size, int(context.sum()) - 1, max_masked_patches)
    if max_target_count < min_masked_patches:
        raise ValueError("Sample cannot provide the minimum targets while retaining visible context.")
    target_count = int(np.floor(eligible_ids.size * mask_ratio + 0.5))
    target_count = min(max(target_count, min_masked_patches), max_target_count)
    target_ids = np.sort(rng.choice(eligible_ids, size=target_count, replace=False))
    target_mask = np.zeros_like(eligible)
    target_mask[target_ids] = True
    return target_mask, context & ~target_mask, target_ids.astype(np.int64)
