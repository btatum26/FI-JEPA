from __future__ import annotations

import numpy as np


# ============================================================================
# PATCH VALIDITY MASKING
# ============================================================================


def compute_batched_patch_masks(
    valid_asset_mask: np.ndarray,
    valid_date_mask: np.ndarray,
    holdout_date_mask: np.ndarray,
    padded_date_mask: np.ndarray,
    asset_slot_mask: np.ndarray,
    *,
    patch_len: int,
    min_valid_days_per_asset_patch: int,
    min_valid_dates_in_patch: int,
    min_valid_asset_fraction: float,
    allow_holdout_targets: bool,
) -> dict[str, np.ndarray]:
    """Build context and target-eligibility masks for a batch of windows.

    Daily masks are reshaped into chronological patches. Assets qualify for
    panel pooling only after meeting the per-patch valid-day threshold, and
    padded asset slots are excluded from coverage denominators. Context accepts
    any patch with a valid date; targets additionally require configured date
    and asset coverage, no padded dates, and split-relative holdout permission.

    Returns:
        Per-patch asset validity ``[B, P, A]`` plus context, holdout, target
        eligibility, and asset-coverage arrays shaped ``[B, P]``.
    """
    # Normalize caller-provided array-likes once so later boolean operations
    # cannot inherit integer or object semantics.
    valid_assets = np.asarray(valid_asset_mask, dtype=bool)
    valid_dates = np.asarray(valid_date_mask, dtype=bool)
    holdout_dates = np.asarray(holdout_date_mask, dtype=bool)
    padded_dates = np.asarray(padded_date_mask, dtype=bool)
    asset_slots = np.asarray(asset_slot_mask, dtype=bool)

    if valid_assets.ndim != 3:
        raise ValueError("valid_asset_mask must have shape [batch, dates, assets].")
    batch_size, n_dates, n_assets = valid_assets.shape
    if n_dates % patch_len:
        raise ValueError("Window length must be divisible by patch_len.")
    if asset_slots.shape != (batch_size, n_assets):
        raise ValueError("asset_slot_mask must have shape [batch, assets].")
    for name, mask in (
        ("valid_date_mask", valid_dates),
        ("holdout_date_mask", holdout_dates),
        ("padded_date_mask", padded_dates),
    ):
        if mask.shape != (batch_size, n_dates):
            raise ValueError(f"{name} must have shape [batch, dates].")

    # [B, W, A] -> [B, P, L, A] and [B, W] -> [B, P, L]. Reshaping preserves
    # chronological order because windows are contiguous oldest-to-newest arrays.
    n_patches = n_dates // patch_len
    valid_assets_patched = valid_assets.reshape(batch_size, n_patches, patch_len, n_assets)
    valid_dates_patched = valid_dates.reshape(batch_size, n_patches, patch_len)
    holdout_patched = holdout_dates.reshape(batch_size, n_patches, patch_len)
    padded_patched = padded_dates.reshape(batch_size, n_patches, patch_len)

    # An asset contributes to panel pooling only when it has enough valid days
    # inside the patch. Padded asset slots never enter the coverage denominator.
    patch_asset_mask = valid_assets_patched.sum(axis=2) >= min_valid_days_per_asset_patch
    patch_asset_mask &= asset_slots[:, None, :]

    # Use one as the empty-panel denominator so a fully padded panel produces
    # zero coverage rather than a divide-by-zero result.
    valid_slot_count = np.maximum(asset_slots.sum(axis=1, keepdims=True), 1)
    valid_asset_fraction = patch_asset_mask.sum(axis=2) / valid_slot_count
    valid_date_count = valid_dates_patched.sum(axis=2)
    patch_has_holdout = holdout_patched.any(axis=2)
    patch_has_padding = padded_patched.any(axis=2)
    patch_context_mask = valid_dates_patched.any(axis=2)

    # Context can be incomplete. JEPA targets must satisfy stricter coverage
    # rules and cannot cross an early-history padding boundary.
    patch_target_eligible = (
        (valid_date_count >= min_valid_dates_in_patch)
        & (valid_asset_fraction >= min_valid_asset_fraction)
        & (~patch_has_padding)
    )
    if not allow_holdout_targets:
        patch_target_eligible &= ~patch_has_holdout

    return {
        "patch_asset_mask": patch_asset_mask,
        "patch_context_mask": patch_context_mask,
        "patch_has_holdout": patch_has_holdout,
        "patch_target_eligible": patch_target_eligible,
        "valid_asset_fraction": valid_asset_fraction.astype(np.float32),
    }


def compute_patch_masks(
    valid_asset_mask: np.ndarray,
    valid_date_mask: np.ndarray,
    holdout_date_mask: np.ndarray,
    padded_date_mask: np.ndarray,
    asset_slot_mask: np.ndarray,
    *,
    patch_len: int,
    min_valid_days_per_asset_patch: int,
    min_valid_dates_in_patch: int,
    min_valid_asset_fraction: float,
    allow_holdout_targets: bool,
) -> dict[str, np.ndarray]:
    """Build patch masks for one window through the batch-first implementation."""
    batched = compute_batched_patch_masks(
        np.asarray(valid_asset_mask)[None, ...],
        np.asarray(valid_date_mask)[None, ...],
        np.asarray(holdout_date_mask)[None, ...],
        np.asarray(padded_date_mask)[None, ...],
        np.asarray(asset_slot_mask)[None, ...],
        patch_len=patch_len,
        min_valid_days_per_asset_patch=min_valid_days_per_asset_patch,
        min_valid_dates_in_patch=min_valid_dates_in_patch,
        min_valid_asset_fraction=min_valid_asset_fraction,
        allow_holdout_targets=allow_holdout_targets,
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
    """Sample independent temporal targets and remove them from visible context.

    The requested mask ratio is rounded half-up, clamped to configured bounds,
    and then capped by the number of eligible patches. Returned target IDs are
    sorted chronologically even though they are sampled independently.

    Returns:
        Target mask ``[P]``, visible context mask ``[P]``, and sorted target IDs
        ``[T]``.
    """
    eligible = np.asarray(patch_target_eligible, dtype=bool)
    context = np.asarray(patch_context_mask, dtype=bool)
    if eligible.ndim != 1 or context.shape != eligible.shape:
        raise ValueError("Patch target and context masks must be one-dimensional and aligned.")
    if np.any(eligible & ~context):
        raise ValueError("Target-eligible patches must also be context-valid.")

    eligible_ids = np.flatnonzero(eligible)
    max_target_count = min(eligible_ids.size, int(context.sum()) - 1, max_masked_patches)
    if max_target_count < min_masked_patches:
        raise ValueError(
            "Sample cannot provide the minimum targets while retaining visible context."
        )

    # Use conventional half-up rounding rather than Python's banker rounding,
    # then enforce the configured target-count bounds.
    target_count = int(np.floor(eligible_ids.size * mask_ratio + 0.5))
    target_count = min(max(target_count, min_masked_patches), max_target_count)

    # Targets are sampled independently rather than as a contiguous block.
    target_ids = np.sort(rng.choice(eligible_ids, size=target_count, replace=False))
    target_mask = np.zeros_like(eligible)
    target_mask[target_ids] = True

    # A target remains context-valid metadata but is hidden from the online encoder.
    return target_mask, context & ~target_mask, target_ids.astype(np.int64)
