from __future__ import annotations

from functools import lru_cache

import numpy as np

from fi_jepa.dataloader.validation import (
    validate_batched_patch_mask_inputs,
    validate_jepa_mask_inputs,
)


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
    batch_size, n_dates, n_assets = validate_batched_patch_mask_inputs(
        valid_assets, valid_dates, target_dates, patch_len
    )

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


def _sample_contiguous_target_ids(
    eligible: np.ndarray,
    rng: np.random.Generator,
    *,
    target_count: int,
    min_target_blocks: int,
    max_target_blocks: int,
) -> np.ndarray:
    """Sample an exact target count arranged into non-adjacent contiguous blocks.

    Dynamic-programming feasibility checks prevent random placement retries
    from failing on target-eligibility gaps. A one-patch gap is required
    between blocks so the returned block count has exact temporal semantics.
    """
    num_patches = int(eligible.size)
    padded_eligible = np.concatenate(([False], eligible, [False]))
    run_starts = np.flatnonzero(~padded_eligible[:-1] & padded_eligible[1:])
    run_ends = np.flatnonzero(padded_eligible[:-1] & ~padded_eligible[1:])

    # Most live samples contain one long eligible run. Sample directly inside
    # that run instead of building a per-sample feasibility table.
    fast_options = [
        (block_count, run_start, run_end)
        for block_count in range(min_target_blocks, max_target_blocks + 1)
        for run_start, run_end in zip(run_starts, run_ends, strict=True)
        if run_end - run_start >= target_count + block_count - 1
    ]
    if fast_options:
        block_count, run_start, run_end = fast_options[int(rng.integers(len(fast_options)))]
        if block_count == 1:
            block_lengths = np.asarray([target_count], dtype=np.int64)
        else:
            cut_points = np.sort(
                rng.choice(np.arange(1, target_count), size=block_count - 1, replace=False)
            )
            block_lengths = np.diff(np.concatenate(([0], cut_points, [target_count])))

        slack = int(run_end - run_start - target_count - block_count + 1)
        gap_extras = rng.multinomial(slack, np.full(block_count + 1, 1.0 / (block_count + 1)))
        gaps = gap_extras.astype(np.int64, copy=False)
        gaps[1:block_count] += 1

        target_mask = np.zeros(num_patches, dtype=bool)
        position = int(run_start + gaps[0])
        for block_index, block_length in enumerate(block_lengths):
            target_mask[position : position + block_length] = True
            position += int(block_length + gaps[block_index + 1])
        return np.flatnonzero(target_mask).astype(np.int64)

    @lru_cache(maxsize=None)
    def can_complete(position: int, remaining_targets: int, remaining_blocks: int) -> bool:
        """Return whether the suffix can satisfy the exact remaining target layout."""
        if remaining_targets == 0 or remaining_blocks == 0:
            return remaining_targets == 0 and remaining_blocks == 0
        if position >= num_patches or remaining_targets < remaining_blocks:
            return False
        if num_patches - position < remaining_targets + remaining_blocks - 1:
            return False
        if can_complete(position + 1, remaining_targets, remaining_blocks):
            return True

        max_block_length = min(
            remaining_targets - remaining_blocks + 1,
            num_patches - position,
        )
        for block_length in range(1, max_block_length + 1):
            if not eligible[position + block_length - 1]:
                break
            if can_complete(
                position + block_length + 1,
                remaining_targets - block_length,
                remaining_blocks - 1,
            ):
                return True
        return False

    feasible_block_counts = [
        block_count
        for block_count in range(min_target_blocks, max_target_blocks + 1)
        if can_complete(0, target_count, block_count)
    ]
    if not feasible_block_counts:
        raise ValueError(
            f"Sample cannot arrange {target_count} targets into "
            f"{min_target_blocks}..{max_target_blocks} contiguous blocks; "
            f"eligible_patch_ids={np.flatnonzero(eligible).tolist()}."
        )

    remaining_targets = target_count
    remaining_blocks = int(rng.choice(feasible_block_counts))
    position = 0
    target_mask = np.zeros(num_patches, dtype=bool)
    while remaining_targets:
        transitions: list[tuple[str, int]] = []
        if can_complete(position + 1, remaining_targets, remaining_blocks):
            transitions.append(("skip", 1))

        max_block_length = min(
            remaining_targets - remaining_blocks + 1,
            num_patches - position,
        )
        for block_length in range(1, max_block_length + 1):
            if not eligible[position + block_length - 1]:
                break
            if can_complete(
                position + block_length + 1,
                remaining_targets - block_length,
                remaining_blocks - 1,
            ):
                transitions.append(("block", block_length))

        transition, length = transitions[int(rng.integers(len(transitions)))]
        if transition == "skip":
            position += 1
            continue
        target_mask[position : position + length] = True
        position += length + 1
        remaining_targets -= length
        remaining_blocks -= 1

    return np.flatnonzero(target_mask).astype(np.int64)


def sample_jepa_target_mask(
    patch_target_eligible: np.ndarray,
    patch_context_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    mask_ratio: float,
    min_masked_patches: int,
    max_masked_patches: int,
    min_target_blocks: int,
    max_target_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample target blocks and remove them from visible context.

    Each target-eligible patch has ``mask_ratio`` probability of contributing
    to the sampled target count. Targets are arranged into a random feasible
    number of contiguous blocks within the configured block bounds. If that is
    impossible, sampling falls back first to one contiguous block and finally
    to unconstrained random eligible patches. Target count bounds and the need
    to retain visible context remain hard constraints.
    """
    eligible = np.asarray(patch_target_eligible, dtype=bool)
    context = np.asarray(patch_context_mask, dtype=bool)
    validate_jepa_mask_inputs(eligible, context)

    eligible_ids = np.flatnonzero(eligible)
    max_target_count = min(eligible_ids.size, int(context.sum()) - 1, max_masked_patches)
    if max_target_count < min_masked_patches:
        raise ValueError("Sample cannot provide the minimum targets while retaining visible context.")
    # Draw the count as well as the target identities so repeated samples do
    # not always expose the model to one fixed target/context partition size.
    target_count = int(rng.binomial(eligible_ids.size, mask_ratio))
    target_count = min(max(target_count, min_masked_patches), max_target_count)
    try:
        target_ids = _sample_contiguous_target_ids(
            eligible,
            rng,
            target_count=target_count,
            min_target_blocks=min_target_blocks,
            max_target_blocks=max_target_blocks,
        )
    except ValueError:
        try:
            target_ids = _sample_contiguous_target_ids(
                eligible,
                rng,
                target_count=target_count,
                min_target_blocks=1,
                max_target_blocks=1,
            )
        except ValueError:
            target_ids = np.sort(rng.choice(eligible_ids, size=target_count, replace=False)).astype(np.int64)
    target_mask = np.zeros_like(eligible)
    target_mask[target_ids] = True
    return target_mask, context & ~target_mask, target_ids.astype(np.int64)
