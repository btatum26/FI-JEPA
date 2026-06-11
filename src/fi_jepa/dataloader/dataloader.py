from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fi_jepa.dataloader.config import FIJepaDataConfig
from fi_jepa.dataloader.dataset import AssetView, EmbeddingSplit, FIJepaEmbeddingDataset, FIJepaWindowDataset
from fi_jepa.dataloader.panel_store import FrozenPanelStore, Split


# ============================================================================
# BATCH COLLATION
# ============================================================================


def _pad_tensor(
    tensor: torch.Tensor,
    size: int,
    dimension: int,
    value: int | float,
) -> torch.Tensor:
    """Right-pad one tensor dimension to ``size`` using a typed fill value."""
    if tensor.shape[dimension] == size:
        return tensor
    shape = list(tensor.shape)
    shape[dimension] = size - tensor.shape[dimension]
    padding = torch.full(shape, value, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=dimension)


def collate_fi_jepa_batch(samples: list[dict[str, object]], *, patch_len: int) -> dict[str, object]:
    """Pad variable axes, stack samples, and expose zero-copy patch views.

    Asset panels and target lists vary by sample, while date-level streams have
    fixed shapes. After padding and stacking those variable axes, the function
    reshapes daily tensors into model-facing patch views without duplicating
    their storage.
    """
    if not samples:
        raise ValueError("Cannot collate an empty FI-JEPA sample list.")

    max_assets = max(int(sample["asset_ids"].shape[0]) for sample in samples)
    has_targets = "target_patch_ids" in samples[0]
    batch: dict[str, object] = {
        "sample_date": [str(sample["sample_date"]) for sample in samples],
        "split_label": [str(sample["split_label"]) for sample in samples],
    }
    for name in ("validation_window_name", "asset_view"):
        if name in samples[0]:
            batch[name] = [str(sample[name]) for sample in samples]

    # Only tensors containing the asset axis need panel-size padding. Date-level
    # streams and patch masks already have fixed shapes across samples.
    asset_dimensions = {
        "asset_x": 1,
        "asset_feature_mask": 1,
        "valid_asset_mask": 1,
        "asset_ids": 0,
        "asset_slot_mask": 0,
        "patch_asset_mask": 1,
    }
    asset_fill = {"asset_ids": -1}
    for name, dimension in asset_dimensions.items():
        batch[name] = torch.stack(
            [
                _pad_tensor(
                    sample[name],
                    max_assets,
                    dimension,
                    asset_fill.get(name, 0),
                )
                for sample in samples
            ]
        )

    # Target IDs use -1 padding so the derived mask is unambiguous.
    if has_targets:
        max_targets = max(int(sample["target_patch_ids"].shape[0]) for sample in samples)
        target_patch_ids = torch.stack(
            [_pad_tensor(sample["target_patch_ids"], max_targets, 0, -1) for sample in samples]
        )
        batch["target_patch_ids"] = target_patch_ids
        batch["target_patch_id_mask"] = target_patch_ids >= 0

    # Stack all remaining fixed-shape tensors after variable axes are handled.
    excluded = {
        *asset_dimensions,
        "target_patch_ids",
        "sample_date",
        "split_label",
        "validation_window_name",
        "asset_view",
    }
    for name in samples[0]:
        if name not in excluded:
            batch[name] = torch.stack([sample[name] for sample in samples])

    asset_x = batch["asset_x"]
    market_x = batch["market_x"]
    macro_x = batch["macro_x"]
    batch_size, lookback_days, n_assets, asset_dim = asset_x.shape
    n_patches = lookback_days // patch_len

    # [B, W, A, F_asset] -> [B, P, L, A, F_asset]. These are views over the
    # stacked daily tensors, not duplicate patch buffers.
    batch["asset_patches"] = asset_x.view(batch_size, n_patches, patch_len, n_assets, asset_dim)
    # [B, W, F_stream] -> [B, P, L, F_stream].
    batch["market_patches"] = market_x.view(batch_size, n_patches, patch_len, market_x.shape[-1])
    batch["macro_patches"] = macro_x.view(batch_size, n_patches, patch_len, macro_x.shape[-1])
    batch["asset_feature_mask_patched"] = batch["asset_feature_mask"].view(
        batch_size, n_patches, patch_len, n_assets, asset_dim
    )
    batch["market_feature_mask_patched"] = batch["market_feature_mask"].view(
        batch_size, n_patches, patch_len, market_x.shape[-1]
    )
    batch["macro_feature_mask_patched"] = batch["macro_feature_mask"].view(
        batch_size, n_patches, patch_len, macro_x.shape[-1]
    )
    batch["valid_asset_mask_patched"] = batch["valid_asset_mask"].view(
        batch_size, n_patches, patch_len, n_assets
    )
    for name in (
        "valid_date_mask",
        "holdout_date_mask",
        "padded_date_mask",
        "valid_market_date_mask",
        "valid_macro_date_mask",
    ):
        batch[f"{name}_patched"] = batch[name].view(batch_size, n_patches, patch_len)
    return batch


# ============================================================================
# DATALOADER CONSTRUCTION
# ============================================================================


def build_fi_jepa_dataloader(
    config: FIJepaDataConfig | Path | str,
    split: Split,
    *,
    store: FrozenPanelStore | None = None,
    view_index: int = 0,
    shuffle: bool | None = None,
) -> DataLoader:
    """Build a deterministic PyTorch loader backed by a shared panel store.

    A supplied store allows multiple split loaders to reuse the same dense
    arrays. Shuffle defaults to training-only, and the loader generator uses
    the data configuration seed for reproducible sample ordering.
    """
    if not isinstance(config, FIJepaDataConfig):
        config = FIJepaDataConfig.from_yaml(config)
    if store is None:
        store = FrozenPanelStore(config.artifact_path)
    elif store.artifact_path.resolve() != config.artifact_path.resolve():
        raise ValueError("The supplied store does not match config.artifact_path.")

    dataset = FIJepaWindowDataset(store, config, split, view_index=view_index)
    if shuffle is None:
        shuffle = split == "train"
    batch_size = config.batch_size if split == "train" else config.validation_batch_size

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last and split == "train",
        collate_fn=partial(collate_fi_jepa_batch, patch_len=config.patch_len),
        generator=generator,
        persistent_workers=False,
    )


def build_fi_jepa_embedding_dataloader(
    config: FIJepaDataConfig | Path | str,
    split: EmbeddingSplit,
    *,
    asset_view: AssetView,
    store: FrozenPanelStore | None = None,
    view_index: int = 0,
) -> DataLoader:
    """Build a deterministic unmasked loader for pooled-state evaluation."""
    if not isinstance(config, FIJepaDataConfig):
        config = FIJepaDataConfig.from_yaml(config)
    if store is None:
        store = FrozenPanelStore(config.artifact_path)
    elif store.artifact_path.resolve() != config.artifact_path.resolve():
        raise ValueError("The supplied store does not match config.artifact_path.")

    dataset = FIJepaEmbeddingDataset(
        store,
        config,
        split,
        asset_view=asset_view,
        view_index=view_index,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=partial(collate_fi_jepa_batch, patch_len=config.patch_len),
        generator=generator,
        persistent_workers=False,
    )
