from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fi_jepa.dataloader.batch_assembler import FIJepaBatchAssembler, collate_fi_jepa_batch
from fi_jepa.dataloader.config import FIJepaDataConfig
from fi_jepa.dataloader.dataset import (
    AssetView,
    EmbeddingSplit,
    FIJepaEmbeddingDataset,
    FIJepaWindowDataset,
)
from fi_jepa.dataloader.panel_store import FrozenPanelStore, Split

__all__ = [
    "build_fi_jepa_dataloader",
    "build_fi_jepa_embedding_dataloader",
    "collate_fi_jepa_batch",
]


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
    """Build a deterministic request loader with batch-first materialization.

    A supplied store allows multiple split loaders to reuse the same dense
    arrays. Shuffle defaults to training-only, and the assembler follows the
    configured fast or compatibility assembly mode.
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
        collate_fn=FIJepaBatchAssembler(store, config),
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
    """Build a deterministic unmasked request loader for pooled-state evaluation."""
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
        collate_fn=FIJepaBatchAssembler(store, config),
        generator=generator,
        persistent_workers=False,
    )
