from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from fi_jepa.dataloader.batch_assembler import DensePanelBatchAssembler
from fi_jepa.dataloader.config import FIJepaDataConfig
from fi_jepa.dataloader.dataset import DensePanelWindowRequestDataset
from fi_jepa.dataloader.panel_store import DensePanelStore, Split

AssetView = Literal["all_valid", "fixed_k"]


# ============================================================================
# DATALOADER CONSTRUCTION
# ============================================================================


def build_fi_jepa_dataloader(
    config: FIJepaDataConfig | Path | str,
    split: Split,
    *,
    store: DensePanelStore | None = None,
    view_index: int = 0,
    shuffle: bool | None = None,
) -> DataLoader:
    """Build a JEPA loader over the split-specific dense panel cache."""
    config, store = _resolve_config_and_store(config, store)
    view_kind = "random_k" if split == "train" else "all_valid"
    dataset = DensePanelWindowRequestDataset(
        store,
        config,
        split,
        request_kind="jepa",
        view_kind=view_kind,
        view_index=view_index,
    )
    if shuffle is None:
        shuffle = split == "train"
    batch_size = config.batch_size if split == "train" else config.validation_batch_size
    return _build_loader(
        dataset,
        store,
        config,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=config.drop_last and split == "train",
    )


def build_fi_jepa_embedding_dataloader(
    config: FIJepaDataConfig | Path | str,
    split: Split,
    *,
    asset_view: AssetView,
    store: DensePanelStore | None = None,
    view_index: int = 0,
) -> DataLoader:
    """Build an unmasked embedding loader over the dense panel cache."""
    config, store = _resolve_config_and_store(config, store)
    dataset = DensePanelWindowRequestDataset(
        store,
        config,
        split,
        request_kind="embedding",
        view_kind=asset_view,
        view_index=view_index,
    )
    return _build_loader(
        dataset,
        store,
        config,
        batch_size=config.validation_batch_size,
        shuffle=False,
        drop_last=False,
    )


def _resolve_config_and_store(
    config: FIJepaDataConfig | Path | str,
    store: DensePanelStore | None,
) -> tuple[FIJepaDataConfig, DensePanelStore]:
    """Normalize config and ensure one matching parent-built store."""
    if not isinstance(config, FIJepaDataConfig):
        config = FIJepaDataConfig.from_yaml(config)
    if store is None:
        store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    elif store.artifact_path != config.artifact_path.resolve():
        raise ValueError("The supplied store does not match config.artifact_path.")
    return config, store


def _build_loader(
    dataset: DensePanelWindowRequestDataset,
    store: DensePanelStore,
    config: FIJepaDataConfig,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """Construct a worker-safe loader after the parent has opened the cache."""
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=drop_last,
        collate_fn=DensePanelBatchAssembler(store, config),
        generator=generator,
        persistent_workers=False,
    )
