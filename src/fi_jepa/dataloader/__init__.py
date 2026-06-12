"""Frozen-panel reconstruction, masking, sampling, and PyTorch loading API."""

from fi_jepa.dataloader.batch_assembler import FIJepaBatchAssembler, collate_fi_jepa_batch
from fi_jepa.dataloader.config import AssemblyMode, FIJepaDataConfig
from fi_jepa.dataloader.dataloader import (
    build_fi_jepa_embedding_dataloader,
    build_fi_jepa_dataloader,
)
from fi_jepa.dataloader.dataset import (
    AssetView,
    EmbeddingSplit,
    FIJepaEmbeddingDataset,
    FIJepaWindowDataset,
    fixed_k_asset_ids,
)
from fi_jepa.dataloader.masking import (
    compute_batched_patch_masks,
    compute_patch_masks,
    sample_jepa_target_mask,
)
from fi_jepa.dataloader.panel_store import FrozenPanelStore, Split
from fi_jepa.dataloader.request import RequestKind, ViewKind, WindowRequest

__all__ = [
    "AssemblyMode",
    "AssetView",
    "EmbeddingSplit",
    "FIJepaBatchAssembler",
    "FIJepaDataConfig",
    "FIJepaEmbeddingDataset",
    "FIJepaWindowDataset",
    "FrozenPanelStore",
    "RequestKind",
    "Split",
    "ViewKind",
    "WindowRequest",
    "build_fi_jepa_embedding_dataloader",
    "build_fi_jepa_dataloader",
    "collate_fi_jepa_batch",
    "compute_batched_patch_masks",
    "compute_patch_masks",
    "fixed_k_asset_ids",
    "sample_jepa_target_mask",
]
