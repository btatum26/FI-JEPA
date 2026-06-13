from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


# ============================================================================
# DATALOADER CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class FIJepaDataConfig:
    """Configure dense-panel gathering, asset views, masking, and batching."""

    artifact_path: Path
    cache_root: Path = Path("data/cache/dense_panel")
    lookback_days: int = 252
    patch_len: int = 21
    train_k_assets: int = 256
    fixed_k_assets: int = 256
    mask_ratio: float = 0.35
    min_masked_patches: int = 3
    max_masked_patches: int = 5
    min_target_blocks: int = 2
    max_target_blocks: int = 4
    min_valid_days_per_asset_patch: int = 10
    min_valid_dates_in_patch: int = 10
    min_valid_asset_fraction: float = 0.25
    batch_size: int = 32
    validation_batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    seed: int = 1337

    def __post_init__(self) -> None:
        """Reject invalid dimensions, target bounds, and loader settings."""
        if self.lookback_days <= 0 or self.patch_len <= 0:
            raise ValueError("lookback_days and patch_len must be positive.")
        if self.lookback_days % self.patch_len:
            raise ValueError("lookback_days must be divisible by patch_len.")
        if not 0.0 < self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in (0, 1].")
        if not 0.0 <= self.min_valid_asset_fraction <= 1.0:
            raise ValueError("min_valid_asset_fraction must be in [0, 1].")
        if not 1 <= self.min_masked_patches <= self.max_masked_patches:
            raise ValueError("Masked patch bounds are invalid.")
        if self.max_masked_patches > self.num_patches:
            raise ValueError("max_masked_patches exceeds the number of patches.")
        if not 1 <= self.min_target_blocks <= self.max_target_blocks:
            raise ValueError("Target block bounds are invalid.")
        if self.min_target_blocks > self.min_masked_patches:
            raise ValueError("min_target_blocks cannot exceed min_masked_patches.")
        if self.max_target_blocks > self.max_masked_patches:
            raise ValueError("max_target_blocks cannot exceed max_masked_patches.")
        if not 1 <= self.min_valid_days_per_asset_patch <= self.patch_len:
            raise ValueError("min_valid_days_per_asset_patch must be within one patch.")
        if not 1 <= self.min_valid_dates_in_patch <= self.patch_len:
            raise ValueError("min_valid_dates_in_patch must be within one patch.")
        if self.train_k_assets <= 0 or self.fixed_k_assets <= 0:
            raise ValueError("Asset view sizes must be positive.")
        if self.batch_size <= 0 or self.validation_batch_size <= 0:
            raise ValueError("Batch sizes must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

    @property
    def num_patches(self) -> int:
        """Return the fixed number of temporal patches in each sample."""
        return self.lookback_days // self.patch_len

    @classmethod
    def from_yaml(cls, path: Path | str) -> FIJepaDataConfig:
        """Load and normalize a dataloader YAML file."""
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Dataloader configuration must be a YAML mapping.")
        values["artifact_path"] = Path(values["artifact_path"])
        values["cache_root"] = Path(values.get("cache_root", "data/cache/dense_panel"))
        return cls(**values)
