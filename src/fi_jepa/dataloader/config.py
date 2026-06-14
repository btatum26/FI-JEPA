from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from fi_jepa.dataloader.validation import validate_data_config, validate_data_yaml


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
    feature_dropout_rate: float = 0.0
    batch_size: int = 32
    validation_batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    seed: int = 1337

    def __post_init__(self) -> None:
        """Reject invalid dimensions, target bounds, and loader settings."""
        validate_data_config(
            lookback_days=self.lookback_days,
            patch_len=self.patch_len,
            mask_ratio=self.mask_ratio,
            min_masked_patches=self.min_masked_patches,
            max_masked_patches=self.max_masked_patches,
            min_target_blocks=self.min_target_blocks,
            max_target_blocks=self.max_target_blocks,
            min_valid_days_per_asset_patch=self.min_valid_days_per_asset_patch,
            min_valid_dates_in_patch=self.min_valid_dates_in_patch,
            min_valid_asset_fraction=self.min_valid_asset_fraction,
            feature_dropout_rate=self.feature_dropout_rate,
            train_k_assets=self.train_k_assets,
            fixed_k_assets=self.fixed_k_assets,
            batch_size=self.batch_size,
            validation_batch_size=self.validation_batch_size,
            num_workers=self.num_workers,
        )

    @property
    def num_patches(self) -> int:
        """Return the fixed number of temporal patches in each sample."""
        return self.lookback_days // self.patch_len

    @classmethod
    def from_yaml(cls, path: Path | str) -> FIJepaDataConfig:
        """Load and normalize a dataloader YAML file."""
        values = validate_data_yaml(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
        values["artifact_path"] = Path(values["artifact_path"])
        values["cache_root"] = Path(values.get("cache_root", "data/cache/dense_panel"))
        return cls(**values)
