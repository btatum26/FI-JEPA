from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from fi_jepa.dataloader.config import FIJepaDataConfig
from fi_jepa.dataloader.masking import compute_patch_masks, sample_jepa_target_mask
from fi_jepa.dataloader.panel_store import FrozenPanelStore, Split

EmbeddingSplit = Literal["train", "validation"]
AssetView = Literal["all_valid", "fixed_k"]


# ============================================================================
# WINDOW DATASET AND LOADER
# ============================================================================


class FIJepaWindowDataset(Dataset[dict[str, object]]):
    """Reconstruct split-safe windows and sample reproducible JEPA views.

    Initialization removes sample dates that can never satisfy the configured
    minimum target count. Each item then chooses a split-appropriate asset
    panel, reconstructs values and validity masks from the shared store, and
    samples a temporal JEPA target/context partition.
    """

    def __init__(
        self,
        store: FrozenPanelStore,
        config: FIJepaDataConfig,
        split: Split,
        *,
        view_index: int = 0,
    ):
        if split not in {"train", "validation", "diagnostic"}:
            raise ValueError(f"Unsupported split: {split}")

        self.store = store
        self.config = config
        self.split = split
        self.view_index = view_index
        self.epoch = 0

        # The artifact and runtime must agree on window length because all
        # reconstructed masks and patch boundaries depend on it.
        artifact_lookback = (store.resolved_config.get("dates") or {}).get("lookback_days")
        if artifact_lookback is not None and int(artifact_lookback) != config.lookback_days:
            raise ValueError(
                f"Configured lookback_days={config.lookback_days} does not match "
                f"artifact lookback_days={artifact_lookback}."
            )

        nominal = store.sample_indices_for(split)
        retained: list[int] = []

        # Filter once using every endpoint-valid asset. Removed dates are
        # structurally unable to produce enough target patches under any view.
        for sample_date_idx in nominal:
            asset_ids = store.endpoint_asset_ids(int(sample_date_idx), split)
            if asset_ids.size == 0:
                continue
            masks = store.window_masks(int(sample_date_idx), asset_ids, split, config.lookback_days)
            patch_masks = self._patch_masks(masks)
            if (
                int(patch_masks["patch_target_eligible"].sum()) >= config.min_masked_patches
                and int(patch_masks["patch_context_mask"].sum()) > config.min_masked_patches
            ):
                retained.append(int(sample_date_idx))

        self.sample_date_indices = np.asarray(retained, dtype=np.int64)
        self.nominal_sample_count = int(len(nominal))
        self.dropped_sample_count = self.nominal_sample_count - len(retained)

    def __len__(self) -> int:
        """Return the number of structurally valid sample dates."""
        return len(self.sample_date_indices)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for reproducible epoch-varying training views."""
        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        self.epoch = epoch

    def _rng(self, sample_date_idx: int, stream: int, *, attempt: int = 0) -> np.random.Generator:
        """Create an independent deterministic RNG for one sampling decision.

        Training views vary by epoch. Validation and diagnostic views ignore
        epoch so repeated evaluation is stable. ``stream`` separates asset
        sampling from temporal masking, while ``attempt`` makes retries stable.
        """
        epoch = self.epoch if self.split == "train" else 0
        seed = np.random.SeedSequence(
            [self.config.seed, sample_date_idx, epoch, self.view_index, stream, attempt]
        )
        return np.random.default_rng(seed)

    def _patch_masks(self, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert daily store masks into split-aware patch-level masks."""
        return compute_patch_masks(
            masks["valid_asset_mask"],
            masks["valid_date_mask"],
            masks["holdout_date_mask"],
            masks["padded_date_mask"],
            masks["asset_slot_mask"],
            patch_len=self.config.patch_len,
            min_valid_days_per_asset_patch=self.config.min_valid_days_per_asset_patch,
            min_valid_dates_in_patch=self.config.min_valid_dates_in_patch,
            min_valid_asset_fraction=self.config.min_valid_asset_fraction,
            allow_holdout_targets=self.split != "train",
        )

    def _pad_asset_ids(self, asset_ids: np.ndarray, target_size: int) -> np.ndarray:
        """Right-pad an asset view with the ``-1`` missing-slot sentinel."""
        if len(asset_ids) >= target_size:
            return asset_ids
        return np.pad(asset_ids, (0, target_size - len(asset_ids)), constant_values=-1)

    def _choose_asset_view(self, sample_date_idx: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Choose assets and return the patch masks induced by that panel.

        Validation always uses the complete endpoint-valid panel. Training and
        diagnostic views sample a fixed-width panel, retry historically sparse
        selections, then fall back to the highest-coverage assets.
        """
        candidates = self.store.endpoint_asset_ids(sample_date_idx, self.split)
        if self.split == "validation":
            # Stable full-panel validation measures the same view every run;
            # collation handles different asset counts across sample dates.
            selected = candidates
            masks = self.store.window_masks(
                sample_date_idx, selected, self.split, self.config.lookback_days
            )
            return selected, self._patch_masks(masks)

        target_size = (
            self.config.train_k_assets if self.split == "train" else self.config.diagnostic_k_assets
        )

        # An endpoint-valid asset can still be sparse earlier in the lookback.
        # Retry deterministic views until enough target patches remain.
        for attempt in range(self.config.max_asset_sampling_attempts):
            rng = self._rng(sample_date_idx, 1, attempt=attempt)
            selected = rng.choice(
                candidates, size=min(target_size, len(candidates)), replace=False
            ).astype(np.int64)
            selected = self._pad_asset_ids(selected, target_size)
            masks = self.store.window_masks(
                sample_date_idx, selected, self.split, self.config.lookback_days
            )
            patch_masks = self._patch_masks(masks)
            if (
                int(patch_masks["patch_target_eligible"].sum()) >= self.config.min_masked_patches
                and int(patch_masks["patch_context_mask"].sum()) > self.config.min_masked_patches
            ):
                return selected, patch_masks

        # If random views fail, rank the full candidate panel by historical
        # coverage so a viable sparse sample is not rejected due to bad draws.
        all_masks = self.store.window_masks(
            sample_date_idx, candidates, self.split, self.config.lookback_days
        )
        coverage = all_masks["valid_asset_mask"].sum(axis=0)
        selected = candidates[np.argsort(-coverage, kind="stable")[:target_size]]
        selected = self._pad_asset_ids(selected, target_size)
        masks = self.store.window_masks(
            sample_date_idx, selected, self.split, self.config.lookback_days
        )
        patch_masks = self._patch_masks(masks)
        if (
            int(patch_masks["patch_target_eligible"].sum()) < self.config.min_masked_patches
            or int(patch_masks["patch_context_mask"].sum()) <= self.config.min_masked_patches
        ):
            raise RuntimeError(
                f"Sample date_idx={sample_date_idx} cannot produce "
                f"{self.config.min_masked_patches} target patches with visible context."
            )
        return selected, patch_masks

    def __getitem__(self, index: int) -> dict[str, object]:
        """Build one dense model sample with daily, patch, and JEPA masks."""
        sample_date_idx = int(self.sample_date_indices[index])
        asset_ids, patch_masks = self._choose_asset_view(sample_date_idx)
        window = self.store.window(
            sample_date_idx, asset_ids, self.split, self.config.lookback_days
        )

        # Temporal targets use a separate RNG stream from asset selection, so
        # changing one sampling policy does not silently perturb the other.
        target_mask, context_mask, target_ids = sample_jepa_target_mask(
            patch_masks["patch_target_eligible"],
            patch_masks["patch_context_mask"],
            self._rng(sample_date_idx, 2),
            mask_ratio=self.config.mask_ratio,
            min_masked_patches=self.config.min_masked_patches,
            max_masked_patches=self.config.max_masked_patches,
        )

        # NumPy arrays are contiguous CPU buffers, so torch.from_numpy avoids
        # copying each reconstructed sample before collation.
        tensors = {
            name: torch.from_numpy(np.asarray(value))
            for name, value in {**window, **patch_masks}.items()
            if name != "date_indices"
        }
        return {
            **tensors,
            "date_indices": torch.from_numpy(window["date_indices"]),
            "asset_ids": torch.from_numpy(asset_ids),
            "sample_date": self.store.date_values[sample_date_idx].isoformat(),
            "sample_date_idx": torch.tensor(sample_date_idx, dtype=torch.int64),
            "split_label": self.split,
            "jepa_target_mask": torch.from_numpy(target_mask),
            "jepa_context_mask": torch.from_numpy(context_mask),
            "target_patch_ids": torch.from_numpy(target_ids),
        }


# ============================================================================
# UNMASKED EMBEDDING DATASET
# ============================================================================


def fixed_k_asset_ids(
    candidates: np.ndarray,
    *,
    dataset_version: str,
    sample_date: str,
    view_index: int,
    k: int,
) -> np.ndarray:
    """Select one hardware- and ordering-independent deterministic fixed-K panel.

    Each candidate is ranked by SHA-256 over the immutable dataset version,
    sample date, view index, requested width, and asset ID. Selected IDs are
    sorted into canonical asset order before optional ``-1`` padding.
    """
    if view_index < 0:
        raise ValueError("view_index cannot be negative.")
    if k <= 0:
        raise ValueError("k must be positive.")
    unique = np.unique(np.asarray(candidates, dtype=np.int64))
    ranked = sorted(
        unique.tolist(),
        key=lambda asset_id: hashlib.sha256(
            f"{dataset_version}|{sample_date}|{view_index}|{k}|{asset_id}".encode("utf-8")
        ).digest(),
    )
    selected = np.sort(np.asarray(ranked[:k], dtype=np.int64))
    if selected.size < k:
        selected = np.pad(selected, (0, k - selected.size), constant_values=-1)
    return selected


class FIJepaEmbeddingDataset(Dataset[dict[str, object]]):
    """Reconstruct deterministic full-context windows without JEPA target masks."""

    def __init__(
        self,
        store: FrozenPanelStore,
        config: FIJepaDataConfig,
        split: EmbeddingSplit,
        *,
        asset_view: AssetView,
        view_index: int = 0,
    ):
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported embedding split: {split}")
        if asset_view not in {"all_valid", "fixed_k"}:
            raise ValueError(f"Unsupported embedding asset view: {asset_view}")
        if view_index < 0:
            raise ValueError("view_index cannot be negative.")

        self.store = store
        self.config = config
        self.split = split
        self.asset_view = asset_view
        self.view_index = view_index

        artifact_lookback = (store.resolved_config.get("dates") or {}).get("lookback_days")
        if artifact_lookback is not None and int(artifact_lookback) != config.lookback_days:
            raise ValueError(
                f"Configured lookback_days={config.lookback_days} does not match "
                f"artifact lookback_days={artifact_lookback}."
            )

        # Endpoint-valid candidates and split permission guarantee that the
        # final patch has context. Avoid reconstructing every all-asset window
        # during initialization; each item still enforces the endpoint rule.
        retained = [
            int(sample_date_idx)
            for sample_date_idx in store.sample_indices_for(split)
            if store.endpoint_asset_ids(int(sample_date_idx), split).size > 0
        ]
        self.sample_date_indices = np.asarray(retained, dtype=np.int64)

    def __len__(self) -> int:
        """Return the number of endpoint-valid embedding dates."""
        return len(self.sample_date_indices)

    def _patch_masks(self, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert daily masks into the full-context embedding patch contract."""
        return compute_patch_masks(
            masks["valid_asset_mask"],
            masks["valid_date_mask"],
            masks["holdout_date_mask"],
            masks["padded_date_mask"],
            masks["asset_slot_mask"],
            patch_len=self.config.patch_len,
            min_valid_days_per_asset_patch=self.config.min_valid_days_per_asset_patch,
            min_valid_dates_in_patch=self.config.min_valid_dates_in_patch,
            min_valid_asset_fraction=self.config.min_valid_asset_fraction,
            allow_holdout_targets=self.split == "validation",
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        """Build one deterministic unmasked encoder window."""
        sample_date_idx = int(self.sample_date_indices[index])
        sample_date = self.store.date_values[sample_date_idx].isoformat()
        candidates = self.store.endpoint_asset_ids(sample_date_idx, self.split)
        asset_ids = candidates
        if self.asset_view == "fixed_k":
            asset_ids = fixed_k_asset_ids(
                candidates,
                dataset_version=self.store.dataset_version,
                sample_date=sample_date,
                view_index=self.view_index,
                k=self.config.diagnostic_k_assets,
            )

        window = self.store.window(
            sample_date_idx, asset_ids, self.split, self.config.lookback_days
        )
        patch_masks = self._patch_masks(window)
        if not bool(patch_masks["patch_context_mask"][-1]):
            raise RuntimeError(
                f"Embedding sample date_idx={sample_date_idx} has no context-valid endpoint patch."
            )
        tensors = {
            name: torch.from_numpy(np.asarray(value))
            for name, value in {**window, **patch_masks}.items()
            if name not in {"date_indices", "patch_target_eligible"}
        }
        window_name = self.store.dates.iloc[sample_date_idx].get("validation_window_name", None)
        return {
            **tensors,
            "date_indices": torch.from_numpy(window["date_indices"]),
            "asset_ids": torch.from_numpy(asset_ids),
            "sample_date": sample_date,
            "sample_date_idx": torch.tensor(sample_date_idx, dtype=torch.int64),
            "split_label": self.split,
            "validation_window_name": "" if window_name is None or str(window_name) == "<NA>" else str(window_name),
            "asset_view": self.asset_view,
            "view_index": torch.tensor(self.view_index, dtype=torch.int64),
        }
