from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from torch.utils.data import Dataset

from fi_jepa.dataloader.config import FIJepaDataConfig
from fi_jepa.dataloader.masking import compute_patch_masks
from fi_jepa.dataloader.panel_store import FrozenPanelStore, Split
from fi_jepa.dataloader.request import ViewKind, WindowRequest

EmbeddingSplit = Literal["train", "validation"]
AssetView = Literal["all_valid", "fixed_k"]


# ============================================================================
# SHARED DATASET VALIDATION
# ============================================================================


def _validate_artifact_lookback(store: FrozenPanelStore, config: FIJepaDataConfig) -> None:
    """Require runtime windows to fit within the artifact's protected lookback."""
    artifact_lookback = (store.resolved_config.get("dates") or {}).get("lookback_days")
    if artifact_lookback is not None and config.lookback_days > int(artifact_lookback):
        raise ValueError(
            f"Configured lookback_days={config.lookback_days} exceeds "
            f"artifact lookback_days={artifact_lookback}."
        )


# ============================================================================
# JEPA WINDOW REQUEST DATASET
# ============================================================================


class FIJepaWindowDataset(Dataset[WindowRequest]):
    """Expose structurally valid JEPA endpoints as lightweight window requests.

    Initialization still removes sample dates that can never satisfy the
    configured minimum target count. Item access performs no asset selection,
    dense gathering, patching, or JEPA masking; those batch-dependent concerns
    are owned by ``FIJepaBatchAssembler``.
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
        if view_index < 0:
            raise ValueError("view_index cannot be negative.")

        _validate_artifact_lookback(store, config)
        self.config = config
        self.split = split
        self.view_index = view_index
        self.epoch = 0

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
        self.sample_dates = tuple(
            store.date_values[sample_date_idx].isoformat()
            for sample_date_idx in self.sample_date_indices
        )
        self.nominal_sample_count = int(len(nominal))
        self.dropped_sample_count = self.nominal_sample_count - len(retained)

    def __len__(self) -> int:
        """Return the number of structurally valid sample dates."""
        return len(self.sample_date_indices)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch encoded into reproducible training request seeds."""
        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        self.epoch = epoch

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

    def _seed_for(self, sample_date_idx: int) -> int:
        """Return one deterministic seed for all assembler decisions for a request."""
        epoch = self.epoch if self.split == "train" else 0
        seed = np.random.SeedSequence(
            [self.config.seed, sample_date_idx, epoch, self.view_index]
        ).generate_state(1, dtype=np.uint64)[0]
        return int(seed)

    def __getitem__(self, index: int) -> WindowRequest:
        """Return one small JEPA request without materializing its window."""
        sample_date_idx = int(self.sample_date_indices[index])
        view_kind: ViewKind = "all_valid" if self.split == "validation" else "random_k"
        return WindowRequest(
            sample_date_idx=sample_date_idx,
            sample_date=self.sample_dates[index],
            split=self.split,
            request_kind="jepa",
            view_kind=view_kind,
            view_index=self.view_index,
            seed=self._seed_for(sample_date_idx),
        )


# ============================================================================
# DETERMINISTIC FIXED-K ASSET VIEWS
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


# ============================================================================
# UNMASKED EMBEDDING REQUEST DATASET
# ============================================================================


class FIJepaEmbeddingDataset(Dataset[WindowRequest]):
    """Expose deterministic embedding endpoints as lightweight window requests."""

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

        _validate_artifact_lookback(store, config)
        self.config = config
        self.split = split
        self.asset_view = asset_view
        self.view_index = view_index

        # Endpoint-valid candidates and split permission guarantee that the
        # final patch has context. The assembler enforces that invariant again.
        retained = [
            int(sample_date_idx)
            for sample_date_idx in store.sample_indices_for(split)
            if store.endpoint_asset_ids(int(sample_date_idx), split).size > 0
        ]
        self.sample_date_indices = np.asarray(retained, dtype=np.int64)
        self.sample_dates = tuple(
            store.date_values[sample_date_idx].isoformat()
            for sample_date_idx in self.sample_date_indices
        )

    def __len__(self) -> int:
        """Return the number of endpoint-valid embedding dates."""
        return len(self.sample_date_indices)

    def __getitem__(self, index: int) -> WindowRequest:
        """Return one small embedding request without materializing its window."""
        sample_date_idx = int(self.sample_date_indices[index])
        seed = np.random.SeedSequence(
            [self.config.seed, sample_date_idx, 0, self.view_index]
        ).generate_state(1, dtype=np.uint64)[0]
        return WindowRequest(
            sample_date_idx=sample_date_idx,
            sample_date=self.sample_dates[index],
            split=self.split,
            request_kind="embedding",
            view_kind=self.asset_view,
            view_index=self.view_index,
            seed=int(seed),
        )
