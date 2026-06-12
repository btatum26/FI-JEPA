from __future__ import annotations

from typing import cast

import numpy as np
import torch

from fi_jepa.dataloader.config import AssemblyMode, FIJepaDataConfig
from fi_jepa.dataloader.dataset import fixed_k_asset_ids
from fi_jepa.dataloader.masking import (
    compute_batched_patch_masks,
    compute_patch_masks,
    sample_jepa_target_mask,
)
from fi_jepa.dataloader.panel_store import FrozenPanelStore, Split
from fi_jepa.dataloader.request import WindowRequest


# ============================================================================
# LEGACY SAMPLE COLLATION
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


def _add_patch_views(batch: dict[str, object], *, patch_len: int) -> dict[str, object]:
    """Add zero-copy chronological patch views to an assembled daily batch."""
    asset_x = cast(torch.Tensor, batch["asset_x"])
    market_x = cast(torch.Tensor, batch["market_x"])
    macro_x = cast(torch.Tensor, batch["macro_x"])
    batch_size, lookback_days, n_assets, asset_dim = asset_x.shape
    n_patches = lookback_days // patch_len

    # [B, W, A, F_asset] -> [B, P, L, A, F_asset]. These are views over the
    # assembled daily tensors, not duplicate patch buffers.
    batch["asset_patches"] = asset_x.view(
        batch_size, n_patches, patch_len, n_assets, asset_dim
    )
    # [B, W, F_stream] -> [B, P, L, F_stream].
    batch["market_patches"] = market_x.view(
        batch_size, n_patches, patch_len, market_x.shape[-1]
    )
    batch["macro_patches"] = macro_x.view(
        batch_size, n_patches, patch_len, macro_x.shape[-1]
    )
    batch["asset_feature_mask_patched"] = cast(
        torch.Tensor, batch["asset_feature_mask"]
    ).view(batch_size, n_patches, patch_len, n_assets, asset_dim)
    batch["market_feature_mask_patched"] = cast(
        torch.Tensor, batch["market_feature_mask"]
    ).view(batch_size, n_patches, patch_len, market_x.shape[-1])
    batch["macro_feature_mask_patched"] = cast(
        torch.Tensor, batch["macro_feature_mask"]
    ).view(batch_size, n_patches, patch_len, macro_x.shape[-1])
    batch["valid_asset_mask_patched"] = cast(torch.Tensor, batch["valid_asset_mask"]).view(
        batch_size, n_patches, patch_len, n_assets
    )
    for name in (
        "valid_date_mask",
        "holdout_date_mask",
        "padded_date_mask",
        "valid_market_date_mask",
        "valid_macro_date_mask",
    ):
        batch[f"{name}_patched"] = cast(torch.Tensor, batch[name]).view(
            batch_size, n_patches, patch_len
        )
    return batch


def collate_fi_jepa_batch(
    samples: list[dict[str, object]], *, patch_len: int
) -> dict[str, object]:
    """Pad and stack already materialized samples for compatibility validation.

    This is the intentionally slow ``per_sample`` fallback. It remains public
    so callers can compare the batch-first path against the original assembly
    structure while the model-facing output contract stays unchanged.
    """
    if not samples:
        raise ValueError("Cannot collate an empty FI-JEPA sample list.")

    max_assets = max(int(cast(torch.Tensor, sample["asset_ids"]).shape[0]) for sample in samples)
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
                    cast(torch.Tensor, sample[name]),
                    max_assets,
                    dimension,
                    asset_fill.get(name, 0),
                )
                for sample in samples
            ]
        )

    # Target IDs use -1 padding so the derived mask is unambiguous.
    if has_targets:
        max_targets = max(
            int(cast(torch.Tensor, sample["target_patch_ids"]).shape[0]) for sample in samples
        )
        target_patch_ids = torch.stack(
            [
                _pad_tensor(cast(torch.Tensor, sample["target_patch_ids"]), max_targets, 0, -1)
                for sample in samples
            ]
        )
        batch["target_patch_ids"] = target_patch_ids
        batch["target_patch_id_mask"] = target_patch_ids >= 0

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
            batch[name] = torch.stack([cast(torch.Tensor, sample[name]) for sample in samples])
    return _add_patch_views(batch, patch_len=patch_len)


# ============================================================================
# BATCH ASSEMBLER
# ============================================================================


class FIJepaBatchAssembler:
    """Materialize lightweight requests into the stable FI-JEPA batch contract.

    Asset selection is performed once per request because panels can differ,
    then the fast path builds date and asset index matrices and gathers every
    dense stream once for the whole batch. ``per_sample`` uses the same chosen
    assets and seeds but reconstructs samples through ``FrozenPanelStore.window``
    before legacy collation, providing a correctness reference.
    """

    def __init__(
        self,
        store: FrozenPanelStore,
        config: FIJepaDataConfig,
        *,
        assembly_mode: AssemblyMode | None = None,
    ):
        self.store = store
        self.config = config
        self.assembly_mode = config.assembly_mode if assembly_mode is None else assembly_mode
        if self.assembly_mode not in {"batched_gather", "per_sample"}:
            raise ValueError(f"Unsupported assembly_mode: {self.assembly_mode}")

    def __call__(self, requests: list[WindowRequest]) -> dict[str, object]:
        """Select assets and assemble one homogeneous request batch."""
        self._validate_requests(requests)
        asset_ids = [self._select_asset_ids(request) for request in requests]
        if self.assembly_mode == "per_sample":
            samples = [
                self._materialize_sample(request, selected)
                for request, selected in zip(requests, asset_ids, strict=True)
            ]
            return collate_fi_jepa_batch(samples, patch_len=self.config.patch_len)
        return self._assemble_batched(requests, asset_ids)

    def _validate_requests(self, requests: list[WindowRequest]) -> None:
        """Require one non-empty homogeneous batch from a single dataset."""
        if not requests:
            raise ValueError("Cannot assemble an empty FI-JEPA request list.")
        if not all(isinstance(request, WindowRequest) for request in requests):
            raise TypeError("FIJepaBatchAssembler accepts only WindowRequest values.")
        first = requests[0]
        for request in requests[1:]:
            if request.split != first.split or request.request_kind != first.request_kind:
                raise ValueError("FI-JEPA request batches must share split and request_kind.")

    def _rng(
        self, request: WindowRequest, stream: int, *, attempt: int = 0
    ) -> np.random.Generator:
        """Create an independent deterministic RNG for one assembler decision."""
        return np.random.default_rng(np.random.SeedSequence([request.seed, stream, attempt]))

    def _patch_masks(
        self, masks: dict[str, np.ndarray], split: Split
    ) -> dict[str, np.ndarray]:
        """Build single-window masks used by filtering and compatibility assembly."""
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
            allow_holdout_targets=split != "train",
        )

    def _has_viable_jepa_masks(self, patch_masks: dict[str, np.ndarray]) -> bool:
        """Return whether masks can sample targets while retaining visible context."""
        return (
            int(patch_masks["patch_target_eligible"].sum()) >= self.config.min_masked_patches
            and int(patch_masks["patch_context_mask"].sum()) > self.config.min_masked_patches
        )

    def _pad_asset_ids(self, asset_ids: np.ndarray, target_size: int) -> np.ndarray:
        """Right-pad one selected asset panel with the ``-1`` slot sentinel."""
        if len(asset_ids) >= target_size:
            return asset_ids
        return np.pad(asset_ids, (0, target_size - len(asset_ids)), constant_values=-1)

    def _select_asset_ids(self, request: WindowRequest) -> np.ndarray:
        """Apply the request's all-valid, fixed-K, or retrying random-K policy."""
        candidates = self.store.endpoint_asset_ids(request.sample_date_idx, request.split)
        if candidates.size == 0:
            raise RuntimeError(
                f"Sample date_idx={request.sample_date_idx} has no endpoint-valid assets."
            )
        if request.view_kind == "all_valid":
            return candidates
        if request.view_kind == "fixed_k":
            return fixed_k_asset_ids(
                candidates,
                dataset_version=self.store.dataset_version,
                sample_date=request.sample_date,
                view_index=request.view_index,
                k=self.config.diagnostic_k_assets,
            )

        target_size = (
            self.config.train_k_assets
            if request.split == "train"
            else self.config.diagnostic_k_assets
        )
        # Endpoint-valid assets can be sparse earlier in the lookback. Retry
        # deterministic panels before falling back to highest coverage.
        for attempt in range(self.config.max_asset_sampling_attempts):
            selected = self._rng(request, 1, attempt=attempt).choice(
                candidates, size=min(target_size, len(candidates)), replace=False
            ).astype(np.int64)
            selected = self._pad_asset_ids(selected, target_size)
            masks = self.store.window_masks(
                request.sample_date_idx, selected, request.split, self.config.lookback_days
            )
            if self._has_viable_jepa_masks(self._patch_masks(masks, request.split)):
                return selected

        all_masks = self.store.window_masks(
            request.sample_date_idx, candidates, request.split, self.config.lookback_days
        )
        coverage = all_masks["valid_asset_mask"].sum(axis=0)
        selected = candidates[np.argsort(-coverage, kind="stable")[:target_size]]
        selected = self._pad_asset_ids(selected, target_size)
        masks = self.store.window_masks(
            request.sample_date_idx, selected, request.split, self.config.lookback_days
        )
        if not self._has_viable_jepa_masks(self._patch_masks(masks, request.split)):
            raise RuntimeError(
                f"Sample date_idx={request.sample_date_idx} cannot produce "
                f"{self.config.min_masked_patches} target patches with visible context."
            )
        return selected

    def _sample_jepa_masks(
        self, request: WindowRequest, patch_masks: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample one reproducible JEPA target/context partition."""
        return sample_jepa_target_mask(
            patch_masks["patch_target_eligible"],
            patch_masks["patch_context_mask"],
            self._rng(request, 2),
            mask_ratio=self.config.mask_ratio,
            min_masked_patches=self.config.min_masked_patches,
            max_masked_patches=self.config.max_masked_patches,
        )

    def _validation_window_name(self, sample_date_idx: int) -> str:
        """Return normalized optional validation-window metadata."""
        value = self.store.dates.iloc[sample_date_idx].get("validation_window_name", None)
        return "" if value is None or str(value) == "<NA>" else str(value)

    def _materialize_sample(
        self, request: WindowRequest, asset_ids: np.ndarray
    ) -> dict[str, object]:
        """Build one reference sample through the original per-window store API."""
        window = self.store.window(
            request.sample_date_idx, asset_ids, request.split, self.config.lookback_days
        )
        patch_masks = self._patch_masks(window, request.split)
        tensor_values = {**window, **patch_masks}
        if request.request_kind == "embedding":
            if not bool(patch_masks["patch_context_mask"][-1]):
                raise RuntimeError(
                    f"Embedding sample date_idx={request.sample_date_idx} "
                    "has no context-valid endpoint patch."
                )
            tensor_values.pop("patch_target_eligible")

        sample: dict[str, object] = {
            name: torch.from_numpy(np.asarray(value))
            for name, value in tensor_values.items()
        }
        sample.update(
            {
                "asset_ids": torch.from_numpy(asset_ids),
                "sample_date": request.sample_date,
                "sample_date_idx": torch.tensor(request.sample_date_idx, dtype=torch.int64),
                "split_label": request.split,
            }
        )
        if request.request_kind == "jepa":
            target_mask, context_mask, target_ids = self._sample_jepa_masks(request, patch_masks)
            sample.update(
                {
                    "jepa_target_mask": torch.from_numpy(target_mask),
                    "jepa_context_mask": torch.from_numpy(context_mask),
                    "target_patch_ids": torch.from_numpy(target_ids),
                }
            )
        else:
            sample.update(
                {
                    "validation_window_name": self._validation_window_name(
                        request.sample_date_idx
                    ),
                    "asset_view": request.view_kind,
                    "view_index": torch.tensor(request.view_index, dtype=torch.int64),
                }
            )
        return sample

    def _assemble_batched(
        self, requests: list[WindowRequest], selected_assets: list[np.ndarray]
    ) -> dict[str, object]:
        """Gather all daily streams once and construct the final batched contract."""
        batch_size = len(requests)
        max_assets = max(len(asset_ids) for asset_ids in selected_assets)
        asset_ids = np.full((batch_size, max_assets), -1, dtype=np.int64)
        for batch_index, selected in enumerate(selected_assets):
            asset_ids[batch_index, : len(selected)] = selected
        asset_slot_mask = asset_ids >= 0
        safe_asset_ids = np.where(asset_slot_mask, asset_ids, 0)

        # [B, W] chronological date matrix, left-padded with -1 before history.
        endpoints = np.asarray([request.sample_date_idx for request in requests], dtype=np.int64)
        offsets = np.arange(self.config.lookback_days - 1, -1, -1, dtype=np.int64)
        raw_date_indices = endpoints[:, None] - offsets[None, :]
        padded_date_mask = raw_date_indices < 0
        safe_date_indices = np.maximum(raw_date_indices, 0)
        date_indices = np.where(padded_date_mask, -1, raw_date_indices)

        split = requests[0].split
        permission = self.store.permission_for(split)[safe_date_indices] & ~padded_date_mask

        # Flatten date/asset pairs into one gather index. ``np.take`` avoids the
        # repeated broadcast bookkeeping of multidimensional advanced indexing
        # while producing each tensor directly at [B, W, A, ...].
        linear_asset_indices = (
            safe_date_indices[:, :, None] * self.store.asset_count
            + safe_asset_ids[:, None, :]
        )
        asset_x = np.take(
            self.store.asset_x.reshape(-1, self.store.asset_x.shape[-1]),
            linear_asset_indices,
            axis=0,
        )
        asset_feature_mask = np.take(
            self.store.asset_feature_mask.reshape(
                -1, self.store.asset_feature_mask.shape[-1]
            ),
            linear_asset_indices,
            axis=0,
        )
        valid_asset_mask = np.take(
            self.store.valid_asset.reshape(-1), linear_asset_indices, axis=0
        )
        market_x = np.take(self.store.market_x, safe_date_indices, axis=0)
        market_feature_mask = np.take(
            self.store.market_feature_mask, safe_date_indices, axis=0
        )
        macro_x = np.take(self.store.macro_x, safe_date_indices, axis=0)
        macro_feature_mask = np.take(
            self.store.macro_feature_mask, safe_date_indices, axis=0
        )

        # Split permission and padded slots are authoritative. Clear only
        # inaccessible rows/slots instead of multiplying every gathered value.
        blocked_dates = ~permission
        asset_x[blocked_dates] = 0
        asset_feature_mask[blocked_dates] = False
        valid_asset_mask[blocked_dates] = False
        market_x[blocked_dates] = 0
        market_feature_mask[blocked_dates] = False
        macro_x[blocked_dates] = 0
        macro_feature_mask[blocked_dates] = False
        for batch_index in np.flatnonzero(~asset_slot_mask.all(axis=1)):
            padded_slots = ~asset_slot_mask[batch_index]
            asset_x[batch_index, :, padded_slots] = 0
            asset_feature_mask[batch_index, :, padded_slots] = False
            valid_asset_mask[batch_index, :, padded_slots] = False

        valid_market_date_mask = self.store.valid_market_date[safe_date_indices] & permission
        valid_macro_date_mask = self.store.valid_macro_date[safe_date_indices] & permission
        valid_date_mask = (
            valid_market_date_mask | valid_macro_date_mask | valid_asset_mask.any(axis=2)
        )
        holdout_date_mask = (
            self.store.protected_holdout[safe_date_indices] & ~padded_date_mask
        )

        patch_masks = compute_batched_patch_masks(
            valid_asset_mask,
            valid_date_mask,
            holdout_date_mask,
            padded_date_mask,
            asset_slot_mask,
            patch_len=self.config.patch_len,
            min_valid_days_per_asset_patch=self.config.min_valid_days_per_asset_patch,
            min_valid_dates_in_patch=self.config.min_valid_dates_in_patch,
            min_valid_asset_fraction=self.config.min_valid_asset_fraction,
            allow_holdout_targets=split != "train",
        )
        arrays = {
            "asset_ids": asset_ids,
            "asset_slot_mask": asset_slot_mask,
            "date_indices": date_indices,
            "asset_x": asset_x,
            "asset_feature_mask": asset_feature_mask,
            "valid_asset_mask": valid_asset_mask,
            "market_x": market_x,
            "market_feature_mask": market_feature_mask,
            "macro_x": macro_x,
            "macro_feature_mask": macro_feature_mask,
            "valid_market_date_mask": valid_market_date_mask,
            "valid_macro_date_mask": valid_macro_date_mask,
            "valid_date_mask": valid_date_mask,
            "holdout_date_mask": holdout_date_mask,
            "padded_date_mask": padded_date_mask,
            **patch_masks,
        }
        if requests[0].request_kind == "embedding":
            if not bool(patch_masks["patch_context_mask"][:, -1].all()):
                raise RuntimeError("Embedding batch contains an endpoint without valid context.")
            arrays.pop("patch_target_eligible")

        # One contiguous conversion per final array keeps patch reshaping cheap.
        batch: dict[str, object] = {
            name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in arrays.items()
        }
        batch.update(
            {
                "sample_date": [request.sample_date for request in requests],
                "sample_date_idx": torch.tensor(endpoints, dtype=torch.int64),
                "split_label": [request.split for request in requests],
            }
        )
        if requests[0].request_kind == "jepa":
            target_masks: list[np.ndarray] = []
            context_masks: list[np.ndarray] = []
            target_ids: list[np.ndarray] = []
            for batch_index, request in enumerate(requests):
                single_patch_masks = {
                    name: value[batch_index] for name, value in patch_masks.items()
                }
                target_mask, context_mask, ids = self._sample_jepa_masks(
                    request, single_patch_masks
                )
                target_masks.append(target_mask)
                context_masks.append(context_mask)
                target_ids.append(ids)
            max_targets = max(len(ids) for ids in target_ids)
            padded_target_ids = np.full((batch_size, max_targets), -1, dtype=np.int64)
            for batch_index, ids in enumerate(target_ids):
                padded_target_ids[batch_index, : len(ids)] = ids
            batch.update(
                {
                    "jepa_target_mask": torch.from_numpy(np.stack(target_masks)),
                    "jepa_context_mask": torch.from_numpy(np.stack(context_masks)),
                    "target_patch_ids": torch.from_numpy(padded_target_ids),
                    "target_patch_id_mask": torch.from_numpy(padded_target_ids >= 0),
                }
            )
        else:
            batch.update(
                {
                    "validation_window_name": [
                        self._validation_window_name(request.sample_date_idx)
                        for request in requests
                    ],
                    "asset_view": [request.view_kind for request in requests],
                    "view_index": torch.tensor(
                        [request.view_index for request in requests], dtype=torch.int64
                    ),
                }
            )
        return _add_patch_views(batch, patch_len=self.config.patch_len)
