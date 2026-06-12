from __future__ import annotations

import hashlib

import numpy as np
import torch

from fi_jepa.dataloader.config import FIJepaDataConfig
from fi_jepa.dataloader.masking import compute_batched_patch_masks, sample_jepa_target_mask
from fi_jepa.dataloader.panel_store import DensePanelStore
from fi_jepa.dataloader.request import DensePanelRequest


# ============================================================================
# DENSE PANEL BATCH ASSEMBLY
# ============================================================================


class DensePanelBatchAssembler:
    """Select assets, gather split panels, and emit the model batch ABI."""

    def __init__(self, store: DensePanelStore, config: FIJepaDataConfig):
        self.store = store
        self.config = config

    def __call__(self, requests: list[DensePanelRequest]) -> dict[str, object]:
        """Gather one homogeneous request batch and derive patch/JEPA masks."""
        self._validate_requests(requests)
        selected_assets = [self._select_asset_ids(request) for request in requests]
        asset_ids = np.stack(selected_assets)
        endpoints = np.asarray(
            [request.sample_date_idx for request in requests], dtype=np.int64
        )
        offsets = np.arange(self.config.lookback_days - 1, -1, -1, dtype=np.int64)
        date_indices = endpoints[:, None] - offsets[None, :]
        if (date_indices < 0).any():
            raise RuntimeError("Dense panel requests cannot use early-history padding.")

        split = requests[0].split
        arrays = self.store.arrays_for(split)
        linear_asset_indices = (
            date_indices[:, :, None] * self.store.asset_count + asset_ids[:, None, :]
        )

        # Each take directly emits [B, W, A, ...] or [B, W, ...]. Values and
        # masks are gathered together; the dataloader never infers feature masks.
        asset_x = np.take(
            arrays["asset_x"].reshape(-1, arrays["asset_x"].shape[-1]),
            linear_asset_indices,
            axis=0,
        )
        asset_feature_mask = np.take(
            arrays["asset_feature_mask"].reshape(
                -1, arrays["asset_feature_mask"].shape[-1]
            ),
            linear_asset_indices,
            axis=0,
        )
        valid_asset_mask = np.take(
            arrays["valid_asset_mask"].reshape(-1),
            linear_asset_indices,
            axis=0,
        )
        market_x = np.take(arrays["market_x"], date_indices, axis=0)
        market_feature_mask = np.take(
            arrays["market_feature_mask"], date_indices, axis=0
        )
        valid_market_date = np.take(arrays["valid_market_date"], date_indices, axis=0)
        macro_x = np.take(arrays["macro_x"], date_indices, axis=0)
        macro_feature_mask = np.take(
            arrays["macro_feature_mask"], date_indices, axis=0
        )
        valid_macro_date = np.take(arrays["valid_macro_date"], date_indices, axis=0)
        valid_date = valid_market_date | valid_macro_date | valid_asset_mask.any(axis=2)
        target_date = (
            np.take(arrays["target_date_mask"], date_indices, axis=0)
            if split == "train"
            else np.ones_like(valid_date, dtype=bool)
        )

        patch_masks = compute_batched_patch_masks(
            valid_asset_mask,
            valid_date,
            target_date,
            patch_len=self.config.patch_len,
            min_valid_days_per_asset_patch=self.config.min_valid_days_per_asset_patch,
            min_valid_dates_in_patch=self.config.min_valid_dates_in_patch,
            min_valid_asset_fraction=self.config.min_valid_asset_fraction,
        )
        target_counts = patch_masks["patch_target_eligible"].sum(axis=1)
        context_counts = patch_masks["patch_context_mask"].sum(axis=1)
        if requests[0].request_kind == "jepa":
            invalid = (target_counts < self.config.min_masked_patches) | (
                context_counts <= self.config.min_masked_patches
            )
            if invalid.any():
                index = int(np.flatnonzero(invalid)[0])
                request = requests[index]
                raise RuntimeError(
                    "Selected JEPA view is not viable: "
                    f"sample_date_idx={request.sample_date_idx}, "
                    f"sample_date={request.sample_date}, seed={request.seed}, "
                    f"view_type={request.view_kind}, k_assets={asset_ids.shape[1]}, "
                    f"n_endpoint_valid_assets={request.n_endpoint_valid_assets}, "
                    f"valid_patch_count={int(target_counts[index])}."
                )
        elif not bool(patch_masks["patch_context_mask"][:, -1].all()):
            raise RuntimeError("Embedding batch contains an endpoint without valid context.")

        # Convert each gathered daily array once. Patch tensors are views over
        # these storages; no patched cache or duplicate patch buffer is created.
        batch_size, lookback_days, n_assets, asset_dim = asset_x.shape
        n_patches = self.config.num_patches
        patch_len = self.config.patch_len
        asset_daily = torch.from_numpy(np.ascontiguousarray(asset_x))
        asset_feature_daily = torch.from_numpy(np.ascontiguousarray(asset_feature_mask))
        valid_asset_daily = torch.from_numpy(np.ascontiguousarray(valid_asset_mask))
        market_daily = torch.from_numpy(np.ascontiguousarray(market_x))
        market_feature_daily = torch.from_numpy(np.ascontiguousarray(market_feature_mask))
        macro_daily = torch.from_numpy(np.ascontiguousarray(macro_x))
        macro_feature_daily = torch.from_numpy(np.ascontiguousarray(macro_feature_mask))
        valid_market_daily = torch.from_numpy(np.ascontiguousarray(valid_market_date))
        valid_macro_daily = torch.from_numpy(np.ascontiguousarray(valid_macro_date))
        batch: dict[str, object] = {
            "asset_patches": asset_daily.view(
                batch_size, n_patches, patch_len, n_assets, asset_dim
            ),
            "market_patches": market_daily.view(
                batch_size, n_patches, patch_len, market_x.shape[-1]
            ),
            "macro_patches": macro_daily.view(
                batch_size, n_patches, patch_len, macro_x.shape[-1]
            ),
            "asset_feature_mask_patched": asset_feature_daily.view(
                batch_size, n_patches, patch_len, n_assets, asset_dim
            ),
            "market_feature_mask_patched": market_feature_daily.view(
                batch_size, n_patches, patch_len, market_x.shape[-1]
            ),
            "macro_feature_mask_patched": macro_feature_daily.view(
                batch_size, n_patches, patch_len, macro_x.shape[-1]
            ),
            "valid_asset_mask_patched": valid_asset_daily.view(
                batch_size, n_patches, patch_len, n_assets
            ),
            "valid_market_date_mask_patched": valid_market_daily.view(
                batch_size, n_patches, patch_len
            ),
            "valid_macro_date_mask_patched": valid_macro_daily.view(
                batch_size, n_patches, patch_len
            ),
            "patch_asset_mask": torch.from_numpy(patch_masks["patch_asset_mask"]),
            "patch_context_mask": torch.from_numpy(patch_masks["patch_context_mask"]),
            "patch_target_eligible": torch.from_numpy(
                patch_masks["patch_target_eligible"]
            ),
            "asset_ids": torch.from_numpy(asset_ids),
            "sample_date_idx": torch.from_numpy(endpoints),
            "sample_date": [request.sample_date for request in requests],
            "split_label": [request.split for request in requests],
            "validation_window_name": [
                request.validation_window_name for request in requests
            ],
            "asset_view": [request.view_kind for request in requests],
            "view_index": torch.tensor(
                [request.view_index for request in requests], dtype=torch.int64
            ),
            "request_seed": [request.seed for request in requests],
            "k_assets": [int(asset_ids.shape[1])] * batch_size,
            "n_endpoint_valid_assets": [
                request.n_endpoint_valid_assets for request in requests
            ],
            "target_eligible_patch_count": torch.from_numpy(
                target_counts.astype(np.int64)
            ),
        }
        if requests[0].request_kind == "jepa":
            batch.update(self._sample_jepa_masks(requests, patch_masks))
        return batch

    def _validate_requests(self, requests: list[DensePanelRequest]) -> None:
        """Require one non-empty homogeneous request batch."""
        if not requests:
            raise ValueError("Cannot assemble an empty dense panel request list.")
        if not all(isinstance(request, DensePanelRequest) for request in requests):
            raise TypeError("DensePanelBatchAssembler accepts only DensePanelRequest values.")
        first = requests[0]
        for request in requests[1:]:
            if (
                request.split != first.split
                or request.request_kind != first.request_kind
                or request.view_kind != first.view_kind
            ):
                raise ValueError("Dense panel request batches must be homogeneous.")

    def _select_asset_ids(self, request: DensePanelRequest) -> np.ndarray:
        """Select the global axis, a random K view, or a deterministic fixed-K view."""
        if request.view_kind == "all_valid":
            return np.arange(self.store.asset_count, dtype=np.int64)

        candidates = self.store.endpoint_asset_ids(request.sample_date_idx, request.split)
        if len(candidates) != request.n_endpoint_valid_assets:
            raise RuntimeError(
                f"Request index endpoint asset count drifted for date_idx={request.sample_date_idx}."
            )
        k = (
            self.config.train_k_assets
            if request.view_kind == "random_k"
            else self.config.fixed_k_assets
        )
        if len(candidates) < k:
            raise RuntimeError(
                f"Fixed-K request requires k_assets={k}, but sample_date_idx="
                f"{request.sample_date_idx} has n_endpoint_valid_assets={len(candidates)}."
            )
        if request.view_kind == "random_k":
            selected = np.random.default_rng(request.seed).choice(
                candidates, size=k, replace=False
            )
            return np.sort(selected.astype(np.int64))

        ranked = sorted(
            candidates.tolist(),
            key=lambda asset_id: hashlib.sha256(
                f"{self.store.dataset_version}|{request.sample_date}|"
                f"{request.view_index}|{k}|{asset_id}".encode("utf-8")
            ).digest(),
        )
        return np.sort(np.asarray(ranked[:k], dtype=np.int64))

    def _sample_jepa_masks(
        self,
        requests: list[DensePanelRequest],
        patch_masks: dict[str, np.ndarray],
    ) -> dict[str, torch.Tensor]:
        """Sample deterministic JEPA targets and pad only target-ID metadata."""
        target_masks: list[np.ndarray] = []
        context_masks: list[np.ndarray] = []
        target_ids: list[np.ndarray] = []
        for index, request in enumerate(requests):
            target, context, ids = sample_jepa_target_mask(
                patch_masks["patch_target_eligible"][index],
                patch_masks["patch_context_mask"][index],
                np.random.default_rng(np.random.SeedSequence([request.seed, 1])),
                mask_ratio=self.config.mask_ratio,
                min_masked_patches=self.config.min_masked_patches,
                max_masked_patches=self.config.max_masked_patches,
            )
            target_masks.append(target)
            context_masks.append(context)
            target_ids.append(ids)

        max_targets = max(len(ids) for ids in target_ids)
        padded_ids = np.full((len(requests), max_targets), -1, dtype=np.int64)
        for index, ids in enumerate(target_ids):
            padded_ids[index, : len(ids)] = ids
        return {
            "jepa_target_mask": torch.from_numpy(np.stack(target_masks)),
            "jepa_context_mask": torch.from_numpy(np.stack(context_masks)),
            "target_patch_ids": torch.from_numpy(padded_ids),
            "target_patch_id_mask": torch.from_numpy(padded_ids >= 0),
        }
