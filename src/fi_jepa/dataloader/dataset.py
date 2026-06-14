from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from fi_jepa.dataloader.config import FIJepaDataConfig
from fi_jepa.dataloader.masking import compute_patch_masks
from fi_jepa.dataloader.panel_store import DensePanelStore, Split
from fi_jepa.dataloader.request import DensePanelWindowRequest, RequestKind, ViewKind
from fi_jepa.dataloader.validation import validate_request_dataset_options, validate_request_history


# ============================================================================
# DENSE PANEL REQUEST DATASET
# ============================================================================


class DensePanelWindowRequestDataset(Dataset[DensePanelWindowRequest]):
    """Expose viable dense-panel endpoints as deterministic metadata requests.

    Cache request indexes contain only artifact-derived facts. JEPA viability
    depends on runtime patch thresholds, so structurally invalid endpoints are
    removed once here in the parent process before workers are constructed.
    """

    def __init__(
        self,
        store: DensePanelStore,
        config: FIJepaDataConfig,
        split: Split,
        *,
        request_kind: RequestKind,
        view_kind: ViewKind,
        view_index: int = 0,
    ):
        artifact_lookback = (store.resolved_config.get("dates") or {}).get("lookback_days")
        validate_request_dataset_options(
            split=split,
            request_kind=request_kind,
            view_kind=view_kind,
            view_index=view_index,
            configured_lookback=config.lookback_days,
            artifact_lookback=artifact_lookback,
        )
        request_index = store.request_index_for(split).reset_index(drop=True)
        validate_request_history(request_index, config.lookback_days)

        self.nominal_request_count = len(request_index)
        if request_kind == "jepa":
            request_index = self._filter_structurally_viable_jepa_requests(
                store, config, split, request_index
            )
        self.dropped_request_count = self.nominal_request_count - len(request_index)

        self.config = config
        self.split = split
        self.request_kind = request_kind
        self.view_kind = view_kind
        self.view_index = view_index
        self.request_index = request_index
        self._epoch = mp.Value("q", 0)

    def __len__(self) -> int:
        """Return the number of runtime-viable request endpoints."""
        return len(self.request_index)

    @property
    def epoch(self) -> int:
        """Return the epoch shared by the parent and persistent worker copies."""
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        """Set the shared epoch encoded into deterministic training request seeds."""
        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        self._epoch.value = epoch

    def __getitem__(self, index: int) -> DensePanelWindowRequest:
        """Return one request without gathering panel arrays."""
        row = self.request_index.iloc[index]
        sample_date_idx = int(row["sample_date_idx"])
        epoch = self.epoch if self.split == "train" and self.request_kind == "jepa" else 0
        seed = np.random.SeedSequence(
            [self.config.seed, sample_date_idx, epoch, self.view_index]
        ).generate_state(1, dtype=np.uint64)[0]
        return DensePanelWindowRequest(
            sample_date_idx=sample_date_idx,
            sample_date=str(row["sample_date"]),
            split=self.split,
            request_kind=self.request_kind,
            view_kind=self.view_kind,
            view_index=self.view_index,
            epoch=epoch,
            seed=int(seed),
            n_endpoint_valid_assets=int(row["n_endpoint_valid_assets"]),
            validation_window_name=str(row["validation_window_name"]),
        )

    @staticmethod
    def _filter_structurally_viable_jepa_requests(
        store: DensePanelStore,
        config: FIJepaDataConfig,
        split: Split,
        request_index: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop endpoints whose full split panel cannot support JEPA masking.

        The global asset axis detects inaccessible lookback history without
        depending on an epoch-specific random-K selection. The batch assembler
        still validates the selected view and fails loudly if that view itself
        cannot support the configured targets.
        """
        arrays = store.arrays_for(split)
        keep = np.zeros(len(request_index), dtype=bool)
        for position, sample_date_idx in enumerate(
            request_index["sample_date_idx"].to_numpy(dtype=np.int64)
        ):
            start = int(sample_date_idx) - config.lookback_days + 1
            stop = int(sample_date_idx) + 1
            valid_assets = arrays["valid_asset_mask"][start:stop]
            valid_dates = (
                arrays["valid_market_date"][start:stop]
                | arrays["valid_macro_date"][start:stop]
                | valid_assets.any(axis=1)
            )
            target_dates = (
                arrays["target_date_mask"][start:stop]
                if split == "train"
                else np.ones(config.lookback_days, dtype=bool)
            )
            masks = compute_patch_masks(
                valid_assets,
                valid_dates,
                target_dates,
                patch_len=config.patch_len,
                min_valid_days_per_asset_patch=config.min_valid_days_per_asset_patch,
                min_valid_dates_in_patch=config.min_valid_dates_in_patch,
                min_valid_asset_fraction=config.min_valid_asset_fraction,
            )
            keep[position] = (
                int(masks["patch_target_eligible"].sum()) >= config.min_masked_patches
                and int(masks["patch_context_mask"].sum()) > config.min_masked_patches
            )
        return request_index.loc[keep].reset_index(drop=True)
