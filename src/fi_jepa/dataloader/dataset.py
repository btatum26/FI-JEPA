from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from fi_jepa.dataloader.config import FIJepaDataConfig
from fi_jepa.dataloader.panel_store import DensePanelStore, Split
from fi_jepa.dataloader.request import DensePanelRequest, RequestKind, ViewKind


# ============================================================================
# DENSE PANEL REQUEST DATASET
# ============================================================================


class DensePanelRequestDataset(Dataset[DensePanelRequest]):
    """Expose one cache request index as deterministic metadata-only requests."""

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
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported split: {split}")
        if request_kind == "jepa" and view_kind not in {"random_k", "all_valid"}:
            raise ValueError("JEPA requests support only random_k and all_valid views.")
        if request_kind == "embedding" and view_kind not in {"fixed_k", "all_valid"}:
            raise ValueError("Embedding requests support only fixed_k and all_valid views.")
        if view_index < 0:
            raise ValueError("view_index cannot be negative.")

        artifact_lookback = (store.resolved_config.get("dates") or {}).get("lookback_days")
        if artifact_lookback is not None and config.lookback_days > int(artifact_lookback):
            raise ValueError(
                f"Configured lookback_days={config.lookback_days} exceeds "
                f"artifact lookback_days={artifact_lookback}."
            )
        request_index = store.request_index_for(split).reset_index(drop=True)
        too_early = request_index["sample_date_idx"] < config.lookback_days - 1
        if too_early.any():
            row = request_index.loc[too_early].iloc[0]
            raise ValueError(
                f"Request sample_date_idx={int(row['sample_date_idx'])} cannot provide "
                f"lookback_days={config.lookback_days} without padding."
            )

        self.config = config
        self.split = split
        self.request_kind = request_kind
        self.view_kind = view_kind
        self.view_index = view_index
        self.request_index = request_index
        self.epoch = 0

    def __len__(self) -> int:
        """Return the number of artifact-defined request endpoints."""
        return len(self.request_index)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch encoded into deterministic training request seeds."""
        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        self.epoch = epoch

    def __getitem__(self, index: int) -> DensePanelRequest:
        """Return one request without gathering panel arrays."""
        row = self.request_index.iloc[index]
        sample_date_idx = int(row["sample_date_idx"])
        epoch = self.epoch if self.split == "train" and self.request_kind == "jepa" else 0
        seed = np.random.SeedSequence(
            [self.config.seed, sample_date_idx, epoch, self.view_index]
        ).generate_state(1, dtype=np.uint64)[0]
        return DensePanelRequest(
            sample_date_idx=sample_date_idx,
            sample_date=str(row["sample_date"]),
            split=self.split,
            request_kind=self.request_kind,
            view_kind=self.view_kind,
            view_index=self.view_index,
            seed=int(seed),
            n_endpoint_valid_assets=int(row["n_endpoint_valid_assets"]),
            validation_window_name=str(row["validation_window_name"]),
        )
