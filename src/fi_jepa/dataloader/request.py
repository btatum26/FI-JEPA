from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fi_jepa.dataloader.panel_store import Split

RequestKind = Literal["jepa", "embedding"]
ViewKind = Literal["random_k", "all_valid", "fixed_k"]


# ============================================================================
# DENSE PANEL WINDOW REQUEST
# ============================================================================


@dataclass(frozen=True)
class DensePanelWindowRequest:
    """Carry only metadata required to gather one dense-panel window."""

    sample_date_idx: int
    sample_date: str
    split: Split
    request_kind: RequestKind
    view_kind: ViewKind
    view_index: int
    seed: int
    n_endpoint_valid_assets: int
    validation_window_name: str
