from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fi_jepa.dataloader.panel_store import Split

RequestKind = Literal["jepa", "embedding"]
ViewKind = Literal["random_k", "all_valid", "fixed_k"]


# ============================================================================
# WINDOW REQUEST
# ============================================================================


@dataclass(frozen=True)
class WindowRequest:
    """Describe one runtime window without materializing any panel arrays.

    The request carries only deterministic sampling inputs and metadata. Dense
    values, selected asset IDs, patch masks, and JEPA masks are intentionally
    deferred to the batch assembler.
    """

    sample_date_idx: int
    sample_date: str
    split: Split
    request_kind: RequestKind
    view_kind: ViewKind
    view_index: int
    seed: int
