from __future__ import annotations

from dataclasses import dataclass

import torch


# ============================================================================
# MODEL OUTPUT
# ============================================================================


@dataclass(frozen=True)
class FIJepaOutput:
    """Expose loss, representations, masks, and shared tokens from one pass."""

    loss: torch.Tensor
    predicted_targets: torch.Tensor
    target_representations: torch.Tensor
    target_patch_mask: torch.Tensor
    context_representations: torch.Tensor
    context_mask: torch.Tensor
    fused_tokens: torch.Tensor
