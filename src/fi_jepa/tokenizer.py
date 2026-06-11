from __future__ import annotations

import torch
from torch import nn


# ============================================================================
# MASKED TENSOR OPERATIONS
# ============================================================================


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dimension: int) -> torch.Tensor:
    """Average a tensor dimension while excluding invalid entries.

    ``mask`` must match ``values`` except for the final feature dimension. The
    added singleton dimension broadcasts validity across features, and the
    clamped denominator makes an entirely invalid slice return zeros.
    """
    # [..., N] -> [..., N, 1], aligned with the final feature axis in values.
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    numerator = (values * weights).sum(dim=dimension)
    denominator = weights.sum(dim=dimension).clamp_min(1.0)
    return numerator / denominator


def pack_masked_sequence(
    tokens: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack chronological valid tokens into a padded Transformer batch.

    Args:
        tokens: Full token sequence shaped ``[B, P, D]``.
        mask: Valid positions shaped ``[B, P]``.

    Returns:
        Packed tokens shaped ``[B, C_max, D]`` and their validity mask shaped
        ``[B, C_max]``, where ``C_max`` is the largest valid count in the batch.
    """
    # [B, P] -> [B], one visible-context count per sample.
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every sample must contain at least one visible context patch.")

    batch_size, sequence_length, token_dim = tokens.shape
    max_count = int(counts.max().item())

    # Invalid positions receive an out-of-range sentinel so sorting moves them
    # behind every chronological valid patch ID.
    patch_ids = torch.arange(sequence_length, device=tokens.device).expand(batch_size, -1)
    ranked_ids = torch.where(mask, patch_ids, sequence_length).sort(dim=1).values
    gather_ids = ranked_ids[:, :max_count].clamp_max(sequence_length - 1)

    # [B, P, D] -> [B, C_max, D].
    packed = tokens.gather(1, gather_ids.unsqueeze(-1).expand(-1, -1, token_dim))
    packed_mask = torch.arange(max_count, device=tokens.device).unsqueeze(0) < counts.unsqueeze(1)
    packed = torch.where(packed_mask.unsqueeze(-1), packed, torch.zeros_like(packed))
    return packed, packed_mask


# ============================================================================
# PATCH TOKENIZATION
# ============================================================================


class MaskedPatchTokenizer(nn.Module):
    """Compress one stream's feature-masked daily patch into one token.

    The tokenizer accepts arbitrary leading dimensions followed by ``[L, F]``.
    It concatenates feature-validity bits to zero-filled values, projects each
    day independently, averages only valid days, and projects the pooled patch
    representation. No dropout is used because this shared path feeds both the
    online and EMA target branches.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        self.feature_dim = feature_dim
        self.daily_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        values: torch.Tensor,
        feature_mask: torch.Tensor,
        day_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenize feature-masked daily values into one vector per patch.

        Shapes:
            values: ``[..., L, F]``.
            feature_mask: ``[..., L, F]``.
            day_mask: ``[..., L]``.
            return: ``[..., D_token]``.
        """
        # Invalid feature values are forced to zero before any learned layer.
        valid_values = torch.where(feature_mask, values, torch.zeros_like(values))

        # [..., L, F] + [..., L, F] -> [..., L, 2F].
        daily_input = torch.cat((valid_values, feature_mask.to(values.dtype)), dim=-1)
        daily_tokens = self.daily_projection(daily_input)  # [..., L, H].
        patch_tokens = masked_mean(daily_tokens, day_mask, dimension=-2)  # [..., H].
        return self.output_projection(patch_tokens)  # [..., D_token].
