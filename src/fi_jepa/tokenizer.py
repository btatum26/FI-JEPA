from __future__ import annotations

import torch
from torch import nn

from fi_jepa.model_validation import (
    validate_asset_pooler_config,
    validate_asset_pooler_inputs,
    validate_attention_tokenizer_config,
    validate_attention_tokenizer_inputs,
    validate_model_feature_dimensions,
)


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
    """Compress one stream's feature-masked daily patch with masked mean pooling.

    The tokenizer accepts arbitrary leading dimensions followed by ``[L, F]``.
    It concatenates feature-validity bits to zero-filled values, projects each
    day independently, averages only valid days, and projects the pooled patch
    representation. No dropout is used so the independently parameterized
    online and EMA target copies remain deterministic.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        validate_model_feature_dimensions({"feature_dim": feature_dim})
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
        """Tokenize feature-masked daily values into one mean-pooled patch token.

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


class MaskedTransformerBlock(nn.Module):
    """Apply one dropout-free pre-norm Transformer block to a masked sequence."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Transform ``[N, S, H]`` tokens while ignoring padded key/value positions."""
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.mlp_norm(tokens))


class MaskedAttentionPatchTokenizer(nn.Module):
    """Compress one stream's feature-masked daily patch with a Transformer.

    The tokenizer accepts arbitrary leading dimensions followed by ``[L, F]``.
    It concatenates feature-validity bits to zero-filled values, projects each
    day independently, prepends a learned summary token, and applies masked
    self-attention across the ordered days. The encoded summary becomes the
    patch representation. No dropout is used so the independently parameterized
    online and EMA target copies remain deterministic.
    """

    def __init__(
        self,
        feature_dim: int,
        patch_len: int,
        hidden_dim: int,
        output_dim: int,
        *,
        layers: int,
        heads: int,
        mlp_ratio: int,
    ):
        super().__init__()
        dimensions = {
            "feature_dim": feature_dim,
            "patch_len": patch_len,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "layers": layers,
            "heads": heads,
            "mlp_ratio": mlp_ratio,
        }
        validate_attention_tokenizer_config(dimensions, hidden_dim, heads)

        self.feature_dim = feature_dim
        self.patch_len = patch_len
        self.output_dim = output_dim
        self.daily_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.summary_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.empty(1, patch_len + 1, hidden_dim))
        self.transformer = nn.ModuleList(
            [MaskedTransformerBlock(hidden_dim, heads, mlp_ratio) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

        nn.init.normal_(self.summary_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

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
        validate_attention_tokenizer_inputs(
            values,
            feature_mask,
            day_mask,
            feature_dim=self.feature_dim,
            patch_len=self.patch_len,
        )

        # Invalid feature values are forced to zero before any learned layer.
        valid_values = torch.where(feature_mask, values, torch.zeros_like(values))

        # [..., L, F] + [..., L, F] -> [..., L, 2F].
        daily_input = torch.cat((valid_values, feature_mask.to(values.dtype)), dim=-1)
        daily_tokens = self.daily_projection(daily_input)  # [..., L, H].

        # Flatten arbitrary leading dimensions so attention operates on one
        # independent patch sequence at a time: [..., L, H] -> [N, L, H].
        leading_shape = daily_tokens.shape[:-2]
        flat_daily_tokens = daily_tokens.reshape(-1, self.patch_len, daily_tokens.shape[-1])
        flat_day_mask = day_mask.reshape(-1, self.patch_len)
        summary_tokens = self.summary_token.expand(flat_daily_tokens.shape[0], -1, -1)
        sequence = torch.cat((summary_tokens, flat_daily_tokens), dim=1)
        sequence = sequence + self.position_embedding

        # The summary token is always visible. Invalid days cannot contribute
        # keys or values to its learned patch representation.
        summary_padding = torch.zeros(
            (flat_day_mask.shape[0], 1),
            dtype=torch.bool,
            device=flat_day_mask.device,
        )
        padding_mask = torch.cat((summary_padding, ~flat_day_mask), dim=1)
        for block in self.transformer:
            sequence = block(sequence, padding_mask)

        patch_tokens = self.output_norm(sequence[:, 0])  # [N, H].
        output = self.output_projection(patch_tokens)  # [N, D_token].
        return output.reshape(*leading_shape, self.output_dim)


# ============================================================================
# CROSS-ASSET POOLING
# ============================================================================


class MaskedAttentionPoolingBlock(nn.Module):
    """Update one summary query by attending over a masked variable-size set."""

    def __init__(self, token_dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(token_dim)
        self.asset_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(token_dim * mlp_ratio, token_dim),
        )

    def forward(
        self,
        summary: torch.Tensor,
        asset_tokens: torch.Tensor,
        asset_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply linear-complexity cross-attention from ``[N, 1, D]`` into assets."""
        normalized_summary = self.query_norm(summary)
        normalized_assets = self.asset_norm(asset_tokens)

        # Including the summary as an always-valid key/value keeps empty asset
        # sets numerically defined while valid assets remain mask-controlled.
        key_values = torch.cat((normalized_summary, normalized_assets), dim=1)
        summary_padding = torch.zeros(
            (asset_padding_mask.shape[0], 1),
            dtype=torch.bool,
            device=asset_padding_mask.device,
        )
        key_padding_mask = torch.cat((summary_padding, asset_padding_mask), dim=1)
        attended, _ = self.attention(
            normalized_summary,
            key_values,
            key_values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        summary = summary + attended
        return summary + self.mlp(self.mlp_norm(summary))


class MaskedAttentionAssetPooler(nn.Module):
    """Pool a variable valid-asset set with permutation-invariant attention.

    The pooler accepts arbitrary leading dimensions followed by ``[A, D]``.
    A learned summary query applies masked cross-attention over asset tokens
    with no asset positional embeddings, preserving invariance to asset
    ordering while scaling linearly with asset count. Invalid asset slots
    cannot contribute keys or values. No dropout is used so the independently
    parameterized online and EMA target copies remain deterministic.
    """

    def __init__(self, token_dim: int, *, layers: int, heads: int, mlp_ratio: int):
        super().__init__()
        dimensions = {
            "token_dim": token_dim,
            "layers": layers,
            "heads": heads,
            "mlp_ratio": mlp_ratio,
        }
        validate_asset_pooler_config(dimensions, token_dim, heads)

        self.token_dim = token_dim
        self.summary_token = nn.Parameter(torch.empty(1, 1, token_dim))
        self.blocks = nn.ModuleList(
            [MaskedAttentionPoolingBlock(token_dim, heads, mlp_ratio) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(token_dim)
        nn.init.normal_(self.summary_token, std=0.02)

    def forward(self, asset_tokens: torch.Tensor, asset_mask: torch.Tensor) -> torch.Tensor:
        """Pool valid assets into one panel token.

        Shapes:
            asset_tokens: ``[..., A, D]``.
            asset_mask: ``[..., A]``.
            return: ``[..., D]``.
        """
        validate_asset_pooler_inputs(asset_tokens, asset_mask, token_dim=self.token_dim)

        # [..., A, D] -> [N, A, D], one independent asset set per patch.
        leading_shape = asset_tokens.shape[:-2]
        asset_count = asset_tokens.shape[-2]
        flat_tokens = asset_tokens.reshape(-1, asset_count, self.token_dim)
        flat_mask = asset_mask.reshape(-1, asset_count)
        summary = self.summary_token.expand(flat_tokens.shape[0], -1, -1)
        for block in self.blocks:
            summary = block(summary, flat_tokens, ~flat_mask)

        pooled = self.output_norm(summary[:, 0])  # [N, D].
        return pooled.reshape(*leading_shape, self.token_dim)
