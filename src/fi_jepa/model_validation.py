from __future__ import annotations

from typing import Mapping

import torch

ENCODER_BATCH_TENSOR_NAMES = frozenset(
    {
        "asset_patches",
        "market_patches",
        "macro_patches",
        "asset_feature_mask_patched",
        "market_feature_mask_patched",
        "macro_feature_mask_patched",
        "valid_asset_mask_patched",
        "valid_market_date_mask_patched",
        "valid_macro_date_mask_patched",
        "patch_asset_mask",
        "patch_context_mask",
    }
)
JEPA_BATCH_TENSOR_NAMES = frozenset(
    {
        "target_patch_ids",
        "target_patch_id_mask",
        "jepa_context_mask",
    }
)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================


def validate_model_config(
    *,
    integer_fields: Mapping[str, int],
    tokenizer_type: str,
    asset_pooling_type: str,
    d_model: int,
    context_heads: int,
    predictor_heads: int,
    tokenizer_heads: int,
    tokenizer_hidden_dims: Mapping[str, int],
    asset_token_dim: int,
    asset_pooling_heads: int,
    context_dropout: float,
    predictor_dropout: float,
) -> None:
    """Validate model dimensions, attention widths, and dropout rates."""
    invalid = [name for name, value in integer_fields.items() if value <= 0]
    if invalid:
        raise ValueError(f"Model dimensions and counts must be positive: {invalid}")
    if tokenizer_type not in {"mean", "attention"}:
        raise ValueError("tokenizer_type must be 'mean' or 'attention'.")
    if asset_pooling_type not in {"mean", "attention"}:
        raise ValueError("asset_pooling_type must be 'mean' or 'attention'.")
    if d_model % context_heads:
        raise ValueError("d_model must be divisible by context_heads.")
    if d_model % predictor_heads:
        raise ValueError("d_model must be divisible by predictor_heads.")
    if tokenizer_type == "attention":
        incompatible = [
            name
            for name, hidden_dim in tokenizer_hidden_dims.items()
            if hidden_dim % tokenizer_heads
        ]
        if incompatible:
            raise ValueError(
                f"Tokenizer hidden dimensions must be divisible by tokenizer_heads: {incompatible}"
            )
    if asset_pooling_type == "attention" and asset_token_dim % asset_pooling_heads:
        raise ValueError("asset_token_dim must be divisible by asset_pooling_heads.")
    if not 0.0 <= context_dropout < 1.0:
        raise ValueError("context_dropout must be in [0, 1).")
    if not 0.0 <= predictor_dropout < 1.0:
        raise ValueError("predictor_dropout must be in [0, 1).")


def validate_model_yaml(values: object) -> dict[str, object]:
    """Validate the top-level model YAML contract and return its mapping."""
    if not isinstance(values, dict):
        raise ValueError("Model configuration must be a YAML mapping.")
    allowed_sections = {
        "input",
        "tokenizers",
        "asset_pooling",
        "fusion",
        "context_encoder",
        "predictor",
    }
    required_sections = {
        "input",
        "tokenizers",
        "fusion",
        "context_encoder",
        "predictor",
    }
    missing = sorted(required_sections - set(values))
    if missing:
        raise ValueError(f"Model configuration is missing sections: {missing}")
    unknown = sorted(set(values) - allowed_sections)
    if unknown:
        raise ValueError(f"Model configuration contains unknown sections: {unknown}")

    tokenizers = values["tokenizers"]
    tokenizer_type = str(tokenizers.get("type", "mean"))
    if tokenizer_type == "attention" and not tokenizers.get("attention"):
        raise ValueError("Attention tokenizer configuration is missing tokenizers.attention.")
    asset_pooling = values.get("asset_pooling") or {}
    asset_pooling_type = str(asset_pooling.get("type", "mean"))
    if asset_pooling_type == "attention" and not asset_pooling.get("attention"):
        raise ValueError(
            "Attention asset pooling configuration is missing asset_pooling.attention."
        )
    if float(values["fusion"].get("dropout", 0.0)) != 0.0:
        raise ValueError(
            "Online and target fusion dropout must remain 0.0 for deterministic inputs."
        )
    return values


def validate_positive_dimensions(dimensions: Mapping[str, int], message_prefix: str) -> None:
    """Require every named constructor dimension or count to be positive."""
    invalid = [name for name, value in dimensions.items() if value <= 0]
    if invalid:
        raise ValueError(f"{message_prefix} dimensions and counts must be positive: {invalid}")


def validate_model_feature_dimensions(dimensions: Mapping[str, int]) -> None:
    """Require positive model input feature dimensions."""
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")


# ============================================================================
# TOKENIZER AND POOLER CONTRACTS
# ============================================================================


def validate_attention_tokenizer_config(
    dimensions: Mapping[str, int], hidden_dim: int, heads: int
) -> None:
    """Validate attention-tokenizer dimensions and head divisibility."""
    validate_positive_dimensions(dimensions, "Tokenizer")
    if hidden_dim % heads:
        raise ValueError("Tokenizer hidden_dim must be divisible by heads.")


def validate_attention_tokenizer_inputs(
    values: torch.Tensor,
    feature_mask: torch.Tensor,
    day_mask: torch.Tensor,
    *,
    feature_dim: int,
    patch_len: int,
) -> None:
    """Validate one attention-tokenizer call without transforming tensors."""
    expected_feature_shape = (*values.shape[:-1], feature_dim)
    expected_day_shape = values.shape[:-1]
    if tuple(values.shape) != expected_feature_shape:
        raise ValueError(
            f"Tokenizer values must end in feature_dim={feature_dim}; "
            f"got shape {tuple(values.shape)}."
        )
    if values.shape[-2] != patch_len:
        raise ValueError(
            f"Tokenizer values must use patch_len={patch_len}; got shape {tuple(values.shape)}."
        )
    if tuple(feature_mask.shape) != tuple(values.shape):
        raise ValueError(
            "Tokenizer feature_mask must match values; "
            f"got {tuple(feature_mask.shape)} and {tuple(values.shape)}."
        )
    if tuple(day_mask.shape) != expected_day_shape:
        raise ValueError(
            f"Tokenizer day_mask must have shape {expected_day_shape}; got {tuple(day_mask.shape)}."
        )
    if feature_mask.dtype != torch.bool or day_mask.dtype != torch.bool:
        raise ValueError("Tokenizer feature_mask and day_mask must have dtype bool.")


def validate_asset_pooler_config(dimensions: Mapping[str, int], token_dim: int, heads: int) -> None:
    """Validate attention asset-pooler dimensions and head divisibility."""
    validate_positive_dimensions(dimensions, "Asset pooler")
    if token_dim % heads:
        raise ValueError("Asset pooler token_dim must be divisible by heads.")


def validate_asset_pooler_inputs(
    asset_tokens: torch.Tensor,
    asset_mask: torch.Tensor,
    *,
    token_dim: int,
) -> None:
    """Validate one attention asset-pooler call without transforming tensors."""
    if asset_tokens.shape[-1] != token_dim:
        raise ValueError(
            f"Asset tokens must end in token_dim={token_dim}; "
            f"got shape {tuple(asset_tokens.shape)}."
        )
    if tuple(asset_mask.shape) != tuple(asset_tokens.shape[:-1]):
        raise ValueError(
            f"Asset mask must have shape {tuple(asset_tokens.shape[:-1])}; "
            f"got {tuple(asset_mask.shape)}."
        )
    if asset_mask.dtype != torch.bool:
        raise ValueError("Asset mask must have dtype bool.")


# ============================================================================
# MODEL BATCH CONTRACT
# ============================================================================


def validate_model_batch(
    batch: dict[str, object],
    *,
    num_patches: int,
    patch_len: int,
    asset_feature_dim: int,
    market_feature_dim: int,
    macro_feature_dim: int,
    require_jepa_targets: bool = True,
) -> dict[str, torch.Tensor]:
    """Validate the complete patched-batch ABI before model computation."""
    required = (
        ENCODER_BATCH_TENSOR_NAMES | JEPA_BATCH_TENSOR_NAMES
        if require_jepa_targets
        else ENCODER_BATCH_TENSOR_NAMES
    )
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"FI-JEPA batch is missing required keys: {missing}")

    tensors: dict[str, torch.Tensor] = {}
    for name in required:
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a Tensor; got {type(value).__name__}.")
        tensors[name] = value

    expected_ranks = {
        "asset_patches": 5,
        "market_patches": 4,
        "macro_patches": 4,
        "asset_feature_mask_patched": 5,
        "market_feature_mask_patched": 4,
        "macro_feature_mask_patched": 4,
        "valid_asset_mask_patched": 4,
        "valid_market_date_mask_patched": 3,
        "valid_macro_date_mask_patched": 3,
        "patch_asset_mask": 3,
        "patch_context_mask": 2,
    }
    if require_jepa_targets:
        expected_ranks.update(
            {"target_patch_ids": 2, "target_patch_id_mask": 2, "jepa_context_mask": 2}
        )
    for name, rank in expected_ranks.items():
        if tensors[name].ndim != rank:
            raise ValueError(
                f"{name} must have rank {rank}; got shape {tuple(tensors[name].shape)}."
            )

    asset_shape = tuple(tensors["asset_patches"].shape)
    batch_size, actual_num_patches, actual_patch_len, num_assets, _ = asset_shape
    expected_shapes = {
        "asset_patches": (batch_size, num_patches, patch_len, num_assets, asset_feature_dim),
        "market_patches": (batch_size, num_patches, patch_len, market_feature_dim),
        "macro_patches": (batch_size, num_patches, patch_len, macro_feature_dim),
        "asset_feature_mask_patched": asset_shape,
        "market_feature_mask_patched": (
            batch_size,
            actual_num_patches,
            actual_patch_len,
            market_feature_dim,
        ),
        "macro_feature_mask_patched": (
            batch_size,
            actual_num_patches,
            actual_patch_len,
            macro_feature_dim,
        ),
        "valid_asset_mask_patched": (batch_size, actual_num_patches, actual_patch_len, num_assets),
        "valid_market_date_mask_patched": (batch_size, actual_num_patches, actual_patch_len),
        "valid_macro_date_mask_patched": (batch_size, actual_num_patches, actual_patch_len),
        "patch_asset_mask": (batch_size, actual_num_patches, num_assets),
        "patch_context_mask": (batch_size, actual_num_patches),
    }
    if require_jepa_targets:
        target_count = tensors["target_patch_ids"].shape[1]
        expected_shapes.update(
            {
                "target_patch_ids": (batch_size, target_count),
                "target_patch_id_mask": (batch_size, target_count),
                "jepa_context_mask": (batch_size, actual_num_patches),
            }
        )
    for name, expected in expected_shapes.items():
        actual = tuple(tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}; got {actual}.")

    for name in ("asset_patches", "market_patches", "macro_patches"):
        if not tensors[name].is_floating_point():
            raise ValueError(f"{name} must be floating point; got {tensors[name].dtype}.")
    mask_names = required - {"asset_patches", "market_patches", "macro_patches", "target_patch_ids"}
    for name in mask_names:
        if tensors[name].dtype != torch.bool:
            raise ValueError(f"{name} must have dtype bool; got {tensors[name].dtype}.")
    if require_jepa_targets and tensors["target_patch_ids"].dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError(
            f"target_patch_ids must have an integer dtype; got {tensors['target_patch_ids'].dtype}."
        )

    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError(f"All FI-JEPA batch tensors must share one device; got {devices}.")
    if batch_size <= 0 or num_assets <= 0:
        raise ValueError("Batch and asset dimensions must be positive.")

    patch_context = tensors["patch_context_mask"]
    if not patch_context.any(dim=1).all():
        raise ValueError("patch_context_mask must enable at least one patch per sample.")
    if not require_jepa_targets:
        return tensors

    jepa_context = tensors["jepa_context_mask"]
    target_ids = tensors["target_patch_ids"]
    target_mask = tensors["target_patch_id_mask"]
    target_count = target_ids.shape[1]
    if target_count <= 0:
        raise ValueError("Target dimension must be positive.")
    if not jepa_context.any(dim=1).all():
        raise ValueError("jepa_context_mask must enable at least one patch per sample.")
    if (jepa_context & ~patch_context).any():
        raise ValueError("jepa_context_mask must be a subset of patch_context_mask.")
    if not target_mask.any(dim=1).all():
        raise ValueError("target_patch_id_mask must enable at least one target per sample.")
    if (target_ids[~target_mask] != -1).any():
        raise ValueError("Disabled target_patch_ids must use the -1 padding sentinel.")
    enabled_ids = target_ids[target_mask]
    if ((enabled_ids < 0) | (enabled_ids >= actual_num_patches)).any():
        raise ValueError(
            f"Enabled target_patch_ids must be within [0, {actual_num_patches}); "
            f"got {enabled_ids.tolist()}."
        )

    safe_ids = target_ids.clamp_min(0)
    target_is_context = patch_context.gather(1, safe_ids)
    target_is_visible = jepa_context.gather(1, safe_ids)
    if not target_is_context[target_mask].all():
        raise ValueError("Every enabled target_patch_id must reference a context-valid patch.")
    if target_is_visible[target_mask].any():
        raise ValueError("Enabled target patches cannot also appear in jepa_context_mask.")

    target_positions = torch.zeros_like(patch_context)
    for row_index, (row_ids, row_mask) in enumerate(zip(target_ids, target_mask, strict=True)):
        selected = row_ids[row_mask]
        if selected.unique().numel() != selected.numel():
            raise ValueError("Enabled target_patch_ids must be unique within each sample.")
        target_positions[row_index, selected] = True
    if not torch.equal(jepa_context, patch_context & ~target_positions):
        raise ValueError(
            "jepa_context_mask must equal patch_context_mask with enabled targets removed."
        )
    return tensors
