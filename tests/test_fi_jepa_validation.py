from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fi_jepa.dataloader.validation import (
    validate_batched_patch_mask_inputs,
    validate_cache,
    validate_data_config,
    validate_request_batch,
    validate_required_artifact_files,
)
from fi_jepa.model_validation import (
    validate_attention_tokenizer_inputs,
    validate_model_batch,
    validate_model_config,
)


# ============================================================================
# DATALOADER VALIDATION
# ============================================================================


def test_dataloader_validators_reject_invalid_boundaries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divisible"):
        validate_data_config(
            lookback_days=5,
            patch_len=2,
            mask_ratio=0.5,
            min_masked_patches=1,
            max_masked_patches=2,
            min_target_blocks=1,
            max_target_blocks=1,
            min_valid_days_per_asset_patch=1,
            min_valid_dates_in_patch=1,
            min_valid_asset_fraction=0.5,
            train_k_assets=1,
            fixed_k_assets=1,
            batch_size=1,
            validation_batch_size=1,
            num_workers=0,
        )
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        validate_required_artifact_files(tmp_path, {"manifest.json"})
    assert not validate_cache(tmp_path, {}, ("dates",), ("config_resolved.yaml",))


def test_request_and_patch_mask_validators_reject_malformed_inputs() -> None:
    first = SimpleNamespace(split="train", request_kind="jepa", view_kind="random_k")
    second = SimpleNamespace(split="validation", request_kind="jepa", view_kind="random_k")
    with pytest.raises(ValueError, match="homogeneous"):
        validate_request_batch([first, second], SimpleNamespace)
    with pytest.raises(ValueError, match="valid_date_mask"):
        validate_batched_patch_mask_inputs(
            np.ones((1, 4, 2), dtype=bool),
            np.ones((1, 3), dtype=bool),
            np.ones((1, 4), dtype=bool),
            2,
        )


# ============================================================================
# MODEL VALIDATION
# ============================================================================


def _valid_model_batch() -> dict[str, torch.Tensor]:
    """Return the smallest complete model batch accepted by the ABI validator."""
    return {
        "asset_patches": torch.zeros((1, 2, 2, 1, 1)),
        "market_patches": torch.zeros((1, 2, 2, 1)),
        "macro_patches": torch.zeros((1, 2, 2, 1)),
        "asset_feature_mask_patched": torch.ones((1, 2, 2, 1, 1), dtype=torch.bool),
        "market_feature_mask_patched": torch.ones((1, 2, 2, 1), dtype=torch.bool),
        "macro_feature_mask_patched": torch.ones((1, 2, 2, 1), dtype=torch.bool),
        "valid_asset_mask_patched": torch.ones((1, 2, 2, 1), dtype=torch.bool),
        "valid_market_date_mask_patched": torch.ones((1, 2, 2), dtype=torch.bool),
        "valid_macro_date_mask_patched": torch.ones((1, 2, 2), dtype=torch.bool),
        "patch_asset_mask": torch.ones((1, 2, 1), dtype=torch.bool),
        "patch_context_mask": torch.ones((1, 2), dtype=torch.bool),
        "target_patch_ids": torch.tensor([[1]]),
        "target_patch_id_mask": torch.ones((1, 1), dtype=torch.bool),
        "jepa_context_mask": torch.tensor([[True, False]]),
    }


def test_model_validators_accept_contract_and_reject_drift() -> None:
    batch = _valid_model_batch()
    tensors = validate_model_batch(
        batch,
        num_patches=2,
        patch_len=2,
        asset_feature_dim=1,
        market_feature_dim=1,
        macro_feature_dim=1,
    )
    assert tensors["asset_patches"] is batch["asset_patches"]

    batch["target_patch_id_mask"] = batch["target_patch_id_mask"].float()
    with pytest.raises(ValueError, match="target_patch_id_mask must have dtype bool"):
        validate_model_batch(
            batch,
            num_patches=2,
            patch_len=2,
            asset_feature_dim=1,
            market_feature_dim=1,
            macro_feature_dim=1,
        )


def test_model_config_and_tokenizer_validators_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="d_model must be divisible"):
        validate_model_config(
            integer_fields={"d_model": 5, "context_heads": 2},
            tokenizer_type="mean",
            asset_pooling_type="mean",
            d_model=5,
            context_heads=2,
            predictor_heads=1,
            tokenizer_heads=1,
            tokenizer_hidden_dims={},
            asset_token_dim=1,
            asset_pooling_heads=1,
            context_dropout=0.0,
            predictor_dropout=0.0,
        )
    with pytest.raises(ValueError, match="day_mask must have shape"):
        validate_attention_tokenizer_inputs(
            torch.zeros((1, 2, 3)),
            torch.ones((1, 2, 3), dtype=torch.bool),
            torch.ones((1, 1), dtype=torch.bool),
            feature_dim=3,
            patch_len=2,
        )
