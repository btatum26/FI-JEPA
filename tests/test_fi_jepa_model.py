from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fi_jepa.model import FIJepaModel
from fi_jepa.model_config import FIJepaModelConfig


# ============================================================================
# SYNTHETIC MODEL INPUTS
# ============================================================================


def _small_config() -> FIJepaModelConfig:
    return FIJepaModelConfig(
        patch_len=2,
        num_patches=4,
        asset_hidden_dim=6,
        asset_token_dim=8,
        market_hidden_dim=4,
        market_token_dim=4,
        macro_hidden_dim=5,
        macro_token_dim=4,
        d_model=8,
        latent_dim=3,
        context_layers=1,
        context_heads=2,
        context_mlp_ratio=2,
        context_dropout=0.2,
        predictor_layers=1,
        predictor_heads=2,
        predictor_mlp_ratio=2,
        predictor_dropout=0.2,
    )


def _model() -> FIJepaModel:
    torch.manual_seed(7)
    return FIJepaModel(_small_config(), 2, 2, 3)


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    batch_size, patches, patch_len, assets = 2, 4, 2, 3
    asset_patches = torch.randn(batch_size, patches, patch_len, assets, 2, generator=generator)
    market_patches = torch.randn(batch_size, patches, patch_len, 2, generator=generator)
    macro_patches = torch.randn(batch_size, patches, patch_len, 3, generator=generator)
    asset_feature_mask = torch.ones_like(asset_patches, dtype=torch.bool)
    market_feature_mask = torch.ones_like(market_patches, dtype=torch.bool)
    macro_feature_mask = torch.ones_like(macro_patches, dtype=torch.bool)
    valid_asset_mask = torch.ones(batch_size, patches, patch_len, assets, dtype=torch.bool)
    valid_market_mask = torch.ones(batch_size, patches, patch_len, dtype=torch.bool)
    valid_macro_mask = torch.ones(batch_size, patches, patch_len, dtype=torch.bool)
    patch_asset_mask = torch.ones(batch_size, patches, assets, dtype=torch.bool)

    # Asset slot two is padding in both samples. Sample one also has a trailing
    # invalid patch so the exporter must not blindly use the final position.
    asset_feature_mask[:, :, :, 2] = False
    valid_asset_mask[:, :, :, 2] = False
    patch_asset_mask[:, :, 2] = False
    asset_feature_mask[1, 3] = False
    market_feature_mask[1, 3] = False
    macro_feature_mask[1, 3] = False
    valid_asset_mask[1, 3] = False
    valid_market_mask[1, 3] = False
    valid_macro_mask[1, 3] = False
    patch_asset_mask[1, 3] = False

    return {
        "asset_patches": asset_patches,
        "market_patches": market_patches,
        "macro_patches": macro_patches,
        "asset_feature_mask_patched": asset_feature_mask,
        "market_feature_mask_patched": market_feature_mask,
        "macro_feature_mask_patched": macro_feature_mask,
        "valid_asset_mask_patched": valid_asset_mask,
        "valid_market_date_mask_patched": valid_market_mask,
        "valid_macro_date_mask_patched": valid_macro_mask,
        "patch_asset_mask": patch_asset_mask,
        "patch_context_mask": torch.tensor([[True, True, True, True], [True, True, True, False]]),
        "target_patch_ids": torch.tensor([[1, 3], [0, -1]]),
        "target_patch_id_mask": torch.tensor([[True, True], [True, False]]),
        "jepa_context_mask": torch.tensor([[True, False, True, False], [False, True, True, False]]),
    }


# ============================================================================
# CONFIGURATION AND BATCH CONTRACT
# ============================================================================


def test_model_config_loads_and_model_derives_feature_dimensions_from_store() -> None:
    config = FIJepaModelConfig.from_yaml(Path("configs/model.yaml"))
    store = SimpleNamespace(
        feature_names={
            "asset": [f"a{index}" for index in range(22)],
            "market": [f"m{index}" for index in range(5)],
            "macro": [f"x{index}" for index in range(33)],
        }
    )
    model = FIJepaModel.from_store(config, store)

    assert config.num_patches == 12
    assert model.asset_feature_dim == 22
    assert model.market_feature_dim == 5
    assert model.macro_feature_dim == 33


def test_model_config_rejects_shared_fusion_dropout(tmp_path: Path) -> None:
    config = Path("configs/model.yaml").read_text(encoding="utf-8")
    path = tmp_path / "model.yaml"
    path.write_text(config.replace("dropout: 0.0", "dropout: 0.1", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="fusion dropout"):
        FIJepaModelConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda batch: batch.pop("macro_patches"), "missing required keys"),
        (
            lambda batch: batch.__setitem__("asset_patches", batch["asset_patches"][..., :1]),
            "asset_patches must have shape",
        ),
        (
            lambda batch: batch.__setitem__(
                "target_patch_id_mask", batch["target_patch_id_mask"].float()
            ),
            "target_patch_id_mask must have dtype bool",
        ),
        (
            lambda batch: batch["target_patch_ids"].__setitem__((0, 0), 4),
            "Enabled target_patch_ids must be within",
        ),
    ],
)
def test_model_rejects_batch_contract_changes(mutation: object, message: str) -> None:
    model = _model()
    batch = _batch()
    mutation(batch)

    with pytest.raises(ValueError, match=message):
        model(batch)


# ============================================================================
# TOKENIZATION AND FORWARD PASS
# ============================================================================


def test_forward_shapes_loss_masking_and_target_gradient_boundary() -> None:
    model = _model()
    model.train()
    batch = _batch()
    output = model(batch)

    assert output.fused_tokens.shape == (2, 4, 8)
    assert output.context_representations.shape == (2, 2, 8)
    assert output.context_mask.shape == (2, 2)
    assert output.predicted_targets.shape == (2, 2, 8)
    assert output.target_representations.shape == (2, 2, 8)
    assert output.target_patch_mask.shape == (2, 2)
    assert torch.isfinite(output.loss)
    assert not output.target_representations.requires_grad
    assert not output.predicted_targets[1, 1].any()
    assert not output.target_representations[1, 1].any()

    expected = (
        (
            F.normalize(output.predicted_targets, dim=-1)
            - F.normalize(output.target_representations, dim=-1)
        )
        .square()
        .sum(dim=-1)[output.target_patch_mask]
        .mean()
    )
    assert torch.allclose(output.loss, expected)

    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.context_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_shared_tokenizers_and_fusion_are_dropout_free_and_deterministic() -> None:
    model = _model()
    model.train()
    batch = _batch()
    tensors = model._validate_batch(batch)

    shared_modules = [
        model.asset_tokenizer,
        model.market_tokenizer,
        model.macro_tokenizer,
        model.fusion,
    ]
    assert not any(
        isinstance(module, nn.Dropout) for shared in shared_modules for module in shared.modules()
    )
    first = model._tokenize_and_fuse(tensors)
    second = model._tokenize_and_fuse(tensors)
    assert torch.equal(first, second)


def test_invalid_feature_and_padded_asset_values_cannot_change_fused_tokens() -> None:
    model = _model()
    model.eval()
    batch = _batch()
    changed = deepcopy(batch)
    changed["asset_patches"][~changed["asset_feature_mask_patched"]] = 1_000_000.0
    changed["market_patches"][~changed["market_feature_mask_patched"]] = -1_000_000.0
    changed["macro_patches"][~changed["macro_feature_mask_patched"]] = 500_000.0

    original_tokens = model._tokenize_and_fuse(model._validate_batch(batch))
    changed_tokens = model._tokenize_and_fuse(model._validate_batch(changed))
    assert torch.equal(original_tokens, changed_tokens)


def test_target_representation_uses_full_context_sequence() -> None:
    model = _model()
    model.eval()
    batch = _batch()
    changed = deepcopy(batch)
    changed["asset_patches"][0, 0] *= 20.0
    changed["market_patches"][0, 0] *= -15.0
    changed["macro_patches"][0, 0] += 8.0

    original_target = model(batch).target_representations[0, 0]
    changed_target = model(changed).target_representations[0, 0]
    assert not torch.allclose(original_target, changed_target)


# ============================================================================
# EMA AND STATE EXPORT
# ============================================================================


def test_target_encoder_stays_frozen_in_eval_mode_and_updates_by_ema() -> None:
    model = _model()
    model.train()
    target_parameter = next(model.target_encoder.parameters())
    online_parameter = next(model.context_encoder.parameters())
    original_target = target_parameter.detach().clone()

    assert not model.target_encoder.training
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    with torch.no_grad():
        online_parameter.add_(2.0)
    expected = original_target * 0.25 + online_parameter.detach() * 0.75
    model.update_target_encoder(0.25)

    assert torch.allclose(target_parameter, expected)
    assert not model.target_encoder.training


def test_encode_pooled_state_is_source_of_truth_and_requires_endpoint_patch() -> None:
    model = _model()
    model.eval()
    batch = _batch()
    with pytest.raises(ValueError, match="final patch"):
        model.encode_pooled_state(batch)

    batch["patch_context_mask"][1, -1] = True
    tensors = model._validate_batch(batch, require_jepa_targets=False)
    fused = model._tokenize_and_fuse(tensors)
    positioned = fused + model.patch_position_embedding.unsqueeze(0)
    patch_context = tensors["patch_context_mask"]
    encoded = model.context_encoder(positioned, src_key_padding_mask=~patch_context)
    weights = patch_context.to(encoded.dtype).unsqueeze(-1)
    mean_state = (encoded * weights).sum(dim=1) / weights.sum(dim=1)
    expected = torch.cat((mean_state, encoded[:, -1]), dim=-1)

    actual = model.encode_pooled_state(batch)
    assert actual.shape == (2, 16)
    assert torch.allclose(actual, expected)

    with torch.no_grad():
        for parameter in model.state_exporter.parameters():
            parameter.fill_(1000.0)
    assert torch.allclose(model.encode_pooled_state(batch), expected)
    assert model.encode(batch).shape == (2, 3)
