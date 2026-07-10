from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Mapping

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from fi_jepa.dataloader import DensePanelStore

from fi_jepa.model_config import FIJepaModelConfig
from fi_jepa.model_output import FIJepaOutput
from fi_jepa.model_validation import (
    ENCODER_BATCH_TENSOR_NAMES as ENCODER_BATCH_TENSOR_NAMES,
    JEPA_BATCH_TENSOR_NAMES as JEPA_BATCH_TENSOR_NAMES,
    validate_model_batch,
    validate_model_feature_dimensions,
)
from fi_jepa.tokenizer import (
    MaskedAttentionAssetPooler,
    MaskedAttentionPatchTokenizer,
    MaskedPatchTokenizer,
    masked_mean,
    pack_masked_sequence,
)

INPUT_ABLATION_MODES = ("all_streams", "without_assets", "without_market", "without_macro")

# ============================================================================
# CHECKPOINT COMPATIBILITY
# ============================================================================


def load_fi_jepa_model_state(
    model: FIJepaModel,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load current or legacy model state into a full-EMA FI-JEPA model.

    Legacy checkpoints contain an EMA temporal encoder but no target tokenizer,
    asset-pooler, fusion, or positional-embedding state because those modules
    were shared with the online branch. For those checkpoints, the missing
    teacher preprocessing state is initialized from the saved online modules,
    which exactly reproduces the old target input at resume time. A partially
    written current-format teacher remains an error instead of being repaired.
    """
    # The removed exporter was never part of the JEPA objective and had no
    # effect on training. Discard only that known legacy parameter prefix.
    migrated = {
        name: value
        for name, value in state_dict.items()
        if not name.startswith("state_exporter.")
    }
    if (
        "target_patch_position_embedding" in migrated
        or "patch_position_embedding" not in migrated
    ):
        model.load_state_dict(migrated)
        return

    prefix_pairs = (
        ("asset_tokenizer.", "target_asset_tokenizer."),
        ("market_tokenizer.", "target_market_tokenizer."),
        ("macro_tokenizer.", "target_macro_tokenizer."),
        ("asset_pooler.", "target_asset_pooler."),
        ("fusion.", "target_fusion."),
    )
    for online_prefix, target_prefix in prefix_pairs:
        for name, value in migrated.copy().items():
            if name.startswith(online_prefix):
                migrated[f"{target_prefix}{name.removeprefix(online_prefix)}"] = value
    migrated["target_patch_position_embedding"] = migrated["patch_position_embedding"]
    model.load_state_dict(migrated)


# ============================================================================
# FI-JEPA MODEL
# ============================================================================


class FIJepaModel(nn.Module):
    """Configurable patch-tokenized variable-asset temporal FI-JEPA core model.

    The model consumes the patched batch dictionary emitted by the dense-panel
    dataloader. Online tokenizers, asset pooling, fusion, positional embeddings,
    and temporal encoding are mirrored by a frozen full-EMA target branch. The
    online temporal encoder and predictor may use dropout, while every target
    module remains deterministic and evaluates the full valid patch sequence
    before target positions are gathered.

    Shape symbols used throughout this module:
        ``B``: batch size.
        ``P``: configured number of temporal patches.
        ``L``: days per patch.
        ``A``: padded asset count.
        ``T``: padded target-patch count.
        ``C``: padded visible-context count.
        ``D``: shared model width.
    """

    def __init__(
        self,
        config: FIJepaModelConfig,
        asset_feature_dim: int,
        market_feature_dim: int,
        macro_feature_dim: int,
    ):
        super().__init__()
        validate_model_feature_dimensions(
            {
                "asset_feature_dim": asset_feature_dim,
                "market_feature_dim": market_feature_dim,
                "macro_feature_dim": macro_feature_dim,
            }
        )

        self.config = config
        self.asset_feature_dim = asset_feature_dim
        self.market_feature_dim = market_feature_dim
        self.macro_feature_dim = macro_feature_dim

        # Each tokenizer maps one stream-specific patch to its configured token
        # width before the three streams are combined.
        if config.tokenizer_type == "mean":
            self.asset_tokenizer = MaskedPatchTokenizer(asset_feature_dim, config.asset_hidden_dim, config.asset_token_dim)
            self.market_tokenizer = MaskedPatchTokenizer(market_feature_dim, config.market_hidden_dim, config.market_token_dim)
            self.macro_tokenizer = MaskedPatchTokenizer(macro_feature_dim, config.macro_hidden_dim, config.macro_token_dim)
        else:
            attention_kwargs = {
                "layers": config.tokenizer_layers,
                "heads": config.tokenizer_heads,
                "mlp_ratio": config.tokenizer_mlp_ratio,
            }
            self.asset_tokenizer = MaskedAttentionPatchTokenizer(
                asset_feature_dim,
                config.patch_len,
                config.asset_hidden_dim,
                config.asset_token_dim,
                **attention_kwargs,
            )
            self.market_tokenizer = MaskedAttentionPatchTokenizer(
                market_feature_dim,
                config.patch_len,
                config.market_hidden_dim,
                config.market_token_dim,
                **attention_kwargs,
            )
            self.macro_tokenizer = MaskedAttentionPatchTokenizer(
                macro_feature_dim,
                config.patch_len,
                config.macro_hidden_dim,
                config.macro_token_dim,
                **attention_kwargs,
            )
        self.asset_pooler = (
            MaskedAttentionAssetPooler(
                config.asset_token_dim,
                layers=config.asset_pooling_layers,
                heads=config.asset_pooling_heads,
                mlp_ratio=config.asset_pooling_mlp_ratio,
            )
            if config.asset_pooling_type == "attention"
            else None
        )
        fusion_input_dim = config.asset_token_dim + config.market_token_dim + config.macro_token_dim
        # [asset token | market token | macro token] -> shared D-dimensional token.
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.patch_position_embedding = nn.Parameter(
            torch.empty(config.num_patches, config.d_model)
        )
        nn.init.normal_(self.patch_position_embedding, std=0.02)

        context_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.context_heads,
            dim_feedforward=config.d_model * config.context_mlp_ratio,
            dropout=config.context_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(context_layer, num_layers=config.context_layers, enable_nested_tensor=False)

        # The teacher owns every learned transformation from raw patches through
        # temporal encoding. It starts as an exact online copy and advances only
        # through EMA updates after successful optimizer steps.
        self.target_asset_tokenizer = deepcopy(self.asset_tokenizer)
        self.target_market_tokenizer = deepcopy(self.market_tokenizer)
        self.target_macro_tokenizer = deepcopy(self.macro_tokenizer)
        self.target_asset_pooler = deepcopy(self.asset_pooler)
        self.target_fusion = deepcopy(self.fusion)
        self.target_patch_position_embedding = nn.Parameter(
            self.patch_position_embedding.detach().clone(),
            requires_grad=False,
        )
        self.target_encoder = deepcopy(self.context_encoder)
        for target_module, _ in self._target_online_module_pairs():
            target_module.requires_grad_(False)
        self._set_target_branch_eval()

        predictor_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.predictor_heads,
            dim_feedforward=config.d_model * config.predictor_mlp_ratio,
            dropout=config.predictor_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerDecoder(predictor_layer, num_layers=config.predictor_layers)
        # One learned query is combined with each target patch's positional embedding.
        self.target_mask_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        nn.init.normal_(self.target_mask_token, std=0.02)

    @classmethod
    def from_store(cls, config: FIJepaModelConfig, store: DensePanelStore) -> FIJepaModel:
        """Construct a model using dense-cache feature-manifest dimensions."""
        return cls(
            config,
            asset_feature_dim=len(store.feature_names["asset"]),
            market_feature_dim=len(store.feature_names["market"]),
            macro_feature_dim=len(store.feature_names["macro"]),
        )

    def train(self, mode: bool = True) -> FIJepaModel:
        """Set online modules to training mode while keeping the EMA target deterministic."""
        super().train(mode)
        self._set_target_branch_eval()
        return self

    def _target_online_module_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        """Return aligned target/online module pairs that define the full EMA branch."""
        pairs = [
            (self.target_asset_tokenizer, self.asset_tokenizer),
            (self.target_market_tokenizer, self.market_tokenizer),
            (self.target_macro_tokenizer, self.macro_tokenizer),
            (self.target_fusion, self.fusion),
            (self.target_encoder, self.context_encoder),
        ]
        if self.target_asset_pooler is not None and self.asset_pooler is not None:
            pairs.insert(3, (self.target_asset_pooler, self.asset_pooler))
        return tuple(pairs)

    def _set_target_branch_eval(self) -> None:
        """Keep every full-EMA target module deterministic regardless of parent mode."""
        for target_module, _ in self._target_online_module_pairs():
            target_module.eval()

    def target_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return every frozen parameter owned by the complete EMA target branch."""
        parameters = [self.target_patch_position_embedding]
        for target_module, _ in self._target_online_module_pairs():
            parameters.extend(target_module.parameters())
        return tuple(parameters)

    def _validate_batch(
        self,
        batch: dict[str, object],
        *,
        require_jepa_targets: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Delegate complete patched-batch interface validation."""
        return validate_model_batch(
            batch,
            num_patches=self.config.num_patches,
            patch_len=self.config.patch_len,
            asset_feature_dim=self.asset_feature_dim,
            market_feature_dim=self.market_feature_dim,
            macro_feature_dim=self.macro_feature_dim,
            require_jepa_targets=require_jepa_targets,
        )

    def _tokenize_and_fuse(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        use_target_branch: bool = False,
        input_mode: str = "all_streams",
    ) -> torch.Tensor:
        """Create one online or EMA target token per temporal patch.

        Asset patches are tokenized per asset and pooled across only valid
        asset slots. Market and macro patches are tokenized directly. The
        resulting stream tokens are concatenated and projected from their
        combined width to ``D``.

        Returns:
            Fused patch tokens shaped ``[B, P, D]``.
        """
        if input_mode not in INPUT_ABLATION_MODES:
            raise ValueError(f"Unknown input ablation mode: {input_mode!r}.")

        asset_tokenizer = (
            self.target_asset_tokenizer if use_target_branch else self.asset_tokenizer
        )
        market_tokenizer = (
            self.target_market_tokenizer if use_target_branch else self.market_tokenizer
        )
        macro_tokenizer = (
            self.target_macro_tokenizer if use_target_branch else self.macro_tokenizer
        )
        asset_pooler = self.target_asset_pooler if use_target_branch else self.asset_pooler
        fusion = self.target_fusion if use_target_branch else self.fusion

        # [B, P, L, A, F_asset] -> [B, P, A, L, F_asset].
        asset_values = tensors["asset_patches"].permute(0, 1, 3, 2, 4)
        asset_features = tensors["asset_feature_mask_patched"].permute(0, 1, 3, 2, 4)
        asset_days = tensors["valid_asset_mask_patched"].permute(0, 1, 3, 2)
        asset_days = asset_days & asset_features.any(dim=-1)
        asset_tokens = asset_tokenizer(
            asset_values, asset_features, asset_days
        )  # [B, P, A, D_asset].
        if asset_pooler is None:
            panel_tokens = masked_mean(
                asset_tokens, tensors["patch_asset_mask"], dimension=2
            )  # [B, P, D_asset].
        else:
            panel_tokens = asset_pooler(
                asset_tokens, tensors["patch_asset_mask"]
            )  # [B, P, D_asset].

        market_features = tensors["market_feature_mask_patched"]
        market_days = tensors["valid_market_date_mask_patched"] & market_features.any(dim=-1)
        market_tokens = market_tokenizer(
            tensors["market_patches"], market_features, market_days
        )  # [B, P, D_market].

        macro_features = tensors["macro_feature_mask_patched"]
        macro_days = tensors["valid_macro_date_mask_patched"] & macro_features.any(dim=-1)
        macro_tokens = macro_tokenizer(
            tensors["macro_patches"], macro_features, macro_days
        )  # [B, P, D_macro].

        # Preserve the fusion input shape while neutralizing only the requested stream.
        if input_mode == "without_assets":
            panel_tokens = torch.zeros_like(panel_tokens)  # [B, P, D_asset].
        elif input_mode == "without_market":
            market_tokens = torch.zeros_like(market_tokens)  # [B, P, D_market].
        elif input_mode == "without_macro":
            macro_tokens = torch.zeros_like(macro_tokens)  # [B, P, D_macro].

        # [B, P, D_asset + D_market + D_macro] -> [B, P, D].
        combined_tokens = torch.cat((panel_tokens, market_tokens, macro_tokens), dim=-1)
        return fusion(combined_tokens)

    def forward(self, batch: dict[str, object]) -> FIJepaOutput:
        """Predict EMA target representations for masked patch positions.

        The online branch sees only packed JEPA context patches. The EMA target
        branch sees every context-valid patch, after which only requested target
        positions are gathered. Prediction and target tensors retain padded
        target slots, but those slots are zeroed and excluded from the loss.
        """
        tensors = self._validate_batch(batch)
        fused_tokens = self._tokenize_and_fuse(tensors)  # [B, P, D].
        positioned_tokens = fused_tokens + self.patch_position_embedding.unsqueeze(0)  # [B, P, D].

        # [B, P, D] -> [B, C, D], removing masked targets and invalid patches.
        context_tokens, context_mask = pack_masked_sequence(positioned_tokens, tensors["jepa_context_mask"])
        context_encoded = self.context_encoder(context_tokens, src_key_padding_mask=~context_mask)  # [B, C, D].

        patch_context = tensors["patch_context_mask"]
        self._set_target_branch_eval()
        with torch.no_grad():
            target_fused_tokens = self._tokenize_and_fuse(
                tensors,
                use_target_branch=True,
            )  # [B, P, D].
            target_positioned_tokens = (
                target_fused_tokens + self.target_patch_position_embedding.unsqueeze(0)
            )  # [B, P, D].
            # The full EMA branch encodes the complete valid sequence: [B, P, D].
            target_full = self.target_encoder(
                target_positioned_tokens,
                src_key_padding_mask=~patch_context,
            )

        target_ids = tensors["target_patch_ids"]
        target_mask = tensors["target_patch_id_mask"]
        safe_ids = target_ids.clamp_min(0)
        gather_ids = safe_ids.unsqueeze(-1).expand(-1, -1, self.config.d_model)
        target_representations = target_full.gather(1, gather_ids)  # [B, T, D].
        target_representations = torch.where(
            target_mask.unsqueeze(-1),
            target_representations,
            torch.zeros_like(target_representations),
        )

        target_positions = self.patch_position_embedding[safe_ids]
        target_queries = self.target_mask_token + target_positions  # [B, T, D].
        target_queries = torch.where(
            target_mask.unsqueeze(-1), target_queries, torch.zeros_like(target_queries)
        )
        predicted_targets = self.predictor(
            target_queries,
            context_encoded,
            tgt_key_padding_mask=~target_mask,
            memory_key_padding_mask=~context_mask,
        )  # [B, T, D].
        predicted_targets = torch.where(
            target_mask.unsqueeze(-1), predicted_targets, torch.zeros_like(predicted_targets)
        )

        # Normalize across D, reduce to one loss per target [B, T], then discard padding.
        normalized_prediction = F.normalize(predicted_targets, dim=-1)
        normalized_target = F.normalize(target_representations, dim=-1)
        per_target_loss = ((normalized_prediction - normalized_target) ** 2).sum(dim=-1)
        loss = per_target_loss[target_mask].mean()
        return FIJepaOutput(
            loss=loss,
            predicted_targets=predicted_targets,
            target_representations=target_representations,
            target_patch_mask=target_mask,
            context_representations=context_encoded,
            context_mask=context_mask,
            fused_tokens=fused_tokens,
        )

    def encode_state_components(
        self,
        batch: dict[str, object],
        *,
        input_mode: str = "all_streams",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the temporal-mean and endpoint encoder states.

        The full unmasked context-valid patch sequence is encoded, then a
        masked temporal mean and the final patch at sample endpoint ``t`` are
        returned separately. Embedding exports require that endpoint patch to
        be context-valid rather than silently substituting an earlier patch.
        """
        tensors = self._validate_batch(batch, require_jepa_targets=False)
        fused_tokens = self._tokenize_and_fuse(tensors, input_mode=input_mode)  # [B, P, D].
        positioned_tokens = fused_tokens + self.patch_position_embedding.unsqueeze(0)  # [B, P, D].
        patch_context = tensors["patch_context_mask"]
        if not patch_context[:, -1].all():
            raise ValueError("Embedding export requires the final patch to be context-valid.")
        full_encoded = self.context_encoder(
            positioned_tokens, src_key_padding_mask=~patch_context
        )  # [B, P, D].

        mean_state = masked_mean(full_encoded, patch_context, dimension=1)  # [B, D].
        endpoint_state = full_encoded[:, -1]  # [B, D].
        return mean_state, endpoint_state

    def encode_pooled_state(
        self, batch: dict[str, object], *, input_mode: str = "all_streams"
    ) -> torch.Tensor:
        """Return the canonical pooled state: temporal mean concatenated with endpoint."""
        mean_state, endpoint_state = self.encode_state_components(
            batch, input_mode=input_mode
        )  # Each [B, D].
        return torch.cat((mean_state, endpoint_state), dim=-1)  # [B, 2D].

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Move the complete target branch toward the online encoder path by EMA.

        ``momentum=1`` leaves the target unchanged, while ``momentum=0`` copies
        the online path exactly. Floating-point parameters use EMA; module
        buffers are copied directly because they represent state rather than
        trainable weights.
        """
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1].")
        for target_module, online_module in self._target_online_module_pairs():
            for target_parameter, online_parameter in zip(
                target_module.parameters(),
                online_module.parameters(),
                strict=True,
            ):
                target_parameter.mul_(momentum).add_(online_parameter, alpha=1.0 - momentum)
            for target_buffer, online_buffer in zip(
                target_module.buffers(),
                online_module.buffers(),
                strict=True,
            ):
                target_buffer.copy_(online_buffer)
        self.target_patch_position_embedding.mul_(momentum).add_(
            self.patch_position_embedding,
            alpha=1.0 - momentum,
        )
        self._set_target_branch_eval()
