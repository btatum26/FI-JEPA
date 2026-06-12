from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from fi_jepa.dataloader import FrozenPanelStore

from fi_jepa.model_config import FIJepaModelConfig
from fi_jepa.model_output import FIJepaOutput
from fi_jepa.tokenizer import (
    MaskedAttentionPatchTokenizer,
    MaskedPatchTokenizer,
    masked_mean,
    pack_masked_sequence,
)

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
# FI-JEPA MODEL
# ============================================================================


class FIJepaModel(nn.Module):
    """Configurable patch-tokenized variable-asset temporal FI-JEPA core model.

    The model consumes the patched batch dictionary emitted by
    ``FIJepaBatchAssembler`` or its compatibility collator. Shared tokenizers
    and fusion are deterministic; the online temporal encoder and predictor may
    use dropout. The target temporal encoder is a frozen EMA copy that always
    evaluates the full valid patch sequence before target positions are
    gathered.

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
        for name, value in (
            ("asset_feature_dim", asset_feature_dim),
            ("market_feature_dim", market_feature_dim),
            ("macro_feature_dim", macro_feature_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        self.config = config
        self.asset_feature_dim = asset_feature_dim
        self.market_feature_dim = market_feature_dim
        self.macro_feature_dim = macro_feature_dim

        # Each tokenizer maps one stream-specific patch to its configured token
        # width before the three streams are combined.
        if config.tokenizer_type == "mean":
            self.asset_tokenizer = MaskedPatchTokenizer(
                asset_feature_dim, config.asset_hidden_dim, config.asset_token_dim
            )
            self.market_tokenizer = MaskedPatchTokenizer(
                market_feature_dim, config.market_hidden_dim, config.market_token_dim
            )
            self.macro_tokenizer = MaskedPatchTokenizer(
                macro_feature_dim, config.macro_hidden_dim, config.macro_token_dim
            )
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

        context_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.context_heads,
            dim_feedforward=config.d_model * config.context_mlp_ratio,
            dropout=config.context_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer, num_layers=config.context_layers, enable_nested_tensor=False
        )
        # The target encoder starts as an exact online-encoder copy and remains
        # gradient-free; training code advances it only through EMA updates.
        self.target_encoder = deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

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
        # Checkpoint compatibility only. The JEPA loss does not train this
        # projection, so evaluation uses encode_pooled_state() plus train-fit PCA.
        self.state_exporter = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.latent_dim),
        )

        nn.init.normal_(self.patch_position_embedding, std=0.02)
        nn.init.normal_(self.target_mask_token, std=0.02)

    @classmethod
    def from_store(cls, config: FIJepaModelConfig, store: FrozenPanelStore) -> FIJepaModel:
        """Construct a model using the frozen artifact's feature manifest dimensions."""
        return cls(
            config,
            asset_feature_dim=len(store.feature_names["asset"]),
            market_feature_dim=len(store.feature_names["market"]),
            macro_feature_dim=len(store.feature_names["macro"]),
        )

    def train(self, mode: bool = True) -> FIJepaModel:
        """Set online modules to training mode while keeping the EMA target deterministic."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _validate_batch(
        self,
        batch: dict[str, object],
        *,
        require_jepa_targets: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Validate the complete patched-batch interface before model computation.

        Shapes, dtypes, devices, configured dimensions, and mask relationships
        are checked here so dataloader drift fails at the model boundary instead
        of producing a plausible but semantically incorrect training run.
        """
        # First establish the complete interface. Optional or silently ignored
        # tensors at this boundary would allow loader/model contracts to drift.
        required = (
            ENCODER_BATCH_TENSOR_NAMES | JEPA_BATCH_TENSOR_NAMES
            if require_jepa_targets
            else ENCODER_BATCH_TENSOR_NAMES
        )
        missing = sorted(required - set(batch))
        if missing:
            raise ValueError(f"FI-JEPA batch is missing required keys: {missing}")

        # Narrow all required values to tensors before shape or device access.
        tensors: dict[str, torch.Tensor] = {}
        for name in required:
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name} must be a Tensor; got {type(value).__name__}.")
            tensors[name] = value

        # Rank checks produce clearer errors than unpacking malformed shapes.
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
                {
                    "target_patch_ids": 2,
                    "target_patch_id_mask": 2,
                    "jepa_context_mask": 2,
                }
            )
        for name, rank in expected_ranks.items():
            if tensors[name].ndim != rank:
                raise ValueError(
                    f"{name} must have rank {rank}; got shape {tuple(tensors[name].shape)}."
                )

        # The asset stream establishes shared B, P, L, and A dimensions.
        # Every other stream and mask must align to those axes.
        asset_shape = tuple(tensors["asset_patches"].shape)
        batch_size, num_patches, patch_len, num_assets, asset_dim = asset_shape
        market_dim = self.market_feature_dim
        macro_dim = self.macro_feature_dim
        expected_shapes = {
            "asset_patches": (
                batch_size,
                self.config.num_patches,
                self.config.patch_len,
                num_assets,
                self.asset_feature_dim,
            ),
            "market_patches": (
                batch_size,
                self.config.num_patches,
                self.config.patch_len,
                market_dim,
            ),
            "macro_patches": (
                batch_size,
                self.config.num_patches,
                self.config.patch_len,
                macro_dim,
            ),
            "asset_feature_mask_patched": asset_shape,
            "market_feature_mask_patched": (
                batch_size,
                num_patches,
                patch_len,
                market_dim,
            ),
            "macro_feature_mask_patched": (
                batch_size,
                num_patches,
                patch_len,
                macro_dim,
            ),
            "valid_asset_mask_patched": (
                batch_size,
                num_patches,
                patch_len,
                num_assets,
            ),
            "valid_market_date_mask_patched": (batch_size, num_patches, patch_len),
            "valid_macro_date_mask_patched": (batch_size, num_patches, patch_len),
            "patch_asset_mask": (batch_size, num_patches, num_assets),
            "patch_context_mask": (batch_size, num_patches),
        }
        if require_jepa_targets:
            target_count = tensors["target_patch_ids"].shape[1]
            expected_shapes.update(
                {
                    "target_patch_ids": (batch_size, target_count),
                    "target_patch_id_mask": (batch_size, target_count),
                    "jepa_context_mask": (batch_size, num_patches),
                }
            )
        for name, expected in expected_shapes.items():
            actual = tuple(tensors[name].shape)
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}; got {actual}.")

        # Values must remain floating point, masks boolean, and target IDs
        # integer so later masking and gather operations have exact semantics.
        for name in ("asset_patches", "market_patches", "macro_patches"):
            if not tensors[name].is_floating_point():
                raise ValueError(f"{name} must be floating point; got {tensors[name].dtype}.")
        mask_names = required - {
            "asset_patches",
            "market_patches",
            "macro_patches",
            "target_patch_ids",
        }
        for name in mask_names:
            if tensors[name].dtype != torch.bool:
                raise ValueError(f"{name} must have dtype bool; got {tensors[name].dtype}.")
        if require_jepa_targets:
            integer_dtypes = {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
            if tensors["target_patch_ids"].dtype not in integer_dtypes:
                raise ValueError(
                    "target_patch_ids must have an integer dtype; "
                    f"got {tensors['target_patch_ids'].dtype}."
                )

        # Mixed-device batches fail here instead of during a later arithmetic op.
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

        # Validate the semantic relationship between full context, hidden
        # targets, and the visible online-encoder context.
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
        if ((enabled_ids < 0) | (enabled_ids >= num_patches)).any():
            raise ValueError(
                f"Enabled target_patch_ids must be within [0, {num_patches}); "
                f"got {enabled_ids.tolist()}."
            )
        # Replace -1 padding only for gathers; target_mask continues to control
        # whether the gathered placeholder has any semantic effect.
        safe_ids = target_ids.clamp_min(0)
        target_is_context = patch_context.gather(1, safe_ids)
        target_is_visible = jepa_context.gather(1, safe_ids)
        if not target_is_context[target_mask].all():
            raise ValueError("Every enabled target_patch_id must reference a context-valid patch.")
        if target_is_visible[target_mask].any():
            raise ValueError("Enabled target patches cannot also appear in jepa_context_mask.")
        # Reconstruct target positions from IDs to verify exact set equality
        # with the patches removed from visible JEPA context.
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

    def _tokenize_and_fuse(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        """Create one deterministic shared token per temporal patch.

        Asset patches are tokenized per asset and pooled across only valid
        asset slots. Market and macro patches are tokenized directly. The
        resulting stream tokens are concatenated and projected from their
        combined width to ``D``.

        Returns:
            Fused patch tokens shaped ``[B, P, D]``.
        """
        # [B, P, L, A, F_asset] -> [B, P, A, L, F_asset].
        asset_values = tensors["asset_patches"].permute(0, 1, 3, 2, 4)
        asset_features = tensors["asset_feature_mask_patched"].permute(0, 1, 3, 2, 4)
        asset_days = tensors["valid_asset_mask_patched"].permute(0, 1, 3, 2)
        asset_days = asset_days & asset_features.any(dim=-1)
        asset_tokens = self.asset_tokenizer(
            asset_values, asset_features, asset_days
        )  # [B, P, A, D_asset].
        panel_tokens = masked_mean(
            asset_tokens, tensors["patch_asset_mask"], dimension=2
        )  # [B, P, D_asset].

        market_features = tensors["market_feature_mask_patched"]
        market_days = tensors["valid_market_date_mask_patched"] & market_features.any(dim=-1)
        market_tokens = self.market_tokenizer(
            tensors["market_patches"], market_features, market_days
        )  # [B, P, D_market].

        macro_features = tensors["macro_feature_mask_patched"]
        macro_days = tensors["valid_macro_date_mask_patched"] & macro_features.any(dim=-1)
        macro_tokens = self.macro_tokenizer(
            tensors["macro_patches"], macro_features, macro_days
        )  # [B, P, D_macro].

        # [B, P, D_asset + D_market + D_macro] -> [B, P, D].
        combined_tokens = torch.cat((panel_tokens, market_tokens, macro_tokens), dim=-1)
        return self.fusion(combined_tokens)

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
        context_tokens, context_mask = pack_masked_sequence(
            positioned_tokens, tensors["jepa_context_mask"]
        )
        context_encoded = self.context_encoder(
            context_tokens, src_key_padding_mask=~context_mask
        )  # [B, C, D].

        patch_context = tensors["patch_context_mask"]
        self.target_encoder.eval()
        with torch.no_grad():
            # The EMA branch encodes the full valid sequence: [B, P, D].
            target_full = self.target_encoder(
                positioned_tokens.detach(), src_key_padding_mask=~patch_context
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

    def encode_pooled_state(self, batch: dict[str, object]) -> torch.Tensor:
        """Return the learned encoder state used by all representation evaluation.

        The full unmasked context-valid patch sequence is encoded, then a
        masked temporal mean is concatenated with the final patch at sample
        endpoint ``t``. Embedding exports require that endpoint patch to be
        context-valid rather than silently substituting an earlier patch.
        """
        tensors = self._validate_batch(batch, require_jepa_targets=False)
        fused_tokens = self._tokenize_and_fuse(tensors)  # [B, P, D].
        positioned_tokens = fused_tokens + self.patch_position_embedding.unsqueeze(0)  # [B, P, D].
        patch_context = tensors["patch_context_mask"]
        if not patch_context[:, -1].all():
            raise ValueError("Embedding export requires the final patch to be context-valid.")
        full_encoded = self.context_encoder(
            positioned_tokens, src_key_padding_mask=~patch_context
        )  # [B, P, D].

        mean_state = masked_mean(full_encoded, patch_context, dimension=1)  # [B, D].
        endpoint_state = full_encoded[:, -1]  # [B, D].
        return torch.cat((mean_state, endpoint_state), dim=-1)  # [B, 2D].

    def encode(self, batch: dict[str, object]) -> torch.Tensor:
        """Apply the legacy untrained state exporter for checkpoint compatibility only."""
        return self.state_exporter(self.encode_pooled_state(batch))

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Move target encoder parameters toward online parameters by EMA.

        ``momentum=1`` leaves the target unchanged, while ``momentum=0`` copies
        the online encoder exactly. Buffers are unaffected because the current
        Transformer encoder does not use running-statistic buffers.
        """
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1].")
        for target_parameter, online_parameter in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            target_parameter.mul_(momentum).add_(online_parameter, alpha=1.0 - momentum)
        self.target_encoder.eval()
