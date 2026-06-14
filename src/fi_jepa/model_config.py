from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from fi_jepa.model_validation import validate_model_config, validate_model_yaml


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class FIJepaModelConfig:
    """Configure tokenizer, Transformer, and predictor dimensions.

    All dimensions used by the model are represented here so construction can
    validate attention divisibility and deterministic preprocessing constraints
    before allocating the online and EMA target copies.
    """

    patch_len: int = 21
    num_patches: int = 12
    tokenizer_type: str = "mean"
    tokenizer_layers: int = 2
    tokenizer_heads: int = 4
    tokenizer_mlp_ratio: int = 4
    asset_pooling_type: str = "mean"
    asset_pooling_layers: int = 2
    asset_pooling_heads: int = 4
    asset_pooling_mlp_ratio: int = 4
    asset_hidden_dim: int = 64
    asset_token_dim: int = 128
    market_hidden_dim: int = 32
    market_token_dim: int = 64
    macro_hidden_dim: int = 64
    macro_token_dim: int = 64
    d_model: int = 128
    context_layers: int = 2
    context_heads: int = 4
    context_mlp_ratio: int = 4
    context_dropout: float = 0.1
    predictor_layers: int = 2
    predictor_heads: int = 4
    predictor_mlp_ratio: int = 4
    predictor_dropout: float = 0.1

    def __post_init__(self) -> None:
        """Reject invalid dimensions, attention widths, and dropout rates."""
        validate_model_config(
            integer_fields={
                "patch_len": self.patch_len,
                "num_patches": self.num_patches,
                "tokenizer_layers": self.tokenizer_layers,
                "tokenizer_heads": self.tokenizer_heads,
                "tokenizer_mlp_ratio": self.tokenizer_mlp_ratio,
                "asset_pooling_layers": self.asset_pooling_layers,
                "asset_pooling_heads": self.asset_pooling_heads,
                "asset_pooling_mlp_ratio": self.asset_pooling_mlp_ratio,
                "asset_hidden_dim": self.asset_hidden_dim,
                "asset_token_dim": self.asset_token_dim,
                "market_hidden_dim": self.market_hidden_dim,
                "market_token_dim": self.market_token_dim,
                "macro_hidden_dim": self.macro_hidden_dim,
                "macro_token_dim": self.macro_token_dim,
                "d_model": self.d_model,
                "context_layers": self.context_layers,
                "context_heads": self.context_heads,
                "context_mlp_ratio": self.context_mlp_ratio,
                "predictor_layers": self.predictor_layers,
                "predictor_heads": self.predictor_heads,
                "predictor_mlp_ratio": self.predictor_mlp_ratio,
            },
            tokenizer_type=self.tokenizer_type,
            asset_pooling_type=self.asset_pooling_type,
            d_model=self.d_model,
            context_heads=self.context_heads,
            predictor_heads=self.predictor_heads,
            tokenizer_heads=self.tokenizer_heads,
            tokenizer_hidden_dims={
                "asset_hidden_dim": self.asset_hidden_dim,
                "market_hidden_dim": self.market_hidden_dim,
                "macro_hidden_dim": self.macro_hidden_dim,
            },
            asset_token_dim=self.asset_token_dim,
            asset_pooling_heads=self.asset_pooling_heads,
            context_dropout=self.context_dropout,
            predictor_dropout=self.predictor_dropout,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> FIJepaModelConfig:
        """Load and flatten the nested architecture configuration YAML.

        The YAML is organized by architecture component for readability. This
        method converts that nested representation into the immutable runtime
        configuration and enforces dropout-free online and target fusion.
        """
        values = validate_model_yaml(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
        tokenizers = values["tokenizers"]
        tokenizer_type = str(tokenizers.get("type", "mean"))
        tokenizer_attention = tokenizers.get("attention") or {}
        asset_pooling = values.get("asset_pooling") or {}
        asset_pooling_type = str(asset_pooling.get("type", "mean"))
        asset_pooling_attention = asset_pooling.get("attention") or {}
        return cls(
            patch_len=int(values["input"]["patch_len"]),
            num_patches=int(values["input"]["num_patches"]),
            tokenizer_type=tokenizer_type,
            tokenizer_layers=int(tokenizer_attention.get("layers", 2)),
            tokenizer_heads=int(tokenizer_attention.get("heads", 4)),
            tokenizer_mlp_ratio=int(tokenizer_attention.get("mlp_ratio", 4)),
            asset_pooling_type=asset_pooling_type,
            asset_pooling_layers=int(asset_pooling_attention.get("layers", 2)),
            asset_pooling_heads=int(asset_pooling_attention.get("heads", 4)),
            asset_pooling_mlp_ratio=int(asset_pooling_attention.get("mlp_ratio", 4)),
            asset_hidden_dim=int(tokenizers["asset"]["hidden_dim"]),
            asset_token_dim=int(tokenizers["asset"]["output_dim"]),
            market_hidden_dim=int(tokenizers["market"]["hidden_dim"]),
            market_token_dim=int(tokenizers["market"]["output_dim"]),
            macro_hidden_dim=int(tokenizers["macro"]["hidden_dim"]),
            macro_token_dim=int(tokenizers["macro"]["output_dim"]),
            d_model=int(values["fusion"]["output_dim"]),
            context_layers=int(values["context_encoder"]["layers"]),
            context_heads=int(values["context_encoder"]["heads"]),
            context_mlp_ratio=int(values["context_encoder"]["mlp_ratio"]),
            context_dropout=float(values["context_encoder"]["dropout"]),
            predictor_layers=int(values["predictor"]["layers"]),
            predictor_heads=int(values["predictor"]["heads"]),
            predictor_mlp_ratio=int(values["predictor"]["mlp_ratio"]),
            predictor_dropout=float(values["predictor"]["dropout"]),
        )
