from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class FIJepaModelConfig:
    """Configure tokenizer, Transformer, predictor, and exporter dimensions.

    All dimensions used by the model are represented here so construction can
    validate attention divisibility and deterministic shared-path constraints
    before allocating modules.
    """

    patch_len: int = 21
    num_patches: int = 12
    asset_hidden_dim: int = 64
    asset_token_dim: int = 128
    market_hidden_dim: int = 32
    market_token_dim: int = 64
    macro_hidden_dim: int = 64
    macro_token_dim: int = 64
    d_model: int = 128
    latent_dim: int = 8
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
        integer_fields = {
            "patch_len": self.patch_len,
            "num_patches": self.num_patches,
            "asset_hidden_dim": self.asset_hidden_dim,
            "asset_token_dim": self.asset_token_dim,
            "market_hidden_dim": self.market_hidden_dim,
            "market_token_dim": self.market_token_dim,
            "macro_hidden_dim": self.macro_hidden_dim,
            "macro_token_dim": self.macro_token_dim,
            "d_model": self.d_model,
            "latent_dim": self.latent_dim,
            "context_layers": self.context_layers,
            "context_heads": self.context_heads,
            "context_mlp_ratio": self.context_mlp_ratio,
            "predictor_layers": self.predictor_layers,
            "predictor_heads": self.predictor_heads,
            "predictor_mlp_ratio": self.predictor_mlp_ratio,
        }
        invalid = [name for name, value in integer_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"Model dimensions and counts must be positive: {invalid}")
        if self.d_model % self.context_heads:
            raise ValueError("d_model must be divisible by context_heads.")
        if self.d_model % self.predictor_heads:
            raise ValueError("d_model must be divisible by predictor_heads.")
        if not 0.0 <= self.context_dropout < 1.0:
            raise ValueError("context_dropout must be in [0, 1).")
        if not 0.0 <= self.predictor_dropout < 1.0:
            raise ValueError("predictor_dropout must be in [0, 1).")

    @classmethod
    def from_yaml(cls, path: Path | str) -> FIJepaModelConfig:
        """Load and flatten the nested architecture configuration YAML.

        The YAML is organized by architecture component for readability. This
        method converts that nested representation into the immutable runtime
        configuration and enforces the dropout-free shared fusion contract.
        """
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Model configuration must be a YAML mapping.")
        required_sections = {
            "input",
            "tokenizers",
            "fusion",
            "context_encoder",
            "predictor",
            "state_exporter",
        }
        missing = sorted(required_sections - set(values))
        if missing:
            raise ValueError(f"Model configuration is missing sections: {missing}")

        tokenizers = values["tokenizers"]
        fusion_dropout = float(values["fusion"].get("dropout", 0.0))
        if fusion_dropout != 0.0:
            raise ValueError(
                "Shared fusion dropout must remain 0.0 for a deterministic target input."
            )
        return cls(
            patch_len=int(values["input"]["patch_len"]),
            num_patches=int(values["input"]["num_patches"]),
            asset_hidden_dim=int(tokenizers["asset"]["hidden_dim"]),
            asset_token_dim=int(tokenizers["asset"]["output_dim"]),
            market_hidden_dim=int(tokenizers["market"]["hidden_dim"]),
            market_token_dim=int(tokenizers["market"]["output_dim"]),
            macro_hidden_dim=int(tokenizers["macro"]["hidden_dim"]),
            macro_token_dim=int(tokenizers["macro"]["output_dim"]),
            d_model=int(values["fusion"]["output_dim"]),
            latent_dim=int(values["state_exporter"]["latent_dim"]),
            context_layers=int(values["context_encoder"]["layers"]),
            context_heads=int(values["context_encoder"]["heads"]),
            context_mlp_ratio=int(values["context_encoder"]["mlp_ratio"]),
            context_dropout=float(values["context_encoder"]["dropout"]),
            predictor_layers=int(values["predictor"]["layers"]),
            predictor_heads=int(values["predictor"]["heads"]),
            predictor_mlp_ratio=int(values["predictor"]["mlp_ratio"]),
            predictor_dropout=float(values["predictor"]["dropout"]),
        )
