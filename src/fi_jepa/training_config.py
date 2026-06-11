from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

import yaml


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class FIJepaTrainingConfig:
    """Configure one resumable FI-JEPA pretraining run.

    Architecture and dataloader settings remain in their existing configuration
    files. This configuration owns only run orchestration, optimization,
    validation cadence, checkpoint cadence, and logging cadence.
    """

    run_name: str = "fi_jepa_v1"
    output_root: Path = Path("runs/pretraining")
    model_config_path: Path = Path("configs/model.yaml")
    dataloader_config_path: Path = Path("configs/dataloader.yaml")
    device: str = "auto"
    optimizer: str = "adamw"
    lr: float = 0.0001
    min_lr: float = 0.0
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    ema_momentum_start: float = 0.99
    ema_momentum_end: float = 0.999
    validation_every_epochs: int = 1
    representation_evaluation_enabled: bool = True
    representation_evaluation_every_epochs: int = 1
    representation_views_per_date: int = 3
    representation_pca_components: int = 8
    representation_export_embeddings: bool = False
    checkpoint_every_steps: int = 1000
    logging_every_steps: int = 10

    def __post_init__(self) -> None:
        """Reject invalid run names, optimization settings, and cadences."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.run_name):
            raise ValueError("run_name must contain only letters, numbers, underscores, and dashes.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")
        if self.optimizer != "adamw":
            raise ValueError("Only the adamw optimizer is supported.")
        if self.lr <= 0.0 or self.min_lr < 0.0 or self.min_lr > self.lr:
            raise ValueError("Learning rates must satisfy 0 <= min_lr <= lr and lr > 0.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        if self.epochs <= 0 or not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("epochs must be positive and warmup_epochs must be in [0, epochs).")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive.")
        if not 0.0 <= self.ema_momentum_start <= self.ema_momentum_end <= 1.0:
            raise ValueError("EMA momentum must satisfy 0 <= start <= end <= 1.")
        for name, value in (
            ("validation_every_epochs", self.validation_every_epochs),
            ("representation_evaluation_every_epochs", self.representation_evaluation_every_epochs),
            ("representation_views_per_date", self.representation_views_per_date),
            ("representation_pca_components", self.representation_pca_components),
            ("checkpoint_every_steps", self.checkpoint_every_steps),
            ("logging_every_steps", self.logging_every_steps),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

    @classmethod
    def from_yaml(cls, path: Path | str) -> FIJepaTrainingConfig:
        """Load the nested pretraining YAML into the immutable runtime config."""
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Training configuration must be a YAML mapping.")
        required = {"run", "configs", "optimization", "ema", "validation", "checkpointing", "logging"}
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"Training configuration is missing sections: {missing}")

        run = values["run"]
        configs = values["configs"]
        optimization = values["optimization"]
        representation = values.get("representation_evaluation") or {}
        return cls(
            run_name=str(run["name"]),
            output_root=Path(run["output_root"]),
            model_config_path=Path(configs["model"]),
            dataloader_config_path=Path(configs["dataloader"]),
            device=str(run.get("device", "auto")),
            optimizer=str(optimization.get("optimizer", "adamw")).lower(),
            lr=float(optimization["lr"]),
            min_lr=float(optimization.get("min_lr", 0.0)),
            weight_decay=float(optimization["weight_decay"]),
            epochs=int(optimization["epochs"]),
            warmup_epochs=int(optimization["warmup_epochs"]),
            grad_clip_norm=float(optimization["grad_clip_norm"]),
            mixed_precision=bool(optimization["mixed_precision"]),
            ema_momentum_start=float(values["ema"]["momentum_start"]),
            ema_momentum_end=float(values["ema"]["momentum_end"]),
            validation_every_epochs=int(values["validation"]["every_epochs"]),
            representation_evaluation_enabled=bool(representation.get("enabled", True)),
            representation_evaluation_every_epochs=int(representation.get("every_epochs", 1)),
            representation_views_per_date=int(representation.get("views_per_date", 3)),
            representation_pca_components=int(representation.get("pca_components", 8)),
            representation_export_embeddings=bool(
                representation.get("export_embeddings_every_validation", False)
            ),
            checkpoint_every_steps=int(values["checkpointing"]["every_steps"]),
            logging_every_steps=int(values["logging"]["every_steps"]),
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FIJepaTrainingConfig:
        """Reconstruct a training config stored inside a resolved checkpoint."""
        normalized = dict(values)
        for name in ("output_root", "model_config_path", "dataloader_config_path"):
            normalized[name] = Path(normalized[name])
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML- and checkpoint-safe representation."""
        values = asdict(self)
        for name in ("output_root", "model_config_path", "dataloader_config_path"):
            values[name] = str(values[name])
        return values
