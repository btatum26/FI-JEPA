from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any
import uuid

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
import yaml

from fi_jepa.dataloader import FIJepaDataConfig, FrozenPanelStore, build_fi_jepa_dataloader
from fi_jepa.model import FIJepaModel
from fi_jepa.model_config import FIJepaModelConfig
from fi_jepa.representation import (
    canonical_version_hash,
    model_state_hash,
    run_representation_evaluation,
)
from fi_jepa.training_config import FIJepaTrainingConfig


# ============================================================================
# STEP SCHEDULES
# ============================================================================


class WarmupCosineLRSchedule:
    """Warm one AdamW learning rate linearly, then decay it with cosine.

    The schedule is indexed by successful optimizer steps. Calls beyond the
    originally planned run clamp to ``min_lr``, which keeps replayed batches
    after a basic epoch resume from extending the cosine curve.
    """

    def __init__(
        self,
        optimizer: AdamW,
        *,
        base_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ):
        if not 0 <= warmup_steps < total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps).")
        if not 0.0 <= min_lr <= base_lr:
            raise ValueError("Learning rates must satisfy 0 <= min_lr <= base_lr.")
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.last_step = -1

    def value_at(self, step: int) -> float:
        """Return the clamped learning rate for a zero-based optimizer step."""
        if step < 0:
            raise ValueError("Schedule step cannot be negative.")
        if step >= self.total_steps:
            return self.min_lr
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)

        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 1:
            return self.min_lr
        progress = float(step - self.warmup_steps) / float(decay_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def apply(self, step: int, *, commit: bool) -> float:
        """Apply one step's LR, optionally recording that the step succeeded."""
        value = self.value_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        if commit:
            self.last_step = step
        return value

    def state_dict(self) -> dict[str, int | float]:
        """Return the complete schedule state stored in checkpoints."""
        return {
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict[str, int | float]) -> None:
        """Restore state while rejecting a schedule with different bounds."""
        expected = self.state_dict()
        for name in ("base_lr", "min_lr", "warmup_steps", "total_steps"):
            if state[name] != expected[name]:
                raise ValueError(f"Checkpoint LR schedule disagrees on {name}.")
        self.last_step = int(state["last_step"])


class LinearEMAMomentumSchedule:
    """Increase target-encoder EMA momentum linearly by optimizer step."""

    def __init__(self, *, start: float, end: float, total_steps: int):
        if not 0.0 <= start <= end <= 1.0:
            raise ValueError("EMA momentum must satisfy 0 <= start <= end <= 1.")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.last_step = -1

    def value_at(self, step: int) -> float:
        """Return momentum for a zero-based step, clamped at the final value."""
        if step < 0:
            raise ValueError("Schedule step cannot be negative.")
        if self.total_steps == 1 or step >= self.total_steps - 1:
            return self.end
        progress = float(step) / float(self.total_steps - 1)
        return self.start + (self.end - self.start) * progress

    def commit(self, step: int) -> float:
        """Record one successful EMA update and return its momentum."""
        value = self.value_at(step)
        self.last_step = step
        return value

    def state_dict(self) -> dict[str, int | float]:
        """Return the complete schedule state stored in checkpoints."""
        return {
            "start": self.start,
            "end": self.end,
            "total_steps": self.total_steps,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict[str, int | float]) -> None:
        """Restore state while rejecting a schedule with different bounds."""
        expected = self.state_dict()
        for name in ("start", "end", "total_steps"):
            if state[name] != expected[name]:
                raise ValueError(f"Checkpoint EMA schedule disagrees on {name}.")
        self.last_step = int(state["last_step"])


# ============================================================================
# RUNTIME AND CONFIGURATION
# ============================================================================


def _utc_timestamp() -> str:
    """Return a compact UTC timestamp suitable for run folders and records."""
    return datetime.now(timezone.utc).isoformat()


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch before model and loader construction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic device and reject unavailable CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def _resolve_amp_dtype(device: torch.device, enabled: bool) -> torch.dtype | None:
    """Choose BF16 on capable CUDA devices, otherwise FP16; CPU AMP stays off."""
    if not enabled or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _serialized_dataclass(value: object) -> dict[str, Any]:
    """Convert a config dataclass to a YAML-safe mapping."""
    serialized = asdict(value)
    for name, item in serialized.items():
        if isinstance(item, Path):
            serialized[name] = str(item)
    return serialized


def _build_resolved_config(
    training_config: FIJepaTrainingConfig,
    model_config: FIJepaModelConfig,
    data_config: FIJepaDataConfig,
    store: FrozenPanelStore,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    train_sample_count: int,
    validation_sample_count: int,
    steps_per_epoch: int,
) -> dict[str, Any]:
    """Capture every runtime input needed to understand or resume one run."""
    artifact_manifest = json.loads(
        (store.artifact_path / "manifest.json").read_text(encoding="utf-8")
    )
    return {
        "training": training_config.to_dict(),
        "model": _serialized_dataclass(model_config),
        "dataloader": _serialized_dataclass(data_config),
        "dataset_artifact": {
            "path": str(store.artifact_path.resolve()),
            "manifest": artifact_manifest,
        },
        "feature_dimensions": {
            group: len(names) for group, names in store.feature_names.items()
        },
        "runtime": {
            "device": str(device),
            "amp_dtype": None if amp_dtype is None else str(amp_dtype).removeprefix("torch."),
            "train_sample_count": train_sample_count,
            "validation_sample_count": validation_sample_count,
            "steps_per_epoch": steps_per_epoch,
            "planned_optimizer_steps": steps_per_epoch * training_config.epochs,
        },
    }


def _write_resolved_config(path: Path, resolved_config: dict[str, Any]) -> None:
    """Write the human-readable resolved run configuration atomically."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


# ============================================================================
# OPTIMIZER, BATCHES, AND METRICS
# ============================================================================


def build_adamw(model: FIJepaModel, config: FIJepaTrainingConfig) -> AdamW:
    """Build AdamW over trainable online parameters only.

    The EMA target encoder is frozen by the model contract and is also
    explicitly excluded here so optimizer state cannot silently grow around it.
    """
    target_ids = {id(parameter) for parameter in model.target_encoder.parameters()}
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in target_ids
    ]
    return AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move every tensor in a collated FI-JEPA batch to the training device."""
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _batch_mask_metrics(batch: dict[str, object]) -> dict[str, float]:
    """Summarize patch eligibility and temporal masking for one batch."""
    eligible = batch["patch_target_eligible"]
    target = batch["jepa_target_mask"]
    context = batch["patch_context_mask"]
    if not all(isinstance(value, torch.Tensor) for value in (eligible, target, context)):
        raise ValueError("Patch metric inputs must be tensors.")
    context_count = max(int(context.sum().item()), 1)
    return {
        "target_patch_eligibility_rate": float(eligible.sum().item()) / context_count,
        "masked_patch_rate": float(target.sum().item()) / context_count,
        "masked_patch_count_mean": float(target.sum(dim=1).float().mean().item()),
    }


def validate_jepa(
    model: FIJepaModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    """Compute deterministic target-count-weighted validation JEPA metrics."""
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    target_count = 0
    eligibility_weighted = 0.0
    masked_rate_weighted = 0.0
    masked_count_weighted = 0.0
    sample_count = 0

    with torch.inference_mode():
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            autocast = (
                torch.amp.autocast(device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with autocast:
                output = model(batch)
            batch_targets = int(output.target_patch_mask.sum().item())
            batch_samples = int(output.target_patch_mask.shape[0])
            metrics = _batch_mask_metrics(batch)
            weighted_loss += float(output.loss.item()) * batch_targets
            target_count += batch_targets
            eligibility_weighted += metrics["target_patch_eligibility_rate"] * batch_samples
            masked_rate_weighted += metrics["masked_patch_rate"] * batch_samples
            masked_count_weighted += metrics["masked_patch_count_mean"] * batch_samples
            sample_count += batch_samples

    if was_training:
        model.train()
    if target_count == 0 or sample_count == 0:
        raise RuntimeError("Validation produced no JEPA targets.")
    return {
        "validation_jepa_loss": weighted_loss / target_count,
        "target_patch_eligibility_rate": eligibility_weighted / sample_count,
        "masked_patch_rate": masked_rate_weighted / sample_count,
        "masked_patch_count_mean": masked_count_weighted / sample_count,
    }


# ============================================================================
# LOGGING AND CHECKPOINTS
# ============================================================================


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one compact JSON record and flush it to disk."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _capture_rng_state() -> dict[str, object]:
    """Capture all process RNG state required by the checkpoint contract."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    """Restore Python, NumPy, Torch CPU, and available Torch CUDA RNG states."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint_state(
    *,
    kind: str,
    model: FIJepaModel,
    optimizer: AdamW,
    lr_schedule: WarmupCosineLRSchedule,
    ema_schedule: LinearEMAMomentumSchedule,
    scaler: torch.amp.GradScaler,
    resume_epoch: int,
    global_step: int,
    best_validation_loss: float | None,
    resolved_config: dict[str, Any],
) -> dict[str, object]:
    """Build a complete checkpoint without any batch or sampler position."""
    return {
        "format_version": 1,
        "kind": kind,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_schedule.state_dict(),
        "ema_scheduler": ema_schedule.state_dict(),
        "scaler": scaler.state_dict(),
        "resume_epoch": resume_epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "rng_state": _capture_rng_state(),
        "resolved_config": resolved_config,
    }


def _write_checkpoint(path: Path, state: dict[str, object]) -> None:
    """Write one Torch checkpoint atomically within its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _resolve_resume_checkpoint(path: Path) -> tuple[Path, Path]:
    """Resolve a run directory or checkpoint path to checkpoint and run roots."""
    checkpoint = path / "checkpoints" / "latest.pt" if path.is_dir() else path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("Resume checkpoint must live inside a run's checkpoints directory.")
    return checkpoint, checkpoint.parent.parent


# ============================================================================
# PRETRAINING ORCHESTRATION
# ============================================================================


def train_fi_jepa(
    config: FIJepaTrainingConfig | Path | str | None = None,
    *,
    resume: Path | str | None = None,
    device_override: str | None = None,
) -> Path:
    """Run or resume FI-JEPA pretraining and return its self-contained run folder.

    Periodic checkpoints resume from the beginning of their saved epoch.
    Epoch-end and best-validation checkpoints resume from the following epoch.
    No batch cursor, sampler state, or processed-batch list is stored.
    """
    checkpoint: dict[str, Any] | None = None
    if resume is not None:
        checkpoint_path, run_dir = _resolve_resume_checkpoint(Path(resume))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        resolved_config = checkpoint["resolved_config"]
        training_config = FIJepaTrainingConfig.from_dict(resolved_config["training"])
        if device_override is not None:
            training_config = replace(training_config, device=device_override)
        model_config = FIJepaModelConfig(**resolved_config["model"])
        data_values = dict(resolved_config["dataloader"])
        data_values["artifact_path"] = Path(data_values["artifact_path"])
        data_config = FIJepaDataConfig(**data_values)
    else:
        if config is None:
            config = Path("configs/pretraining.yaml")
        training_config = (
            config if isinstance(config, FIJepaTrainingConfig) else FIJepaTrainingConfig.from_yaml(config)
        )
        if device_override is not None:
            training_config = replace(training_config, device=device_override)
        model_config = FIJepaModelConfig.from_yaml(training_config.model_config_path)
        data_config = FIJepaDataConfig.from_yaml(training_config.dataloader_config_path)
        data_config = replace(data_config, artifact_path=data_config.artifact_path.resolve())
        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = training_config.output_root / f"{run_stamp}_{training_config.run_name}"
        run_dir.mkdir(parents=True, exist_ok=False)

    if model_config.patch_len != data_config.patch_len:
        raise ValueError("Model and dataloader patch_len values must match.")
    if model_config.num_patches != data_config.num_patches:
        raise ValueError("Model and dataloader num_patches values must match.")

    device = _resolve_device(training_config.device)
    amp_dtype = _resolve_amp_dtype(device, training_config.mixed_precision)
    _seed_everything(data_config.seed)

    store = FrozenPanelStore(data_config.artifact_path)
    train_loader = build_fi_jepa_dataloader(data_config, "train", store=store)
    validation_loader = build_fi_jepa_dataloader(
        data_config, "validation", store=store, shuffle=False
    )
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise RuntimeError("Training and validation loaders must both contain at least one batch.")

    model = FIJepaModel.from_store(model_config, store).to(device)
    optimizer = build_adamw(model, training_config)
    total_steps = len(train_loader) * training_config.epochs
    lr_schedule = WarmupCosineLRSchedule(
        optimizer,
        base_lr=training_config.lr,
        min_lr=training_config.min_lr,
        warmup_steps=len(train_loader) * training_config.warmup_epochs,
        total_steps=total_steps,
    )
    ema_schedule = LinearEMAMomentumSchedule(
        start=training_config.ema_momentum_start,
        end=training_config.ema_momentum_end,
        total_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_dtype == torch.float16,
    )

    if checkpoint is None:
        resolved_config = _build_resolved_config(
            training_config,
            model_config,
            data_config,
            store,
            device=device,
            amp_dtype=amp_dtype,
            train_sample_count=len(train_loader.dataset),
            validation_sample_count=len(validation_loader.dataset),
            steps_per_epoch=len(train_loader),
        )
        _write_resolved_config(run_dir / "resolved_config.yaml", resolved_config)
        start_epoch = 0
        global_step = 0
        best_validation_loss: float | None = None
    else:
        runtime = resolved_config["runtime"]
        current_manifest = json.loads(
            (store.artifact_path / "manifest.json").read_text(encoding="utf-8")
        )
        if str(store.artifact_path.resolve()) != resolved_config["dataset_artifact"]["path"]:
            raise ValueError("Current frozen artifact path differs from the resumed run.")
        if current_manifest != resolved_config["dataset_artifact"]["manifest"]:
            raise ValueError("Current frozen artifact manifest differs from the resumed run.")
        if int(runtime["steps_per_epoch"]) != len(train_loader):
            raise ValueError("Current training loader length differs from the resumed run.")
        if int(runtime["planned_optimizer_steps"]) != total_steps:
            raise ValueError("Current planned optimizer steps differ from the resumed run.")
        if int(runtime["train_sample_count"]) != len(train_loader.dataset):
            raise ValueError("Current training sample count differs from the resumed run.")
        if int(runtime["validation_sample_count"]) != len(validation_loader.dataset):
            raise ValueError("Current validation sample count differs from the resumed run.")
        # Device is the only supported resume-time override. Persist the actual
        # resumed runtime so later checkpoints do not claim a stale device.
        resolved_config["training"] = training_config.to_dict()
        runtime["device"] = str(device)
        runtime["amp_dtype"] = (
            None if amp_dtype is None else str(amp_dtype).removeprefix("torch.")
        )
        _write_resolved_config(run_dir / "resolved_config.yaml", resolved_config)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_schedule.load_state_dict(checkpoint["lr_scheduler"])
        ema_schedule.load_state_dict(checkpoint["ema_scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["resume_epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation_loss = checkpoint["best_validation_loss"]
        _restore_rng_state(checkpoint["rng_state"])
        _append_jsonl(
            run_dir / "train_log.jsonl",
            {
                "event": "resume",
                "timestamp": _utc_timestamp(),
                "epoch": start_epoch,
                "step": global_step,
                "checkpoint_kind": checkpoint["kind"],
            },
        )

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    print(f"FI-JEPA run: {run_dir}")

    for epoch in range(start_epoch, training_config.epochs):
        dataset = train_loader.dataset
        if not hasattr(dataset, "set_epoch"):
            raise TypeError("FI-JEPA training dataset must implement set_epoch(epoch).")
        dataset.set_epoch(epoch)
        model.train()

        interval_start = time.perf_counter()
        interval_samples = 0
        interval_targets = 0
        interval_weighted_loss = 0.0
        interval_gradient_norm = 0.0
        interval_successful_steps = 0
        interval_metrics = {
            "target_patch_eligibility_rate": 0.0,
            "masked_patch_rate": 0.0,
            "masked_patch_count_mean": 0.0,
        }

        for batch_index, cpu_batch in enumerate(train_loader):
            batch = _move_batch(cpu_batch, device)
            step_index = global_step
            learning_rate = lr_schedule.apply(step_index, commit=False)
            ema_momentum = ema_schedule.value_at(step_index)
            optimizer.zero_grad(set_to_none=True)

            autocast = (
                torch.amp.autocast(device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with autocast:
                output = model(batch)

            if scaler.is_enabled():
                scaler.scale(output.loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.grad_clip_norm
                )
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_succeeded = scaler.get_scale() >= old_scale
            else:
                output.loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.grad_clip_norm
                )
                optimizer.step()
                step_succeeded = True

            if not step_succeeded:
                _append_jsonl(
                    log_path,
                    {
                        "event": "amp_overflow",
                        "timestamp": _utc_timestamp(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                )
                continue

            lr_schedule.apply(step_index, commit=True)
            ema_momentum = ema_schedule.commit(step_index)
            model.update_target_encoder(ema_momentum)
            global_step += 1

            batch_targets = int(output.target_patch_mask.sum().item())
            batch_samples = int(output.target_patch_mask.shape[0])
            mask_metrics = _batch_mask_metrics(batch)
            interval_samples += batch_samples
            interval_targets += batch_targets
            interval_weighted_loss += float(output.loss.item()) * batch_targets
            interval_gradient_norm += float(gradient_norm.item())
            interval_successful_steps += 1
            for name in interval_metrics:
                interval_metrics[name] += mask_metrics[name] * batch_samples

            if (
                global_step % training_config.logging_every_steps == 0
                or batch_index == len(train_loader) - 1
            ):
                elapsed = max(time.perf_counter() - interval_start, 1e-12)
                _append_jsonl(
                    log_path,
                    {
                        "event": "train",
                        "timestamp": _utc_timestamp(),
                        "epoch": epoch,
                        "step": global_step,
                        "train_jepa_loss": interval_weighted_loss / interval_targets,
                        "learning_rate": learning_rate,
                        "ema_momentum": ema_momentum,
                        "gradient_norm": interval_gradient_norm / interval_successful_steps,
                        **{
                            name: value / interval_samples
                            for name, value in interval_metrics.items()
                        },
                        "samples_per_second": interval_samples / elapsed,
                    },
                )
                interval_start = time.perf_counter()
                interval_samples = 0
                interval_targets = 0
                interval_weighted_loss = 0.0
                interval_gradient_norm = 0.0
                interval_successful_steps = 0
                interval_metrics = {name: 0.0 for name in interval_metrics}

            if global_step % training_config.checkpoint_every_steps == 0:
                state = _checkpoint_state(
                    kind="periodic",
                    model=model,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    ema_schedule=ema_schedule,
                    scaler=scaler,
                    resume_epoch=epoch,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    resolved_config=resolved_config,
                )
                periodic_path = checkpoints_dir / f"step_{global_step:09d}.pt"
                # Resuming an older periodic checkpoint can replay global step
                # numbers already present in the run. Preserve those immutable
                # recovery points while still advancing latest.pt.
                if not periodic_path.exists():
                    _write_checkpoint(periodic_path, state)
                _write_checkpoint(checkpoints_dir / "latest.pt", state)

        should_validate = (epoch + 1) % training_config.validation_every_epochs == 0
        if should_validate:
            validation_metrics = validate_jepa(model, validation_loader, device, amp_dtype)
            representation_result: dict[str, object] | None = None
            if training_config.representation_evaluation_enabled:
                checkpoint_id = (
                    f"step_{global_step:09d}_{model_state_hash(model)[:12]}"
                )
                representation_result = run_representation_evaluation(
                    model,
                    store,
                    data_config,
                    device=device,
                    amp_dtype=amp_dtype,
                    n_components=training_config.representation_pca_components,
                    views_per_date=training_config.representation_views_per_date,
                    output_dir=(
                        run_dir
                        / "representation_diagnostics"
                        / f"step_{global_step:09d}"
                    ),
                    checkpoint_id=checkpoint_id,
                    checkpoint_step=global_step,
                    checkpoint_format_version=1,
                    model_version=canonical_version_hash(resolved_config["model"]),
                    export_embeddings=training_config.representation_export_embeddings,
                )
            _append_jsonl(
                log_path,
                {
                    "event": "validation",
                    "timestamp": _utc_timestamp(),
                    "epoch": epoch,
                    "step": global_step,
                    **validation_metrics,
                    **(
                        {"representation_diagnostics": representation_result["summary"]}
                        if representation_result is not None
                        else {}
                    ),
                },
            )
            if (
                best_validation_loss is None
                or validation_metrics["validation_jepa_loss"] < best_validation_loss
            ):
                best_validation_loss = validation_metrics["validation_jepa_loss"]
                best_state = _checkpoint_state(
                    kind="epoch_end",
                    model=model,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    ema_schedule=ema_schedule,
                    scaler=scaler,
                    resume_epoch=epoch + 1,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    resolved_config=resolved_config,
                )
                _write_checkpoint(checkpoints_dir / "best_validation.pt", best_state)

        latest_state = _checkpoint_state(
            kind="epoch_end",
            model=model,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            ema_schedule=ema_schedule,
            scaler=scaler,
            resume_epoch=epoch + 1,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            resolved_config=resolved_config,
        )
        _write_checkpoint(checkpoints_dir / "latest.pt", latest_state)
        print(f"Completed epoch {epoch + 1}/{training_config.epochs} at step {global_step}.")

    return run_dir


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse the config-driven new-run or checkpoint-resume CLI."""
    parser = argparse.ArgumentParser(description="Train or resume the FI-JEPA model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretraining.yaml"),
        help="Pretraining YAML used for a new run.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Existing run directory or checkpoint. Its resolved config is authoritative.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override run device.")
    return parser.parse_args()


def main() -> None:
    """Run the FI-JEPA pretraining CLI."""
    args = parse_args()
    train_fi_jepa(args.config, resume=args.resume, device_override=args.device)


if __name__ == "__main__":
    main()
