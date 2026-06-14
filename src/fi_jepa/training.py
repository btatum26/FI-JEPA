from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
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
from tqdm.auto import tqdm
import yaml

from fi_jepa.dataloader import DensePanelStore, FIJepaDataConfig, build_fi_jepa_dataloader
from fi_jepa.losses import pooled_variance_covariance_loss
from fi_jepa.model import (
    ENCODER_BATCH_TENSOR_NAMES,
    JEPA_BATCH_TENSOR_NAMES,
    FIJepaModel,
    load_fi_jepa_model_state,
)
from fi_jepa.model_config import FIJepaModelConfig
from fi_jepa.model_output import FIJepaOutput
from fi_jepa.representation import (
    canonical_version_hash,
    model_state_hash,
    run_representation_evaluation,
)
from fi_jepa.training_config import FIJepaTrainingConfig
from fi_jepa.training_profiler import (
    PythonStackSampler,
    TimingRecord,
    append_runtime_timing_to_profile_summary,
    build_training_profiler,
    profile_range,
    write_runtime_timing_summary,
)
from fi_jepa.tokenizer import masked_mean

from fi_jepa.schedulers import WarmupCosineLRSchedule, LinearEMAMomentumSchedule

CHECKPOINT_FORMAT_VERSION = 2


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
    store: DensePanelStore,
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

    The complete EMA target branch is frozen by the model contract and is also
    explicitly excluded here so optimizer state cannot silently grow around it.
    """
    target_ids = {id(parameter) for parameter in model.target_parameters()}
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in target_ids
    ]
    return AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move only model-required tensors, leaving duplicate views and metrics on CPU."""
    required = ENCODER_BATCH_TENSOR_NAMES | JEPA_BATCH_TENSOR_NAMES
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in required
        if isinstance(batch[name], torch.Tensor)
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


@torch.no_grad()
def _effective_rank(values: torch.Tensor) -> float:
    """Return covariance-spectrum effective rank for one representation matrix."""
    if values.ndim != 2:
        raise ValueError(f"Effective-rank values must have shape [N, D]; got {values.shape}.")
    if values.shape[0] < 2:
        return 0.0

    # Center valid target rows so rank measures represented variation rather than a large shared mean direction.
    centered = values.detach().float()
    centered = centered - centered.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.svdvals(centered).square()
    spectrum_sum = spectrum.sum()
    if float(spectrum_sum.item()) <= 0.0:
        return 0.0
    probabilities = spectrum[spectrum > 0.0] / spectrum_sum
    return float(torch.exp(-(probabilities * probabilities.log()).sum()).item())


@torch.no_grad()
def _batch_representation_metrics(output: FIJepaOutput) -> dict[str, float]:
    """Measure matched cosine and valid-target rank for one logged training batch."""
    target_mask = output.target_patch_mask
    predicted = output.predicted_targets[target_mask]
    targets = output.target_representations[target_mask]
    if predicted.shape[0] == 0:
        raise RuntimeError("A logged training batch produced no valid JEPA targets.")

    normalized_prediction = nn.functional.normalize(predicted.float(), dim=-1)
    normalized_target = nn.functional.normalize(targets.float(), dim=-1)
    matched_cosine = (normalized_prediction * normalized_target).sum(dim=-1).mean()
    return {
        "matched_target_cosine": float(matched_cosine.item()),
        "predictor_effective_rank": _effective_rank(predicted),
        "target_effective_rank": _effective_rank(targets),
    }


def _training_objective(
    output: FIJepaOutput,
    config: FIJepaTrainingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine JEPA prediction loss with weak visible-context anti-collapse terms.

    The regularizer uses one masked-mean visible-context state per sample. It
    therefore adds no second encoder pass, never exposes hidden target patches
    to the online encoder, and estimates batch variation without treating patch
    positions as interchangeable samples.
    """
    pooled_context = masked_mean(
        output.context_representations,
        output.context_mask,
        dimension=1,
    )  # [B, C, D] -> [B, D].
    variance_loss, covariance_loss, mean_feature_std = pooled_variance_covariance_loss(
        pooled_context,
        variance_floor=config.anti_collapse_variance_floor,
        epsilon=config.anti_collapse_epsilon,
    )
    weighted_variance = variance_loss * config.anti_collapse_variance_weight
    weighted_covariance = covariance_loss * config.anti_collapse_covariance_weight
    total_loss = output.loss.float() + weighted_variance + weighted_covariance
    return total_loss, {
        "anti_collapse_variance_loss": variance_loss,
        "anti_collapse_covariance_loss": covariance_loss,
        "anti_collapse_weighted_variance_loss": weighted_variance,
        "anti_collapse_weighted_covariance_loss": weighted_covariance,
        "context_pooled_mean_feature_std": mean_feature_std,
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

    print(f"Starting JEPA validation over {len(loader)} batches.")
    with torch.inference_mode():
        for cpu_batch in tqdm(
            loader,
            desc="Validation",
            total=len(loader),
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ):
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
            metrics = _batch_mask_metrics(cpu_batch)
            weighted_loss += float(output.loss.item()) * batch_targets
            target_count += batch_targets
            eligibility_weighted += metrics["target_patch_eligibility_rate"] * batch_samples
            masked_rate_weighted += metrics["masked_patch_rate"] * batch_samples
            masked_count_weighted += metrics["masked_patch_count_mean"] * batch_samples
            sample_count += batch_samples

    print(f"validation_jepa_loss={weighted_loss / target_count:.4f} | target_patch_eligibility_rate={eligibility_weighted / sample_count:.4f} | masked_patch_rate={masked_rate_weighted / sample_count:.4f} | masked_patch_count_mean={masked_count_weighted / sample_count:.4f}")
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
        "format_version": CHECKPOINT_FORMAT_VERSION,
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

def _log_training_step(
    log_path: Path,
    log_metrics: dict[str, float],
    epoch: int,
    global_step: int,
    progress: tqdm,
    
):  
    
    # add metrics to log file
    _append_jsonl(
        log_path,
        {
            "event": "train",
            "timestamp": _utc_timestamp(),
            "epoch": epoch,
            "step": global_step,
            **log_metrics,
        },
    )
    progress.set_postfix(
        loss=f"{log_metrics['train_loss']:.4f}",
        cos=f"{log_metrics['matched_target_cosine']:.4f}",
        rank=(
            f"{log_metrics['predictor_effective_rank']:.1f}"
            f"/{log_metrics['target_effective_rank']:.1f}"
        ),
        lr=f"{log_metrics['learning_rate']:.2e}",
        samples_per_second=f"{log_metrics['samples_per_second']:.1f}",
        refresh=False,
    )
    



# ============================================================================
# PRETRAINING ORCHESTRATION
# ============================================================================


def train_fi_jepa(
    config: FIJepaTrainingConfig | Path | str | None = None,
    *,
    resume: Path | str | None = None,
    device_override: str | None = None,
    profile: bool = False,
    profile_wait_steps: int = 5,
    profile_warmup_steps: int = 5,
    profile_active_steps: int = 20,
    profile_python_stacks: bool = False,
) -> Path:
    """Run or resume FI-JEPA pretraining and return its self-contained run folder.

    Periodic checkpoints resume from the beginning of their saved epoch.
    Epoch-end and best-validation checkpoints resume from the following epoch.
    No batch cursor, sampler state, or processed-batch list is stored.
    Profiling runs are bounded, non-resumable, and skip validation/checkpoints.
    """
    if profile and resume is not None:
        raise ValueError("Profiling runs cannot resume an existing run.")
    if profile_python_stacks and not profile:
        raise ValueError("Python stack sampling requires profile=True.")
    if profile_wait_steps < 0 or profile_warmup_steps < 0 or profile_active_steps <= 0:
        raise ValueError("Profiler steps require wait >= 0, warmup >= 0, and active > 0.")
    
    
    # =========================================
    # Is this a resumed run or a new run?
    # =========================================

    checkpoint: dict[str, Any] | None = None
    if resume is not None:
        # Resume training from an existing checkpoint, ignoring any provided config. The
        checkpoint_path, run_dir = _resolve_resume_checkpoint(Path(resume))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        resolved_config = checkpoint["resolved_config"]
        training_config = FIJepaTrainingConfig.from_dict(resolved_config["training"])
        if device_override is not None:
            training_config = replace(training_config, device=device_override)
        model_config = FIJepaModelConfig(**resolved_config["model"])

        data_values = dict(resolved_config["dataloader"])
        data_values["artifact_path"] = Path(data_values["artifact_path"])
        data_values["cache_root"] = Path(data_values["cache_root"])
        data_config = FIJepaDataConfig(**data_values)

    else:
        # start a new training run with the provided config or its default path.
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



    # =======================================
    # Do all the init stuff
    # =======================================
    
    # check configs make sense
    if model_config.patch_len != data_config.patch_len:
        raise ValueError("Model and dataloader patch_len values must match.")
    if model_config.num_patches != data_config.num_patches:
        raise ValueError("Model and dataloader num_patches values must match.")

    device = _resolve_device(training_config.device)
    amp_dtype = _resolve_amp_dtype(device, training_config.mixed_precision)
    _seed_everything(data_config.seed)

    # build datastore and dataloaders
    store = DensePanelStore(data_config.artifact_path, cache_root=data_config.cache_root)
    train_loader = build_fi_jepa_dataloader(data_config, "train", store=store)
    validation_loader = build_fi_jepa_dataloader(data_config, "validation", store=store, shuffle=False)
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise RuntimeError("Training and validation loaders must both contain at least one batch.")

    model = FIJepaModel.from_store(model_config, store).to(device)
    optimizer = build_adamw(model, training_config)
    total_steps = len(train_loader) * training_config.epochs
    
    profiler_steps = profile_wait_steps + profile_warmup_steps + profile_active_steps
    if profile and total_steps < profiler_steps:
        raise ValueError(f"Profiler requires {profiler_steps} training steps, but this run only has {total_steps}.")
    
    
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
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    if checkpoint is None:          # if this is a new run
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
        
    else:                           # if this is a resumed run
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
        
        # Device is the only supported resume-time override. 
        # Persist the actual resumed runtime so later checkpoints do not claim a stale device.
        resolved_config["training"] = training_config.to_dict()
        runtime["device"] = str(device)
        runtime["amp_dtype"] = (None if amp_dtype is None else str(amp_dtype).removeprefix("torch."))
        
        _write_resolved_config(run_dir / "resolved_config.yaml", resolved_config)
        load_fi_jepa_model_state(model, checkpoint["model"])
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

    # pathing and logging setup
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    runtime_summary_path = run_dir / "runtime_summary.txt"
    warmup_timing_records: list[TimingRecord] = []
    boundary_timing_records: list[TimingRecord] = []
    profiler_dir = run_dir / "profiler"
    
    # sets to None if profile = false
    training_profiler = (
        build_training_profiler(
            profiler_dir,
            device,
            wait_steps=profile_wait_steps,
            warmup_steps=profile_warmup_steps,
            active_steps=profile_active_steps,
            python_stacks=profile_python_stacks,
        )
        if profile
        else None
    )
    # sets to None if profile_python_stacks = false
    python_stack_sampler = (PythonStackSampler(profiler_dir / "cpu_stacks.txt") if profile and profile_python_stacks else None)
    
    
    print(f"FI-JEPA run: {run_dir}")
    print(
        "Training plan: "
        f"device={device} | train_samples={len(train_loader.dataset)} | "
        f"batch_size={data_config.batch_size} | steps_per_epoch={len(train_loader)} | "
        f"epochs={training_config.epochs} | total_steps={total_steps} | "
        f"validation_samples={len(validation_loader.dataset)} | "
        f"validation_batches={len(validation_loader)}"
    )
    if training_profiler is not None:
        print(
            "Profiler plan: "
            f"wait_steps={profile_wait_steps} | warmup_steps={profile_warmup_steps} | "
            f"active_steps={profile_active_steps} | python_stacks={profile_python_stacks} | "
            f"output={profiler_dir}"
        )
        training_profiler.start()
        if python_stack_sampler is not None:
            python_stack_sampler.start()

    profile_complete = False
    profiled_steps = 0
    
    
    # =======================================
    # Main training loop 
    # =======================================
    
    print(f"Starting training from epoch {start_epoch + 1}/{training_config.epochs}, global step {global_step}.")
    for epoch in range(start_epoch, training_config.epochs):
        epoch_warmup_started = time.perf_counter()
        dataset = train_loader.dataset
        if not hasattr(dataset, "set_epoch"):
            raise TypeError("FI-JEPA training dataset must implement set_epoch(epoch).")
        dataset_update_started = time.perf_counter()
        with profile_range("epoch_warmup.dataset_epoch_update", enabled=training_profiler is not None):
            dataset.set_epoch(epoch)
            model.train()
        # print(f"Starting epoch {epoch + 1}/{training_config.epochs}: creating training iterator.")
        
        
        with profile_range("epoch_warmup.dataloader_iterator_startup", enabled=training_profiler is not None,):
                train_iterator = iter(train_loader)
                
        # profiler setup
        if training_profiler is not None:
            iterator_started = time.perf_counter()
            dataset_epoch_update_seconds = time.perf_counter() - dataset_update_started
            dataloader_iterator_startup_seconds = time.perf_counter() - iterator_started
            warmup_record: TimingRecord = {
                "epoch": epoch,
                "dataset_epoch_update_seconds": dataset_epoch_update_seconds,
                "dataloader_iterator_startup_seconds": dataloader_iterator_startup_seconds,
                "total_seconds": time.perf_counter() - epoch_warmup_started,
            }
            warmup_timing_records.append(warmup_record)
            _append_jsonl(
                log_path,
                {
                    "event": "epoch_warmup",
                    "timestamp": _utc_timestamp(),
                    "step": global_step,
                    **warmup_record,
                },
            )
            write_runtime_timing_summary(runtime_summary_path, warmup_timing_records, boundary_timing_records,)
            print(
                "Epoch warm-up: "
                f"dataset_epoch_update={dataset_epoch_update_seconds:.3f}s | "
                f"dataloader_iterator_startup={dataloader_iterator_startup_seconds:.3f}s | "
                f"total={float(warmup_record['total_seconds']):.3f}s"
            )

        interval_start = time.perf_counter()
        interval_samples = 0
        interval_targets = 0
        interval_total_loss = 0.0
        interval_weighted_jepa_loss = 0.0
        interval_gradient_norm = 0.0
        interval_successful_steps = 0
        interval_loss_components = {
            "anti_collapse_variance_loss": 0.0,
            "anti_collapse_covariance_loss": 0.0,
            "anti_collapse_weighted_variance_loss": 0.0,
            "anti_collapse_weighted_covariance_loss": 0.0,
            "context_pooled_mean_feature_std": 0.0,
        }
        interval_metrics = {
            "target_patch_eligibility_rate": 0.0,
            "masked_patch_rate": 0.0,
            "masked_patch_count_mean": 0.0,
        }
        
        progress = tqdm(
            range(len(train_loader)),
            desc=f"Epoch {epoch + 1}/{training_config.epochs}: ",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}",
            total=len(train_loader),
            unit="step",
            dynamic_ncols=True,
        )
        
        # =========================================
        # Training steps within the epoch
        # =========================================
        for batch_index in progress:
            with profile_range("next_batch_cpu", enabled=training_profiler is not None):
                cpu_batch = next(train_iterator)
            with profile_range("move_batch_to_device", enabled=training_profiler is not None):
                batch = _move_batch(cpu_batch, device)
            step_index = global_step
            learning_rate = lr_schedule.apply(step_index, commit=False)
            ema_momentum = ema_schedule.value_at(step_index)
            optimizer.zero_grad(set_to_none=True)

            autocast = torch.amp.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
            
            # forward step
            with profile_range("forward_and_loss", enabled=training_profiler is not None):
                with autocast:
                    output = model(batch)
                    training_loss, loss_components = _training_objective(output, training_config)

            # back propigation and optimization
            with profile_range("backward_and_optimizer", enabled=training_profiler is not None):
                if scaler.is_enabled():
                    scaler.scale(training_loss).backward()
                    scaler.unscale_(optimizer)
                    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    step_succeeded = scaler.get_scale() >= old_scale
                else:
                    training_loss.backward()
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
                if training_profiler is not None:
                    training_profiler.step()
                    profiled_steps += 1
                    profile_complete = profiled_steps >= profiler_steps
                if profile_complete:
                    break
                continue

            # Update the complete EMA target branch after a successful online step.
            with profile_range("ema_update", enabled=training_profiler is not None):
                lr_schedule.apply(step_index, commit=True)
                ema_momentum = ema_schedule.commit(step_index)
                model.update_target_encoder(ema_momentum)
                global_step += 1

            batch_targets = int(output.target_patch_mask.sum().item())
            batch_samples = int(output.target_patch_mask.shape[0])
            mask_metrics = _batch_mask_metrics(cpu_batch)
            interval_samples += batch_samples
            interval_targets += batch_targets
            interval_total_loss += float(training_loss.item())
            interval_weighted_jepa_loss += float(output.loss.item()) * batch_targets
            interval_gradient_norm += float(gradient_norm.item())
            interval_successful_steps += 1
            for name, value in loss_components.items():
                interval_loss_components[name] += float(value.item())
            for name in interval_metrics:
                interval_metrics[name] += mask_metrics[name] * batch_samples

            # if this is a log step or the last step of the epoch
            if (  global_step % training_config.logging_every_steps == 0 or batch_index == len(train_loader) - 1):
                elapsed = max(time.perf_counter() - interval_start, 1e-12)
                log_metrics = {
                    "train_loss": interval_total_loss / interval_successful_steps,
                    "train_jepa_loss": interval_weighted_jepa_loss / interval_targets,
                    "learning_rate": learning_rate,
                    "ema_momentum": ema_momentum,
                    "gradient_norm": interval_gradient_norm / interval_successful_steps,
                    **{
                        name: value / interval_successful_steps
                        for name, value in interval_loss_components.items()
                    },
                    **_batch_representation_metrics(output),
                    **{name: value / interval_samples for name, value in interval_metrics.items()},
                    "samples_per_second": interval_samples / elapsed,
                }
                _log_training_step(log_path, log_metrics, epoch, global_step, progress)
                interval_start = time.perf_counter()
                interval_samples = 0
                interval_targets = 0
                interval_total_loss = 0.0
                interval_weighted_jepa_loss = 0.0
                interval_gradient_norm = 0.0
                interval_successful_steps = 0
                interval_loss_components = {
                    name: 0.0 for name in interval_loss_components
                }
                interval_metrics = {name: 0.0 for name in interval_metrics}
                
            # write checkpoint if its time
            if training_profiler is None and global_step % training_config.checkpoint_every_steps == 0:
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
                
                # Resuming an older periodic checkpoint can replay global step numbers already present in the run. 
                # Preserve those immutable recovery points while still advancing latest.pt.
                if not periodic_path.exists():
                    _write_checkpoint(periodic_path, state)
                _write_checkpoint(checkpoints_dir / "latest.pt", state)

            if training_profiler is not None:
                training_profiler.step()
                profiled_steps += 1
                profile_complete = profiled_steps >= profiler_steps
            if profile_complete:
                break

        if training_profiler is not None:
            if profile_complete:
                break
            continue

        epoch_boundary_started = time.perf_counter()
        validation_seconds = 0.0
        representation_evaluation_seconds = 0.0
        best_checkpoint_seconds = 0.0
        
        # runs a validation and representation eval if its time
        if (epoch + 1) % training_config.validation_every_epochs == 0:
            validation_started = time.perf_counter()
            # run the validation
            with profile_range("epoch_boundary.validation", enabled=training_profiler is not None):
                validation_metrics = validate_jepa(model, validation_loader, device, amp_dtype)
            validation_seconds = time.perf_counter() - validation_started
            representation_result: dict[str, object] | None = None
            
            # whether to run the representation eval
            should_evaluate_representations = (
                training_config.representation_evaluation_enabled
                and (
                    (epoch + 1) % training_config.representation_evaluation_every_epochs == 0
                    or epoch + 1 == training_config.epochs
                )
            )
            if should_evaluate_representations:
                checkpoint_id = (
                    f"step_{global_step:09d}_{model_state_hash(model)[:12]}"
                )
                representation_started = time.perf_counter()
                with profile_range(
                    "epoch_boundary.representation_evaluation",
                    enabled=training_profiler is not None,
                ):
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
                        checkpoint_format_version=CHECKPOINT_FORMAT_VERSION,
                        model_version=canonical_version_hash(resolved_config["model"]),
                        export_embeddings=training_config.representation_export_embeddings,
                    )
                representation_evaluation_seconds = time.perf_counter() - representation_started
                
            # writes to the jsonl log
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
            
            # if this is the best validation so far, write a best_validation.pt checkpoint
            if (
                best_validation_loss is None
                or validation_metrics["validation_jepa_loss"] < best_validation_loss
            ):
                best_validation_loss = validation_metrics["validation_jepa_loss"]
                best_checkpoint_started = time.perf_counter()
                with profile_range(
                    "epoch_boundary.best_checkpoint",
                    enabled=training_profiler is not None,
                ):
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
                best_checkpoint_seconds = time.perf_counter() - best_checkpoint_started

        latest_checkpoint_started = time.perf_counter()
        with profile_range("epoch_boundary.latest_checkpoint", enabled=training_profiler is not None):
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
        latest_checkpoint_seconds = time.perf_counter() - latest_checkpoint_started
        boundary_record: TimingRecord = {
            "epoch": epoch,
            "validation_seconds": validation_seconds,
            "representation_evaluation_seconds": representation_evaluation_seconds,
            "best_checkpoint_seconds": best_checkpoint_seconds,
            "latest_checkpoint_seconds": latest_checkpoint_seconds,
            "total_seconds": time.perf_counter() - epoch_boundary_started,
        }
        boundary_timing_records.append(boundary_record)
        _append_jsonl(
            log_path,
            {
                "event": "epoch_boundary",
                "timestamp": _utc_timestamp(),
                "step": global_step,
                **boundary_record,
            },
        )
        write_runtime_timing_summary(
            runtime_summary_path,
            warmup_timing_records,
            boundary_timing_records,
        )
        #print(
        #    "Epoch boundary: "
        #    f"validation={validation_seconds:.3f}s | "
        #    f"representation_evaluation={representation_evaluation_seconds:.3f}s | "
        #    f"best_checkpoint={best_checkpoint_seconds:.3f}s | "
        #    f"latest_checkpoint={latest_checkpoint_seconds:.3f}s | "
        #    f"total={float(boundary_record['total_seconds']):.3f}s"
        #)
        #print(f"Completed epoch {epoch + 1}/{training_config.epochs} at step {global_step}.")

    if training_profiler is not None:
        training_profiler.stop()
        if python_stack_sampler is not None:
            python_stack_sampler.stop()
        append_runtime_timing_to_profile_summary(
            profiler_dir / "summary.txt",
            runtime_summary_path,
        )
        append_runtime_timing_to_profile_summary(
            profiler_dir / "cpu_summary.txt",
            runtime_summary_path,
        )
        print(f"Profiler capture complete after {profiled_steps} steps: {profiler_dir}")

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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture a bounded training-only PyTorch profiler trace, then stop.",
    )
    parser.add_argument(
        "--profile-wait-steps",
        type=int,
        default=5,
        help="Unrecorded profiler steps before warmup.",
    )
    parser.add_argument(
        "--profile-warmup-steps",
        type=int,
        default=5,
        help="Profiler warmup steps before trace capture.",
    )
    parser.add_argument(
        "--profile-active-steps",
        type=int,
        default=20,
        help="Training steps recorded in the profiler trace.",
    )
    parser.add_argument(
        "--profile-python-stacks",
        action="store_true",
        help="Sample Python stacks into profiler/cpu_stacks.txt; adds substantial overhead.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the FI-JEPA pretraining CLI."""
    args = parse_args()
    train_fi_jepa(
        args.config,
        resume=args.resume,
        device_override=args.device,
        profile=args.profile,
        profile_wait_steps=args.profile_wait_steps,
        profile_warmup_steps=args.profile_warmup_steps,
        profile_active_steps=args.profile_active_steps,
        profile_python_stacks=args.profile_python_stacks,
    )


if __name__ == "__main__":
    main()
