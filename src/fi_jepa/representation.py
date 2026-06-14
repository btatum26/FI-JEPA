from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from fi_jepa.dataloader import (
    FIJepaDataConfig,
    DensePanelStore,
    build_fi_jepa_embedding_dataloader,
)
from fi_jepa.model import ENCODER_BATCH_TENSOR_NAMES, FIJepaModel, load_fi_jepa_model_state
from fi_jepa.model_config import FIJepaModelConfig

EMBEDDING_SCHEMA_VERSION = 1
PCA_FORMAT_VERSION = 1


# ============================================================================
# VERSIONING AND PCA EXPORTER
# ============================================================================


def canonical_version_hash(value: object) -> str:
    """Hash one JSON-compatible value using stable canonical serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_state_hash(model: FIJepaModel) -> str:
    """Hash the complete checkpoint model state without serializing a checkpoint."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PCAExporter:
    """Train-only non-whitened PCA projection used for versioned 8D exports."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    version: str

    def transform(self, states: np.ndarray) -> np.ndarray:
        """Project pooled states into checkpoint-specific PCA coordinates."""
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("PCA input dimensions do not match the fitted exporter.")
        return (values - self.mean) @ self.components.T

    def save(self, path: Path) -> None:
        """Write the complete PCA projection contract as a compressed NPZ."""
        np.savez_compressed(
            path,
            format_version=np.asarray(PCA_FORMAT_VERSION, dtype=np.int64),
            mean=self.mean,
            components=self.components,
            explained_variance=self.explained_variance,
            explained_variance_ratio=self.explained_variance_ratio,
            version=np.asarray(self.version),
            whiten=np.asarray(False),
        )


def fit_pca_exporter(states: np.ndarray, n_components: int) -> PCAExporter:
    """Fit deterministic, sign-canonicalized, non-whitened PCA on train states."""
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("PCA states must have shape [samples, dimensions].")
    if not np.isfinite(values).all():
        raise ValueError("PCA states must be finite.")
    if not 1 <= n_components <= min(values.shape):
        raise ValueError("PCA component count exceeds available samples or dimensions.")

    mean = values.mean(axis=0)
    centered = values - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = right_vectors[:n_components].copy()

    # SVD component signs are arbitrary. Canonicalizing the largest loading
    # makes repeated fits deterministic, but does not make axes comparable
    # across checkpoints when eigenvalues are close.
    for index, component in enumerate(components):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            components[index] *= -1.0

    denominator = max(values.shape[0] - 1, 1)
    all_variance = np.square(singular_values) / denominator
    explained_variance = all_variance[:n_components]
    total_variance = float(all_variance.sum())
    explained_ratio = (
        explained_variance / total_variance
        if total_variance > 0.0
        else np.zeros_like(explained_variance)
    )
    version_hasher = hashlib.sha256()
    for array in (mean, components, explained_variance, explained_ratio):
        version_hasher.update(np.ascontiguousarray(array).tobytes())
    return PCAExporter(
        mean=mean,
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_ratio,
        version=version_hasher.hexdigest(),
    )


# ============================================================================
# REPRESENTATION DIAGNOSTICS
# ============================================================================


def _pairwise_cosine_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize off-diagonal pairwise cosine geometry without storing a matrix."""
    count = values.shape[0]
    if count < 2:
        return {
            "pair_count": 0,
            "similarity_mean": 0.0,
            "similarity_median": 0.0,
            "similarity_std": 0.0,
            "similarity_min": 0.0,
            "similarity_max": 0.0,
            "distance_mean": 0.0,
            "distance_median": 0.0,
            "distance_std": 0.0,
            "distance_min": 0.0,
            "distance_max": 0.0,
        }

    norms = np.linalg.norm(values, axis=1)
    normalized = np.divide(
        values,
        norms[:, None],
        out=np.zeros_like(values, dtype=np.float64),
        where=norms[:, None] > 0.0,
    )
    chunks: list[np.ndarray] = []
    block_size = 256
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        similarities = normalized[start:stop] @ normalized.T
        for local_index, global_index in enumerate(range(start, stop)):
            chunks.append(similarities[local_index, global_index + 1 :])
    pairwise = np.concatenate(chunks)
    distances = 1.0 - pairwise
    return {
        "pair_count": int(pairwise.size),
        "similarity_mean": float(pairwise.mean()),
        "similarity_median": float(np.median(pairwise)),
        "similarity_std": float(pairwise.std()),
        "similarity_min": float(pairwise.min()),
        "similarity_max": float(pairwise.max()),
        "distance_mean": float(distances.mean()),
        "distance_median": float(np.median(distances)),
        "distance_std": float(distances.std()),
        "distance_min": float(distances.min()),
        "distance_max": float(distances.max()),
    }


def representation_diagnostics(values: np.ndarray) -> dict[str, object]:
    """Compute collapse and geometry diagnostics for one representation matrix."""
    states = np.asarray(values, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("Diagnostic states must have shape [nonzero samples, dimensions].")
    if not np.isfinite(states).all():
        raise ValueError("Diagnostic states must be finite.")

    sample_variance = states.var(axis=0, ddof=1) if states.shape[0] > 1 else np.zeros(states.shape[1])
    covariance = (
        np.cov(states, rowvar=False, ddof=1)
        if states.shape[0] > 1
        else np.zeros((states.shape[1], states.shape[1]))
    )
    covariance = np.atleast_2d(covariance)
    standard_deviation = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance)[::-1], 0.0, None)
    eigenvalue_sum = float(eigenvalues.sum())
    if eigenvalue_sum > 0.0:
        probabilities = eigenvalues[eigenvalues > 0.0] / eigenvalue_sum
        effective_rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    else:
        effective_rank = 0.0
    norms = np.linalg.norm(states, axis=1)
    return {
        "sample_count": int(states.shape[0]),
        "dimension_count": int(states.shape[1]),
        "mean_per_dimension": states.mean(axis=0).tolist(),
        "sample_variance_per_dimension": sample_variance.tolist(),
        "variance_spectrum": eigenvalues.tolist(),
        "covariance": covariance.tolist(),
        "correlation": correlation.tolist(),
        "effective_rank": effective_rank,
        "zero_norm_count": int((norms == 0.0).sum()),
        "near_zero_norm_count": int((norms <= 1e-12).sum()),
        "pairwise_cosine": _pairwise_cosine_summary(states),
    }


def _cosine_stability(reference: np.ndarray, view: np.ndarray) -> dict[str, object]:
    """Compare aligned all-valid and fixed-K states for the same sample dates."""
    if reference.shape != view.shape:
        raise ValueError("K-view states must align exactly with all-valid states.")
    reference_norm = np.linalg.norm(reference, axis=1)
    view_norm = np.linalg.norm(view, axis=1)
    denominator = reference_norm * view_norm
    similarity = np.divide(
        (reference * view).sum(axis=1),
        denominator,
        out=np.zeros(reference.shape[0], dtype=np.float64),
        where=denominator > 0.0,
    )
    distance = 1.0 - similarity
    return {
        "sample_count": int(similarity.size),
        "similarity_mean": float(similarity.mean()),
        "similarity_median": float(np.median(similarity)),
        "similarity_std": float(similarity.std()),
        "similarity_min": float(similarity.min()),
        "similarity_max": float(similarity.max()),
        "distance_mean": float(distance.mean()),
        "distance_median": float(np.median(distance)),
        "distance_std": float(distance.std()),
        "distance_min": float(distance.min()),
        "distance_max": float(distance.max()),
    }


# ============================================================================
# STATE COLLECTION AND SUITE EXECUTION
# ============================================================================


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move only encoder-required tensors, leaving aliases and metadata on CPU."""
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in ENCODER_BATCH_TENSOR_NAMES
        if isinstance(batch[name], torch.Tensor)
    }


def collect_pooled_states(
    model: FIJepaModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    *,
    description: str = "Representations",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Encode one deterministic loader into metadata and raw pooled states."""
    was_training = model.training
    model.eval()
    metadata: list[dict[str, object]] = []
    states: list[np.ndarray] = []
    with torch.inference_mode():
        for cpu_batch in tqdm(
            loader,
            desc=description,
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
                pooled = model.encode_pooled_state(batch)
            states.append(pooled.float().cpu().numpy().astype(np.float64))
            dates = cpu_batch["sample_date"]
            splits = cpu_batch["split_label"]
            windows = cpu_batch.get("validation_window_name", [""] * len(dates))
            views = cpu_batch.get("asset_view", ["all_valid"] * len(dates))
            view_indices = cpu_batch.get("view_index")
            for index, date in enumerate(dates):
                metadata.append(
                    {
                        "date": str(date),
                        "split": str(splits[index]),
                        "validation_window_name": str(windows[index]),
                        "asset_view": str(views[index]),
                        "view_index": (
                            int(view_indices[index].item())
                            if isinstance(view_indices, torch.Tensor)
                            else 0
                        ),
                    }
                )
    if was_training:
        model.train()
    if not states:
        raise RuntimeError("Embedding loader produced no pooled states.")
    return pd.DataFrame(metadata), np.concatenate(states, axis=0)


def _embedding_frame(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    versions: dict[str, object],
) -> pd.DataFrame:
    """Attach PCA coordinates and immutable version fields to embedding metadata."""
    frame = metadata.copy()
    for index in range(embeddings.shape[1]):
        frame[f"z_{index + 1}"] = embeddings[:, index]
    for name, value in versions.items():
        frame[name] = value
    return frame


def _invariant_summary(report: dict[str, object]) -> dict[str, object]:
    """Return compact cross-checkpoint-safe diagnostics for the training log."""
    raw_validation = report["representations"]["raw_pooled_state"]["validation"]
    pca_validation = report["representations"]["pca_export"]["validation"]
    return {
        "raw_validation_effective_rank": raw_validation["effective_rank"],
        "raw_validation_variance_spectrum": raw_validation["variance_spectrum"],
        "raw_validation_pairwise_cosine": raw_validation["pairwise_cosine"],
        "pca_validation_effective_rank": pca_validation["effective_rank"],
        "pca_explained_variance_ratio": report["pca"]["explained_variance_ratio"],
        "raw_k_view_stability": report["k_view_stability"]["raw_pooled_state"]["aggregate"],
        "pca_k_view_stability": report["k_view_stability"]["pca_export"]["aggregate"],
    }


def run_representation_evaluation(
    model: FIJepaModel,
    store: DensePanelStore,
    data_config: FIJepaDataConfig,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    n_components: int,
    views_per_date: int,
    output_dir: Path,
    checkpoint_id: str,
    checkpoint_step: int,
    checkpoint_format_version: int,
    model_version: str,
    export_embeddings: bool,
    checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    """Run the complete raw-state, PCA-export, and deterministic K-view suite."""
    if views_per_date <= 0:
        raise ValueError("views_per_date must be positive.")
    output_dir.mkdir(parents=True, exist_ok=False)

    train_loader = build_fi_jepa_embedding_dataloader(
        data_config,
        "train",
        asset_view="all_valid",
        store=store,
        num_workers=0,
    )
    validation_loader = build_fi_jepa_embedding_dataloader(
        data_config,
        "validation",
        asset_view="all_valid",
        store=store,
        num_workers=0,
    )
    train_metadata, train_raw = collect_pooled_states(
        model,
        train_loader,
        device,
        amp_dtype,
        description="Representations: train all-valid",
    )
    validation_metadata, validation_raw = collect_pooled_states(
        model,
        validation_loader,
        device,
        amp_dtype,
        description="Representations: validation all-valid",
    )
    pca = fit_pca_exporter(train_raw, n_components)
    train_z = pca.transform(train_raw)
    validation_z = pca.transform(validation_raw)

    raw_view_states: list[np.ndarray] = []
    pca_view_states: list[np.ndarray] = []
    view_frames: list[pd.DataFrame] = []
    raw_view_reports: list[dict[str, object]] = []
    pca_view_reports: list[dict[str, object]] = []
    for view_index in range(views_per_date):
        loader = build_fi_jepa_embedding_dataloader(
            data_config,
            "validation",
            asset_view="fixed_k",
            store=store,
            view_index=view_index,
            num_workers=0,
        )
        metadata, raw = collect_pooled_states(
            model,
            loader,
            device,
            amp_dtype,
            description=f"Representations: validation fixed-k {view_index + 1}/{views_per_date}",
        )
        if metadata["date"].tolist() != validation_metadata["date"].tolist():
            raise AssertionError("Fixed-K and all-valid validation dates do not align.")
        z = pca.transform(raw)
        raw_view_states.append(raw)
        pca_view_states.append(z)
        raw_view_reports.append(_cosine_stability(validation_raw, raw))
        pca_view_reports.append(_cosine_stability(validation_z, z))
        view_frames.append(metadata.assign(**{f"z_{index + 1}": z[:, index] for index in range(n_components)}))

    raw_aggregate = _cosine_stability(
        np.tile(validation_raw, (views_per_date, 1)),
        np.concatenate(raw_view_states, axis=0),
    )
    pca_aggregate = _cosine_stability(
        np.tile(validation_z, (views_per_date, 1)),
        np.concatenate(pca_view_states, axis=0),
    )
    report: dict[str, object] = {
        "format_version": 1,
        "checkpoint_id": checkpoint_id,
        "checkpoint_step": checkpoint_step,
        "checkpoint_format_version": checkpoint_format_version,
        "dataset_version": store.dataset_version,
        "model_version": model_version,
        "representation_source": "encode_pooled_state",
        "collapse_source_of_truth": "raw_pooled_state",
        "pca_axis_warning": (
            "PCA coordinates are checkpoint-specific; sign canonicalization does not make "
            "axes comparable across epochs when eigenvalues are close."
        ),
        "pca": {
            "version": pca.version,
            "n_components": n_components,
            "whiten": False,
            "explained_variance": pca.explained_variance.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio.tolist(),
        },
        "representations": {
            "raw_pooled_state": {
                "train": representation_diagnostics(train_raw),
                "validation": representation_diagnostics(validation_raw),
            },
            "pca_export": {
                "train": representation_diagnostics(train_z),
                "validation": representation_diagnostics(validation_z),
            },
        },
        "k_view_stability": {
            "raw_pooled_state": {"views": raw_view_reports, "aggregate": raw_aggregate},
            "pca_export": {"views": pca_view_reports, "aggregate": pca_aggregate},
        },
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pca.save(output_dir / "pca_exporter.npz")

    versions = {
        "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "checkpoint_step": checkpoint_step,
        "checkpoint_format_version": checkpoint_format_version,
        "model_version": model_version,
        "dataset_version": store.dataset_version,
        "pca_version": pca.version,
    }
    if export_embeddings:
        embeddings = pd.concat(
            [
                _embedding_frame(train_metadata, train_z, versions),
                _embedding_frame(validation_metadata, validation_z, versions),
            ],
            ignore_index=True,
        )
        embeddings.to_parquet(output_dir / "embeddings.parquet", index=False, compression="zstd")
        fixed_views = pd.concat(view_frames, ignore_index=True)
        for name, value in versions.items():
            fixed_views[name] = value
        fixed_views.to_parquet(
            output_dir / "validation_k_view_embeddings.parquet",
            index=False,
            compression="zstd",
        )

    manifest = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "checkpoint_format_version": checkpoint_format_version,
        "model_version": model_version,
        "dataset_version": store.dataset_version,
        "source_database": store.manifest.get("source_database"),
        "source_database_sha256": store.manifest.get("source_database_sha256"),
        "pca_version": pca.version,
        "representation_source": "encode_pooled_state",
        "targets_included": False,
        "embeddings_exported": export_embeddings,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"report": report, "summary": _invariant_summary(report), "output_dir": output_dir}


# ============================================================================
# ON-DEMAND CHECKPOINT EVALUATION
# ============================================================================


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    output_root: Path = Path("runs/evaluation"),
    device_name: str = "auto",
    export_embeddings: bool = True,
    batch_size: int | None = None,
) -> Path:
    """Load one checkpoint and write an immutable representation evaluation artifact.

    ``batch_size`` overrides the checkpoint's validation batch size only for
    this evaluation run. All embedding loaders use that value, including the
    memory-heavy all-valid train and validation views.
    """
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resolved = checkpoint["resolved_config"]
    model_config = FIJepaModelConfig.from_dict(resolved["model"])
    data_values = dict(resolved["dataloader"])
    data_values["artifact_path"] = Path(data_values["artifact_path"])
    data_values["cache_root"] = Path(data_values["cache_root"])
    if batch_size is not None:
        data_values["validation_batch_size"] = batch_size
    data_config = FIJepaDataConfig(**data_values)
    store = DensePanelStore(data_config.artifact_path, cache_root=data_config.cache_root)
    model = FIJepaModel.from_store(model_config, store)
    load_fi_jepa_model_state(model, checkpoint["model"])

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    model.to(device)
    training = resolved["training"]
    components = int(training["representation_pca_components"])
    views = int(training.get("representation_views_per_date", 3))
    checkpoint_sha = _file_sha256(checkpoint_path)
    checkpoint_id = f"step_{int(checkpoint['global_step']):09d}_{checkpoint_sha[:12]}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = output_root / f"{timestamp}_{checkpoint_id}_{store.dataset_version}"
    model_version = canonical_version_hash(resolved["model"])
    run_representation_evaluation(
        model,
        store,
        data_config,
        device=device,
        amp_dtype=None,
        n_components=components,
        views_per_date=views,
        output_dir=output_dir,
        checkpoint_id=checkpoint_id,
        checkpoint_step=int(checkpoint["global_step"]),
        checkpoint_format_version=int(checkpoint["format_version"]),
        model_version=model_version,
        export_embeddings=export_embeddings,
        checkpoint_sha256=checkpoint_sha,
    )
    print(f"Built representation evaluation: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    """Parse the on-demand checkpoint evaluation CLI."""
    parser = argparse.ArgumentParser(description="Evaluate and export FI-JEPA representations.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the checkpoint batch size for all representation-evaluation loaders.",
    )
    parser.add_argument("--no-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run on-demand representation evaluation."""
    args = parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        output_root=args.output_root,
        device_name=args.device,
        export_embeddings=not args.no_embeddings,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
