from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, replace
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
from fi_jepa.model import (
    ENCODER_BATCH_TENSOR_NAMES,
    INPUT_ABLATION_MODES,
    FIJepaModel,
    load_fi_jepa_model_state,
)
from fi_jepa.model_config import FIJepaModelConfig

EMBEDDING_SCHEMA_VERSION = 1
PCA_FORMAT_VERSION = 1
INITIAL_REPRESENTATION_VARIANTS = (
    "mean_pca_16",
    "endpoint_pca_16",
    "pooled_pca_16",
    "pooled_pca_32",
    "pooled_pca_64",
    "pooled_raw_256",
)
ASSET_COUNT_ABLATIONS = (32, 128, 256)


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


def build_representation_variants(
    train_states: dict[str, np.ndarray],
    validation_states: dict[str, np.ndarray],
    variants: tuple[str, ...] | list[str] = INITIAL_REPRESENTATION_VARIANTS,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, object]], dict[str, PCAExporter]]:
    """Build the explicit initial representation suite with train-only PCA fits.

    Raw state dictionaries must contain ``mean_state``, ``endpoint_state``, and
    ``pooled_state``. PCA exporters are fit only from the corresponding training
    matrix, then applied unchanged to validation rows.
    """
    requested = tuple(dict.fromkeys(str(name) for name in variants))
    unknown = sorted(set(requested).difference(INITIAL_REPRESENTATION_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown representation variants: {unknown}")

    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metadata: dict[str, dict[str, object]] = {}
    exporters: dict[str, PCAExporter] = {}
    for variant in requested:
        source, transform, dimension_text = variant.rsplit("_", 2)
        dimension = int(dimension_text)
        source_name = f"{source}_state"
        train = np.asarray(train_states[source_name], dtype=np.float64)
        validation = np.asarray(validation_states[source_name], dtype=np.float64)
        if transform == "raw":
            if train.shape[1] != dimension:
                raise ValueError(
                    f"{variant} requires a {dimension}-dimensional {source_name}; got {train.shape[1]}."
                )
            train_output, validation_output = train, validation
            explained_variance: list[float] | None = None
            explained_variance_ratio: list[float] | None = None
        else:
            exporter = fit_pca_exporter(train, dimension)
            exporters[variant] = exporter
            train_output = exporter.transform(train)
            validation_output = exporter.transform(validation)
            explained_variance = exporter.explained_variance.tolist()
            explained_variance_ratio = exporter.explained_variance_ratio.tolist()
        outputs[variant] = (train_output, validation_output)
        metadata[variant] = {
            "representation_variant": variant,
            "representation_source": source_name,
            "dimension": dimension,
            "transform": transform,
            "explained_variance": explained_variance,
            "explained_variance_ratio": explained_variance_ratio,
            "pca_fit_split": "train" if transform == "pca" else None,
        }
    return outputs, metadata, exporters


def representation_distance_summary(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """Compare aligned representation rows using direction and magnitude metrics."""
    reference_values = np.asarray(reference, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    if reference_values.shape != candidate_values.shape or reference_values.ndim != 2:
        raise ValueError("Representation distance inputs must have the same [samples, dimensions] shape.")
    reference_norm = np.linalg.norm(reference_values, axis=1)
    candidate_norm = np.linalg.norm(candidate_values, axis=1)
    denominator = reference_norm * candidate_norm
    cosine = np.divide(
        np.einsum("nd,nd->n", reference_values, candidate_values),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1.0e-12,
    )
    relative_l2 = np.linalg.norm(candidate_values - reference_values, axis=1) / np.maximum(
        reference_norm, 1.0e-12
    )
    return {
        "mean_cosine_similarity": float(cosine.mean()),
        "median_cosine_similarity": float(np.median(cosine)),
        "mean_relative_l2_distance": float(relative_l2.mean()),
        "median_relative_l2_distance": float(np.median(relative_l2)),
    }


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


def _validation_rank_metrics(values: np.ndarray) -> dict[str, float | int]:
    """Compute only the seven geometry metrics used for validation-rank interpretation."""
    states = np.asarray(values, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("Rank-diagnostic states must have shape [nonzero samples, dimensions].")
    if not np.isfinite(states).all():
        raise ValueError("Rank-diagnostic states must be finite.")

    centered = states - states.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = np.square(singular_values) / max(states.shape[0] - 1, 1)
    eigenvalue_sum = float(eigenvalues.sum())
    if eigenvalue_sum > 0.0:
        probabilities = eigenvalues[eigenvalues > 0.0] / eigenvalue_sum
        effective_rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
        top_eigenvalue_share = float(eigenvalues[0] / eigenvalue_sum)
        top_5_eigenvalue_share = float(eigenvalues[:5].sum() / eigenvalue_sum)
    else:
        effective_rank = 0.0
        top_eigenvalue_share = 0.0
        top_5_eigenvalue_share = 0.0
    pairwise = _pairwise_cosine_summary(states)
    return {
        "sample_count": int(states.shape[0]),
        "effective_rank": effective_rank,
        "top_eigenvalue_share": top_eigenvalue_share,
        "top_5_eigenvalue_share": top_5_eigenvalue_share,
        "mean_pairwise_cosine": pairwise["similarity_mean"],
        "median_pairwise_cosine": pairwise["similarity_median"],
        "mean_vector_norm": float(np.linalg.norm(states, axis=1).mean()),
    }


def windowed_validation_rank_diagnostics(
    train_metadata: pd.DataFrame,
    train_representations: dict[str, np.ndarray],
    validation_metadata: pd.DataFrame,
    validation_representations: dict[str, np.ndarray],
) -> dict[str, object]:
    """Compare each outer validation window with exact-date-length contiguous train windows.

    The caller supplies only the raw pooled state and selected PCA representation.
    Training windows are non-overlapping exact-length blocks plus an exact-length
    trailing block when the split leaves a remainder. The validation percentile
    is the empirical percentage of matched training values less than or equal to
    the validation value.
    """
    representation_names = tuple(train_representations)
    if representation_names != tuple(validation_representations):
        raise ValueError("Train and validation diagnostic representations must have identical names.")
    if len(train_metadata) == 0 or len(validation_metadata) == 0:
        raise ValueError("Windowed diagnostics require non-empty train and validation metadata.")
    for name in representation_names:
        if len(train_representations[name]) != len(train_metadata):
            raise ValueError(f"Train metadata and {name!r} rows do not align.")
        if len(validation_representations[name]) != len(validation_metadata):
            raise ValueError(f"Validation metadata and {name!r} rows do not align.")

    train_dates = pd.to_datetime(train_metadata["date"], errors="raise")
    validation_dates = pd.to_datetime(validation_metadata["date"], errors="raise")
    unique_train_dates = np.asarray(sorted(train_dates.unique()))
    window_labels = (
        validation_metadata["validation_window_name"].fillna("").astype(str)
        if "validation_window_name" in validation_metadata
        else pd.Series("validation", index=validation_metadata.index, dtype=object)
    )
    window_labels = window_labels.mask(window_labels.eq(""), "validation")

    report: dict[str, object] = {
        "matched_train_window_method": (
            "historical_contiguous_exact_date_length_nonoverlapping_plus_trailing"
        ),
        "validation_percentile_definition": "100 * mean(matched_train_value <= validation_value)",
        "representations": {name: {"windows": []} for name in representation_names},
    }
    for window_name in sorted(window_labels.unique()):
        validation_mask = window_labels.eq(window_name).to_numpy()
        validation_date_count = int(validation_dates[validation_mask].nunique())
        validation_start = validation_dates[validation_mask].min()
        eligible_train_dates = unique_train_dates[unique_train_dates < validation_start.to_datetime64()]
        if validation_date_count > len(eligible_train_dates):
            raise ValueError(
                f"Validation window {window_name!r} has {validation_date_count} dates, "
                f"but has only {len(eligible_train_dates)} historical training dates."
            )

        starts = list(range(0, len(eligible_train_dates) - validation_date_count + 1, validation_date_count))
        trailing_start = len(eligible_train_dates) - validation_date_count
        if starts[-1] != trailing_start:
            starts.append(trailing_start)
        train_masks = [
            train_dates.isin(eligible_train_dates[start : start + validation_date_count]).to_numpy()
            for start in starts
        ]

        for representation_name in representation_names:
            train_values = np.asarray(train_representations[representation_name], dtype=np.float64)
            validation_values = np.asarray(validation_representations[representation_name], dtype=np.float64)
            validation_metrics = _validation_rank_metrics(validation_values[validation_mask])
            matched_metrics = [_validation_rank_metrics(train_values[mask]) for mask in train_masks]
            comparisons: dict[str, dict[str, float | int]] = {}
            for metric_name, validation_value in validation_metrics.items():
                distribution = np.asarray([metrics[metric_name] for metrics in matched_metrics], dtype=np.float64)
                comparisons[metric_name] = {
                    "matched_train_median": float(np.median(distribution)),
                    "matched_train_5th_percentile": float(np.percentile(distribution, 5)),
                    "matched_train_95th_percentile": float(np.percentile(distribution, 95)),
                    "validation_value": validation_value,
                    "validation_percentile": float(100.0 * np.mean(distribution <= float(validation_value))),
                }
            report["representations"][representation_name]["windows"].append(
                {
                    "validation_window_name": str(window_name),
                    "validation_date_count": validation_date_count,
                    "matched_train_window_count": len(train_masks),
                    "matched_train_window_date_counts": [
                        int(train_dates[mask].nunique()) for mask in train_masks
                    ],
                    "metrics": comparisons,
                }
            )
    return report


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


def collect_representation_states(
    model: FIJepaModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    *,
    description: str = "Representations",
    input_mode: str = "all_streams",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Encode one deterministic loader into metadata and source-separated states."""
    was_training = model.training
    model.eval()
    metadata: list[dict[str, object]] = []
    mean_states: list[np.ndarray] = []
    endpoint_states: list[np.ndarray] = []
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
                mean_state, endpoint_state = model.encode_state_components(
                    batch, input_mode=input_mode
                )  # Each [B, D].
            mean_states.append(mean_state.float().cpu().numpy().astype(np.float64))
            endpoint_states.append(endpoint_state.float().cpu().numpy().astype(np.float64))
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
    if not mean_states:
        raise RuntimeError("Embedding loader produced no representation states.")
    mean = np.concatenate(mean_states, axis=0)
    endpoint = np.concatenate(endpoint_states, axis=0)
    return pd.DataFrame(metadata), {
        "mean_state": mean,
        "endpoint_state": endpoint,
        "pooled_state": np.concatenate((mean, endpoint), axis=1),
    }


def run_compact_ablation_suite(
    model: FIJepaModel,
    store: DensePanelStore,
    data_config: FIJepaDataConfig,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    train_metadata: pd.DataFrame,
    train_states: dict[str, np.ndarray],
    validation_metadata: pd.DataFrame,
    validation_states: dict[str, np.ndarray],
    collect_probe_states: bool,
) -> tuple[dict[str, object], dict[str, tuple[np.ndarray, np.ndarray, str]]]:
    """Evaluate the exact compact asset-count and input-stream ablations.

    Every candidate is aligned by date to the all-valid reference. Only the
    fixed-K 128 state is retained for asset-count probes; the other K values are
    distance diagnostics unless later results justify probing them.
    """
    reference = validation_states["pooled_state"]
    asset_reports: dict[str, dict[str, object]] = {
        "all_valid": {
            "asset_view": "all_valid",
            **representation_distance_summary(reference, reference),
            "effective_rank": representation_diagnostics(reference)["effective_rank"],
            "probe_variant": "pooled_pca_16",
            "probe_performance_change": 0.0,
        }
    }
    probe_states: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for asset_count in ASSET_COUNT_ABLATIONS:
        ablation_config = replace(data_config, fixed_k_assets=asset_count)
        validation_loader = build_fi_jepa_embedding_dataloader(
            ablation_config,
            "validation",
            asset_view="fixed_k",
            store=store,
            view_index=0,
            num_workers=0,
        )
        metadata, states = collect_representation_states(
            model,
            validation_loader,
            device,
            amp_dtype,
            description=f"Ablation: validation fixed-k {asset_count}",
        )
        if metadata["date"].tolist() != validation_metadata["date"].tolist():
            raise AssertionError(f"Fixed-K {asset_count} and all-valid validation dates do not align.")
        candidate = states["pooled_state"]
        probe_variant = f"asset_k_{asset_count}_pooled_pca_16" if asset_count == 128 else None
        asset_reports[f"k_{asset_count}"] = {
            "asset_view": "fixed_k",
            "asset_count": asset_count,
            **representation_distance_summary(reference, candidate),
            "effective_rank": representation_diagnostics(candidate)["effective_rank"],
            "probe_variant": probe_variant,
            "probe_performance_change": None,
        }
        if collect_probe_states and asset_count == 128:
            train_loader = build_fi_jepa_embedding_dataloader(
                ablation_config,
                "train",
                asset_view="fixed_k",
                store=store,
                view_index=0,
                num_workers=0,
            )
            ablation_train_metadata, ablation_train_states = collect_representation_states(
                model,
                train_loader,
                device,
                amp_dtype,
                description="Ablation: train fixed-k 128",
            )
            if ablation_train_metadata["date"].tolist() != train_metadata["date"].tolist():
                raise AssertionError("Fixed-K 128 and all-valid training dates do not align.")
            probe_states[probe_variant] = (
                ablation_train_states["pooled_state"],
                candidate,
                "pooled_state_asset_k_128",
            )

    input_reports: dict[str, dict[str, object]] = {}
    all_valid_loader = build_fi_jepa_embedding_dataloader(
        data_config,
        "validation",
        asset_view="all_valid",
        store=store,
        num_workers=0,
    )
    for input_mode in INPUT_ABLATION_MODES:
        if input_mode == "all_streams":
            candidate = reference
        else:
            metadata, states = collect_representation_states(
                model,
                all_valid_loader,
                device,
                amp_dtype,
                description=f"Ablation: validation {input_mode}",
                input_mode=input_mode,
            )
            if metadata["date"].tolist() != validation_metadata["date"].tolist():
                raise AssertionError(f"{input_mode} and all-stream validation dates do not align.")
            candidate = states["pooled_state"]
        probe_variant = "pooled_pca_16" if input_mode == "all_streams" else f"{input_mode}_pooled_pca_16"
        input_reports[input_mode] = {
            **representation_distance_summary(reference, candidate),
            "effective_rank": representation_diagnostics(candidate)["effective_rank"],
            "probe_variant": probe_variant,
            "probe_performance_change": 0.0 if input_mode == "all_streams" else None,
        }
        if collect_probe_states and input_mode != "all_streams":
            train_loader = build_fi_jepa_embedding_dataloader(
                data_config,
                "train",
                asset_view="all_valid",
                store=store,
                num_workers=0,
            )
            ablation_train_metadata, ablation_train_states = collect_representation_states(
                model,
                train_loader,
                device,
                amp_dtype,
                description=f"Ablation: train {input_mode}",
                input_mode=input_mode,
            )
            if ablation_train_metadata["date"].tolist() != train_metadata["date"].tolist():
                raise AssertionError(f"{input_mode} and all-stream training dates do not align.")
            probe_states[probe_variant] = (
                ablation_train_states["pooled_state"],
                candidate,
                f"pooled_state_{input_mode}",
            )
    return {
        "asset_count_ablations": asset_reports,
        "input_branch_ablations": input_reports,
        "default_asset_probe_variants": ["pooled_pca_16", "asset_k_128_pooled_pca_16"],
    }, probe_states


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
    representation_variant: str,
    checkpoint_sha256: str | None = None,
    representation_variants: tuple[str, ...] | list[str] | None = None,
    run_compact_ablations: bool = False,
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
    train_metadata, train_states = collect_representation_states(
        model,
        train_loader,
        device,
        amp_dtype,
        description="Representations: train all-valid",
    )
    validation_metadata, validation_states = collect_representation_states(
        model,
        validation_loader,
        device,
        amp_dtype,
        description="Representations: validation all-valid",
    )
    train_raw = train_states["pooled_state"]
    validation_raw = validation_states["pooled_state"]
    pca = fit_pca_exporter(train_raw, n_components)
    train_z = pca.transform(train_raw)
    validation_z = pca.transform(validation_raw)
    variant_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    variant_metadata: dict[str, dict[str, object]] = {}
    variant_exporters: dict[str, PCAExporter] = {}
    if representation_variants is not None:
        variant_outputs, variant_metadata, variant_exporters = build_representation_variants(
            train_states, validation_states, representation_variants
        )

    ablation_report: dict[str, object] = {}
    if run_compact_ablations:
        ablation_report, ablation_probe_states = run_compact_ablation_suite(
            model,
            store,
            data_config,
            device=device,
            amp_dtype=amp_dtype,
            train_metadata=train_metadata,
            train_states=train_states,
            validation_metadata=validation_metadata,
            validation_states=validation_states,
            collect_probe_states=export_embeddings,
        )
        for variant, (ablation_train, ablation_validation, source) in ablation_probe_states.items():
            exporter = fit_pca_exporter(ablation_train, 16)
            variant_exporters[variant] = exporter
            variant_outputs[variant] = (
                exporter.transform(ablation_train),
                exporter.transform(ablation_validation),
            )
            variant_metadata[variant] = {
                "representation_variant": variant,
                "representation_source": source,
                "dimension": 16,
                "transform": "pca",
                "explained_variance": exporter.explained_variance.tolist(),
                "explained_variance_ratio": exporter.explained_variance_ratio.tolist(),
                "pca_fit_split": "train",
            }

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
        metadata, states = collect_representation_states(
            model,
            loader,
            device,
            amp_dtype,
            description=f"Representations: validation fixed-k {view_index + 1}/{views_per_date}",
        )
        if metadata["date"].tolist() != validation_metadata["date"].tolist():
            raise AssertionError("Fixed-K and all-valid validation dates do not align.")
        raw = states["pooled_state"]
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
    created_at_utc = datetime.now(timezone.utc).isoformat()
    resolved_representation_config = {
        "pca_components": n_components,
        "views_per_date": views_per_date,
        "export_embeddings": export_embeddings,
        "representation_variants": list(variant_metadata) or [representation_variant],
        "compact_ablations": run_compact_ablations,
        "asset_count_ablations": [*ASSET_COUNT_ABLATIONS, "all_valid"] if run_compact_ablations else [],
        "input_branch_ablations": list(INPUT_ABLATION_MODES) if run_compact_ablations else [],
    }
    validation_rank_diagnostics = windowed_validation_rank_diagnostics(
        train_metadata,
        {
            "raw_pooled_state": train_raw,
            "selected_pca_representation": train_z,
        },
        validation_metadata,
        {
            "raw_pooled_state": validation_raw,
            "selected_pca_representation": validation_z,
        },
    )
    validation_rank_diagnostics["representations"]["raw_pooled_state"].update(
        {"representation_source": "pooled_state", "representation_variant": "pooled_raw_256"}
    )
    validation_rank_diagnostics["representations"]["selected_pca_representation"].update(
        {"representation_source": "pooled_state", "representation_variant": representation_variant}
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": created_at_utc,
        "checkpoint_id": checkpoint_id,
        "checkpoint_step": checkpoint_step,
        "checkpoint_format_version": checkpoint_format_version,
        "dataset_version": store.dataset_version,
        "model_version": model_version,
        "representation_source": "encode_pooled_state",
        "representation_variant": representation_variant,
        "resolved_probe_config": {},
        "resolved_representation_config": resolved_representation_config,
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
        "representation_variants": variant_metadata,
        "compact_ablations": ablation_report,
        "validation_rank_diagnostics": validation_rank_diagnostics,
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
    for variant, exporter in variant_exporters.items():
        exporter.save(output_dir / f"pca_exporter_{variant}.npz")

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
        for variant, (variant_train, variant_validation) in variant_outputs.items():
            details = variant_metadata[variant]
            variant_versions = {
                **versions,
                "pca_version": (
                    variant_exporters[variant].version if variant in variant_exporters else None
                ),
                "representation_variant": variant,
                "representation_source": details["representation_source"],
                "representation_dimension": details["dimension"],
            }
            variant_frame = pd.concat(
                [
                    _embedding_frame(train_metadata, variant_train, variant_versions),
                    _embedding_frame(validation_metadata, variant_validation, variant_versions),
                ],
                ignore_index=True,
            )
            variant_frame.to_parquet(
                output_dir / f"embeddings_{variant}.parquet", index=False, compression="zstd"
            )
        fixed_views = pd.concat(view_frames, ignore_index=True)
        for name, value in versions.items():
            fixed_views[name] = value
        fixed_views.to_parquet(
            output_dir / "validation_k_view_embeddings.parquet",
            index=False,
            compression="zstd",
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": created_at_utc,
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
        "representation_variant": representation_variant,
        "resolved_representation_config": resolved_representation_config,
        "representation_variants": {
            variant: {
                **details,
                "embedding_file": f"embeddings_{variant}.parquet" if export_embeddings else None,
                "pca_file": f"pca_exporter_{variant}.npz" if variant in variant_exporters else None,
            }
            for variant, details in variant_metadata.items()
        },
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
        representation_variant=f"pooled_pca_{components}",
        checkpoint_sha256=checkpoint_sha,
        representation_variants=INITIAL_REPRESENTATION_VARIANTS,
        run_compact_ablations=True,
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
