from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FORBIDDEN_FEATURE_PATTERN = re.compile(r"future_|target|label", re.IGNORECASE)


# ============================================================================
# DATALOADER CONFIGURATION
# ============================================================================


def validate_data_config(
    *,
    lookback_days: int,
    patch_len: int,
    mask_ratio: float,
    min_masked_patches: int,
    max_masked_patches: int,
    min_target_blocks: int,
    max_target_blocks: int,
    min_valid_days_per_asset_patch: int,
    min_valid_dates_in_patch: int,
    min_valid_asset_fraction: float,
    feature_dropout_rate: float,
    train_k_assets: int,
    fixed_k_assets: int,
    batch_size: int,
    validation_batch_size: int,
    num_workers: int,
) -> None:
    """Validate dense-panel dimensions, masking bounds, and loader settings."""
    if lookback_days <= 0 or patch_len <= 0:
        raise ValueError("lookback_days and patch_len must be positive.")
    if lookback_days % patch_len:
        raise ValueError("lookback_days must be divisible by patch_len.")
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1].")
    if not 0.0 <= min_valid_asset_fraction <= 1.0:
        raise ValueError("min_valid_asset_fraction must be in [0, 1].")
    if not 0.0 <= feature_dropout_rate < 1.0:
        raise ValueError("feature_dropout_rate must be in [0, 1).")
    if not 1 <= min_masked_patches <= max_masked_patches:
        raise ValueError("Masked patch bounds are invalid.")
    if max_masked_patches > lookback_days // patch_len:
        raise ValueError("max_masked_patches exceeds the number of patches.")
    if not 1 <= min_target_blocks <= max_target_blocks:
        raise ValueError("Target block bounds are invalid.")
    if min_target_blocks > min_masked_patches:
        raise ValueError("min_target_blocks cannot exceed min_masked_patches.")
    if max_target_blocks > max_masked_patches:
        raise ValueError("max_target_blocks cannot exceed max_masked_patches.")
    if not 1 <= min_valid_days_per_asset_patch <= patch_len:
        raise ValueError("min_valid_days_per_asset_patch must be within one patch.")
    if not 1 <= min_valid_dates_in_patch <= patch_len:
        raise ValueError("min_valid_dates_in_patch must be within one patch.")
    if train_k_assets <= 0 or fixed_k_assets <= 0:
        raise ValueError("Asset view sizes must be positive.")
    if batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")


def validate_data_yaml(values: object) -> dict[str, object]:
    """Validate the top-level dataloader YAML contract and return its mapping."""
    if not isinstance(values, dict):
        raise ValueError("Dataloader configuration must be a YAML mapping.")
    return values


# ============================================================================
# ARTIFACT AND CACHE CONTRACTS
# ============================================================================


def validate_cache_root(artifact_path: Path, cache_root: Path) -> None:
    """Require mutable cache files to live outside the immutable artifact."""
    if cache_root.is_relative_to(artifact_path):
        raise ValueError("cache_root must be outside the immutable artifact directory.")


def validate_build_id(manifest: Mapping[str, object]) -> str:
    """Return the stable sparse-artifact build ID or reject the manifest."""
    build_id = manifest.get("build_id")
    if not build_id:
        raise ValueError("Sparse artifact manifest.json must contain a stable build_id.")
    return str(build_id)


def validate_required_artifact_files(artifact_path: Path, required_files: set[str]) -> None:
    """Require the complete sparse model-artifact input contract."""
    missing = sorted(name for name in required_files if not (artifact_path / name).is_file())
    if missing:
        raise FileNotFoundError(f"Sparse artifact is missing required files: {missing}")


def validate_source_manifests(
    dates: pd.DataFrame,
    assets: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """Validate dense axes, feature groups, and leakage-sensitive feature names."""
    required_dates = {
        "date_idx",
        "date",
        "sample_eligible",
        "validation_sample",
        "protected_holdout",
        "train_fact_allowed",
        "validation_fact_allowed",
    }
    missing_dates = sorted(required_dates - set(dates.columns))
    if missing_dates:
        raise ValueError(f"dates.parquet is missing columns: {missing_dates}")
    if dates["date_idx"].tolist() != list(range(len(dates))):
        raise ValueError("dates.parquet date_idx must be contiguous and ordered.")
    if not dates["date"].is_monotonic_increasing:
        raise ValueError("dates.parquet dates must be ordered.")

    required_assets = {"asset_id", "symbol", "trainable"}
    missing_assets = sorted(required_assets - set(assets.columns))
    if missing_assets:
        raise ValueError(f"assets.parquet is missing columns: {missing_assets}")
    if assets["asset_id"].tolist() != list(range(len(assets))):
        raise ValueError("assets.parquet asset_id must be contiguous and ordered.")

    required_features = {"feature_name", "feature_index", "input_group", "dtype"}
    missing_features = sorted(required_features - set(features.columns))
    if missing_features:
        raise ValueError(f"feature_manifest.parquet is missing columns: {missing_features}")
    if set(features["input_group"]) != {"asset", "market", "macro"}:
        raise ValueError("Feature manifest must contain asset, market, and macro groups.")
    forbidden = features["feature_name"].astype(str).str.contains(FORBIDDEN_FEATURE_PATTERN)
    if forbidden.any():
        names = features.loc[forbidden, "feature_name"].tolist()
        raise ValueError(f"Forbidden target-like features in artifact: {names}")
    for group, frame in features.groupby("input_group"):
        indices = frame.sort_values("feature_index")["feature_index"].tolist()
        if indices != list(range(len(indices))):
            raise ValueError(f"{group} feature indices must be contiguous from zero.")


def validate_cache(
    cache_path: Path,
    source_identity: Mapping[str, object],
    array_names: tuple[str, ...],
    metadata_files: tuple[str, ...],
) -> bool:
    """Return whether a published cache exactly matches the source contract."""
    try:
        actual = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
        if any(actual.get(name) != value for name, value in source_identity.items()):
            return False
        if any(not (cache_path / name).is_file() for name in metadata_files):
            return False

        shapes = actual["array_shapes"]
        dtypes = actual["array_dtypes"]
        if set(shapes) != set(array_names) or set(dtypes) != set(array_names):
            return False
        for name in array_names:
            array = np.load(cache_path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            try:
                if list(array.shape) != shapes[name]:
                    return False
                if array.dtype.str != np.dtype(str(dtypes[name])).str:
                    return False
            finally:
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
    except (
        AttributeError,
        EOFError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    return True


def validate_fact_schema(path: Path, group: str, feature_names: list[str]) -> None:
    """Require sparse values, validity masks, keys, and row-validity fields."""
    columns = set(pq.read_schema(path).names)
    required = {"date", "date_idx", *feature_names}
    required.update(f"{name}__valid" for name in feature_names)
    required.add("valid_asset" if group == "asset" else "valid_date")
    if group == "asset":
        required.add("asset_id")
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    forbidden = sorted(name for name in columns if FORBIDDEN_FEATURE_PATTERN.search(name))
    if forbidden:
        raise ValueError(f"{path.name} contains forbidden target-like columns: {forbidden}")


# ============================================================================
# REQUEST AND MASK CONTRACTS
# ============================================================================


def validate_request_batch(requests: list[object], request_type: type) -> None:
    """Require one non-empty homogeneous dense-panel request batch."""
    if not requests:
        raise ValueError("Cannot assemble an empty dense panel request list.")
    if not all(isinstance(request, request_type) for request in requests):
        raise TypeError("DensePanelBatchAssembler accepts only DensePanelWindowRequest values.")
    first = requests[0]
    for request in requests[1:]:
        if (
            request.split != first.split
            or request.request_kind != first.request_kind
            or request.view_kind != first.view_kind
            or request.epoch != first.epoch
        ):
            raise ValueError("Dense panel request batches must be homogeneous.")


def validate_request_dataset_options(
    *,
    split: str,
    request_kind: str,
    view_kind: str,
    view_index: int,
    configured_lookback: int,
    artifact_lookback: object,
) -> None:
    """Validate request-mode combinations and artifact lookback compatibility."""
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported split: {split}")
    if request_kind == "jepa" and view_kind not in {"random_k", "all_valid"}:
        raise ValueError("JEPA requests support only random_k and all_valid views.")
    if request_kind == "embedding" and view_kind not in {"fixed_k", "all_valid"}:
        raise ValueError("Embedding requests support only fixed_k and all_valid views.")
    if view_index < 0:
        raise ValueError("view_index cannot be negative.")
    if artifact_lookback is not None and configured_lookback > int(artifact_lookback):
        raise ValueError(
            f"Configured lookback_days={configured_lookback} exceeds "
            f"artifact lookback_days={artifact_lookback}."
        )


def validate_request_history(request_index: pd.DataFrame, configured_lookback: int) -> None:
    """Require every request endpoint to provide the configured lookback."""
    too_early = request_index["sample_date_idx"] < configured_lookback - 1
    if too_early.any():
        row = request_index.loc[too_early].iloc[0]
        raise ValueError(
            f"Request sample_date_idx={int(row['sample_date_idx'])} cannot provide "
            f"lookback_days={configured_lookback} without padding."
        )


def validate_store_artifact_path(store_artifact_path: Path, configured_artifact_path: Path) -> None:
    """Require a supplied parent-built store to match the dataloader artifact."""
    if store_artifact_path != configured_artifact_path.resolve():
        raise ValueError("The supplied store does not match config.artifact_path.")


def validate_batched_patch_mask_inputs(
    valid_assets: np.ndarray,
    valid_dates: np.ndarray,
    target_dates: np.ndarray,
    patch_len: int,
) -> tuple[int, int, int]:
    """Validate batched daily mask axes and return their shared dimensions."""
    if valid_assets.ndim != 3:
        raise ValueError("valid_asset_mask must have shape [batch, dates, assets].")
    batch_size, n_dates, n_assets = valid_assets.shape
    if n_dates % patch_len:
        raise ValueError("Window length must be divisible by patch_len.")
    for name, mask in (("valid_date_mask", valid_dates), ("target_date_mask", target_dates)):
        if mask.shape != (batch_size, n_dates):
            raise ValueError(f"{name} must have shape [batch, dates].")
    return batch_size, n_dates, n_assets


def validate_jepa_mask_inputs(eligible: np.ndarray, context: np.ndarray) -> None:
    """Validate aligned one-dimensional target-eligibility and context masks."""
    if eligible.ndim != 1 or context.shape != eligible.shape:
        raise ValueError("Patch target and context masks must be one-dimensional and aligned.")
    if np.any(eligible & ~context):
        raise ValueError("Target-eligible patches must also be context-valid.")
