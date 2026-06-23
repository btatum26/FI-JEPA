from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fi_jepa.probes.artifacts import (
    clean_temporary_artifact,
    publish_artifact,
    read_manifest,
    readable_artifact_destination,
)


# ============================================================================
# REUSABLE PROBE DATASET
# ============================================================================


def assemble_probe_dataset(
    embedding_artifact: Path,
    target_artifact: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join frozen embeddings and separate targets into the Phase 1 probe-dataset contract."""
    embedding_manifest = read_manifest(embedding_artifact)
    target_manifest = read_manifest(target_artifact)
    embedding_database = embedding_manifest.get("source_database_sha256")
    target_database = target_manifest.get("source_database_sha256")
    if not embedding_database or embedding_database != target_database:
        raise ValueError("Embedding and probe-target artifacts have different source database hashes.")

    embeddings = pd.read_parquet(embedding_artifact / "embeddings.parquet")
    targets = pd.read_parquet(target_artifact / "targets.parquet")
    baseline_features_path = target_artifact / "baseline_features.parquet"
    baseline_features = (
        pd.read_parquet(baseline_features_path)
        if baseline_features_path.is_file()
        else pd.DataFrame({"date": targets["date"].drop_duplicates()})
    )
    forbidden_embedding_columns = [
        name for name in embeddings.columns if name.startswith("future_")
    ]
    if forbidden_embedding_columns:
        raise ValueError(f"Embedding artifact contains forbidden targets: {forbidden_embedding_columns}")
    if embeddings["date"].duplicated().any():
        raise ValueError("All-valid embedding export must contain one row per date.")

    embeddings["date"] = pd.to_datetime(embeddings["date"])
    targets["date"] = pd.to_datetime(targets["date"])
    baseline_features["date"] = pd.to_datetime(baseline_features["date"])
    target_columns = [str(name) for name in target_manifest["target_columns"]]
    baseline_feature_columns = [
        str(name)
        for name in target_manifest.get("baseline_feature_columns", [])
        if str(name) in baseline_features.columns
    ]
    forbidden_baseline_columns = [
        name for name in baseline_feature_columns if name.startswith("future_") or name.startswith("z_")
    ]
    if forbidden_baseline_columns:
        raise ValueError(f"Baseline feature artifact contains forbidden columns: {forbidden_baseline_columns}")
    z_columns = sorted(
        (name for name in embeddings.columns if name.startswith("z_")),
        key=lambda name: int(name.removeprefix("z_")),
    )
    if not z_columns:
        raise ValueError("Embedding artifact contains no z_ columns.")

    probe_dataset = embeddings.merge(
        targets[["date", *target_columns]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    if baseline_feature_columns:
        probe_dataset = probe_dataset.merge(
            baseline_features[["date", *baseline_feature_columns]],
            on="date",
            how="left",
            validate="one_to_one",
        )
    for target_name in target_columns:
        # Availability masks make missing future horizons explicit and reusable by every head.
        probe_dataset[f"target_available__{target_name}"] = np.isfinite(
            probe_dataset[target_name].to_numpy(dtype=np.float64)
        )

    validation = probe_dataset.loc[probe_dataset["split"].eq("validation")]
    windows: list[dict[str, object]] = []
    for window_name, frame in validation.groupby("validation_window_name", sort=True):
        if not window_name:
            continue
        fold_start = frame["date"].min()
        fold_end = frame["date"].max()
        windows.append(
            {
                "validation_window_name": str(window_name),
                "validation_start": str(fold_start.date()),
                "validation_end": str(fold_end.date()),
                "train_cutoff_exclusive": str(fold_start.date()),
                "validation_row_count": int(len(frame)),
            }
        )
    if not windows:
        raise ValueError("Embedding artifact contains no named validation windows.")

    metadata = {
        "embedding_manifest": embedding_manifest,
        "target_manifest": target_manifest,
        "source_database_sha256": target_database,
        "target_columns": target_columns,
        "target_availability_columns": [
            f"target_available__{target_name}" for target_name in target_columns
        ],
        "baseline_feature_columns": baseline_feature_columns,
        "z_columns": z_columns,
        "validation_windows": windows,
    }
    return probe_dataset, metadata


def build_probe_dataset(
    embedding_artifact: Path,
    target_artifact: Path,
    *,
    output_root: Path = Path("data/probe_targets"),
    name: str | None = None,
) -> Path:
    """Build one immutable joined dataset that can be reused by multiple probe heads."""
    probe_dataset, metadata = assemble_probe_dataset(embedding_artifact, target_artifact)
    destination, temporary = readable_artifact_destination(
        output_root, name or f"{embedding_artifact.name}_probe_dataset"
    )
    try:
        probe_dataset.to_parquet(
            temporary / "probe_dataset.parquet", index=False, compression="zstd"
        )
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_artifact": str(embedding_artifact.resolve()),
            "target_artifact": str(target_artifact.resolve()),
            "source_database_sha256": metadata["source_database_sha256"],
            "target_columns": metadata["target_columns"],
            "target_availability_columns": metadata["target_availability_columns"],
            "baseline_feature_columns": metadata["baseline_feature_columns"],
            "z_columns": metadata["z_columns"],
            "validation_windows": metadata["validation_windows"],
            "row_count": int(len(probe_dataset)),
            "targets_joined_into_pretraining_artifact": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        publish_artifact(temporary, destination)
    except Exception:
        clean_temporary_artifact(temporary)
        raise
    print(f"Built probe dataset: {destination}")
    return destination


def load_probe_dataset(probe_dataset_artifact: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and validate one reusable probe-dataset artifact."""
    manifest = read_manifest(probe_dataset_artifact)
    probe_dataset = pd.read_parquet(probe_dataset_artifact / "probe_dataset.parquet")
    probe_dataset["date"] = pd.to_datetime(probe_dataset["date"])
    forbidden = [
        name
        for name in probe_dataset.columns
        if name.startswith("future_") and name not in manifest["target_columns"]
    ]
    if forbidden:
        raise ValueError(f"Probe dataset contains undeclared future targets: {forbidden}")
    if probe_dataset["date"].duplicated().any():
        raise ValueError("Probe dataset must contain one row per date.")
    return probe_dataset, manifest
