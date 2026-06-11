from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import duckdb
import numpy as np
import pandas as pd

PROBE_TARGET_SCHEMA_VERSION = 1
PROBE_REPORT_VERSION = 1


# ============================================================================
# IMMUTABLE PROBE-TARGET EXPORT
# ============================================================================


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_probe_targets(
    database_path: Path,
    *,
    output_root: Path = Path("data/probe_targets"),
) -> Path:
    """Export canonical future targets into a separate immutable probe artifact."""
    database_path = database_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Canonical database does not exist: {database_path}")
    source_sha256 = _file_sha256(database_path)
    with duckdb.connect(str(database_path), read_only=True) as connection:
        targets = connection.execute("SELECT * FROM targets ORDER BY date").fetchdf()
    if targets.empty:
        raise RuntimeError("Canonical targets table is empty.")
    target_columns = sorted(name for name in targets.columns if name.startswith("future_"))
    if not target_columns:
        raise ValueError("Canonical targets table contains no future_ target columns.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_id = hashlib.sha256(
        f"{source_sha256}|{','.join(target_columns)}|{PROBE_TARGET_SCHEMA_VERSION}".encode("utf-8")
    ).hexdigest()[:16]
    destination = output_root / f"{timestamp}_{artifact_id}"
    temporary = output_root / f".tmp-{artifact_id}-{uuid.uuid4().hex}"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        targets.to_parquet(temporary / "targets.parquet", index=False, compression="zstd")
        manifest = {
            "format_version": PROBE_TARGET_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_database": str(database_path),
            "source_database_sha256": source_sha256,
            "target_columns": target_columns,
            "row_count": int(len(targets)),
            "encoder_features_included": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        temporary.replace(destination)
    except Exception:
        for path in temporary.glob("*"):
            path.unlink()
        temporary.rmdir()
        raise
    print(f"Built probe-target export: {destination}")
    return destination


# ============================================================================
# FROZEN RIDGE PROBES
# ============================================================================


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one train-standardized ridge model and predict validation rows."""
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    standardized_train = (train_x - x_mean) / x_std
    standardized_validation = (validation_x - x_mean) / x_std
    y_mean = float(train_y.mean())
    centered_y = train_y - y_mean
    gram = standardized_train.T @ standardized_train
    coefficients = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
        standardized_train.T @ centered_y,
    )
    return standardized_validation @ coefficients + y_mean, coefficients


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute fixed regression metrics with finite constant-series behavior."""
    residual = actual - predicted
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - float(np.sum(np.square(residual))) / denominator if denominator > 0.0 else 0.0
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if actual.std() > 0.0 and predicted.std() > 0.0
        else 0.0
    )
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": r2,
        "pearson_correlation": correlation,
    }


def run_frozen_probes(
    embedding_artifact: Path,
    target_artifact: Path,
    *,
    output_root: Path = Path("runs/probes"),
    alpha: float = 1.0,
) -> Path:
    """Join separate artifacts and run leakage-safe walk-forward frozen ridge probes."""
    if alpha <= 0.0:
        raise ValueError("Ridge alpha must be positive.")
    embedding_manifest = json.loads(
        (embedding_artifact / "manifest.json").read_text(encoding="utf-8")
    )
    target_manifest = json.loads(
        (target_artifact / "manifest.json").read_text(encoding="utf-8")
    )
    embedding_database = embedding_manifest.get("source_database_sha256")
    target_database = target_manifest.get("source_database_sha256")
    if not embedding_database or embedding_database != target_database:
        raise ValueError("Embedding and probe-target artifacts have different database versions.")

    embeddings = pd.read_parquet(embedding_artifact / "embeddings.parquet")
    targets = pd.read_parquet(target_artifact / "targets.parquet")
    forbidden_embedding_columns = [
        name for name in embeddings.columns if name.startswith("future_")
    ]
    if forbidden_embedding_columns:
        raise ValueError(f"Embedding artifact contains forbidden targets: {forbidden_embedding_columns}")
    if embeddings["date"].duplicated().any():
        raise ValueError("All-valid embedding export must contain one row per date.")
    embeddings["date"] = pd.to_datetime(embeddings["date"])
    targets["date"] = pd.to_datetime(targets["date"])
    target_columns = list(target_manifest["target_columns"])
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
    validation = probe_dataset.loc[probe_dataset["split"].eq("validation")].copy()
    train = probe_dataset.loc[probe_dataset["split"].eq("train")].copy()
    windows = sorted(name for name in validation["validation_window_name"].unique() if name)
    if not windows:
        raise ValueError("Embedding artifact contains no named validation windows.")

    report_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for window_name in windows:
        fold_validation = validation.loc[validation["validation_window_name"].eq(window_name)]
        fold_start = fold_validation["date"].min()
        fold_train = train.loc[train["date"] < fold_start]
        if fold_train.empty:
            raise ValueError(f"Validation window {window_name} has no prior train embeddings.")

        for target_name in target_columns:
            target_train = fold_train.dropna(subset=[target_name])
            target_validation = fold_validation.dropna(subset=[target_name])
            if target_train.empty or target_validation.empty:
                continue
            train_x = target_train[z_columns].to_numpy(dtype=np.float64)
            train_y = target_train[target_name].to_numpy(dtype=np.float64)
            validation_x = target_validation[z_columns].to_numpy(dtype=np.float64)
            actual = target_validation[target_name].to_numpy(dtype=np.float64)
            predicted, coefficients = _ridge_predict(
                train_x, train_y, validation_x, alpha=alpha
            )
            baseline = np.full_like(actual, train_y.mean())
            report_rows.append(
                {
                    "validation_window_name": window_name,
                    "target": target_name,
                    "train_start": str(target_train["date"].min().date()),
                    "train_end": str(target_train["date"].max().date()),
                    "validation_start": str(target_validation["date"].min().date()),
                    "validation_end": str(target_validation["date"].max().date()),
                    "train_count": int(len(target_train)),
                    "validation_count": int(len(target_validation)),
                    "ridge_metrics": _regression_metrics(actual, predicted),
                    "train_mean_baseline_metrics": _regression_metrics(actual, baseline),
                    "ridge_coefficients": coefficients.tolist(),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "date": target_validation["date"].to_numpy(),
                        "validation_window_name": window_name,
                        "target": target_name,
                        "actual": actual,
                        "ridge_prediction": predicted,
                        "train_mean_prediction": baseline,
                    }
                )
            )

    if not prediction_rows:
        raise RuntimeError("No finite probe folds were available.")
    predictions = pd.concat(prediction_rows, ignore_index=True)
    aggregate: dict[str, object] = {}
    for target_name, frame in predictions.groupby("target"):
        aggregate[str(target_name)] = {
            "validation_count": int(len(frame)),
            "ridge_metrics": _regression_metrics(
                frame["actual"].to_numpy(), frame["ridge_prediction"].to_numpy()
            ),
            "train_mean_baseline_metrics": _regression_metrics(
                frame["actual"].to_numpy(), frame["train_mean_prediction"].to_numpy()
            ),
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = hashlib.sha256(
        f"{embedding_manifest['pca_version']}|{target_database}|{alpha}".encode("utf-8")
    ).hexdigest()[:16]
    destination = output_root / f"{timestamp}_{run_id}"
    destination.mkdir(parents=True, exist_ok=False)
    probe_dataset.to_parquet(
        destination / "probe_dataset.parquet", index=False, compression="zstd"
    )
    predictions.to_parquet(destination / "predictions.parquet", index=False, compression="zstd")
    report = {
        "format_version": PROBE_REPORT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "alpha": alpha,
        "embedding_artifact": str(embedding_artifact.resolve()),
        "target_artifact": str(target_artifact.resolve()),
        "source_database_sha256": target_database,
        "target_columns": target_columns,
        "z_columns": z_columns,
        "folds": report_rows,
        "aggregate_out_of_fold": aggregate,
        "targets_joined_into_pretraining_artifact": False,
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Built frozen probe report: {destination}")
    return destination


# ============================================================================
# COMMAND-LINE ENTRY POINTS
# ============================================================================


def export_targets_main() -> None:
    """Run the separate canonical probe-target export CLI."""
    parser = argparse.ArgumentParser(description="Export separate FI-JEPA probe targets.")
    parser.add_argument(
        "--database", type=Path, default=Path("data/processed/market_data.duckdb")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/probe_targets"))
    args = parser.parse_args()
    export_probe_targets(args.database, output_root=args.output_root)


def run_probes_main() -> None:
    """Run leakage-safe frozen probes from separate embedding and target artifacts."""
    parser = argparse.ArgumentParser(description="Run FI-JEPA frozen linear probes.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/probes"))
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    run_frozen_probes(
        args.embeddings,
        args.targets,
        output_root=args.output_root,
        alpha=args.alpha,
    )
