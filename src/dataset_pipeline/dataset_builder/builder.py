from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid

import duckdb
import yaml

from dataset_pipeline.dataset_builder.config import (
    OAS_PATTERNS,
    TARGET_PATTERNS,
    artifact_name_for,
    build_id_for,
    find_existing_artifact,
    load_model_dataset_config,
    quote_identifier,
    resolve_feature_manifest,
    sha256_file,
)
from dataset_pipeline.dataset_builder.export import (
    export_fact_file,
    validate_and_report,
    write_frame,
)
from dataset_pipeline.dataset_builder.manifests import (
    build_asset_manifest,
    build_date_manifest,
)
from dataset_pipeline.dataset_builder.normalization import fit_normalization


# ============================================================================
# MODEL DATASET BUILD
# ============================================================================


def build_model_dataset(config_path: Path) -> Path:
    """Build one immutable sparse normalized model dataset from the canonical DuckDB."""
    config = load_model_dataset_config(config_path)
    source_database = Path(config["source_database"])
    if not source_database.exists():
        raise FileNotFoundError(f"Source database does not exist: {source_database}")
    feature_manifest = resolve_feature_manifest(config)
    forbidden = [
        name
        for name in feature_manifest["feature_name"].tolist()
        if any(pattern in name.lower() for pattern in OAS_PATTERNS + TARGET_PATTERNS)
    ]
    if forbidden:
        raise ValueError(f"Forbidden features configured for model export: {forbidden}")

    source_sha256 = sha256_file(source_database)
    build_id = build_id_for(config, source_sha256)
    output_root = Path(config["output_root"]) / config["dataset_name"]
    output_root.mkdir(parents=True, exist_ok=True)
    existing = find_existing_artifact(output_root, build_id)
    if existing is not None:
        return existing

    created_at = datetime.now(timezone.utc)
    artifact_name = artifact_name_for(build_id, created_at)
    destination = output_root / artifact_name
    temporary = output_root / f".tmp-{build_id}-{uuid.uuid4().hex}"
    temporary.mkdir()
    connection = duckdb.connect(str(source_database), read_only=True)
    try:
        schemas = {
            table: {
                row[0]
                for row in connection.execute(f"DESCRIBE {quote_identifier(table)}").fetchall()
            }
            for table in ("features", "ticker_features")
        }
        missing = sorted(
            set(feature_manifest.loc[feature_manifest["input_group"].eq("asset"), "feature_name"])
            - schemas["ticker_features"]
        )
        missing += sorted(
            set(
                feature_manifest.loc[
                    feature_manifest["input_group"].isin(["market", "macro"]), "feature_name"
                ]
            )
            - schemas["features"]
        )
        if missing:
            raise ValueError(f"Configured features missing from source database: {missing}")

        date_config = config["dates"]
        dates = connection.execute("SELECT date FROM features ORDER BY date").fetchdf()["date"]
        sample_reference_symbol = date_config["sample_reference_symbol"]
        sample_dates = connection.execute(
            """
            SELECT ticker.date
            FROM ticker_features AS ticker
            INNER JOIN features USING (date)
            WHERE ticker.symbol = ? AND ticker.valid_observation
            ORDER BY ticker.date
            """,
            [sample_reference_symbol],
        ).fetchdf()["date"]
        if sample_dates.empty:
            raise ValueError(
                "Sample reference symbol has no valid dates joined to features: "
                f"{sample_reference_symbol}"
            )
        date_manifest = build_date_manifest(
            dates,
            sample_dates=sample_dates,
            context_start=date_config["context_start"],
            sample_start=date_config["sample_start"],
            sample_end=date_config.get("sample_end"),
            lookback_days=int(date_config["lookback_days"]),
            max_forward_horizon=int(date_config["max_forward_horizon"]),
            validation_windows=config["splits"]["validation_windows"],
        )
        connection.register("date_manifest", date_manifest)
        asset_manifest = build_asset_manifest(connection, config)
        connection.register("asset_manifest", asset_manifest)
        normalization = fit_normalization(connection, feature_manifest, config)

        write_frame(date_manifest, temporary / "dates.parquet")
        write_frame(asset_manifest, temporary / "assets.parquet")
        write_frame(feature_manifest, temporary / "feature_manifest.parquet")
        write_frame(normalization, temporary / "normalization.parquet")
        for input_group in ("asset", "market", "macro"):
            for split, permission in (
                ("train", "train_fact_allowed"),
                ("validation", "validation_fact_allowed"),
            ):
                export_fact_file(
                    connection,
                    temporary / f"{split}_{input_group}_features.parquet",
                    input_group,
                    permission,
                    feature_manifest,
                    normalization,
                )

        quality = validate_and_report(
            connection, temporary, feature_manifest, date_manifest, asset_manifest
        )
        resolved_config = {
            **config,
            "resolved": {
                "build_id": build_id,
                "artifact_name": artifact_name,
                "created_at_utc": created_at.isoformat(),
                "source_database_sha256": source_sha256,
                "date_start": str(date_manifest["date"].min()),
                "date_end": str(date_manifest["date"].max()),
                "sample_reference_symbol": sample_reference_symbol,
                "minimum_train_observations": int(config["assets"]["minimum_train_observations"]),
            },
        }
        (temporary / "config_resolved.yaml").write_text(
            yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
        )
        manifest = {
            "dataset_name": config["dataset_name"],
            "build_id": build_id,
            "artifact_name": artifact_name,
            "created_at_utc": created_at.isoformat(),
            "source_database": str(source_database),
            "source_database_sha256": source_sha256,
            "sparse_asset_facts": True,
            "train_validation_facts_date_disjoint": True,
            "zero_fill_requires_masks": True,
            "jepa_target_rules": config["jepa_target_rules"],
            "quality": quality,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        connection.close()
    print(f"Built model dataset: {destination}")
    return destination


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sparse FI-JEPA model dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_dataset.yaml"),
        help="Model dataset YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    build_model_dataset(parse_args().config)


if __name__ == "__main__":
    main()
