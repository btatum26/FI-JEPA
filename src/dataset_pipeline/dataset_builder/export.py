from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from dataset_pipeline.dataset_builder.config import quote_identifier, quote_literal
from dataset_pipeline.dataset_builder.normalization import normalized_expression


# ============================================================================
# SPARSE PARQUET EXPORT
# ============================================================================


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def export_fact_file(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    input_group: str,
    permission_column: str,
    feature_manifest: pd.DataFrame,
    normalization: pd.DataFrame,
) -> None:
    """Stream one sparse normalized fact table from DuckDB directly to Parquet."""
    group_features = feature_manifest.loc[feature_manifest["input_group"].eq(input_group)]
    columns: list[str] = []
    valid_masks: list[str] = []
    for _, feature in group_features.iterrows():
        stats = normalization.loc[
            normalization["input_group"].eq(input_group)
            & normalization["feature_name"].eq(feature["feature_name"])
        ].iloc[0]
        value, valid = normalized_expression(feature, stats, "source")
        name = feature["feature_name"]
        columns.extend(
            [
                f"{value} AS {quote_identifier(name)}",
                f"{valid} AS {quote_identifier(f'{name}__valid')}",
            ]
        )
        valid_masks.append(valid)

    if input_group == "asset":
        keys = [
            "source.date",
            "CAST(dates.date_idx AS INTEGER) AS date_idx",
            "CAST(assets.asset_id AS INTEGER) AS asset_id",
            "TRUE AS valid_asset",
        ]
        source = (
            "ticker_features AS source "
            "INNER JOIN date_manifest AS dates USING (date) "
            "INNER JOIN asset_manifest AS assets USING (symbol)"
        )
        condition = (
            f"dates.{quote_identifier(permission_column)} "
            "AND assets.trainable AND source.valid_observation"
        )
        order_by = "source.date, assets.asset_id"
    else:
        keys = [
            "source.date",
            "CAST(dates.date_idx AS INTEGER) AS date_idx",
            f"({' OR '.join(valid_masks)}) AS valid_date",
        ]
        source = "features AS source INNER JOIN date_manifest AS dates USING (date)"
        condition = f"dates.{quote_identifier(permission_column)}"
        order_by = "source.date"

    query = (
        f"SELECT {', '.join(keys + columns)} FROM {source} "
        f"WHERE {condition} ORDER BY {order_by}"
    )
    connection.execute(
        f"COPY ({query}) TO {quote_literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def validate_and_report(
    connection: duckdb.DuckDBPyConnection,
    output_dir: Path,
    feature_manifest: pd.DataFrame,
    date_manifest: pd.DataFrame,
    asset_manifest: pd.DataFrame,
) -> dict[str, object]:
    """Enforce the model-dataset contract and write a compact quality summary."""
    if (date_manifest["train_fact_allowed"] & date_manifest["validation_fact_allowed"]).any():
        raise AssertionError("Train and validation fact permissions overlap.")
    protection_columns = [
        "validation_sample",
        "protected_input_lookback",
        "protected_forward_target",
    ]
    if date_manifest[protection_columns].sum(axis=1).gt(1).any():
        raise AssertionError("Date protection meanings must be disjoint.")
    if not (
        date_manifest["protected_holdout"]
        == date_manifest[protection_columns].any(axis=1)
    ).all():
        raise AssertionError("protected_holdout must be the union of explicit protections.")
    if not (date_manifest["protected_holdout"] == ~date_manifest["train_fact_allowed"]).all():
        raise AssertionError("protected_holdout must be the inverse of train_fact_allowed.")
    forbidden = feature_manifest["feature_name"].str.contains(
        "oas|future_|target|label", case=False, regex=True
    )
    if forbidden.any():
        raise AssertionError(
            "Forbidden features entered the export: "
            f"{feature_manifest.loc[forbidden, 'feature_name'].tolist()}"
        )

    row_counts: dict[str, int] = {}
    for input_group in ("asset", "market", "macro"):
        keys = "date, asset_id" if input_group == "asset" else "date"
        for split in ("train", "validation"):
            path = output_dir / f"{split}_{input_group}_features.parquet"
            row_counts[path.name] = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet({quote_literal(path)})"
                ).fetchone()[0]
            )
            duplicates = int(
                connection.execute(
                    f"""
                    SELECT count(*) FROM (
                        SELECT {keys}
                        FROM read_parquet({quote_literal(path)})
                        GROUP BY {keys}
                        HAVING count(*) > 1
                    )
                    """
                ).fetchone()[0]
            )
            if duplicates:
                raise AssertionError(f"{path.name} contains {duplicates} duplicate fact keys.")
        overlap = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                    SELECT DISTINCT date
                    FROM read_parquet({quote_literal(output_dir / f'train_{input_group}_features.parquet')})
                    INTERSECT
                    SELECT DISTINCT date
                    FROM read_parquet({quote_literal(output_dir / f'validation_{input_group}_features.parquet')})
                )
                """
            ).fetchone()[0]
        )
        if overlap:
            raise AssertionError(f"{input_group} train and validation facts overlap by date.")

    report = {
        "row_counts": row_counts,
        "date_count": int(len(date_manifest)),
        "sample_eligible_dates": int(date_manifest["sample_eligible"].sum()),
        "validation_sample_dates": int(date_manifest["validation_sample"].sum()),
        "protected_input_lookback_dates": int(
            date_manifest["protected_input_lookback"].sum()
        ),
        "protected_forward_target_dates": int(
            date_manifest["protected_forward_target"].sum()
        ),
        "trainable_asset_count": int(asset_manifest["trainable"].sum()),
        "excluded_asset_count": int((~asset_manifest["trainable"]).sum()),
        "feature_counts": {
            group: int(count)
            for group, count in feature_manifest.groupby("input_group").size().items()
        },
        "oas_features_excluded": True,
        "targets_excluded": True,
        "stores_windows": False,
        "stores_complete_asset_grid": False,
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
