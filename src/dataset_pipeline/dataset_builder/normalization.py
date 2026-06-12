from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from dataset_pipeline.dataset_builder.config import quote_identifier


# ============================================================================
# NORMALIZATION
# ============================================================================


def source_expression(feature_name: str, transform: str, alias: str) -> str:
    column = f"{alias}.{quote_identifier(feature_name)}"
    if transform == "none":
        return column
    if transform == "log":
        return f"ln({column})"
    if transform == "log1p":
        return f"ln(1.0 + {column})"
    raise ValueError(f"Unsupported feature transform: {transform}")


def valid_expression(feature_name: str, transform: str, alias: str) -> str:
    column = f"{alias}.{quote_identifier(feature_name)}"
    valid = f"{column} IS NOT NULL AND isfinite(CAST({column} AS DOUBLE))"
    if transform == "log":
        valid += f" AND {column} > 0.0"
    if transform == "log1p":
        valid += f" AND {column} > -1.0"
    return f"({valid})"


def fit_normalization(
    connection: duckdb.DuckDBPyConnection,
    feature_manifest: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    """Fit robust normalization only on real finite train facts."""
    low_q, high_q = config["normalization"]["winsorize_quantiles"]
    rows: list[dict[str, object]] = []
    sources = {
        "asset": (
            "ticker_features AS source "
            "INNER JOIN date_manifest AS dates USING (date) "
            "INNER JOIN asset_manifest AS assets USING (symbol)",
            "dates.train_fact_allowed AND source.valid_observation AND assets.trainable",
        ),
        "market": (
            "features AS source INNER JOIN date_manifest AS dates USING (date)",
            "dates.train_fact_allowed",
        ),
        "macro": (
            "features AS source INNER JOIN date_manifest AS dates USING (date)",
            "dates.train_fact_allowed",
        ),
    }
    for input_group, group_features in feature_manifest.groupby("input_group", sort=False):
        aggregates: list[str] = []
        for _, feature in group_features.iterrows():
            index = int(feature["feature_index"])
            valid = valid_expression(feature["feature_name"], feature["transform"], "source")
            value = source_expression(feature["feature_name"], feature["transform"], "source")
            real = f"CASE WHEN {valid} THEN CAST({value} AS DOUBLE) END"
            aggregates.extend(
                [
                    f"count({real}) AS f{index}_count",
                    f"quantile_cont({real}, {float(low_q)}) AS f{index}_lower",
                    f"quantile_cont({real}, 0.25) AS f{index}_q25",
                    f"median({real}) AS f{index}_center",
                    f"quantile_cont({real}, 0.75) AS f{index}_q75",
                    f"quantile_cont({real}, {float(high_q)}) AS f{index}_upper",
                ]
            )
        source, condition = sources[input_group]
        result = connection.execute(
            f"SELECT {', '.join(aggregates)} FROM {source} WHERE {condition}"
        ).fetchone()
        values = dict(zip([item[0] for item in connection.description], result, strict=True))
        for _, feature in group_features.iterrows():
            index = int(feature["feature_index"])
            count = int(values[f"f{index}_count"])
            if count == 0:
                raise ValueError(
                    f"No finite train facts available for {input_group}.{feature['feature_name']}"
                )
            scale = float(values[f"f{index}_q75"] - values[f"f{index}_q25"])
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            rows.append(
                {
                    "feature_name": feature["feature_name"],
                    "input_group": input_group,
                    "transform": feature["transform"],
                    "normalization_method": feature["normalization_method"],
                    "fit_count": count,
                    "lower_bound": float(values[f"f{index}_lower"]),
                    "center": float(values[f"f{index}_center"]),
                    "scale": scale,
                    "upper_bound": float(values[f"f{index}_upper"]),
                }
            )
    return pd.DataFrame(rows)


def normalized_expression(
    feature: pd.Series,
    normalization: pd.Series,
    alias: str,
) -> tuple[str, str]:
    valid = valid_expression(feature["feature_name"], feature["transform"], alias)
    value = source_expression(feature["feature_name"], feature["transform"], alias)
    lower = float(normalization["lower_bound"])
    upper = float(normalization["upper_bound"])
    center = float(normalization["center"])
    scale = float(normalization["scale"])
    clipped = f"least(greatest(CAST({value} AS DOUBLE), {lower!r}), {upper!r})"
    normalized = (
        f"CAST(CASE WHEN {valid} THEN "
        f"({clipped} - {center!r}) / {scale!r} ELSE 0.0 END AS FLOAT)"
    )
    return normalized, valid
