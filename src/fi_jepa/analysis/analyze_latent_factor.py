from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd

ANALYSIS_FORMAT_VERSION = 2
DATASET_TABLES = ("features", "ticker_features", "targets")
NUMERIC_DUCKDB_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "REAL",
    "DOUBLE",
    "DECIMAL",
}
TIME_CONTROL_FEATURES = ("elapsed_calendar_days", "elapsed_trading_rows")
DEFAULT_FEATURES = (
    *TIME_CONTROL_FEATURES,
    "vix_level",
    "vix_change_1d",
    "vix_change_5d",
    "vix_change_21d",
    "vix_change_63d",
    "realized_vol_5d",
    "realized_vol_21d",
    "realized_vol_63d",
    "realized_vol_126d",
    "drawdown_5d",
    "drawdown_21d",
    "drawdown_63d",
    "drawdown_126d",
    "breadth_1d",
    "pct_above_ma_63d",
    "xs_dispersion_1d",
    "xs_iqr_1d",
    "future_realized_vol_21d",
    "future_realized_vol_63d",
    "future_realized_vol_126d",
)


@dataclass(frozen=True)
class FeatureSpec:
    """One validated numeric feature available for latent-coordinate analysis."""

    selector: str
    source_table: str
    source_column: str
    output_name: str
    data_type: str
    uses_future_data: bool


# ============================================================================
# ARTIFACT AND DATASET FEATURE LOADING
# ============================================================================


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinate_sort_key(name: str) -> tuple[int, str]:
    """Sort standard z_N coordinates numerically and unknown names lexically."""
    suffix = name.removeprefix("z_")
    return (int(suffix), name) if name.startswith("z_") and suffix.isdigit() else (10**9, name)


def _normalize_cli_values(values: Sequence[str] | None) -> list[str] | None:
    """Expand comma-separated and space-separated CLI selections into one list."""
    if values is None:
        return None
    normalized = [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ]
    return normalized


def _load_embeddings(
    embedding_artifact: Path,
    coordinates: Sequence[str] | None,
) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    """Load one all-valid embedding export and resolve selected PCA coordinates."""
    manifest_path = embedding_artifact / "manifest.json"
    embeddings_path = embedding_artifact / "embeddings.parquet"
    if not manifest_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(
            f"Embedding artifact must contain manifest.json and embeddings.parquet: "
            f"{embedding_artifact}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embeddings = pd.read_parquet(embeddings_path)
    available_coordinates = sorted(
        (name for name in embeddings.columns if name.startswith("z_")),
        key=_coordinate_sort_key,
    )
    if not available_coordinates:
        raise ValueError("Embedding artifact contains no z_ coordinates.")

    selected_coordinates = list(coordinates) if coordinates is not None else available_coordinates
    selected_coordinates = list(dict.fromkeys(selected_coordinates))
    missing = [name for name in selected_coordinates if name not in available_coordinates]
    if missing:
        raise ValueError(
            f"Embedding artifact does not contain coordinates {missing}. "
            f"Available coordinates: {available_coordinates}"
        )

    forbidden = [name for name in embeddings.columns if name.startswith("future_")]
    if forbidden:
        raise ValueError(f"Embedding artifact contains forbidden future targets: {forbidden}")
    if embeddings["date"].duplicated().any():
        raise ValueError("All-valid embedding export must contain one row per date.")

    embeddings = embeddings.copy()
    embeddings["date"] = pd.to_datetime(embeddings["date"])
    embeddings = embeddings.sort_values("date").reset_index(drop=True)
    return embeddings, manifest, selected_coordinates


def _feature_catalog(database_path: Path) -> list[FeatureSpec]:
    """Build a selectable numeric-feature catalog from the live canonical schema."""
    placeholders = ", ".join("?" for _ in DATASET_TABLES)
    query = f"""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_name IN ({placeholders})
        ORDER BY table_name, ordinal_position
    """
    with duckdb.connect(str(database_path), read_only=True) as connection:
        columns = connection.execute(query, list(DATASET_TABLES)).fetchdf()

    numeric = columns.loc[
        columns["data_type"].astype(str).str.upper().map(
            lambda value: value in NUMERIC_DUCKDB_TYPES or value.startswith("DECIMAL")
        )
    ].copy()
    counts = numeric["column_name"].value_counts()
    catalog = [
        FeatureSpec(
            selector=f"{row.table_name}.{row.column_name}",
            source_table=str(row.table_name),
            source_column=str(row.column_name),
            output_name=(
                str(row.column_name)
                if counts[str(row.column_name)] == 1
                else f"{row.table_name}__{row.column_name}"
            ),
            data_type=str(row.data_type),
            uses_future_data=str(row.table_name) == "targets",
        )
        for row in numeric.itertuples(index=False)
    ]
    catalog.extend(
        FeatureSpec(
            selector=name,
            source_table="derived",
            source_column=name,
            output_name=name,
            data_type="DOUBLE",
            uses_future_data=False,
        )
        for name in TIME_CONTROL_FEATURES
    )
    return catalog


def _resolve_features(
    catalog: Sequence[FeatureSpec],
    requested: Sequence[str] | None,
) -> list[FeatureSpec]:
    """Resolve qualified or unique unqualified feature names against the catalog."""
    selectors = list(requested) if requested is not None else list(DEFAULT_FEATURES)
    if selectors == ["all"]:
        return list(catalog)
    if "all" in selectors:
        raise ValueError("Feature selector 'all' must be used by itself.")

    by_selector = {spec.selector: spec for spec in catalog}
    by_unqualified: dict[str, list[FeatureSpec]] = {}
    for spec in catalog:
        by_unqualified.setdefault(spec.source_column, []).append(spec)

    resolved: list[FeatureSpec] = []
    for selector in selectors:
        if selector in by_selector:
            spec = by_selector[selector]
        else:
            matches = by_unqualified.get(selector, [])
            if not matches:
                raise ValueError(
                    f"Unknown feature {selector!r}. Use --list-features to inspect the live schema."
                )
            if len(matches) > 1:
                qualified = [match.selector for match in matches]
                raise ValueError(f"Ambiguous feature {selector!r}; use one of {qualified}.")
            spec = matches[0]
        if spec not in resolved:
            resolved.append(spec)
    if not resolved:
        raise ValueError("At least one feature must be selected.")
    return resolved


def _quoted_identifier(name: str) -> str:
    """Quote one already-validated DuckDB identifier."""
    return '"' + name.replace('"', '""') + '"'


def _load_selected_features(
    database_path: Path,
    market_symbol: str,
    features: Sequence[FeatureSpec],
) -> pd.DataFrame:
    """Load selected canonical features while preserving their source-table semantics."""
    selected_expressions: list[str] = []
    for spec in features:
        if spec.source_table == "derived":
            continue
        selected_expressions.append(
            f"{spec.source_table}.{_quoted_identifier(spec.source_column)} "
            f"AS {_quoted_identifier(spec.output_name)}"
        )

    select_clause = ",\n            ".join(selected_expressions)
    if select_clause:
        select_clause = ",\n            " + select_clause
    query = f"""
        SELECT
            features.date
            {select_clause}
        FROM features
        LEFT JOIN ticker_features
            ON features.date = ticker_features.date
            AND ticker_features.symbol = ?
            AND ticker_features.valid_observation
        LEFT JOIN targets
            ON features.date = targets.date
            AND targets.symbol = ?
        ORDER BY features.date
    """
    with duckdb.connect(str(database_path), read_only=True) as connection:
        frame = connection.execute(query, [market_symbol, market_symbol]).fetchdf()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================


def _finite_pair(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    """Return finite numeric pairs for one correlation calculation."""
    pair = frame[[left, right]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(pair[left].to_numpy()) & np.isfinite(pair[right].to_numpy())
    return pair.loc[finite]


def _linear_time_detrend(pair: pd.DataFrame) -> pd.DataFrame:
    """Remove a least-squares linear time trend from both columns of one pair."""
    time = np.linspace(-1.0, 1.0, len(pair), dtype=np.float64)
    design = np.column_stack([np.ones(len(pair), dtype=np.float64), time])
    detrended = pair.copy()
    for name in pair.columns:
        values = pair[name].to_numpy(dtype=np.float64)
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        residuals = values - design @ coefficients
        if np.std(residuals) <= np.std(values) * 1e-10:
            residuals = np.zeros_like(residuals)
        detrended[name] = residuals
    return detrended


def _correlation_row(
    frame: pd.DataFrame,
    *,
    coordinate: str,
    feature: FeatureSpec,
    segment: str,
    transform: str,
) -> dict[str, object]:
    """Calculate one pairwise Pearson and Spearman correlation record."""
    pair = _finite_pair(frame, coordinate, feature.output_name)
    if transform == "linear_time_detrended" and len(pair) >= 3:
        pair = _linear_time_detrend(pair)

    pearson = float("nan")
    spearman = float("nan")
    if (
        len(pair) >= 3
        and pair[coordinate].std() > 0.0
        and pair[feature.output_name].std() > 0.0
    ):
        pearson = float(pair[coordinate].corr(pair[feature.output_name], method="pearson"))
        coordinate_ranks = pair[coordinate].rank(method="average")
        feature_ranks = pair[feature.output_name].rank(method="average")
        spearman = float(coordinate_ranks.corr(feature_ranks, method="pearson"))
    return {
        "segment": segment,
        "transform": transform,
        "coordinate": coordinate,
        "feature_selector": feature.selector,
        "source_table": feature.source_table,
        "variable": feature.output_name,
        "uses_future_data": feature.uses_future_data,
        "observation_count": int(len(pair)),
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "pearson_r_squared": pearson * pearson,
    }


def calculate_correlations(
    dataset: pd.DataFrame,
    coordinates: Sequence[str],
    features: Sequence[FeatureSpec],
) -> pd.DataFrame:
    """Calculate transformed correlations for every coordinate, feature, and segment.

    First differences are calculated before segment filtering so every date uses
    the immediately preceding exported date. Linear detrending is fitted within
    each reported segment and coordinate-feature pair.
    """
    difference_columns = [*coordinates, *(feature.output_name for feature in features)]
    differences = dataset.copy()
    differences[difference_columns] = differences[difference_columns].diff()

    segments: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=dataset.index)),
        ("train", dataset["split"].eq("train")),
        ("validation", dataset["split"].eq("validation")),
    ]
    for window_name in sorted(
        name for name in dataset["validation_window_name"].dropna().unique() if name
    ):
        segments.append(
            (
                f"validation_window:{window_name}",
                dataset["validation_window_name"].eq(window_name),
            )
        )

    rows: list[dict[str, object]] = []
    for segment, mask in segments:
        for coordinate in coordinates:
            for feature in features:
                for transform, source in (
                    ("level", dataset),
                    ("first_difference", differences),
                    ("linear_time_detrended", dataset),
                ):
                    rows.append(
                        _correlation_row(
                            source.loc[mask],
                            coordinate=coordinate,
                            feature=feature,
                            segment=segment,
                            transform=transform,
                        )
                    )
    return pd.DataFrame(rows)


# ============================================================================
# ANALYSIS ARTIFACT
# ============================================================================


def analyze_latent_coordinates(
    embedding_artifact: Path,
    database_path: Path,
    *,
    coordinates: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    market_symbol: str = "ETF_SPY",
    output_root: Path = Path("runs/latent_factor_analysis"),
    verify_database_hash: bool = True,
) -> Path:
    """Correlate selected exported PCA coordinates with selected canonical features.

    Coordinate selection defaults to every exported ``z_*`` dimension. Feature
    selection accepts qualified names from ``features``, ``ticker_features``,
    and ``targets`` plus derived time controls. Future targets are joined only
    inside this analysis output and never mutate representation or pretraining
    artifacts.
    """
    embedding_artifact = embedding_artifact.resolve()
    database_path = database_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Canonical database does not exist: {database_path}")

    embeddings, embedding_manifest, selected_coordinates = _load_embeddings(
        embedding_artifact,
        coordinates,
    )
    catalog = _feature_catalog(database_path)
    selected_features = _resolve_features(catalog, feature_names)
    expected_database_sha256 = embedding_manifest.get("source_database_sha256")
    actual_database_sha256 = _file_sha256(database_path) if verify_database_hash else None
    if (
        verify_database_hash
        and expected_database_sha256
        and actual_database_sha256 != expected_database_sha256
    ):
        raise ValueError("Embedding artifact and canonical database have different versions.")

    feature_frame = _load_selected_features(database_path, market_symbol, selected_features)
    dataset = embeddings.merge(feature_frame, on="date", how="left", validate="one_to_one")
    dataset["elapsed_calendar_days"] = (
        dataset["date"] - dataset["date"].min()
    ).dt.days.astype(np.float64)
    dataset["elapsed_trading_rows"] = np.arange(len(dataset), dtype=np.float64)

    correlations = calculate_correlations(dataset, selected_coordinates, selected_features)
    strongest_candidates = correlations.loc[correlations["segment"].eq("all")].copy()
    strongest_candidates = strongest_candidates.loc[
        np.isfinite(strongest_candidates["pearson_correlation"])
    ]
    strongest = (
        strongest_candidates
        .assign(absolute_pearson=lambda frame: frame["pearson_correlation"].abs())
        .sort_values(
            ["coordinate", "transform", "absolute_pearson"],
            ascending=[True, True, False],
        )
        .groupby(["coordinate", "transform"], sort=False)
        .head(8)
        .drop(columns="absolute_pearson")
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    selection_payload = json.dumps(
        {
            "coordinates": selected_coordinates,
            "features": [feature.selector for feature in selected_features],
        },
        sort_keys=True,
    )
    analysis_id = hashlib.sha256(
        (
            f"{embedding_manifest.get('pca_version')}|{expected_database_sha256}|"
            f"{selection_payload}|{market_symbol}|{ANALYSIS_FORMAT_VERSION}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    coordinate_label = (
        selected_coordinates[0] if len(selected_coordinates) == 1 else f"{len(selected_coordinates)}_z"
    )
    destination = output_root / f"{timestamp}_{coordinate_label}_{analysis_id}"
    destination.mkdir(parents=True, exist_ok=False)
    dataset.to_parquet(destination / "analysis_dataset.parquet", index=False, compression="zstd")
    correlations.to_csv(destination / "correlations.csv", index=False)
    report = {
        "format_version": ANALYSIS_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_artifact": str(embedding_artifact),
        "database": str(database_path),
        "source_database_sha256": expected_database_sha256,
        "database_hash_verified": verify_database_hash,
        "coordinates": selected_coordinates,
        "features": [asdict(feature) for feature in selected_features],
        "market_symbol": market_symbol,
        "row_count": int(len(dataset)),
        "first_date": str(dataset["date"].min().date()),
        "last_date": str(dataset["date"].max().date()),
        "strongest_absolute_correlations": strongest.to_dict(orient="records"),
        "future_targets_joined_only_for_analysis": True,
        "pretraining_artifact_mutated": False,
        "interpretation_warning": (
            "Correlation does not establish latent-axis meaning. Level correlations can be "
            "dominated by common time trends; inspect first_difference and "
            "linear_time_detrended results before assigning semantics."
        ),
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Built latent-factor analysis: {destination}")
    return destination


def analyze_latent_coordinate(
    embedding_artifact: Path,
    database_path: Path,
    *,
    coordinate: str = "z_1",
    feature_names: Sequence[str] | None = None,
    market_symbol: str = "ETF_SPY",
    output_root: Path = Path("runs/latent_factor_analysis"),
    verify_database_hash: bool = True,
) -> Path:
    """Backward-compatible wrapper for analyzing one selected coordinate."""
    return analyze_latent_coordinates(
        embedding_artifact,
        database_path,
        coordinates=[coordinate],
        feature_names=feature_names,
        market_symbol=market_symbol,
        output_root=output_root,
        verify_database_hash=verify_database_hash,
    )


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse the latent-factor analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Correlate FI-JEPA PCA coordinates with selected canonical dataset features."
    )
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/market_data.duckdb"),
    )
    parser.add_argument(
        "--coordinates",
        nargs="+",
        help="Coordinates to analyze, separated by spaces or commas. Defaults to all exported z_*.",
    )
    parser.add_argument(
        "--coordinate",
        action="append",
        dest="legacy_coordinates",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--features",
        nargs="+",
        help=(
            "Numeric feature selectors separated by spaces or commas. Accepts unique column "
            "names, qualified table.column names, or 'all'. Defaults to the original risk set."
        ),
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Print selectable numeric feature names from the live database and exit.",
    )
    parser.add_argument("--market-symbol", default="ETF_SPY")
    parser.add_argument("--output-root", type=Path, default=Path("runs/latent_factor_analysis"))
    parser.add_argument("--skip-database-hash-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run configurable latent-coordinate dataset-feature analysis."""
    args = parse_args()
    if args.list_features:
        for feature in _feature_catalog(args.database.resolve()):
            print(f"{feature.selector}\t{feature.data_type}")
        return
    if args.embeddings is None:
        raise ValueError("--embeddings is required unless --list-features is used.")
    if args.coordinates is not None and args.legacy_coordinates is not None:
        raise ValueError("Use either --coordinates or --coordinate, not both.")

    coordinates = _normalize_cli_values(args.coordinates or args.legacy_coordinates)
    features = _normalize_cli_values(args.features)
    analyze_latent_coordinates(
        args.embeddings,
        args.database,
        coordinates=coordinates,
        feature_names=features,
        market_symbol=args.market_symbol,
        output_root=args.output_root,
        verify_database_hash=not args.skip_database_hash_check,
    )


if __name__ == "__main__":
    main()
