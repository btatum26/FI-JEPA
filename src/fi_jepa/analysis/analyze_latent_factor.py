from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ANALYSIS_FORMAT_VERSION = 1
VARIABLE_GROUPS = {
    "time_controls": ("elapsed_calendar_days", "elapsed_trading_rows"),
    "vix": (
        "vix_level",
        "vix_change_1d",
        "vix_change_5d",
        "vix_change_21d",
        "vix_change_63d",
    ),
    "spy_realized_volatility": (
        "realized_vol_5d",
        "realized_vol_21d",
        "realized_vol_63d",
        "realized_vol_126d",
    ),
    "spy_drawdown": (
        "drawdown_5d",
        "drawdown_21d",
        "drawdown_63d",
        "drawdown_126d",
    ),
    "market_breadth": ("breadth_1d", "pct_above_ma_63d"),
    "cross_sectional_dispersion": ("xs_dispersion_1d", "xs_iqr_1d"),
    "future_volatility": (
        "future_realized_vol_21d",
        "future_realized_vol_63d",
        "future_realized_vol_126d",
    ),
}
VARIABLE_COLUMNS = tuple(name for names in VARIABLE_GROUPS.values() for name in names)


# ============================================================================
# ARTIFACT AND MARKET DATA LOADING
# ============================================================================


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_embeddings(
    embedding_artifact: Path,
    coordinate: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load one all-valid embedding export and validate its analysis contract."""
    manifest_path = embedding_artifact / "manifest.json"
    embeddings_path = embedding_artifact / "embeddings.parquet"
    if not manifest_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(
            f"Embedding artifact must contain manifest.json and embeddings.parquet: "
            f"{embedding_artifact}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embeddings = pd.read_parquet(embeddings_path)
    if coordinate not in embeddings.columns:
        raise ValueError(f"Embedding artifact does not contain coordinate {coordinate!r}.")
    forbidden = [name for name in embeddings.columns if name.startswith("future_")]
    if forbidden:
        raise ValueError(f"Embedding artifact contains forbidden future targets: {forbidden}")
    if embeddings["date"].duplicated().any():
        raise ValueError("All-valid embedding export must contain one row per date.")

    embeddings = embeddings.copy()
    embeddings["date"] = pd.to_datetime(embeddings["date"])
    embeddings = embeddings.sort_values("date").reset_index(drop=True)
    return embeddings, manifest


def _load_market_variables(database_path: Path, market_symbol: str) -> pd.DataFrame:
    """Load raw, interpretable market variables and separate future-volatility targets."""
    query = """
        SELECT
            features.date,
            features.vix_level,
            features.vix_change_1d,
            features.vix_change_5d,
            features.vix_change_21d,
            features.vix_change_63d,
            spy.realized_vol_5d,
            spy.realized_vol_21d,
            spy.realized_vol_63d,
            spy.realized_vol_126d,
            spy.drawdown_5d,
            spy.drawdown_21d,
            spy.drawdown_63d,
            spy.drawdown_126d,
            features.breadth_1d,
            features.pct_above_ma_63d,
            features.xs_dispersion_1d,
            features.xs_iqr_1d,
            targets.future_realized_vol_21d,
            targets.future_realized_vol_63d,
            targets.future_realized_vol_126d
        FROM features
        LEFT JOIN ticker_features AS spy
            ON features.date = spy.date
            AND spy.symbol = ?
            AND spy.valid_observation
        LEFT JOIN targets
            ON features.date = targets.date
            AND targets.symbol = ?
        ORDER BY features.date
    """
    with duckdb.connect(str(database_path), read_only=True) as connection:
        market = connection.execute(query, [market_symbol, market_symbol]).fetchdf()
    market["date"] = pd.to_datetime(market["date"])
    return market


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
        detrended[name] = values - design @ coefficients
    return detrended


def _correlation_row(
    frame: pd.DataFrame,
    *,
    coordinate: str,
    variable: str,
    variable_group: str,
    segment: str,
    transform: str,
) -> dict[str, object]:
    """Calculate one pairwise Pearson and Spearman correlation record."""
    pair = _finite_pair(frame, coordinate, variable)
    if transform == "linear_time_detrended" and len(pair) >= 3:
        pair = _linear_time_detrend(pair)

    pearson = float("nan")
    spearman = float("nan")
    if len(pair) >= 3 and pair[coordinate].std() > 0.0 and pair[variable].std() > 0.0:
        pearson = float(pair[coordinate].corr(pair[variable], method="pearson"))
        # Spearman is Pearson correlation over ranks. Calculate it directly so
        # this analysis utility does not require the optional SciPy dependency.
        coordinate_ranks = pair[coordinate].rank(method="average")
        variable_ranks = pair[variable].rank(method="average")
        spearman = float(coordinate_ranks.corr(variable_ranks, method="pearson"))
    return {
        "segment": segment,
        "transform": transform,
        "coordinate": coordinate,
        "variable_group": variable_group,
        "variable": variable,
        "uses_future_data": variable.startswith("future_"),
        "observation_count": int(len(pair)),
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "pearson_r_squared": pearson * pearson,
    }


def calculate_correlations(dataset: pd.DataFrame, coordinate: str) -> pd.DataFrame:
    """Calculate level, first-difference, and detrended correlations by data segment.

    First differences are calculated before segment filtering so every date uses
    the immediately preceding exported date. Linear detrending is fitted within
    each reported segment and variable pair.
    """
    difference_columns = [coordinate, *VARIABLE_COLUMNS]
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
        for variable_group, variables in VARIABLE_GROUPS.items():
            for variable in variables:
                for transform, source in (
                    ("level", dataset),
                    ("first_difference", differences),
                    ("linear_time_detrended", dataset),
                ):
                    rows.append(
                        _correlation_row(
                            source.loc[mask],
                            coordinate=coordinate,
                            variable=variable,
                            variable_group=variable_group,
                            segment=segment,
                            transform=transform,
                        )
                    )
    return pd.DataFrame(rows)


# ============================================================================
# ANALYSIS ARTIFACT
# ============================================================================


def analyze_latent_coordinate(
    embedding_artifact: Path,
    database_path: Path,
    *,
    coordinate: str = "z_1",
    market_symbol: str = "ETF_SPY",
    output_root: Path = Path("runs/latent_factor_analysis"),
    verify_database_hash: bool = True,
) -> Path:
    """Correlate one exported latent coordinate with interpretable market variables.

    The function reads raw market variables from the canonical database and
    joins separate future-volatility targets only inside the analysis output.
    It never mutates the representation export, model-ready dataset, or
    pretraining artifacts.
    """
    embedding_artifact = embedding_artifact.resolve()
    database_path = database_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Canonical database does not exist: {database_path}")

    embeddings, embedding_manifest = _load_embeddings(embedding_artifact, coordinate)
    expected_database_sha256 = embedding_manifest.get("source_database_sha256")
    actual_database_sha256 = _file_sha256(database_path) if verify_database_hash else None
    if (
        verify_database_hash
        and expected_database_sha256
        and actual_database_sha256 != expected_database_sha256
    ):
        raise ValueError("Embedding artifact and canonical database have different versions.")

    market = _load_market_variables(database_path, market_symbol)
    dataset = embeddings.merge(market, on="date", how="left", validate="one_to_one")
    # Explicit time controls expose latent drift that can masquerade as an
    # interpretable market-state relationship in raw level correlations.
    dataset["elapsed_calendar_days"] = (
        dataset["date"] - dataset["date"].min()
    ).dt.days.astype(np.float64)
    dataset["elapsed_trading_rows"] = np.arange(len(dataset), dtype=np.float64)
    if dataset[list(VARIABLE_COLUMNS)].notna().sum().eq(0).any():
        missing = dataset[list(VARIABLE_COLUMNS)].notna().sum()
        missing = missing.loc[missing.eq(0)].index.tolist()
        raise ValueError(f"Market variables have no observations on embedding dates: {missing}")

    correlations = calculate_correlations(dataset, coordinate)
    strongest_candidates = correlations.loc[correlations["segment"].eq("all")].copy()
    strongest_candidates = strongest_candidates.loc[
        np.isfinite(strongest_candidates["pearson_correlation"])
    ]
    strongest = (
        strongest_candidates
        .assign(absolute_pearson=lambda frame: frame["pearson_correlation"].abs())
        .sort_values(["transform", "absolute_pearson"], ascending=[True, False])
        .groupby("transform", sort=False)
        .head(8)
        .drop(columns="absolute_pearson")
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    analysis_id = hashlib.sha256(
        (
            f"{embedding_manifest.get('pca_version')}|{expected_database_sha256}|"
            f"{coordinate}|{market_symbol}|{ANALYSIS_FORMAT_VERSION}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    destination = output_root / f"{timestamp}_{coordinate}_{analysis_id}"
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
        "coordinate": coordinate,
        "market_symbol": market_symbol,
        "row_count": int(len(dataset)),
        "first_date": str(dataset["date"].min().date()),
        "last_date": str(dataset["date"].max().date()),
        "variable_groups": {name: list(values) for name, values in VARIABLE_GROUPS.items()},
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


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse the latent-factor analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Correlate one FI-JEPA PCA coordinate with interpretable market variables."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/market_data.duckdb"),
    )
    parser.add_argument("--coordinate", default="z_1")
    parser.add_argument("--market-symbol", default="ETF_SPY")
    parser.add_argument("--output-root", type=Path, default=Path("runs/latent_factor_analysis"))
    parser.add_argument("--skip-database-hash-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run latent-coordinate market-meaning analysis."""
    args = parse_args()
    analyze_latent_coordinate(
        args.embeddings,
        args.database,
        coordinate=args.coordinate,
        market_symbol=args.market_symbol,
        output_root=args.output_root,
        verify_database_hash=not args.skip_database_hash_check,
    )


if __name__ == "__main__":
    main()
