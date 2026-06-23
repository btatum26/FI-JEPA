from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import duckdb

from fi_jepa.probes.artifacts import (
    clean_temporary_artifact,
    file_sha256,
    publish_artifact,
    readable_artifact_destination,
)

DEFAULT_MARKET_SYMBOL = "ETF_SPY"
BASELINE_FEATURE_PREFIX = "baseline__"


# ============================================================================
# BASELINE FEATURE EXPORT
# ============================================================================


def _table_exists(connection: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return whether the canonical database exposes one optional probe-support table."""
    count = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchone()[0]
    return bool(count)


def _quoted_identifier(name: str) -> str:
    """Quote one DuckDB identifier from the fixed canonical schema."""
    return '"' + name.replace('"', '""') + '"'


def _numeric_columns(connection: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    """Return numeric columns from one canonical table, excluding leakage metadata."""
    if not _table_exists(connection, table_name):
        return []
    columns = connection.execute(f"DESCRIBE {_quoted_identifier(table_name)}").fetchdf()
    numeric_types = {
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
    }
    excluded = {"date", "symbol", "uses_future_data"}
    result: list[str] = []
    for row in columns.itertuples(index=False):
        name = str(row.column_name)
        data_type = str(row.column_type).upper()
        if name in excluded or name.endswith("_source_date_used"):
            continue
        if data_type in numeric_types or data_type.startswith("DECIMAL"):
            result.append(name)
    return result


def _baseline_feature_frame(
    connection: duckdb.DuckDBPyConnection,
    *,
    market_symbol: str,
) -> tuple[object, list[str]]:
    """Load past-only hand-built market-state features for Phase 3 probe baselines.

    The target artifact remains future-label-only in spirit: these columns are
    copied from canonical past-only encoder tables and are prefixed so they
    cannot be confused with `future_*` targets or model `z_*` coordinates.
    """
    target_dates = connection.execute("SELECT DISTINCT date FROM targets ORDER BY date").fetchdf()
    if target_dates.empty:
        raise RuntimeError("Canonical targets table is empty.")

    if not _table_exists(connection, "features") and not _table_exists(connection, "ticker_features"):
        target_dates["date"] = target_dates["date"]
        return target_dates, []

    feature_columns = _numeric_columns(connection, "features")
    ticker_columns = [
        name
        for name in _numeric_columns(connection, "ticker_features")
        if name
        in {
            "return_21d",
            "return_63d",
            "return_126d",
            "realized_vol_21d",
            "realized_vol_63d",
            "realized_vol_126d",
            "drawdown_21d",
            "drawdown_63d",
            "drawdown_126d",
            "ma_distance_21d",
            "ma_distance_63d",
            "ma_distance_126d",
            "path_efficiency_21d",
            "path_efficiency_63d",
            "path_efficiency_126d",
        }
    ]

    selected: list[str] = []
    expressions: list[str] = []
    for name in feature_columns:
        output_name = f"{BASELINE_FEATURE_PREFIX}{name}"
        selected.append(output_name)
        expressions.append(f"features.{_quoted_identifier(name)} AS {_quoted_identifier(output_name)}")
    for name in ticker_columns:
        if name.startswith("return_"):
            suffix = name.removeprefix("return_")
            output_name = f"{BASELINE_FEATURE_PREFIX}trailing_return_{suffix}"
        elif name.startswith("realized_vol_"):
            suffix = name.removeprefix("realized_vol_")
            output_name = f"{BASELINE_FEATURE_PREFIX}trailing_realized_vol_{suffix}"
        elif name.startswith("drawdown_"):
            suffix = name.removeprefix("drawdown_")
            output_name = f"{BASELINE_FEATURE_PREFIX}trailing_max_drawdown_{suffix}"
        else:
            output_name = f"{BASELINE_FEATURE_PREFIX}{name}"
        selected.append(output_name)
        expressions.append(
            f"ticker_features.{_quoted_identifier(name)} AS {_quoted_identifier(output_name)}"
        )

    trend_expressions: list[str] = []
    for horizon in (21, 63, 126):
        return_name = f"{BASELINE_FEATURE_PREFIX}trailing_return_{horizon}d"
        vol_name = f"{BASELINE_FEATURE_PREFIX}trailing_realized_vol_{horizon}d"
        trend_name = f"{BASELINE_FEATURE_PREFIX}current_trend_score_{horizon}d"
        if return_name in selected and vol_name in selected:
            selected.append(trend_name)
            trend_expressions.append(
                f"{_quoted_identifier(return_name)} / ({_quoted_identifier(vol_name)} + 1e-12) "
                f"AS {_quoted_identifier(trend_name)}"
            )

    select_clause = ",\n            ".join(expressions)
    if select_clause:
        select_clause = ",\n            " + select_clause
    query = f"""
        SELECT
            dates.date
            {select_clause}
        FROM (SELECT DISTINCT date FROM targets) AS dates
        LEFT JOIN features
            ON dates.date = features.date
        LEFT JOIN ticker_features
            ON dates.date = ticker_features.date
            AND ticker_features.symbol = ?
            AND ticker_features.valid_observation
        ORDER BY dates.date
    """
    frame = connection.execute(query, [market_symbol]).fetchdf()
    for expression in trend_expressions:
        frame = connection.execute(
            f"SELECT *, {expression} FROM frame"
        ).fetchdf()
    return frame, selected


# ============================================================================
# IMMUTABLE PROBE-TARGET EXPORT
# ============================================================================


def export_probe_targets(
    database_path: Path,
    *,
    output_root: Path = Path("data/probe_targets"),
    name: str | None = None,
    market_symbol: str = DEFAULT_MARKET_SYMBOL,
) -> Path:
    """Export canonical future targets into a separate immutable probe artifact."""
    database_path = database_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Canonical database does not exist: {database_path}")
    source_sha256 = file_sha256(database_path)
    with duckdb.connect(str(database_path), read_only=True) as connection:
        targets = connection.execute("SELECT * FROM targets ORDER BY date").fetchdf()
        baseline_features, baseline_feature_columns = _baseline_feature_frame(
            connection, market_symbol=market_symbol
        )
    if targets.empty:
        raise RuntimeError("Canonical targets table is empty.")
    target_columns = sorted(name for name in targets.columns if name.startswith("future_"))
    if not target_columns:
        raise ValueError("Canonical targets table contains no future_ target columns.")

    destination, temporary = readable_artifact_destination(
        output_root, name or f"{database_path.stem}_targets"
    )
    try:
        targets.to_parquet(temporary / "targets.parquet", index=False, compression="zstd")
        baseline_features.to_parquet(
            temporary / "baseline_features.parquet", index=False, compression="zstd"
        )
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_database": str(database_path),
            "source_database_sha256": source_sha256,
            "target_columns": target_columns,
            "baseline_feature_columns": baseline_feature_columns,
            "baseline_market_symbol": market_symbol,
            "row_count": int(len(targets)),
            "encoder_features_included": False,
            "past_only_baseline_features_included": bool(baseline_feature_columns),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        publish_artifact(temporary, destination)
    except Exception:
        clean_temporary_artifact(temporary)
        raise
    print(f"Built probe-target export: {destination}")
    return destination
