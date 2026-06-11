from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd


# ============================================================================
# DUCKDB DATASET I/O
# ============================================================================


def write_market_database(
    database_path: Path,
    tables: dict[str, pd.DataFrame],
    derived_tables: dict[str, str] | None = None,
    views: dict[str, str] | None = None,
    drop_tables: tuple[str, ...] = (),
    validation_queries: dict[str, str] | None = None,
) -> None:
    """Atomically replace the canonical market database.

    DataFrames are loaded as source tables. Derived tables can then materialize
    the canonical query contract inside DuckDB before temporary component
    tables are dropped.
    """
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_suffix(f"{database_path.suffix}.tmp")
    temporary_path.unlink(missing_ok=True)

    connection = duckdb.connect(str(temporary_path))
    try:
        for table_name, frame in tables.items():
            connection.register("source_frame", frame)
            connection.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM source_frame')
            connection.unregister("source_frame")
        for table_name, query in (derived_tables or {}).items():
            connection.execute(f'CREATE TABLE "{table_name}" AS {query}')
        for view_name, query in (views or {}).items():
            connection.execute(f'CREATE VIEW "{view_name}" AS {query}')
        for table_name in drop_tables:
            connection.execute(f'DROP TABLE "{table_name}"')
        for check_name, query in (validation_queries or {}).items():
            failures = int(connection.execute(query).fetchone()[0])
            if failures:
                raise AssertionError(f"Database validation failed for {check_name}: {failures}")
        connection.execute("CHECKPOINT")
    finally:
        connection.close()

    os.replace(temporary_path, database_path)


def read_market_table(database_path: Path, table_name: str) -> pd.DataFrame:
    """Read one table or view from the canonical market database."""
    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(f'SELECT * FROM "{table_name}"').fetchdf()
