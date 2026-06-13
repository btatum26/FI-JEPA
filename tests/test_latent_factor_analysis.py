from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from fi_jepa.analysis.analyze_latent_factor import analyze_latent_coordinate


# ============================================================================
# TEST ARTIFACT BUILDERS
# ============================================================================


def _file_sha256(path: Path) -> str:
    """Return a test fixture file hash."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_market_database(path: Path, dates: pd.DatetimeIndex) -> None:
    """Write the minimal canonical tables required by latent-factor analysis."""
    values = np.arange(len(dates), dtype=np.float64)
    features = pd.DataFrame(
        {
            "date": dates,
            "vix_level": 10.0 + values,
            "vix_change_1d": values,
            "vix_change_5d": values,
            "vix_change_21d": values,
            "vix_change_63d": values,
            "breadth_1d": 1.0 - values / 20.0,
            "pct_above_ma_63d": 1.0 - values / 20.0,
            "xs_dispersion_1d": values / 100.0,
            "xs_iqr_1d": values / 100.0,
        }
    )
    ticker_features = pd.DataFrame(
        {
            "date": dates,
            "symbol": "ETF_SPY",
            "valid_observation": True,
            "realized_vol_5d": values / 100.0,
            "realized_vol_21d": values / 100.0,
            "realized_vol_63d": values / 100.0,
            "realized_vol_126d": values / 100.0,
            "drawdown_5d": -values / 100.0,
            "drawdown_21d": -values / 100.0,
            "drawdown_63d": -values / 100.0,
            "drawdown_126d": -values / 100.0,
        }
    )
    targets = pd.DataFrame(
        {
            "date": dates,
            "symbol": "ETF_SPY",
            "future_realized_vol_21d": values / 100.0,
            "future_realized_vol_63d": values / 100.0,
            "future_realized_vol_126d": values / 100.0,
        }
    )
    with duckdb.connect(str(path)) as connection:
        for table_name, frame in (
            ("features", features),
            ("ticker_features", ticker_features),
            ("targets", targets),
        ):
            connection.register("source_frame", frame)
            connection.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM source_frame')
            connection.unregister("source_frame")


def _write_embedding_artifact(root: Path, dates: pd.DatetimeIndex, database_sha256: str) -> Path:
    """Write a valid all-date embedding artifact with a known z_1 relationship."""
    root.mkdir()
    values = np.arange(len(dates), dtype=np.float64)
    embeddings = pd.DataFrame(
        {
            "date": dates,
            "split": ["train"] * 6 + ["validation"] * 6,
            "validation_window_name": [""] * 6 + ["test_window"] * 6,
            "z_1": values,
            "z_2": np.sin(values),
        }
    )
    embeddings.to_parquet(root / "embeddings.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "source_database_sha256": database_sha256,
                "pca_version": "pca-test",
            }
        ),
        encoding="utf-8",
    )
    return root


# ============================================================================
# LATENT-FACTOR ANALYSIS
# ============================================================================


def test_latent_factor_analysis_writes_segmented_market_correlations(tmp_path: Path) -> None:
    dates = pd.bdate_range(date(2024, 1, 1), periods=12)
    database = tmp_path / "market.duckdb"
    _write_market_database(database, dates)
    embeddings = _write_embedding_artifact(
        tmp_path / "embeddings",
        dates,
        _file_sha256(database),
    )

    output = analyze_latent_coordinate(
        embeddings,
        database,
        output_root=tmp_path / "analysis",
    )
    correlations = pd.read_csv(output / "correlations.csv")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    dataset = pd.read_parquet(output / "analysis_dataset.parquet")

    vix = correlations.loc[
        correlations["segment"].eq("all")
        & correlations["transform"].eq("level")
        & correlations["variable"].eq("vix_level")
    ].iloc[0]
    drawdown = correlations.loc[
        correlations["segment"].eq("all")
        & correlations["transform"].eq("level")
        & correlations["variable"].eq("drawdown_21d")
    ].iloc[0]
    detrended_time = correlations.loc[
        correlations["segment"].eq("all")
        & correlations["transform"].eq("linear_time_detrended")
        & correlations["variable"].eq("elapsed_trading_rows")
    ].iloc[0]

    assert vix["pearson_correlation"] == pytest.approx(1.0)
    assert drawdown["pearson_correlation"] == pytest.approx(-1.0)
    assert np.isnan(detrended_time["pearson_correlation"])
    assert "validation_window:test_window" in correlations["segment"].unique()
    assert report["future_targets_joined_only_for_analysis"] is True
    assert report["pretraining_artifact_mutated"] is False
    assert "future_realized_vol_21d" in dataset.columns


def test_latent_factor_analysis_rejects_database_version_mismatch(tmp_path: Path) -> None:
    dates = pd.bdate_range(date(2024, 1, 1), periods=12)
    database = tmp_path / "market.duckdb"
    _write_market_database(database, dates)
    embeddings = _write_embedding_artifact(tmp_path / "embeddings", dates, "wrong")

    with pytest.raises(ValueError, match="different versions"):
        analyze_latent_coordinate(
            embeddings,
            database,
            output_root=tmp_path / "analysis",
        )
