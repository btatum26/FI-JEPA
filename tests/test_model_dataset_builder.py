from __future__ import annotations

from datetime import date
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd
import yaml

from dataset_pipeline.dataset_builder import build_model_dataset
from dataset_pipeline.dataset_builder.manifests import (
    build_date_manifest,
)
from fi_jepa.dataloader.masking import compute_patch_masks


# ============================================================================
# SYNTHETIC FROZEN DATASET
# ============================================================================


def _write_source_database(path: Path, dates: list[date]) -> None:
    features = pd.DataFrame(
        {
            "date": dates,
            "breadth_1d": [float(index) for index in range(len(dates))],
            "vix_level": [20.0 + index for index in range(len(dates))],
            "high_yield_oas_level": [5.0] * len(dates),
            "max_source_date_used": dates,
            "available_asof": dates,
            "uses_future_data": False,
        }
    )
    features.loc[9:14, "breadth_1d"] = 10_000.0
    features.loc[2, "breadth_1d"] = np.nan

    ticker_rows = []
    for index, day in enumerate(dates):
        ticker_rows.append(
            {
                "date": day,
                "symbol": "CALENDAR",
                "asset_type": "index",
                "valid_observation": True,
                "close": float(index + 1),
            }
        )
        ticker_rows.append(
            {
                "date": day,
                "symbol": "ASSET_A",
                "asset_type": "stock",
                "valid_observation": index != 1,
                "close": float(index + 1),
            }
        )
        ticker_rows.append(
            {
                "date": day,
                "symbol": "ASSET_B",
                "asset_type": "stock",
                "valid_observation": index == 0,
                "close": float(index + 1),
            }
        )
    ticker_features = pd.DataFrame(ticker_rows)
    symbol_manifest = pd.DataFrame(
        {
            "symbol": ["CALENDAR", "ASSET_A", "ASSET_B"],
            "asset_type": ["index", "stock", "stock"],
            "first_available_date": [dates[0], dates[0], dates[0]],
            "last_available_date": [dates[-1], dates[-1], dates[0]],
        }
    )
    targets = pd.DataFrame({"date": dates, "future_return_21d": np.arange(len(dates), dtype=float)})

    with duckdb.connect(str(path)) as connection:
        for table_name, frame in {
            "features": features,
            "ticker_features": ticker_features,
            "symbol_manifest": symbol_manifest,
            "targets": targets,
        }.items():
            connection.register("source_frame", frame)
            connection.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM source_frame')
            connection.unregister("source_frame")


def _write_config(path: Path, database_path: Path, output_root: Path, dates: list[date]) -> None:
    config = {
        "dataset_name": "synthetic_sparse",
        "source_database": str(database_path),
        "output_root": str(output_root),
        "dates": {
            "context_start": "2000-01-01",
            "sample_start": "2005-01-03",
            "sample_end": None,
            "sample_reference_symbol": "CALENDAR",
            "lookback_days": 3,
            "max_forward_horizon": 2,
        },
        "splits": {
            "validation_windows": [
                {"name": "anchor", "start": str(dates[11]), "end": str(dates[12])}
            ]
        },
        "assets": {
            "include_asset_types": ["stock"],
            "include_symbols": None,
            "exclude_symbols": [],
            "minimum_train_observations": 2,
        },
        "features": {
            "asset": [
                {
                    "feature_family": "price",
                    "series_source": "stooq",
                    "names": ["close"],
                }
            ],
            "market": [
                {
                    "feature_family": "breadth",
                    "series_source": "market_derived",
                    "names": ["breadth_1d"],
                }
            ],
            "macro": [
                {
                    "feature_family": "macro",
                    "series_source": "fred",
                    "names": ["vix_level"],
                }
            ],
        },
        "normalization": {
            "method": "train_fold_robust_zscore",
            "winsorize_quantiles": [0.0, 1.0],
            "transforms": {"log": ["close"]},
        },
        "jepa_target_rules": {
            "min_valid_dates_in_patch": 2,
            "min_valid_asset_fraction": 0.25,
            "allow_holdout_patches_as_targets": False,
            "allow_padded_patches_as_targets": False,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


# ============================================================================
# FROZEN DATASET CONTRACT
# ============================================================================


def test_date_manifest_separates_protection_meanings() -> None:
    dates = pd.Series(pd.bdate_range("2005-01-03", periods=12))
    manifest = build_date_manifest(
        dates,
        context_start="2000-01-01",
        sample_start="2005-01-03",
        sample_end=None,
        lookback_days=3,
        max_forward_horizon=2,
        validation_windows=[{"name": "anchor", "start": "2005-01-11", "end": "2005-01-12"}],
    )

    protection_columns = [
        "validation_sample",
        "protected_input_lookback",
        "protected_forward_target",
    ]
    assert manifest[protection_columns].sum(axis=1).le(1).all()
    assert (manifest["protected_holdout"] == manifest[protection_columns].any(axis=1)).all()
    assert (manifest["train_fact_allowed"] == ~manifest["protected_holdout"]).all()
    assert (manifest["validation_fact_allowed"] == manifest["protected_holdout"]).all()


def test_build_model_dataset_exports_sparse_disjoint_normalized_facts(tmp_path) -> None:
    dates = [timestamp.date() for timestamp in pd.bdate_range("2004-12-20", periods=15)]
    database_path = tmp_path / "source.duckdb"
    config_path = tmp_path / "frozen.yaml"
    _write_source_database(database_path, dates)
    _write_config(config_path, database_path, tmp_path / "model_ready", dates)

    output = build_model_dataset(config_path)

    exported_dates = pd.read_parquet(output / "dates.parquet")
    assets = pd.read_parquet(output / "assets.parquet")
    features = pd.read_parquet(output / "feature_manifest.parquet")
    normalization = pd.read_parquet(output / "normalization.parquet")
    train_assets = pd.read_parquet(output / "train_asset_features.parquet")
    validation_assets = pd.read_parquet(output / "validation_asset_features.parquet")
    train_market = pd.read_parquet(output / "train_market_features.parquet")
    validation_market = pd.read_parquet(output / "validation_market_features.parquet")
    train_macro = pd.read_parquet(output / "train_macro_features.parquet")

    assert output == build_model_dataset(config_path)
    assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{16}", output.name)
    assert set(train_assets["date"]).isdisjoint(validation_assets["date"])
    assert set(train_market["date"]).isdisjoint(validation_market["date"])
    assert not set(exported_dates.loc[exported_dates["protected_holdout"], "date"]).intersection(
        train_market["date"]
    )
    assert set(exported_dates.loc[exported_dates["validation_fact_allowed"], "date"]) == set(
        validation_market["date"]
    )

    assert assets.set_index("symbol").loc["ASSET_A", "trainable"]
    assert not assets.set_index("symbol").loc["ASSET_B", "trainable"]
    trainable_assets = int(assets["trainable"].sum())
    allowed_dates = int(exported_dates["train_fact_allowed"].sum())
    assert len(train_assets) < trainable_assets * allowed_dates
    assert train_assets["valid_asset"].all()
    assert train_assets["close__valid"].all()

    assert train_macro["date"].min().year == 2004
    assert train_macro["date"].min() < date(2005, 1, 1)
    assert set(features["input_group"]) == {"asset", "market", "macro"}
    assert set(features["feature_family"]) == {"price", "breadth", "macro"}
    assert set(features["series_source"]) == {"stooq", "market_derived", "fred"}
    assert features.loc[features["feature_name"].eq("close"), "transform"].item() == "log"
    assert not features["feature_name"].str.contains("oas|future|target", case=False).any()
    assert (
        features.groupby("input_group")["feature_index"]
        .apply(lambda values: values.tolist() == list(range(len(values))))
        .all()
    )

    breadth_stats = normalization.loc[normalization["feature_name"].eq("breadth_1d")].iloc[0]
    train_fact_dates = set(exported_dates.loc[exported_dates["train_fact_allowed"], "date"])
    expected_center = np.median(
        [float(index) for index, day in enumerate(dates) if day in train_fact_dates and index != 2]
    )
    assert breadth_stats["center"] == expected_center
    assert np.isfinite(train_market["breadth_1d"]).all()
    assert np.isfinite(validation_market["breadth_1d"]).all()
    assert "breadth_1d__valid" in train_market
    missing_row = train_market.loc[train_market["date"].eq(dates[2])].iloc[0]
    assert not missing_row["breadth_1d__valid"]
    assert missing_row["breadth_1d"] == 0.0


def test_build_model_dataset_migrates_legacy_hash_only_directory(tmp_path) -> None:
    dates = [timestamp.date() for timestamp in pd.bdate_range("2004-12-20", periods=15)]
    database_path = tmp_path / "source.duckdb"
    config_path = tmp_path / "model.yaml"
    _write_source_database(database_path, dates)
    _write_config(config_path, database_path, tmp_path / "model_ready", dates)
    output = build_model_dataset(config_path)
    build_id = output.name.rsplit("_", maxsplit=1)[1]
    legacy = output.with_name(build_id)
    output.rename(legacy)

    migrated = build_model_dataset(config_path)

    assert migrated != legacy
    assert migrated.is_dir()
    assert not legacy.exists()
    assert migrated.name.endswith(f"_{build_id}")


# ============================================================================
# JEPA PATCH TARGET RULES
# ============================================================================


def test_jepa_target_eligibility_rejects_protected_padded_and_sparse_patches() -> None:
    valid_dates = np.array([True, True, True, False])
    valid_assets = np.ones((4, 4), dtype=bool)
    target_dates = np.ones(4, dtype=bool)
    rules = {
        "patch_len": 4,
        "min_valid_days_per_asset_patch": 1,
        "min_valid_dates_in_patch": 3,
        "min_valid_asset_fraction": 0.25,
    }

    assert compute_patch_masks(valid_assets, valid_dates, target_dates, **rules)[
        "patch_target_eligible"
    ][0]
    assert not compute_patch_masks(
        valid_assets,
        valid_dates,
        np.array([True, False, True, True]),
        **rules,
    )["patch_target_eligible"][0]
    assert not compute_patch_masks(
        valid_assets, np.array([True, False, False, True]), target_dates, **rules
    )["patch_target_eligible"][0]
    assert not compute_patch_masks(
        np.zeros((4, 4), dtype=bool),
        valid_dates,
        target_dates,
        **rules,
    )["patch_target_eligible"][0]
