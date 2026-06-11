from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from fi_jepa.dataloader import (
    FIJepaDataConfig,
    FIJepaEmbeddingDataset,
    FIJepaWindowDataset,
    FrozenPanelStore,
    build_fi_jepa_embedding_dataloader,
    build_fi_jepa_dataloader,
    fixed_k_asset_ids,
)


# ============================================================================
# SYNTHETIC MODEL ARTIFACT
# ============================================================================


def _write_model_artifact(root: Path) -> FIJepaDataConfig:
    root.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=22)
    validation_sample = np.zeros(22, dtype=bool)
    validation_sample[16:19] = True
    protected_input = np.zeros(22, dtype=bool)
    protected_input[8:16] = True
    protected_forward = np.zeros(22, dtype=bool)
    protected_forward[19:21] = True
    protected = validation_sample | protected_input | protected_forward
    sample_eligible = np.zeros(22, dtype=bool)
    sample_eligible[[7, 16, 17, 18, 21]] = True
    date_manifest = pd.DataFrame(
        {
            "date_idx": np.arange(22, dtype=np.int32),
            "date": dates.date,
            "sample_eligible": sample_eligible,
            "validation_sample": validation_sample,
            "protected_input_lookback": protected_input,
            "protected_forward_target": protected_forward,
            "protected_holdout": protected,
            "train_fact_allowed": ~protected,
            "validation_fact_allowed": protected,
            "validation_window_name": pd.Series([pd.NA] * 22, dtype="string"),
        }
    )
    date_manifest.to_parquet(root / "dates.parquet", index=False)

    assets = pd.DataFrame(
        {
            "asset_id": np.arange(4, dtype=np.int32),
            "symbol": [f"ASSET_{index}" for index in range(4)],
            "asset_type": ["stock"] * 4,
            "first_available_date": [dates[0]] * 4,
            "last_available_date": [dates[-1]] * 4,
            "valid_train_observations": [9] * 4,
            "trainable": [True] * 4,
            "exclusion_reason": pd.Series([pd.NA] * 4, dtype="string"),
        }
    )
    assets.to_parquet(root / "assets.parquet", index=False)

    features = pd.DataFrame(
        [
            {
                "feature_name": "asset_a",
                "feature_index": 0,
                "input_group": "asset",
                "feature_family": "returns",
                "series_source": "synthetic",
                "dtype": "float32",
                "normalized": True,
                "normalization_method": "synthetic",
                "transform": "none",
            },
            {
                "feature_name": "asset_b",
                "feature_index": 1,
                "input_group": "asset",
                "feature_family": "returns",
                "series_source": "synthetic",
                "dtype": "float32",
                "normalized": True,
                "normalization_method": "synthetic",
                "transform": "none",
            },
            {
                "feature_name": "market_a",
                "feature_index": 0,
                "input_group": "market",
                "feature_family": "market",
                "series_source": "synthetic",
                "dtype": "float32",
                "normalized": True,
                "normalization_method": "synthetic",
                "transform": "none",
            },
            {
                "feature_name": "macro_a",
                "feature_index": 0,
                "input_group": "macro",
                "feature_family": "macro",
                "series_source": "synthetic",
                "dtype": "float32",
                "normalized": True,
                "normalization_method": "synthetic",
                "transform": "none",
            },
        ]
    )
    features.to_parquet(root / "feature_manifest.parquet", index=False)
    pd.DataFrame({"feature_name": features["feature_name"]}).to_parquet(
        root / "normalization.parquet", index=False
    )

    for split, allowed in (
        ("train", date_manifest["train_fact_allowed"].to_numpy()),
        ("validation", date_manifest["validation_fact_allowed"].to_numpy()),
    ):
        asset_rows = []
        for date_idx in np.flatnonzero(allowed):
            for asset_id in range(4):
                if date_idx == 16 and asset_id == 3:
                    continue
                second_valid = not (date_idx == 7 and asset_id == 1)
                asset_rows.append(
                    {
                        "date": dates[date_idx].date(),
                        "date_idx": date_idx,
                        "asset_id": asset_id,
                        "valid_asset": True,
                        "asset_a": float(date_idx * 10 + asset_id),
                        "asset_a__valid": True,
                        "asset_b": float(date_idx + asset_id) if second_valid else 0.0,
                        "asset_b__valid": second_valid,
                    }
                )
        pd.DataFrame(asset_rows).to_parquet(root / f"{split}_asset_features.parquet", index=False)

        date_ids = np.flatnonzero(allowed)
        pd.DataFrame(
            {
                "date": dates[date_ids].date,
                "date_idx": date_ids,
                "valid_date": True,
                "market_a": date_ids.astype(np.float32),
                "market_a__valid": True,
            }
        ).to_parquet(root / f"{split}_market_features.parquet", index=False)
        pd.DataFrame(
            {
                "date": dates[date_ids].date,
                "date_idx": date_ids,
                "valid_date": True,
                "macro_a": date_ids.astype(np.float32),
                "macro_a__valid": True,
            }
        ).to_parquet(root / f"{split}_macro_features.parquet", index=False)

    (root / "manifest.json").write_text(json.dumps({"sparse_asset_facts": True}))
    (root / "quality_report.json").write_text(json.dumps({"stores_windows": False}))
    (root / "config_resolved.yaml").write_text(yaml.safe_dump({"synthetic": True}))
    return FIJepaDataConfig(
        artifact_path=root,
        lookback_days=8,
        patch_len=2,
        train_k_assets=2,
        diagnostic_k_assets=2,
        mask_ratio=0.5,
        min_masked_patches=3,
        max_masked_patches=3,
        min_valid_days_per_asset_patch=1,
        min_valid_dates_in_patch=2,
        min_valid_asset_fraction=0.25,
        batch_size=1,
        validation_batch_size=2,
        seed=17,
    )


# ============================================================================
# STORE AND DATASET CONTRACT
# ============================================================================


def test_store_reconstructs_sparse_masks_and_blocks_protected_train_facts(
    tmp_path: Path,
) -> None:
    config = _write_model_artifact(tmp_path / "artifact")
    store = FrozenPanelStore(config.artifact_path)

    assert store.feature_names == {
        "asset": ["asset_a", "asset_b"],
        "market": ["market_a"],
        "macro": ["macro_a"],
    }
    partial = store.window(7, np.array([1]), "train", config.lookback_days)
    assert partial["valid_asset_mask"].all()
    assert partial["asset_feature_mask"][-1, 0].tolist() == [True, False]
    assert partial["asset_x"][-1, 0, 1] == 0.0

    protected = store.window(21, np.array([0]), "train", config.lookback_days)
    assert protected["holdout_date_mask"][:-1].all()
    assert not protected["valid_date_mask"][:-1].any()
    assert not protected["asset_feature_mask"][:-1].any()
    assert np.count_nonzero(protected["asset_x"][:-1]) == 0


def test_dataset_filters_invalid_samples_and_builds_split_relative_masks(
    tmp_path: Path,
) -> None:
    config = _write_model_artifact(tmp_path / "artifact")
    store = FrozenPanelStore(config.artifact_path)
    train = FIJepaWindowDataset(store, config, "train")
    validation = FIJepaWindowDataset(store, config, "validation")

    assert train.nominal_sample_count == 2
    assert train.dropped_sample_count == 1
    assert len(train) == 1
    assert validation.nominal_sample_count == 3
    assert validation.dropped_sample_count == 0
    assert len(validation) == 3

    train_sample = train[0]
    assert train_sample["asset_ids"].shape == (2,)
    assert train_sample["asset_slot_mask"].all()
    assert train_sample["patch_target_eligible"].all()
    assert int(train_sample["jepa_target_mask"].sum()) == 3

    validation_sample = validation[0]
    assert validation_sample["holdout_date_mask"].all()
    assert validation_sample["patch_target_eligible"].all()
    assert validation_sample["asset_ids"].tolist() == [0, 1, 2]


def test_training_views_change_by_epoch_and_validation_views_are_frozen(
    tmp_path: Path,
) -> None:
    config = _write_model_artifact(tmp_path / "artifact")
    store = FrozenPanelStore(config.artifact_path)
    train = FIJepaWindowDataset(store, config, "train")
    validation = FIJepaWindowDataset(store, config, "validation")
    diagnostic = FIJepaWindowDataset(store, config, "diagnostic", view_index=2)

    training_views = []
    for epoch in range(5):
        train.set_epoch(epoch)
        sample = train[0]
        training_views.append(
            (tuple(sample["asset_ids"].tolist()), tuple(sample["jepa_target_mask"].tolist()))
        )
    assert len(set(training_views)) > 1

    first_validation = validation[0]
    validation.set_epoch(9)
    second_validation = validation[0]
    assert torch.equal(first_validation["asset_ids"], second_validation["asset_ids"])
    assert torch.equal(first_validation["jepa_target_mask"], second_validation["jepa_target_mask"])
    assert torch.equal(diagnostic[0]["asset_ids"], diagnostic[0]["asset_ids"])


def test_dataloader_pads_variable_assets_and_exposes_patched_views(tmp_path: Path) -> None:
    config = _write_model_artifact(tmp_path / "artifact")
    loader = build_fi_jepa_dataloader(config, "validation", shuffle=False)
    batch = next(iter(loader))

    assert batch["asset_x"].shape == (2, 8, 4, 2)
    assert batch["asset_patches"].shape == (2, 4, 2, 4, 2)
    assert batch["market_patches"].shape == (2, 4, 2, 1)
    assert batch["macro_patches"].shape == (2, 4, 2, 1)
    assert batch["asset_ids"][0].tolist() == [0, 1, 2, -1]
    assert batch["asset_slot_mask"][0].tolist() == [True, True, True, False]
    assert batch["target_patch_ids"].shape == (2, 3)
    assert batch["target_patch_id_mask"].all()
    assert (
        batch["asset_patches"].untyped_storage().data_ptr()
        == batch["asset_x"].untyped_storage().data_ptr()
    )


def test_embedding_views_are_unmasked_all_valid_and_deterministic_fixed_k(
    tmp_path: Path,
) -> None:
    config = _write_model_artifact(tmp_path / "artifact")
    store = FrozenPanelStore(config.artifact_path)
    all_valid = FIJepaEmbeddingDataset(
        store, config, "validation", asset_view="all_valid"
    )
    fixed_first = FIJepaEmbeddingDataset(
        store, config, "validation", asset_view="fixed_k", view_index=0
    )
    fixed_second = FIJepaEmbeddingDataset(
        store, config, "validation", asset_view="fixed_k", view_index=1
    )

    assert all_valid[1]["asset_ids"].tolist() == [0, 1, 2, 3]
    assert len(fixed_first[1]["asset_ids"]) == config.diagnostic_k_assets
    assert torch.equal(fixed_first[1]["asset_ids"], fixed_first[1]["asset_ids"])
    assert not torch.equal(fixed_first[1]["asset_ids"], fixed_second[1]["asset_ids"])

    loader = build_fi_jepa_embedding_dataloader(
        config, "validation", asset_view="all_valid", store=store
    )
    batch = next(iter(loader))
    assert "jepa_target_mask" not in batch
    assert "jepa_context_mask" not in batch
    assert "target_patch_ids" not in batch
    assert batch["patch_context_mask"][:, -1].all()


def test_fixed_k_asset_hash_selection_ignores_candidate_order() -> None:
    candidates = np.arange(20, dtype=np.int64)
    expected = fixed_k_asset_ids(
        candidates,
        dataset_version="build-a",
        sample_date="2024-01-05",
        view_index=2,
        k=6,
    )
    reversed_order = fixed_k_asset_ids(
        candidates[::-1],
        dataset_version="build-a",
        sample_date="2024-01-05",
        view_index=2,
        k=6,
    )
    other_view = fixed_k_asset_ids(
        candidates,
        dataset_version="build-a",
        sample_date="2024-01-05",
        view_index=3,
        k=6,
    )

    assert expected.tolist() == sorted(expected.tolist())
    assert np.array_equal(expected, reversed_order)
    assert not np.array_equal(expected, other_view)


# ============================================================================
# ARTIFACT REJECTION
# ============================================================================


def test_store_rejects_noncontiguous_features_and_target_columns(tmp_path: Path) -> None:
    config = _write_model_artifact(tmp_path / "bad_indices")
    features = pd.read_parquet(config.artifact_path / "feature_manifest.parquet")
    features.loc[features["feature_name"].eq("asset_b"), "feature_index"] = 3
    features.to_parquet(config.artifact_path / "feature_manifest.parquet", index=False)
    with pytest.raises(ValueError, match="contiguous"):
        FrozenPanelStore(config.artifact_path)

    config = _write_model_artifact(tmp_path / "bad_target")
    path = config.artifact_path / "train_market_features.parquet"
    market = pd.read_parquet(path)
    market["future_return_1d"] = 0.0
    market.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="forbidden target-like columns"):
        FrozenPanelStore(config.artifact_path)


def test_dataset_rejects_lookback_that_disagrees_with_artifact(tmp_path: Path) -> None:
    config = _write_model_artifact(tmp_path / "artifact")
    resolved = {"dates": {"lookback_days": 10}}
    (config.artifact_path / "config_resolved.yaml").write_text(yaml.safe_dump(resolved))
    store = FrozenPanelStore(config.artifact_path)

    with pytest.raises(ValueError, match="does not match artifact"):
        FIJepaWindowDataset(store, config, "train")


@pytest.mark.parametrize(
    ("filename", "expected_message"),
    [
        ("train_asset_features.parquet", r"duplicate \(date_idx, asset_id\) rows"),
        ("train_market_features.parquet", "duplicate date_idx rows"),
    ],
)
def test_store_rejects_duplicates_within_one_parquet_batch(
    tmp_path: Path,
    filename: str,
    expected_message: str,
) -> None:
    config = _write_model_artifact(tmp_path / filename.removesuffix(".parquet"))
    path = config.artifact_path / filename
    facts = pd.read_parquet(path)
    facts = pd.concat([facts, facts.iloc[[0]]], ignore_index=True)
    facts.to_parquet(path, index=False)

    with pytest.raises(ValueError, match=expected_message):
        FrozenPanelStore(config.artifact_path)
