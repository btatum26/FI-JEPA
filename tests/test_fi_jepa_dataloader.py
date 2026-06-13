from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from fi_jepa.dataloader import (
    DensePanelStore,
    FIJepaDataConfig,
    build_fi_jepa_dataloader,
    build_fi_jepa_embedding_dataloader,
)
from fi_jepa.dataloader.panel_store import CACHE_FORMAT_VERSION


# ============================================================================
# SYNTHETIC SPARSE ARTIFACT
# ============================================================================


def _write_sparse_artifact(root: Path) -> FIJepaDataConfig:
    """Write a complete sparse artifact with disjoint train/validation panels."""
    root.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=16)
    train_allowed = np.arange(16) < 8
    validation_allowed = ~train_allowed
    sample_eligible = np.zeros(16, dtype=bool)
    sample_eligible[[3, 5, 7]] = True
    validation_sample = np.zeros(16, dtype=bool)
    validation_sample[[11, 13]] = True
    date_manifest = pd.DataFrame(
        {
            "date_idx": np.arange(16, dtype=np.int32),
            "date": dates.date,
            "sample_eligible": sample_eligible,
            "validation_sample": validation_sample,
            "protected_holdout": validation_allowed,
            "train_fact_allowed": train_allowed,
            "validation_fact_allowed": validation_allowed,
            "validation_window_name": [""] * 11 + ["window_a"] * 3 + ["window_b"] * 2,
        }
    )
    date_manifest.to_parquet(root / "dates.parquet", index=False)
    pd.DataFrame(
        {
            "asset_id": np.arange(4, dtype=np.int32),
            "symbol": [f"ASSET_{index}" for index in range(4)],
            "trainable": [True] * 4,
        }
    ).to_parquet(root / "assets.parquet", index=False)
    features = pd.DataFrame(
        [
            {"feature_name": "asset_a", "feature_index": 0, "input_group": "asset", "dtype": "float32"},
            {"feature_name": "asset_b", "feature_index": 1, "input_group": "asset", "dtype": "float32"},
            {"feature_name": "market_a", "feature_index": 0, "input_group": "market", "dtype": "float32"},
            {"feature_name": "macro_a", "feature_index": 0, "input_group": "macro", "dtype": "float32"},
        ]
    )
    features.to_parquet(root / "feature_manifest.parquet", index=False)

    for split, allowed in (("train", train_allowed), ("validation", validation_allowed)):
        asset_rows = []
        for date_idx in np.flatnonzero(allowed):
            for asset_id in range(4):
                if split == "validation" and date_idx == 11 and asset_id == 3:
                    continue
                second_valid = not (split == "train" and date_idx == 7 and asset_id == 1)
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
        pd.DataFrame(asset_rows).to_parquet(
            root / f"{split}_asset_features.parquet", index=False
        )
        date_ids = np.flatnonzero(allowed)
        for group in ("market", "macro"):
            feature = f"{group}_a"
            pd.DataFrame(
                {
                    "date": dates[date_ids].date,
                    "date_idx": date_ids,
                    "valid_date": True,
                    feature: date_ids.astype(np.float32),
                    f"{feature}__valid": True,
                }
            ).to_parquet(root / f"{split}_{group}_features.parquet", index=False)

    (root / "manifest.json").write_text(
        json.dumps({"build_id": "synthetic-build", "source_database": "synthetic.duckdb"}),
        encoding="utf-8",
    )
    (root / "config_resolved.yaml").write_text(
        yaml.safe_dump({"dates": {"lookback_days": 4}}), encoding="utf-8"
    )
    return FIJepaDataConfig(
        artifact_path=root,
        cache_root=root.parent / "cache",
        lookback_days=4,
        patch_len=2,
        train_k_assets=2,
        fixed_k_assets=2,
        mask_ratio=0.5,
        min_masked_patches=1,
        max_masked_patches=1,
        min_valid_days_per_asset_patch=1,
        min_valid_dates_in_patch=1,
        min_valid_asset_fraction=0.25,
        batch_size=2,
        validation_batch_size=2,
        seed=17,
    )


def _assert_batches_equal(first: dict[str, object], second: dict[str, object]) -> None:
    """Require exact tensor and metadata parity for two runtime batches."""
    assert first.keys() == second.keys()
    for name, first_value in first.items():
        second_value = second[name]
        if isinstance(first_value, torch.Tensor):
            assert isinstance(second_value, torch.Tensor)
            assert torch.equal(first_value, second_value), name
        else:
            assert first_value == second_value, name


# ============================================================================
# DENSE CACHE CONTRACT
# ============================================================================


def test_dense_cache_layout_split_isolation_and_request_indexes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_sparse_artifact(tmp_path / "artifact")
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)

    assert store.cache_path == (
        config.cache_root / f"synthetic-build_v{CACHE_FORMAT_VERSION}"
    ).resolve()
    expected = {
        "manifest.json",
        "config_resolved.yaml",
        "dates.npy",
        "assets.npy",
        "feature_manifest.parquet",
        "train_request_index.parquet",
        "validation_request_index.parquet",
        "train_target_date_mask.npy",
    }
    expected.update(
        f"{split}_{name}.npy"
        for split in ("train", "validation")
        for name in (
            "asset_x",
            "asset_feature_mask",
            "valid_asset_mask",
            "market_x",
            "market_feature_mask",
            "valid_market_date",
            "macro_x",
            "macro_feature_mask",
            "valid_macro_date",
        )
    )
    assert {path.name for path in store.cache_path.iterdir()} == expected
    assert not store.train_valid_asset_mask[8:].any()
    assert not store.validation_valid_asset_mask[:8].any()
    assert np.count_nonzero(store.train_asset_x[8:]) == 0
    assert np.count_nonzero(store.validation_asset_x[:8]) == 0
    assert store.train_asset_feature_mask[7, 1].tolist() == [True, False]
    assert store.train_asset_x[7, 1, 1] == 0.0
    assert store.train_request_index["sample_date_idx"].tolist() == [3, 5, 7]
    assert store.validation_request_index["sample_date_idx"].tolist() == [11, 13]
    assert store.validation_request_index["n_endpoint_valid_assets"].tolist() == [3, 4]
    assert "checking" in capsys.readouterr().out


def test_cache_reuse_strict_invalidation_and_status_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_sparse_artifact(tmp_path / "artifact")
    first = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    first_output = capsys.readouterr().out
    first.close()
    second = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    second_output = capsys.readouterr().out
    second.close()

    config_path = config.artifact_path / "config_resolved.yaml"
    config_path.write_text(
        yaml.safe_dump({"dates": {"lookback_days": 4}, "revision": 2}),
        encoding="utf-8",
    )
    rebuilt = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    rebuilt_output = capsys.readouterr().out

    assert "rebuilding" in first_output
    assert "published" in first_output
    assert "reusing" in second_output
    assert "rebuilding" in rebuilt_output
    manifest = json.loads((rebuilt.cache_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_config_sha256"]
    assert manifest["array_shapes"]["train_asset_x"] == [16, 4, 2]
    assert manifest["array_dtypes"]["train_asset_x"] == "float32"


def test_worker_pickle_reopens_existing_cache_read_only(tmp_path: Path) -> None:
    config = _write_sparse_artifact(tmp_path / "artifact")
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    state = store.__getstate__()

    assert "train_asset_x" not in state
    assert "validation_macro_x" not in state
    restored = pickle.loads(pickle.dumps(store))
    assert isinstance(restored.train_asset_x, np.memmap)
    assert not restored.train_asset_x.flags.writeable
    assert np.array_equal(restored.train_asset_x, store.train_asset_x)


# ============================================================================
# RUNTIME GATHERING AND VIEWS
# ============================================================================


def test_runtime_gathers_model_batch_and_exposes_debug_metadata(tmp_path: Path) -> None:
    config = _write_sparse_artifact(tmp_path / "artifact")
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    train = build_fi_jepa_dataloader(config, "train", store=store, shuffle=False)
    validation = build_fi_jepa_dataloader(config, "validation", store=store, shuffle=False)
    train_batch = next(iter(train))
    validation_batch = next(iter(validation))

    assert train_batch["asset_patches"].shape == (2, 2, 2, 2, 2)
    assert validation_batch["asset_patches"].shape == (2, 2, 2, 4, 2)
    assert train_batch["asset_view"] == ["random_k", "random_k"]
    assert validation_batch["asset_view"] == ["all_valid", "all_valid"]
    assert train_batch["k_assets"] == [2, 2]
    assert validation_batch["k_assets"] == [4, 4]
    assert train_batch["n_endpoint_valid_assets"] == [4, 4]
    assert validation_batch["n_endpoint_valid_assets"] == [3, 4]
    assert "asset_x" not in train_batch
    assert "valid_date_mask" not in train_batch
    assert train_batch["patch_context_mask"].all()
    assert train_batch["jepa_target_mask"].sum(dim=1).tolist() == [1, 1]


def test_random_k_changes_by_epoch_and_fixed_k_is_deterministic(tmp_path: Path) -> None:
    config = _write_sparse_artifact(tmp_path / "artifact")
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    train = build_fi_jepa_dataloader(config, "train", store=store, shuffle=False)
    train.dataset.set_epoch(0)
    first_train = next(iter(train))
    train.dataset.set_epoch(1)
    second_train = next(iter(train))

    fixed_first = next(
        iter(
            build_fi_jepa_embedding_dataloader(
                config, "validation", asset_view="fixed_k", store=store, view_index=2
            )
        )
    )
    fixed_repeat = next(
        iter(
            build_fi_jepa_embedding_dataloader(
                config, "validation", asset_view="fixed_k", store=store, view_index=2
            )
        )
    )
    assert not torch.equal(first_train["asset_ids"], second_train["asset_ids"])
    assert torch.equal(fixed_first["asset_ids"], fixed_repeat["asset_ids"])
    assert "jepa_context_mask" not in fixed_first
    assert fixed_first["patch_context_mask"][:, -1].all()


def test_structurally_invalid_jepa_endpoints_are_filtered(tmp_path: Path) -> None:
    config = _write_sparse_artifact(tmp_path / "artifact")
    path = config.artifact_path / "train_asset_features.parquet"
    facts = pd.read_parquet(path)
    facts = facts.loc[
        ~facts["asset_id"].eq(3) | facts["date_idx"].isin([3, 5, 7])
    ]
    facts.to_parquet(path, index=False)
    (config.artifact_path / "manifest.json").write_text(
        json.dumps({"build_id": "structurally-invalid-build"}), encoding="utf-8"
    )
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    impossible = FIJepaDataConfig(
        **{
            **config.__dict__,
            "min_valid_days_per_asset_patch": 2,
            "min_valid_asset_fraction": 1.0,
        }
    )
    loader = build_fi_jepa_dataloader(impossible, "train", store=store, shuffle=False)

    assert loader.dataset.nominal_request_count == 3
    assert loader.dataset.dropped_request_count == 3
    assert loader.dataset.request_index.empty


def test_fixed_k_and_selected_jepa_views_fail_loudly(tmp_path: Path) -> None:
    config = _write_sparse_artifact(tmp_path / "artifact")
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    too_wide = FIJepaDataConfig(**{**config.__dict__, "fixed_k_assets": 4})
    loader = build_fi_jepa_embedding_dataloader(
        too_wide, "validation", asset_view="fixed_k", store=store
    )
    with pytest.raises(RuntimeError, match="n_endpoint_valid_assets=3"):
        next(iter(loader))

    selected_view_config = _write_sparse_artifact(tmp_path / "selected_view_artifact")
    first_seed = np.random.SeedSequence(
        [selected_view_config.seed, 3, 0, 0]
    ).generate_state(1, dtype=np.uint64)[0]
    selected_asset = int(
        np.random.default_rng(first_seed).choice(np.arange(4), size=1)[0]
    )
    path = selected_view_config.artifact_path / "train_asset_features.parquet"
    facts = pd.read_parquet(path)
    facts = facts.loc[
        ~facts["asset_id"].eq(selected_asset) | facts["date_idx"].isin([3, 5, 7])
    ]
    facts.to_parquet(path, index=False)
    (selected_view_config.artifact_path / "manifest.json").write_text(
        json.dumps({"build_id": "selected-view-build"}), encoding="utf-8"
    )
    selected_view_store = DensePanelStore(
        selected_view_config.artifact_path, cache_root=selected_view_config.cache_root
    )
    selected_view_invalid = FIJepaDataConfig(
        **{
            **selected_view_config.__dict__,
            "train_k_assets": 1,
            "min_valid_days_per_asset_patch": 2,
            "min_valid_asset_fraction": 0.5,
        }
    )
    loader = build_fi_jepa_dataloader(
        selected_view_invalid, "train", store=selected_view_store, shuffle=False
    )
    with pytest.raises(RuntimeError, match="Selected JEPA view is not viable"):
        next(iter(loader))


def test_worker_loader_matches_serial_across_repeated_iterators(tmp_path: Path) -> None:
    config = _write_sparse_artifact(tmp_path / "artifact")
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    serial = build_fi_jepa_dataloader(config, "validation", store=store, shuffle=False)
    worker = build_fi_jepa_dataloader(
        FIJepaDataConfig(**{**config.__dict__, "num_workers": 2}),
        "validation",
        store=store,
        shuffle=False,
    )
    expected = list(serial)
    for actual in (list(worker), list(worker)):
        assert len(actual) == len(expected)
        for first, second in zip(expected, actual, strict=True):
            _assert_batches_equal(first, second)


# ============================================================================
# REJECTION
# ============================================================================


def test_store_requires_build_id_and_rejects_duplicate_sparse_keys(tmp_path: Path) -> None:
    config = _write_sparse_artifact(tmp_path / "missing_id")
    (config.artifact_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="stable build_id"):
        DensePanelStore(config.artifact_path, cache_root=config.cache_root)

    config = _write_sparse_artifact(tmp_path / "duplicates")
    path = config.artifact_path / "train_asset_features.parquet"
    facts = pd.read_parquet(path)
    pd.concat([facts, facts.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match=r"duplicate \(date_idx, asset_id\)"):
        DensePanelStore(config.artifact_path, cache_root=config.cache_root)


def test_request_dataset_rejects_lookback_requiring_padding(tmp_path: Path) -> None:
    config = _write_sparse_artifact(tmp_path / "artifact")
    (config.artifact_path / "config_resolved.yaml").write_text(
        yaml.safe_dump({"dates": {"lookback_days": 8}}), encoding="utf-8"
    )
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    invalid = FIJepaDataConfig(**{**config.__dict__, "lookback_days": 6, "patch_len": 2})
    with pytest.raises(ValueError, match="without padding"):
        build_fi_jepa_dataloader(invalid, "train", store=store)
