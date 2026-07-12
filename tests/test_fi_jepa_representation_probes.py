from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

import fi_jepa.representation as representation_module
from fi_jepa.dataloader import FIJepaDataConfig
from fi_jepa.model_config import FIJepaModelConfig
from fi_jepa.probes import build_probe_dataset, export_probe_targets, run_frozen_probes
from fi_jepa.representation import (
    INITIAL_REPRESENTATION_VARIANTS,
    _embedding_frame,
    build_representation_variants,
    fit_pca_exporter,
    representation_distance_summary,
    representation_diagnostics,
    run_compact_ablation_suite,
    windowed_validation_rank_diagnostics,
)


# ============================================================================
# PCA AND REPRESENTATION DIAGNOSTICS
# ============================================================================


def test_pca_exporter_is_deterministic_sign_canonicalized_and_not_whitened() -> None:
    states = np.asarray(
        [
            [1.0, 0.0, 2.0, 1.0],
            [2.0, 1.0, 4.0, 0.0],
            [3.0, 1.0, 6.0, -1.0],
            [4.0, 2.0, 8.0, -2.0],
            [5.0, 3.0, 10.0, -3.0],
        ]
    )
    first = fit_pca_exporter(states, 2)
    second = fit_pca_exporter(states, 2)
    transformed = first.transform(states)

    assert first.version == second.version
    assert np.array_equal(first.components, second.components)
    for component in first.components:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] >= 0.0
    assert np.allclose(transformed.var(axis=0, ddof=1), first.explained_variance)


def test_representation_diagnostics_report_rank_geometry_and_zero_norms() -> None:
    values = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0]]
    )
    diagnostics = representation_diagnostics(values)

    assert diagnostics["sample_count"] == 5
    assert diagnostics["dimension_count"] == 2
    assert diagnostics["zero_norm_count"] == 1
    assert diagnostics["near_zero_norm_count"] == 1
    assert diagnostics["effective_rank"] == pytest.approx(2.0)
    assert diagnostics["pairwise_cosine"]["pair_count"] == 10


def test_windowed_rank_diagnostics_keep_outer_validation_windows_separate() -> None:
    """Compute one result per named outer window for raw and selected PCA states only."""
    train_metadata = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=12)})
    validation_metadata = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=6),
            "validation_window_name": ["first"] * 3 + ["second"] * 3,
        }
    )
    axis = np.arange(12, dtype=np.float64)
    train_raw = np.column_stack((axis, axis % 3, np.ones(12)))
    validation_raw = np.column_stack((np.arange(6), np.arange(6) % 2, np.ones(6)))
    diagnostics = windowed_validation_rank_diagnostics(
        train_metadata,
        {"raw_pooled_state": train_raw, "selected_pca_representation": train_raw[:, :2]},
        validation_metadata,
        {
            "raw_pooled_state": validation_raw,
            "selected_pca_representation": validation_raw[:, :2],
        },
    )

    assert set(diagnostics["representations"]) == {
        "raw_pooled_state",
        "selected_pca_representation",
    }
    for representation in diagnostics["representations"].values():
        assert [window["validation_window_name"] for window in representation["windows"]] == [
            "first",
            "second",
        ]
        assert all(window["validation_date_count"] == 3 for window in representation["windows"])
        assert set(representation["windows"][0]["metrics"]) == {
            "sample_count",
            "effective_rank",
            "top_eigenvalue_share",
            "top_5_eigenvalue_share",
            "mean_pairwise_cosine",
            "median_pairwise_cosine",
            "mean_vector_norm",
        }


def test_matched_train_windows_have_exact_validation_date_length() -> None:
    train_metadata = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=10)})
    validation_metadata = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=4),
            "validation_window_name": ["outer"] * 4,
        }
    )
    train = np.column_stack((np.arange(10, dtype=np.float64), np.ones(10)))
    validation = np.column_stack((np.arange(4, dtype=np.float64), np.ones(4)))
    diagnostics = windowed_validation_rank_diagnostics(
        train_metadata,
        {"raw_pooled_state": train},
        validation_metadata,
        {"raw_pooled_state": validation},
    )

    window = diagnostics["representations"]["raw_pooled_state"]["windows"][0]
    assert window["matched_train_window_count"] == 3
    assert window["matched_train_window_date_counts"] == [4, 4, 4]


def test_validation_percentile_uses_weak_empirical_rank() -> None:
    """Count matched train values less than or equal to the validation value."""
    train_metadata = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=6)})
    validation_metadata = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=2),
            "validation_window_name": ["outer", "outer"],
        }
    )
    # Exact two-date windows have mean norms 1, 2, and 3; validation ties the middle value.
    train = np.asarray([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
    validation = np.asarray([[2.0], [2.0]])
    diagnostics = windowed_validation_rank_diagnostics(
        train_metadata,
        {"raw_pooled_state": train},
        validation_metadata,
        {"raw_pooled_state": validation},
    )

    comparison = diagnostics["representations"]["raw_pooled_state"]["windows"][0]["metrics"][
        "mean_vector_norm"
    ]
    assert comparison == {
        "matched_train_median": 2.0,
        "matched_train_5th_percentile": pytest.approx(1.1),
        "matched_train_95th_percentile": pytest.approx(2.9),
        "validation_value": 2.0,
        "validation_percentile": pytest.approx(200.0 / 3.0),
    }


def test_representation_report_contains_minimal_artifact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record the frozen checkpoint, variant, and resolved evaluation settings."""
    train_metadata = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5)})
    validation_metadata = pd.DataFrame({"date": pd.bdate_range("2024-02-01", periods=3)})
    train_base = np.arange(20, dtype=np.float64).reshape(5, 4)
    validation_base = np.arange(12, dtype=np.float64).reshape(3, 4)

    def fake_loader(config: object, split: str, *, asset_view: str, **kwargs: object) -> tuple[str, str]:
        return split, asset_view

    def fake_collect(
        model: object,
        loader: tuple[str, str],
        device: object,
        amp_dtype: object,
        **kwargs: object,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        metadata = train_metadata if loader[0] == "train" else validation_metadata
        base = train_base if loader[0] == "train" else validation_base
        states = {
            "mean_state": base,
            "endpoint_state": base + 0.25,
            "pooled_state": np.concatenate([base, base + 0.5], axis=1),
        }
        return metadata, states

    class FakeStore:
        """Expose the immutable identity read by representation evaluation."""

        dataset_version = "dataset-a"
        manifest = {"source_database": "market.duckdb", "source_database_sha256": "db-hash"}

    monkeypatch.setattr(representation_module, "build_fi_jepa_embedding_dataloader", fake_loader)
    monkeypatch.setattr(representation_module, "collect_representation_states", fake_collect)
    output_dir = tmp_path / "evaluation"
    representation_module.run_representation_evaluation(
        object(),
        FakeStore(),
        FIJepaDataConfig(artifact_path=tmp_path / "artifact"),
        device=representation_module.torch.device("cpu"),
        amp_dtype=None,
        n_components=2,
        views_per_date=1,
        output_dir=output_dir,
        checkpoint_id="checkpoint-a",
        checkpoint_step=10,
        checkpoint_format_version=2,
        model_version="model-a",
        export_embeddings=False,
        representation_variant="pooled_pca_2",
    )

    report = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert {
        "schema_version",
        "checkpoint_id",
        "checkpoint_step",
        "representation_source",
        "representation_variant",
        "resolved_probe_config",
        "resolved_representation_config",
        "created_at_utc",
    }.issubset(report)
    assert report["checkpoint_id"] == "checkpoint-a"
    assert report["checkpoint_step"] == 10
    assert report["representation_variant"] == "pooled_pca_2"


def test_initial_representation_variants_use_train_only_pca_and_exact_dimensions() -> None:
    rng = np.random.default_rng(17)
    train_mean = rng.normal(size=(80, 128))
    train_endpoint = rng.normal(size=(80, 128))
    validation_mean = rng.normal(loc=100.0, size=(12, 128))
    validation_endpoint = rng.normal(loc=-100.0, size=(12, 128))
    train_states = {
        "mean_state": train_mean,
        "endpoint_state": train_endpoint,
        "pooled_state": np.concatenate((train_mean, train_endpoint), axis=1),
    }
    validation_states = {
        "mean_state": validation_mean,
        "endpoint_state": validation_endpoint,
        "pooled_state": np.concatenate((validation_mean, validation_endpoint), axis=1),
    }

    outputs, metadata, exporters = build_representation_variants(
        train_states, validation_states
    )

    assert tuple(outputs) == INITIAL_REPRESENTATION_VARIANTS
    assert {name: values[0].shape[1] for name, values in outputs.items()} == {
        "mean_pca_16": 16,
        "endpoint_pca_16": 16,
        "pooled_pca_16": 16,
        "pooled_pca_32": 32,
        "pooled_pca_64": 64,
        "pooled_raw_256": 256,
    }
    for variant, exporter in exporters.items():
        source = str(metadata[variant]["representation_source"])
        assert np.allclose(exporter.mean, train_states[source].mean(axis=0))
        assert metadata[variant]["pca_fit_split"] == "train"
        assert metadata[variant]["explained_variance_ratio"]
    assert np.array_equal(outputs["pooled_raw_256"][0], train_states["pooled_state"])
    assert np.array_equal(outputs["pooled_raw_256"][1], validation_states["pooled_state"])

    exported = _embedding_frame(
        pd.DataFrame({"date": pd.date_range("2024-01-01", periods=80)}),
        outputs["pooled_pca_16"][0],
        {
            "representation_variant": "pooled_pca_16",
            "representation_source": "pooled_state",
            "representation_dimension": 16,
        },
    )
    assert not any(column.startswith("future_") for column in exported.columns)


def test_representation_distance_retains_magnitude_information() -> None:
    reference = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    scaled = reference * 2.0

    summary = representation_distance_summary(reference, scaled)

    assert summary["mean_cosine_similarity"] == pytest.approx(1.0)
    assert summary["mean_relative_l2_distance"] == pytest.approx(1.0)


def test_compact_ablation_suite_uses_exact_counts_modes_and_default_probe_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    metadata = pd.DataFrame({"date": dates})
    reference = np.arange(48, dtype=np.float64).reshape(6, 8) + 1.0
    states = {
        "mean_state": reference[:, :4],
        "endpoint_state": reference[:, 4:],
        "pooled_state": reference,
    }

    def fake_loader(
        config: FIJepaDataConfig, split: str, *, asset_view: str, **kwargs: object
    ) -> tuple[str, str, int]:
        return split, asset_view, config.fixed_k_assets

    def fake_collect(
        model: object,
        loader: tuple[str, str, int],
        device: object,
        amp_dtype: object,
        *,
        input_mode: str = "all_streams",
        **kwargs: object,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        _, asset_view, asset_count = loader
        scale = 1.0 if asset_view == "all_valid" else 1.0 + asset_count / 1_000.0
        if input_mode != "all_streams":
            scale += {"without_assets": 0.1, "without_market": 0.2, "without_macro": 0.3}[input_mode]
        values = reference * scale
        return metadata.copy(), {
            "mean_state": values[:, :4],
            "endpoint_state": values[:, 4:],
            "pooled_state": values,
        }

    monkeypatch.setattr(representation_module, "build_fi_jepa_embedding_dataloader", fake_loader)
    monkeypatch.setattr(representation_module, "collect_representation_states", fake_collect)
    report, probe_states = run_compact_ablation_suite(
        object(),
        object(),
        FIJepaDataConfig(artifact_path=tmp_path / "artifact"),
        device=representation_module.torch.device("cpu"),
        amp_dtype=None,
        train_metadata=metadata,
        train_states=states,
        validation_metadata=metadata,
        validation_states=states,
        collect_probe_states=True,
    )

    assert set(report["asset_count_ablations"]) == {"k_32", "k_128", "k_256", "all_valid"}
    assert set(report["input_branch_ablations"]) == {
        "all_streams",
        "without_assets",
        "without_market",
        "without_macro",
    }
    assert report["default_asset_probe_variants"] == [
        "pooled_pca_16",
        "asset_k_128_pooled_pca_16",
    ]
    assert "asset_k_128_pooled_pca_16" in probe_states
    assert "asset_k_32_pooled_pca_16" not in probe_states
    assert "asset_k_256_pooled_pca_16" not in probe_states


# ============================================================================
# ON-DEMAND EVALUATION OVERRIDES
# ============================================================================


def test_evaluate_checkpoint_overrides_only_runtime_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply the CLI batch override without mutating checkpoint configuration."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    embedded_data_config = asdict(
        FIJepaDataConfig(artifact_path=tmp_path / "artifact", validation_batch_size=16)
    )
    checkpoint = {
        "format_version": 1,
        "global_step": 7,
        "model": {},
        "resolved_config": {
            "model": asdict(FIJepaModelConfig()),
            "dataloader": embedded_data_config,
            "training": {
                "representation_pca_components": 8,
                "representation_views_per_date": 3,
            },
        },
    }

    class FakeStore:
        """Expose only metadata required by on-demand checkpoint evaluation."""

        def __init__(self, artifact_path: Path, *, cache_root: Path):
            self.artifact_path = artifact_path
            self.dataset_version = "dataset-test"

    class FakeModel:
        """Capture model loading while avoiding a real evaluation allocation."""

        @classmethod
        def from_store(cls, model_config: FIJepaModelConfig, store: FakeStore) -> FakeModel:
            return cls()

        def load_state_dict(self, state: dict[str, object]) -> None:
            return None

        def to(self, device: object) -> FakeModel:
            return self

    captured: dict[str, object] = {}

    def fake_run_representation_evaluation(
        model: FakeModel,
        store: FakeStore,
        data_config: FIJepaDataConfig,
        **kwargs: object,
    ) -> None:
        captured["data_config"] = data_config

    monkeypatch.setattr(representation_module.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(representation_module, "DensePanelStore", FakeStore)
    monkeypatch.setattr(representation_module, "FIJepaModel", FakeModel)
    monkeypatch.setattr(
        representation_module,
        "run_representation_evaluation",
        fake_run_representation_evaluation,
    )

    representation_module.evaluate_checkpoint(
        checkpoint_path,
        output_root=tmp_path / "evaluation",
        device_name="cpu",
        batch_size=1,
    )

    runtime_config = captured["data_config"]
    assert isinstance(runtime_config, FIJepaDataConfig)
    assert runtime_config.validation_batch_size == 1
    assert checkpoint["resolved_config"]["dataloader"]["validation_batch_size"] == 16


def test_evaluation_cli_parses_batch_size_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the on-demand evaluation batch override through the CLI."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate-fi-jepa", "--checkpoint", "checkpoint.pt", "--batch-size", "2"],
    )

    assert representation_module.parse_args().batch_size == 2


# ============================================================================
# SEPARATE PROBE ARTIFACTS AND WALK-FORWARD FITTING
# ============================================================================


def _write_probe_database(path: Path) -> pd.DatetimeIndex:
    dates = pd.bdate_range("2022-01-03", periods=380)
    feature_axis = np.linspace(0.0, 1.0, len(dates))
    features = pd.DataFrame(
        {
            "date": dates,
            "xs_dispersion_1d": 0.1 + feature_axis,
            "breadth_1d": 0.4 + 0.2 * feature_axis,
            "pct_above_ma_63d": 0.3 + 0.3 * feature_axis,
            "vix_level": 15.0 + feature_axis,
            "uses_future_data": False,
        }
    )
    ticker_features = pd.DataFrame(
        {
            "date": dates,
            "symbol": "ETF_SPY",
            "valid_observation": True,
            "return_21d": np.linspace(-0.1, 0.2, len(dates)),
            "return_63d": np.linspace(-0.2, 0.3, len(dates)),
            "return_126d": np.linspace(-0.3, 0.4, len(dates)),
            "realized_vol_21d": np.linspace(0.1, 0.5, len(dates)),
            "realized_vol_63d": np.linspace(0.2, 0.6, len(dates)),
            "realized_vol_126d": np.linspace(0.3, 0.7, len(dates)),
            "drawdown_21d": np.linspace(-0.3, -0.01, len(dates)),
            "drawdown_63d": np.linspace(-0.4, -0.02, len(dates)),
            "drawdown_126d": np.linspace(-0.5, -0.03, len(dates)),
            "uses_future_data": False,
        }
    )
    targets = pd.DataFrame(
        {
            "date": dates,
            "symbol": "ETF_SPY",
            "future_return_21d": np.linspace(-0.2, 0.3, len(dates)),
            "future_realized_vol_21d": np.linspace(0.1, 0.5, len(dates)),
            "future_max_drawdown_21d": np.linspace(-0.4, -0.05, len(dates)),
            "uses_future_data": True,
        }
    )
    with duckdb.connect(str(path)) as connection:
        connection.register("feature_frame", features)
        connection.register("ticker_feature_frame", ticker_features)
        connection.register("target_frame", targets)
        connection.execute("CREATE TABLE features AS SELECT * FROM feature_frame")
        connection.execute("CREATE TABLE ticker_features AS SELECT * FROM ticker_feature_frame")
        connection.execute("CREATE TABLE targets AS SELECT * FROM target_frame")
    return dates


def _write_embedding_artifact(
    root: Path,
    dates: pd.DatetimeIndex,
    source_database_sha256: str,
) -> Path:
    root.mkdir()
    split = ["train"] * 280 + ["validation"] * 20 + ["train"] * 60 + ["validation"] * 20
    windows = [""] * 280 + ["window_one"] * 20 + [""] * 60 + ["window_two"] * 20
    embeddings = pd.DataFrame(
        {
            "date": dates,
            "z_1": np.linspace(-1.0, 1.0, len(dates)),
            "z_2": np.cos(np.linspace(0.0, 2.0, len(dates))),
            "split": split,
            "validation_window_name": windows,
            "embedding_schema_version": 1,
            "checkpoint_id": "checkpoint-a",
            "checkpoint_step": 10,
            "checkpoint_format_version": 1,
            "model_version": "model-a",
            "dataset_version": "dataset-a",
            "pca_version": "pca-a",
        }
    )
    embeddings.to_parquet(root / "embeddings.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "source_database_sha256": source_database_sha256,
                "pca_version": "pca-a",
                "checkpoint_id": "checkpoint-a",
                "checkpoint_step": 10,
                "representation_source": "encode_pooled_state",
                "representation_variant": "pooled_pca_16",
                "resolved_representation_config": {
                    "pca_components": 16,
                    "views_per_date": 3,
                    "export_embeddings": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_probe_dataset_selects_one_explicit_exported_variant(tmp_path: Path) -> None:
    database = tmp_path / "market.duckdb"
    dates = _write_probe_database(database)
    target_artifact = export_probe_targets(database, output_root=tmp_path / "probe_targets")
    target_manifest = json.loads((target_artifact / "manifest.json").read_text(encoding="utf-8"))
    embedding_artifact = _write_embedding_artifact(
        tmp_path / "embeddings", dates, target_manifest["source_database_sha256"]
    )
    base = pd.read_parquet(embedding_artifact / "embeddings.parquet")
    variant = base.drop(columns=["z_1", "z_2"]).copy()
    for index in range(16):
        variant[f"z_{index + 1}"] = np.linspace(index, index + 1.0, len(variant))
    variant["representation_variant"] = "mean_pca_16"
    variant["representation_source"] = "mean_state"
    variant["representation_dimension"] = 16
    variant.to_parquet(embedding_artifact / "embeddings_mean_pca_16.parquet", index=False)
    manifest = json.loads((embedding_artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["representation_variants"] = {
        "mean_pca_16": {
            "representation_variant": "mean_pca_16",
            "representation_source": "mean_state",
            "dimension": 16,
            "embedding_file": "embeddings_mean_pca_16.parquet",
        }
    }
    (embedding_artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset_artifact = build_probe_dataset(
        embedding_artifact,
        target_artifact,
        output_root=tmp_path / "probe_targets",
        representation_variant="mean_pca_16",
    )
    dataset_manifest = json.loads((dataset_artifact / "manifest.json").read_text(encoding="utf-8"))
    dataset = pd.read_parquet(dataset_artifact / "probe_dataset.parquet")

    assert dataset_manifest["representation_variant"] == "mean_pca_16"
    assert dataset_manifest["representation_source"] == "mean_state"
    assert dataset_manifest["representation_dimension"] == 16
    assert len([column for column in dataset if column.startswith("z_")]) == 16
    assert not any(column.startswith("future_") for column in variant.columns)


def test_probe_targets_stay_separate_and_probes_are_walk_forward(tmp_path: Path) -> None:
    database = tmp_path / "market.duckdb"
    dates = _write_probe_database(database)
    target_artifact = export_probe_targets(database, output_root=tmp_path / "probe_targets")
    target_manifest = json.loads(
        (target_artifact / "manifest.json").read_text(encoding="utf-8")
    )
    embedding_artifact = _write_embedding_artifact(
        tmp_path / "embeddings",
        dates,
        target_manifest["source_database_sha256"],
    )

    dataset_artifact = build_probe_dataset(
        embedding_artifact,
        target_artifact,
        output_root=tmp_path / "probe_targets",
    )
    output = run_frozen_probes(
        probe_dataset_artifact=dataset_artifact,
        output_root=tmp_path / "probe_runs",
        ridge_alphas=(0.1,),
        huber_alphas=(0.01,),
        elastic_net_alphas=(0.001,),
        elastic_net_l1_ratios=(0.5,),
        logistic_alphas=(1.0,),
    )
    exported_targets = pd.read_parquet(target_artifact / "targets.parquet")
    probe_dataset = pd.read_parquet(dataset_artifact / "probe_dataset.parquet")
    predictions = pd.read_parquet(output / "predictions.parquet")
    dataset_manifest = json.loads(
        (dataset_artifact / "manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    summary = (output / "summary.md").read_text(encoding="utf-8")

    assert target_artifact == tmp_path / "probe_targets" / "market_targets"
    assert dataset_artifact == tmp_path / "probe_targets" / "embeddings_probe_dataset"
    assert all(name.startswith("future_") for name in target_manifest["target_columns"])
    assert target_manifest["baseline_feature_columns"]
    assert not any(name.startswith("z_") for name in exported_targets.columns)
    assert any(name.startswith("z_") for name in probe_dataset.columns)
    assert any(name.startswith("future_") for name in probe_dataset.columns)
    assert any(name.startswith("baseline__") for name in probe_dataset.columns)
    assert all(
        f"target_available__{name}" in probe_dataset.columns
        for name in target_manifest["target_columns"]
    )
    assert len(dataset_manifest["validation_windows"]) == 2
    assert {
        "train_mean",
        "trailing_target_proxy",
        "ridge__z_only",
        "huber__z_only",
        "elastic_net__z_only",
        "ridge__hand_market_features",
        "ridge__hand_market_pca",
        "ridge__hand_market_features_plus_z",
        "ridge__hand_market_features_plus_residual_z",
        "ridge__feature_residualized_z_only",
        "class_prior",
        "logistic__z_only",
        "logistic__hand_market_features",
    }.issubset(set(predictions["predictor_name"]))
    assert {
        "prediction",
        "invalid_prediction",
        "invalid_prediction_reason",
        "selected_alpha",
        "alpha_selection",
        "task_type",
        "model_name",
        "feature_family",
    }.issubset(predictions.columns)
    assert "format_version" not in target_manifest
    assert "format_version" not in dataset_manifest
    assert "format_version" not in report
    assert {
        "schema_version",
        "checkpoint_id",
        "checkpoint_step",
        "representation_source",
        "representation_variant",
        "resolved_probe_config",
        "resolved_representation_config",
        "created_at_utc",
    }.issubset(report)
    assert report["checkpoint_id"] == "checkpoint-a"
    assert report["checkpoint_step"] == 10
    assert report["representation_source"] == "encode_pooled_state"
    assert report["representation_variant"] == "pooled_pca_16"
    assert report["probe_dataset_artifact"] == str(dataset_artifact.resolve())
    assert report["alpha_selection"] == "three_fold_expanding_purged"
    assert report["resolved_probe_config"]["ridge_alphas"] == [0.1]
    assert report["resolved_probe_config"]["huber_alphas"] == [0.01]
    assert report["resolved_probe_config"]["elastic_net_alphas"] == [0.001]
    assert report["resolved_probe_config"]["elastic_net_l1_ratios"] == [0.5]
    assert report["resolved_probe_config"]["logistic_alphas"] == [1.0]
    assert report["resolved_probe_config"]["bootstrap_samples"] == 500
    assert report["resolved_probe_config"]["bootstrap_block_length"] == "target_horizon"
    assert report["bootstrap_by_window"]
    assert report["stability_summary"]
    assert report["evaluated_target_model_variant_combination_count"] == len(
        report["stability_summary"]
    )
    assert report["multiple_comparison_correction_applied"] is False
    assert all(
        row["representation_variant"] == "pooled_pca_16"
        and row["sample_count"] == 500
        and row["block_length"] in {21, 63, 126}
        and row["confidence_intervals_95"]
        for row in report["bootstrap_by_window"]
    )
    assert report["baseline_feature_columns"]
    assert "hand_market_features" in report["feature_families"]
    assert "hand_market_features_plus_residual_z" in report["feature_families"]
    assert "feature_residualized_z_only" in report["feature_families"]
    assert "huber" in report["regression_heads"]
    assert "logistic" in report["classification_heads"]
    assert report["phase4_notes"]["hand_plus_residual_z_enabled"] is True
    assert report["final_regression_summary"]
    assert report["pass_fail_gate_counts"]["regression"]
    assert "oracle_validation_recalibration" not in json.dumps(
        report["final_regression_summary"]
    )
    assert "validation_recalibration" not in json.dumps(report["window_summaries"])
    assert report["targets_joined_into_pretraining_artifact"] is False
    assert "recalibration_is_diagnostic_only" not in report
    assert report["incremental_hand_plus_z_comparisons"]
    assert "## Run Identity" in summary
    assert "## Incremental Regression Results" in summary
    assert "## Classification Results" in summary
    assert "## Warnings" in summary
    assert report["window_summaries"]
    assert "future_realized_vol_21d__log_realized_vol" in report["transformed_target_columns"]
    assert "future_max_drawdown_21d__log_drawdown_magnitude" in report["transformed_target_columns"]
    assert report["parameter_selection_by_fold"]
    assert {row["model_name"] for row in report["parameter_selection_by_fold"]} == {
        "ridge",
        "huber",
        "elastic_net",
        "logistic",
    }
    assert "coefficients_by_fold" not in report
    assert all(
        "candidate_diagnostics" not in row and "selection_status" not in row
        for row in report["parameter_selection_by_fold"]
    )

    required_metrics = {
        "rmse",
        "mae",
        "r2",
        "pearson_correlation",
        "spearman_correlation",
        "rmse_ratio_vs_baseline",
        "mae_ratio_vs_baseline",
        "actual_mean",
        "actual_std",
        "prediction_mean",
        "prediction_std",
        "bias",
        "std_ratio",
        "invalid_prediction_count",
        "invalid_prediction_rate",
    }
    regression_results = [
        result for result in report["results"] if result["task_type"] == "regression"
    ]
    for result in regression_results:
        assert required_metrics.issubset(result)
        assert "validation_recalibration" not in result
        assert "oracle_validation_recalibration" not in result
        assert pd.Timestamp(result["train_end"]) < pd.Timestamp(result["validation_start"])

    ridge_results = [
        result for result in regression_results if result["model_name"] == "ridge"
    ]
    baseline_results = [
        result for result in regression_results if result["predictor_name"] == "train_mean"
    ]
    classification_results = [
        result for result in report["results"] if result["task_type"] == "classification"
    ]
    assert all(result["rmse_ratio_vs_baseline"] >= 0.0 for result in ridge_results)
    assert all(result["selected_alpha"] > 0.0 for result in ridge_results)
    model_results = [result for result in report["results"] if result["predictor_kind"] == "model"]
    assert all(len(result["selected_inner_fold_scores"]) == 3 for result in model_results)
    assert all(result["inner_validation_score"] is not None for result in model_results)
    assert all(isinstance(result["selected_at_grid_boundary"], bool) for result in model_results)
    assert all(
        "selected_l1_ratio" in result
        for result in model_results
        if result["model_name"] == "elastic_net"
    )
    assert all(
        "selected_l1_ratio" not in result
        for result in model_results
        if result["model_name"] != "elastic_net"
    )
    assert all(result["rmse_ratio_vs_baseline"] == 1.0 for result in baseline_results)
    assert any(
        result["feature_family"] == "hand_market_features_plus_residual_z"
        for result in regression_results
    )
    assert any(
        result["feature_family"] == "feature_residualized_z_only"
        for result in regression_results
    )
    assert classification_results
    assert all("roc_auc" in result for result in classification_results)
    required_incremental_metrics = {
        "hand_only_rmse",
        "hand_plus_z_rmse",
        "delta_rmse",
        "rmse_ratio_vs_hand",
        "hand_only_mae",
        "hand_plus_z_mae",
        "delta_mae",
        "hand_only_r2",
        "hand_plus_z_r2",
    }
    assert all(
        required_incremental_metrics.issubset(comparison)
        and comparison["score_space"] == "original"
        for comparison in report["incremental_hand_plus_z_comparisons"]
    )


def test_probes_reject_mismatched_source_database_hashes(tmp_path: Path) -> None:
    database = tmp_path / "market.duckdb"
    dates = _write_probe_database(database)
    target_artifact = export_probe_targets(database, output_root=tmp_path / "probe_targets")
    embedding_artifact = _write_embedding_artifact(tmp_path / "embeddings", dates, "wrong")

    with pytest.raises(ValueError, match="different source database hashes"):
        run_frozen_probes(
            embedding_artifact,
            target_artifact,
            output_root=tmp_path / "probe_runs",
        )
