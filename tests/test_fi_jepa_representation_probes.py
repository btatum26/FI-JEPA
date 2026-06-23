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
from fi_jepa.representation import fit_pca_exporter, representation_diagnostics


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
    dates = pd.bdate_range("2024-01-01", periods=12)
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
    split = ["train"] * 5 + ["validation"] * 2 + ["train"] * 3 + ["validation"] * 2
    windows = [""] * 5 + ["window_one"] * 2 + [""] * 3 + ["window_two"] * 2
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
            }
        ),
        encoding="utf-8",
    )
    return root


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
    )
    exported_targets = pd.read_parquet(target_artifact / "targets.parquet")
    probe_dataset = pd.read_parquet(dataset_artifact / "probe_dataset.parquet")
    predictions = pd.read_parquet(output / "predictions.parquet")
    dataset_manifest = json.loads(
        (dataset_artifact / "manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))

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
    assert report["probe_dataset_artifact"] == str(dataset_artifact.resolve())
    assert report["alpha_selection"] == "inner_walk_forward"
    assert 1.0 in report["alpha_grid"]
    assert report["baseline_feature_columns"]
    assert "hand_market_features" in report["feature_families"]
    assert "huber" in report["regression_heads"]
    assert "logistic" in report["classification_heads"]
    assert report["targets_joined_into_pretraining_artifact"] is False
    assert report["recalibration_is_diagnostic_only"] is True
    assert report["window_summaries"]
    assert "future_realized_vol_21d__log_realized_vol" in report["transformed_target_columns"]
    assert "future_max_drawdown_21d__log_drawdown_magnitude" in report["transformed_target_columns"]
    assert report["alpha_selection_by_fold"]

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
        "validation_recalibration",
    }
    regression_results = [
        result for result in report["results"] if result["task_type"] == "regression"
    ]
    for result in regression_results:
        assert required_metrics.issubset(result)
        assert result["validation_recalibration"]["uses_validation_labels"] is True
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
    assert all(result["rmse_ratio_vs_baseline"] == 1.0 for result in baseline_results)
    assert classification_results
    assert all("roc_auc" in result for result in classification_results)


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
