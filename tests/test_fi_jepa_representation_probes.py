from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from fi_jepa.probes import export_probe_targets, run_frozen_probes
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
# SEPARATE PROBE ARTIFACTS AND WALK-FORWARD FITTING
# ============================================================================


def _write_probe_database(path: Path) -> pd.DatetimeIndex:
    dates = pd.bdate_range("2024-01-01", periods=12)
    targets = pd.DataFrame(
        {
            "date": dates,
            "symbol": "ETF_SPY",
            "future_return_21d": np.linspace(-0.2, 0.3, len(dates)),
            "future_realized_vol_21d": np.linspace(0.1, 0.5, len(dates)),
            "uses_future_data": True,
        }
    )
    with duckdb.connect(str(path)) as connection:
        connection.register("target_frame", targets)
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

    output = run_frozen_probes(
        embedding_artifact,
        target_artifact,
        output_root=tmp_path / "probe_runs",
    )
    exported_targets = pd.read_parquet(target_artifact / "targets.parquet")
    probe_dataset = pd.read_parquet(output / "probe_dataset.parquet")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))

    assert all(name.startswith("future_") for name in target_manifest["target_columns"])
    assert not any(name.startswith("z_") for name in exported_targets.columns)
    assert any(name.startswith("z_") for name in probe_dataset.columns)
    assert any(name.startswith("future_") for name in probe_dataset.columns)
    assert report["targets_joined_into_pretraining_artifact"] is False
    for fold in report["folds"]:
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_start"])


def test_probes_reject_mismatched_database_versions(tmp_path: Path) -> None:
    database = tmp_path / "market.duckdb"
    dates = _write_probe_database(database)
    target_artifact = export_probe_targets(database, output_root=tmp_path / "probe_targets")
    embedding_artifact = _write_embedding_artifact(tmp_path / "embeddings", dates, "wrong")

    with pytest.raises(ValueError, match="different database versions"):
        run_frozen_probes(
            embedding_artifact,
            target_artifact,
            output_root=tmp_path / "probe_runs",
        )
