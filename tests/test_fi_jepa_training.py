from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from fi_jepa.dataloader import DensePanelStore, FIJepaDataConfig, build_fi_jepa_dataloader
from fi_jepa.model import FIJepaModel
from fi_jepa.model_config import FIJepaModelConfig
from fi_jepa.model_output import FIJepaOutput
from fi_jepa.training import (
    CHECKPOINT_FORMAT_VERSION,
    LinearEMAMomentumSchedule,
    WarmupCosineLRSchedule,
    _create_run_directory,
    _training_objective,
    build_adamw,
    train_fi_jepa,
    validate_jepa,
)
from fi_jepa.training_config import FIJepaTrainingConfig


# ============================================================================
# SYNTHETIC PRETRAINING INPUTS
# ============================================================================


def _write_training_artifact(root: Path) -> FIJepaDataConfig:
    """Write a minimal complete frozen artifact with train and validation dates."""
    root.mkdir()
    dates = pd.bdate_range("2021-01-01", periods=12)
    validation_sample = np.zeros(12, dtype=bool)
    validation_sample[[9, 11]] = True
    sample_eligible = np.zeros(12, dtype=bool)
    sample_eligible[[3, 4, 5]] = True
    protected = np.zeros(12, dtype=bool)
    protected[6:] = True
    date_manifest = pd.DataFrame(
        {
            "date_idx": np.arange(12, dtype=np.int32),
            "date": dates.date,
            "sample_eligible": sample_eligible,
            "validation_sample": validation_sample,
            "protected_holdout": protected,
            "train_fact_allowed": ~protected,
            "validation_fact_allowed": protected,
        }
    )
    date_manifest.to_parquet(root / "dates.parquet", index=False)
    pd.DataFrame(
        {
            "asset_id": [0, 1],
            "symbol": ["A", "B"],
            "trainable": [True, True],
        }
    ).to_parquet(root / "assets.parquet", index=False)

    features = pd.DataFrame(
        [
            {
                "feature_name": "asset_a",
                "feature_index": 0,
                "input_group": "asset",
                "dtype": "float32",
            },
            {
                "feature_name": "market_a",
                "feature_index": 0,
                "input_group": "market",
                "dtype": "float32",
            },
            {
                "feature_name": "macro_a",
                "feature_index": 0,
                "input_group": "macro",
                "dtype": "float32",
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
        date_ids = np.flatnonzero(allowed)
        asset_rows = [
            {
                "date": dates[date_idx].date(),
                "date_idx": date_idx,
                "asset_id": asset_id,
                "valid_asset": True,
                "asset_a": float(date_idx + asset_id),
                "asset_a__valid": True,
            }
            for date_idx in date_ids
            for asset_id in range(2)
        ]
        pd.DataFrame(asset_rows).to_parquet(root / f"{split}_asset_features.parquet", index=False)
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
        json.dumps({"build_id": "training-test"}), encoding="utf-8"
    )
    (root / "quality_report.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
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
        min_target_blocks=1,
        max_target_blocks=1,
        min_valid_days_per_asset_patch=1,
        min_valid_dates_in_patch=1,
        min_valid_asset_fraction=0.25,
        batch_size=1,
        validation_batch_size=1,
        seed=19,
    )


def _small_model_config() -> FIJepaModelConfig:
    return FIJepaModelConfig(
        patch_len=2,
        num_patches=2,
        tokenizer_type="attention",
        tokenizer_layers=1,
        tokenizer_heads=1,
        tokenizer_mlp_ratio=2,
        asset_pooling_type="attention",
        asset_pooling_layers=1,
        asset_pooling_heads=1,
        asset_pooling_mlp_ratio=2,
        asset_hidden_dim=4,
        asset_token_dim=4,
        market_hidden_dim=4,
        market_token_dim=2,
        macro_hidden_dim=4,
        macro_token_dim=2,
        d_model=4,
        context_layers=1,
        context_heads=2,
        context_mlp_ratio=2,
        context_dropout=0.0,
        predictor_layers=1,
        predictor_heads=2,
        predictor_mlp_ratio=2,
        predictor_dropout=0.0,
    )


def _write_run_configs(root: Path) -> FIJepaTrainingConfig:
    data_config = _write_training_artifact(root / "artifact")
    model_path = root / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "input": {"patch_len": 2, "num_patches": 2},
                "tokenizers": {
                    "type": "attention",
                    "attention": {"layers": 1, "heads": 1, "mlp_ratio": 2},
                    "asset": {"hidden_dim": 4, "output_dim": 4},
                    "market": {"hidden_dim": 4, "output_dim": 2},
                    "macro": {"hidden_dim": 4, "output_dim": 2},
                },
                "asset_pooling": {
                    "type": "attention",
                    "attention": {"layers": 1, "heads": 1, "mlp_ratio": 2},
                },
                "fusion": {"output_dim": 4, "dropout": 0.0},
                "context_encoder": {"layers": 1, "heads": 2, "mlp_ratio": 2, "dropout": 0.0},
                "predictor": {"layers": 1, "heads": 2, "mlp_ratio": 2, "dropout": 0.0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    data_path = root / "dataloader.yaml"
    data_values = data_config.__dict__.copy()
    for name in ("artifact_path", "cache_root"):
        data_values[name] = str(data_values[name])
    data_path.write_text(yaml.safe_dump(data_values, sort_keys=False), encoding="utf-8")
    return FIJepaTrainingConfig(
        run_name="smoke",
        output_root=root / "runs",
        model_config_path=model_path,
        dataloader_config_path=data_path,
        device="cpu",
        epochs=2,
        warmup_epochs=0,
        mixed_precision=True,
        validation_every_epochs=1,
        representation_pca_components=2,
        representation_views_per_date=1,
        representation_export_embeddings=True,
        checkpoint_every_steps=1,
        logging_every_steps=1,
    )


# ============================================================================
# CONFIGURATION, SCHEDULES, AND OPTIMIZER
# ============================================================================


def test_training_config_and_schedules_validate_endpoints_and_clamp() -> None:
    with pytest.raises(ValueError, match="warmup_epochs"):
        FIJepaTrainingConfig(epochs=2, warmup_epochs=2)
    with pytest.raises(ValueError, match="variance_weight"):
        FIJepaTrainingConfig(
            epochs=2,
            warmup_epochs=0,
            anti_collapse_variance_weight=-0.001,
        )

    configured = FIJepaTrainingConfig.from_yaml(Path("configs/pretraining.yaml"))
    assert configured.anti_collapse_variance_weight == pytest.approx(0.001)
    assert configured.anti_collapse_covariance_weight == pytest.approx(0.0001)

    legacy_checkpoint_config = FIJepaTrainingConfig(epochs=2, warmup_epochs=0).to_dict()
    for name in (
        "anti_collapse_variance_weight",
        "anti_collapse_covariance_weight",
        "anti_collapse_variance_floor",
        "anti_collapse_epsilon",
    ):
        legacy_checkpoint_config.pop(name)
    restored_legacy = FIJepaTrainingConfig.from_dict(legacy_checkpoint_config)
    assert restored_legacy.anti_collapse_variance_weight == 0.0
    assert restored_legacy.anti_collapse_covariance_weight == 0.0

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    lr_schedule = WarmupCosineLRSchedule(
        optimizer, base_lr=1.0, min_lr=0.1, warmup_steps=2, total_steps=6
    )
    ema_schedule = LinearEMAMomentumSchedule(start=0.99, end=0.999, total_steps=6)

    assert lr_schedule.value_at(0) == pytest.approx(0.5)
    assert lr_schedule.value_at(1) == pytest.approx(1.0)
    assert lr_schedule.value_at(5) == pytest.approx(0.1)
    assert lr_schedule.value_at(9) == pytest.approx(0.1)
    assert ema_schedule.value_at(0) == pytest.approx(0.99)
    assert ema_schedule.value_at(5) == pytest.approx(0.999)
    assert ema_schedule.value_at(9) == pytest.approx(0.999)


def test_same_name_run_directories_append_readable_timestamp(tmp_path: Path) -> None:
    created_at = datetime(2026, 6, 14, 17, 23, 45, tzinfo=timezone.utc)

    first = _create_run_directory(tmp_path, "experiment", created_at=created_at)
    second = _create_run_directory(tmp_path, "experiment", created_at=created_at)
    third = _create_run_directory(tmp_path, "experiment", created_at=created_at)

    assert first.name == "experiment"
    assert second.name == "experiment-2026-06-14-17-23-45"
    assert third.name == "experiment-2026-06-14-17-23-45-2"


def test_adamw_excludes_complete_frozen_target_branch() -> None:
    model = FIJepaModel(_small_model_config(), 1, 1, 1)
    optimizer = build_adamw(model, FIJepaTrainingConfig(epochs=2, warmup_epochs=0))
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert not optimized.intersection(id(parameter) for parameter in model.target_parameters())
    assert optimized == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_training_objective_adds_weak_regularizer_and_preserves_jepa_loss() -> None:
    context = torch.tensor(
        [
            [[-1.0, -1.0], [-1.0, -1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        requires_grad=True,
    )
    jepa_loss = torch.tensor(2.0, requires_grad=True)
    output = FIJepaOutput(
        loss=jepa_loss,
        predicted_targets=torch.zeros(3, 1, 2),
        target_representations=torch.zeros(3, 1, 2),
        target_patch_mask=torch.ones(3, 1, dtype=torch.bool),
        context_representations=context,
        context_mask=torch.ones(3, 2, dtype=torch.bool),
        fused_tokens=torch.zeros(3, 2, 2),
    )
    config = FIJepaTrainingConfig(
        epochs=2,
        warmup_epochs=0,
        anti_collapse_variance_weight=0.1,
        anti_collapse_covariance_weight=0.2,
        anti_collapse_variance_floor=2.0,
    )

    total_loss, components = _training_objective(output, config)
    expected = (
        jepa_loss
        + 0.1 * components["anti_collapse_variance_loss"]
        + 0.2 * components["anti_collapse_covariance_loss"]
    )

    assert total_loss.item() > jepa_loss.item()
    assert torch.allclose(total_loss, expected)
    total_loss.backward()
    assert jepa_loss.grad == pytest.approx(1.0)
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()


# ============================================================================
# VALIDATION, SMOKE TRAINING, AND RESUME
# ============================================================================


def test_validation_is_deterministic(tmp_path: Path) -> None:
    data_config = _write_training_artifact(tmp_path / "artifact")
    store = DensePanelStore(data_config.artifact_path, cache_root=data_config.cache_root)
    loader = build_fi_jepa_dataloader(data_config, "validation", store=store, shuffle=False)
    torch.manual_seed(5)
    model = FIJepaModel.from_store(_small_model_config(), store)

    first = validate_jepa(model, loader, torch.device("cpu"), None)
    second = validate_jepa(model, loader, torch.device("cpu"), None)

    assert first == second


def test_smoke_training_and_basic_epoch_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_run_configs(tmp_path)
    run_dir = train_fi_jepa(config)
    checkpoints = run_dir / "checkpoints"
    latest = torch.load(checkpoints / "latest.pt", map_location="cpu", weights_only=False)
    periodic = torch.load(checkpoints / "step_000000001.pt", map_location="cpu", weights_only=False)

    assert run_dir == config.output_root / config.run_name
    assert run_dir.name == "smoke"
    assert (run_dir / "resolved_config.yaml").is_file()
    assert (run_dir / "train_log.jsonl").is_file()
    assert (checkpoints / "best_validation.pt").is_file()
    assert (
        run_dir / "representation_diagnostics" / "step_000000003" / "diagnostics.json"
    ).is_file()
    assert (
        run_dir / "representation_diagnostics" / "step_000000006" / "pca_exporter.npz"
    ).is_file()
    embedding_path = (
        run_dir / "representation_diagnostics" / "step_000000006" / "embeddings.parquet"
    )
    embeddings = pd.read_parquet(embedding_path)
    assert {
        "date",
        "z_1",
        "z_2",
        "split",
        "checkpoint_id",
        "model_version",
        "dataset_version",
    } <= set(embeddings.columns)
    assert not any(name.startswith("future_") for name in embeddings.columns)
    assert latest["kind"] == "epoch_end"
    assert latest["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert latest["resume_epoch"] == 2
    assert periodic["kind"] == "periodic"
    assert periodic["resume_epoch"] == 0
    assert "batch" not in periodic
    assert "sampler" not in periodic
    assert latest["global_step"] == 6

    # Epoch-end resume has no remaining work; a periodic resume replays its
    # saved epoch and continues global optimizer-step scheduling.
    train_fi_jepa(resume=checkpoints / "latest.pt")
    train_fi_jepa(resume=checkpoints / "step_000000001.pt")
    resumed = torch.load(checkpoints / "latest.pt", map_location="cpu", weights_only=False)
    records = [
        json.loads(line)
        for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert resumed["resume_epoch"] == 2
    assert resumed["global_step"] == 7
    assert resumed["lr_scheduler"]["last_step"] == 6
    assert resumed["ema_scheduler"]["last_step"] == 6
    assert resumed["lr_scheduler"]["min_lr"] == pytest.approx(
        resumed["optimizer"]["param_groups"][0]["lr"]
    )
    train_records = [record for record in records if record["event"] == "train"]
    assert all(
        {
            "train_loss",
            "train_jepa_loss",
            "anti_collapse_variance_loss",
            "anti_collapse_covariance_loss",
            "anti_collapse_weighted_variance_loss",
            "anti_collapse_weighted_covariance_loss",
            "context_pooled_mean_feature_std",
            "matched_target_cosine",
            "predictor_effective_rank",
            "target_effective_rank",
        }
        <= set(record)
        for record in train_records
    )
    assert sum(record["event"] == "resume" for record in records) == 2
    assert sum(record["event"] == "epoch_warmup" for record in records) == 4
    assert sum(record["event"] == "epoch_boundary" for record in records) == 4
    assert "EPOCH BOUNDARY" in (run_dir / "runtime_summary.txt").read_text(encoding="utf-8")
    assert (checkpoints / "step_000000006.pt").is_file()
    terminal_output = capsys.readouterr()
    assert "Training plan:" in terminal_output.out
    assert "Epoch warm-up:" in terminal_output.out
    assert "Epoch boundary:" in terminal_output.out
    assert "rank=" in terminal_output.err
