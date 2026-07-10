from __future__ import annotations

import inspect

import numpy as np
import pytest

from fi_jepa.probes.runner import (
    _incremental_comparison_metrics,
    _incremental_residual_predictions,
    _invalid_predictions,
    _score_predictions,
)


# ============================================================================
# TRAIN-ONLY INCREMENTAL RESIDUALIZATION
# ============================================================================


def test_incremental_prediction_reconstructs_hand_plus_predicted_residual() -> None:
    hand_train = np.arange(8, dtype=np.float64)[:, None]
    z_train = np.column_stack([np.sin(hand_train[:, 0]), np.cos(hand_train[:, 0])])
    train_y = 0.5 * hand_train[:, 0] + 0.4 * z_train[:, 0]
    hand_validation = np.asarray([[8.0], [9.0]], dtype=np.float64)
    z_validation = np.column_stack(
        [np.sin(hand_validation[:, 0]), np.cos(hand_validation[:, 0])]
    )

    hand, residual, reconstructed = _incremental_residual_predictions(
        "ridge",
        hand_train,
        z_train,
        train_y,
        hand_validation,
        z_validation,
        hand_alpha=0.01,
        residual_alpha=0.01,
    )

    assert np.allclose(reconstructed, hand + residual)
    assert "validation_y" not in inspect.signature(_incremental_residual_predictions).parameters


def test_residual_score_space_does_not_apply_original_target_bounds() -> None:
    predicted = np.asarray([-0.25, 0.25], dtype=np.float64)

    residual_invalid, _ = _invalid_predictions(
        "future_realized_vol_21d", predicted, score_space="residual"
    )
    original_invalid, _ = _invalid_predictions(
        "future_realized_vol_21d", predicted, score_space="original"
    )

    assert residual_invalid.tolist() == [False, False]
    assert original_invalid.tolist() == [True, False]


def test_incremental_metrics_compare_matched_predictions_against_hand_only() -> None:
    actual = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    hand = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
    hand_plus_z = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    metrics = _incremental_comparison_metrics(actual, hand, hand_plus_z)

    assert metrics["hand_only_rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert metrics["hand_plus_z_rmse"] == 0.0
    assert metrics["delta_rmse"] == pytest.approx(-np.sqrt(2.0 / 3.0))
    assert metrics["rmse_ratio_vs_hand"] == 0.0
    assert metrics["hand_only_mae"] == pytest.approx(2.0 / 3.0)
    assert metrics["hand_plus_z_mae"] == 0.0
    assert metrics["delta_mae"] == pytest.approx(-2.0 / 3.0)
    assert metrics["hand_plus_z_r2"] == 1.0


def test_oracle_validation_recalibration_is_absent_from_scored_metrics() -> None:
    metrics = _score_predictions(
        "future_return_5d",
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([0.1, 0.9, 2.1], dtype=np.float64),
        score_space="original",
        baseline_metrics=None,
    )

    assert "validation_recalibration" not in metrics
    assert "oracle_validation_recalibration" not in metrics
