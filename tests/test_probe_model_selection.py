from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fi_jepa.probes.runner import (
    _expanding_time_folds,
    _moving_block_bootstrap_indices,
    _paired_moving_block_bootstrap,
    _purged_training_mask,
    _ridge_predict,
    _select_model_parameters,
)


# ============================================================================
# CHRONOLOGICAL SPLITS AND PURGING
# ============================================================================


@pytest.mark.parametrize("horizon", [21, 63, 126])
def test_three_expanding_folds_are_chronological_and_horizon_purged(horizon: int) -> None:
    dates = pd.bdate_range("2018-01-01", periods=520).to_numpy()

    folds = _expanding_time_folds(dates[:450], horizon=horizon, calendar_dates=dates)

    assert len(folds) == 3
    prior_train_count = 0
    positions = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    for expected_fold, (train_mask, validation_mask, metadata) in enumerate(folds, start=1):
        train_dates = dates[:450][train_mask]
        validation_dates = dates[:450][validation_mask]
        assert metadata["fold"] == expected_fold
        assert len(train_dates) > prior_train_count
        assert train_dates.max() < validation_dates.min()
        assert positions[pd.Timestamp(train_dates.max())] + horizon < positions[pd.Timestamp(validation_dates.min())]
        prior_train_count = len(train_dates)


@pytest.mark.parametrize("horizon", [21, 63, 126])
def test_outer_purge_removes_every_overlapping_target_interval(horizon: int) -> None:
    dates = pd.bdate_range("2019-01-01", periods=400).to_numpy()
    validation_start = pd.Timestamp(dates[300])

    mask = _purged_training_mask(
        dates[:300],
        validation_start=validation_start,
        horizon=horizon,
        calendar_dates=dates,
    )

    retained_positions = np.flatnonzero(mask)
    removed_positions = np.flatnonzero(~mask)
    assert retained_positions[-1] + horizon < 300
    assert np.all(removed_positions + horizon >= 300)


# ============================================================================
# MODEL-SPECIFIC SELECTION AND PENALTY SCALING
# ============================================================================


def test_moving_block_bootstrap_preserves_contiguous_sequences() -> None:
    indices = _moving_block_bootstrap_indices(12, 4, 8, seed=19)

    assert indices.shape == (8, 12)
    for sample in indices:
        for start in range(0, 12, 4):
            assert np.array_equal(np.diff(sample[start : start + 4]), np.ones(3, dtype=np.int64))


def test_paired_bootstrap_keeps_candidate_and_hand_predictions_aligned() -> None:
    actual = np.arange(10, dtype=np.float64)
    result = _paired_moving_block_bootstrap(
        actual,
        actual,
        actual + 1.0,
        task_type="regression",
        block_length=3,
        sample_count=40,
        seed=23,
    )

    assert result["point_estimates"]["rmse_difference_vs_hand"] == pytest.approx(-1.0)
    assert result["confidence_intervals_95"]["rmse_difference_vs_hand"] == pytest.approx(
        [-1.0, -1.0]
    )
    with pytest.raises(ValueError, match="identical aligned lengths"):
        _paired_moving_block_bootstrap(
            actual,
            actual[:-1],
            actual,
            task_type="regression",
            block_length=3,
        )


def test_paired_bootstrap_is_deterministic_with_fixed_seed() -> None:
    actual = (np.arange(30) % 2).astype(np.float64)
    probability = np.linspace(0.1, 0.9, 30)
    hand_probability = np.full(30, 0.5)

    first = _paired_moving_block_bootstrap(
        actual,
        probability,
        hand_probability,
        task_type="classification",
        block_length=5,
        sample_count=50,
        seed=29,
    )
    second = _paired_moving_block_bootstrap(
        actual,
        probability,
        hand_probability,
        task_type="classification",
        block_length=5,
        sample_count=50,
        seed=29,
    )

    assert first == second
    assert set(first["confidence_intervals_95"]) == {
        "roc_auc",
        "brier_score_difference_vs_hand",
    }


def test_each_probe_head_selects_from_its_own_parameter_grid() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2017-01-01", periods=360).to_numpy()
    x = rng.normal(size=(320, 3))
    regression_y = x[:, 0] - 0.3 * x[:, 1] + rng.normal(scale=0.2, size=320)
    classification_y = (x[:, 0] + rng.normal(scale=0.5, size=320) > 0.0).astype(np.float64)

    ridge = _select_model_parameters(
        "ridge", x, regression_y, dates[:320], (0.01,), horizon=21, calendar_dates=dates
    )
    huber = _select_model_parameters(
        "huber", x, regression_y, dates[:320], (0.1,), horizon=21, calendar_dates=dates
    )
    elastic_net = _select_model_parameters(
        "elastic_net",
        x,
        regression_y,
        dates[:320],
        (1.0,),
        horizon=21,
        calendar_dates=dates,
        l1_ratios=(0.9,),
    )
    logistic = _select_model_parameters(
        "logistic", x, classification_y, dates[:320], (10.0,), horizon=21, calendar_dates=dates
    )

    assert ridge["selected_alpha"] == 0.01
    assert huber["selected_alpha"] == 0.1
    assert elastic_net["selected_alpha"] == 1.0
    assert elastic_net["selected_l1_ratio"] == 0.9
    assert logistic["selected_alpha"] == 10.0
    standard_keys = {
        "selected_alpha",
        "inner_validation_score",
        "selected_inner_fold_scores",
        "selected_at_grid_boundary",
    }
    assert set(ridge) == standard_keys
    assert set(elastic_net) == standard_keys | {"selected_l1_ratio"}
    for selection in (ridge, huber, elastic_net, logistic):
        assert selection["inner_validation_score"] is not None
        assert len(selection["selected_inner_fold_scores"]) == 3
        assert selection["selected_at_grid_boundary"] is True
        assert "candidate_diagnostics" not in selection
        assert "selection_status" not in selection


def test_parameter_grid_candidates_are_only_retained_for_debug_reports() -> None:
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2017-01-01", periods=360).to_numpy()
    x = rng.normal(size=(320, 3))
    y = x[:, 0] + rng.normal(scale=0.2, size=320)

    selection = _select_model_parameters(
        "ridge",
        x,
        y,
        dates[:320],
        (0.01, 0.1, 1.0),
        horizon=21,
        calendar_dates=dates,
        include_debug_diagnostics=True,
    )

    assert selection["selection_status"] == "selected"
    assert len(selection["candidate_diagnostics"]) == 3
    assert all(
        len(candidate["fold_scores"]) == 3
        for candidate in selection["candidate_diagnostics"]
    )


def test_ridge_penalty_is_stable_when_samples_are_duplicated() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(80, 4))
    y = x @ np.asarray([0.8, -0.4, 0.2, 0.0]) + rng.normal(scale=0.1, size=80)
    validation_x = rng.normal(size=(20, 4))

    prediction, coefficients = _ridge_predict(x, y, validation_x, alpha=0.3)
    duplicated_prediction, duplicated_coefficients = _ridge_predict(
        np.repeat(x, 3, axis=0),
        np.repeat(y, 3),
        validation_x,
        alpha=0.3,
    )

    np.testing.assert_allclose(duplicated_coefficients, coefficients, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(duplicated_prediction, prediction, rtol=1.0e-10, atol=1.0e-10)


def test_insufficient_inner_history_is_reported_as_fallback() -> None:
    dates = pd.bdate_range("2024-01-01", periods=30).to_numpy()
    x = np.arange(60, dtype=np.float64).reshape(30, 2)
    selection = _select_model_parameters(
        "ridge", x, x[:, 0], dates, (0.1, 1.0), horizon=21, calendar_dates=dates
    )

    assert selection["inner_validation_score"] is None
    assert selection["selected_inner_fold_scores"] == []
    assert selection["selected_at_grid_boundary"] is True
    assert "candidate_diagnostics" not in selection
    assert "selection_status" not in selection
