from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from fi_jepa.probes.artifacts import (
    artifact_destination,
    clean_temporary_artifact,
    publish_artifact,
)
from fi_jepa.probes.dataset import assemble_probe_dataset, build_probe_dataset, load_probe_dataset
from fi_jepa.probes.summary import build_summary_markdown
from fi_jepa.probes.targets import export_probe_targets
from fi_jepa.probes.target_transforms import (
    TARGET_TRANSFORM_EPSILON,
    inverse_transform_predictions,
    target_transform_specs,
    transform_target_values,
)

DEFAULT_RIDGE_ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_HUBER_ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
DEFAULT_ELASTIC_NET_ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0)
DEFAULT_ELASTIC_NET_L1_RATIOS = (0.1, 0.5, 0.9)
DEFAULT_LOGISTIC_ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
TRAIN_MEAN_BASELINE = "train_mean"
TRAILING_TARGET_PROXY_BASELINE = "trailing_target_proxy"
Z_ONLY_FAMILY = "z_only"
HAND_FEATURE_FAMILY = "hand_market_features"
HAND_PCA_FAMILY = "hand_market_pca"
HAND_PLUS_Z_FAMILY = "hand_market_features_plus_z"
HAND_PLUS_RESIDUAL_Z_FAMILY = "hand_market_features_plus_residual_z"
FEATURE_RESIDUALIZED_Z_FAMILY = "feature_residualized_z_only"
REGRESSION_HEADS = ("ridge", "huber", "elastic_net")
CLASSIFICATION_QUANTILE = 0.8
DEFAULT_BOOTSTRAP_SAMPLES = 500
DEFAULT_BOOTSTRAP_SEED = 1729


# ============================================================================
# REGRESSION AND DIAGNOSTIC METRICS
# ============================================================================


def _standardized_train_validation(
    train_x: np.ndarray,
    validation_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Impute and standardize feature matrices using train-only statistics."""
    train = np.asarray(train_x, dtype=np.float64)
    validation = np.asarray(validation_x, dtype=np.float64)
    finite_train = np.isfinite(train)
    usable = finite_train.any(axis=0)
    if not usable.any():
        raise ValueError("No usable finite features are available for this fold.")

    train = train[:, usable]
    validation = validation[:, usable]
    medians = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(train), train, medians)
    validation = np.where(np.isfinite(validation), validation, medians)

    x_mean = train.mean(axis=0)
    x_std = train.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    return (train - x_mean) / x_std, (validation - x_mean) / x_std, x_mean, x_std


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one train-standardized ridge model and predict validation rows."""
    standardized_train, standardized_validation, _, _ = _standardized_train_validation(
        train_x, validation_x
    )
    y_mean = float(train_y.mean())
    centered_y = train_y - y_mean

    n_samples = max(len(centered_y), 1)
    gram = standardized_train.T @ standardized_train / n_samples

    coefficients = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
        standardized_train.T @ centered_y / n_samples,
    )
    return standardized_validation @ coefficients + y_mean, coefficients


def _huber_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a small deterministic Huber ridge model with train-standardized features."""
    standardized_train, standardized_validation, _, _ = _standardized_train_validation(
        train_x, validation_x
    )
    y_mean = float(train_y.mean())
    centered_y = train_y - y_mean
    coefficients = np.zeros(standardized_train.shape[1], dtype=np.float64)
    scale = max(float(np.std(centered_y)), 1.0e-12)
    delta = 1.345 * scale

    for _ in range(30):
        residual = centered_y - standardized_train @ coefficients
        abs_residual = np.abs(residual)
        weights = np.ones_like(abs_residual)
        large = abs_residual > delta
        weights[large] = delta / np.maximum(abs_residual[large], 1.0e-12)
        weighted_x = standardized_train * weights[:, None]
        n_samples = max(len(centered_y), 1)
        coefficients = np.linalg.solve(
            standardized_train.T @ weighted_x / n_samples + alpha * np.eye(standardized_train.shape[1]),
            weighted_x.T @ centered_y / n_samples,
        )
    return standardized_validation @ coefficients + y_mean, coefficients


def _soft_threshold(value: float, penalty: float) -> float:
    """Apply scalar soft-thresholding for coordinate-descent ElasticNet."""
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def _elastic_net_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a compact ElasticNet head with train-standardized features and fixed L1 mix."""
    standardized_train, standardized_validation, _, _ = _standardized_train_validation(
        train_x, validation_x
    )
    y_mean = float(train_y.mean())
    centered_y = train_y - y_mean
    n_samples = max(len(centered_y), 1)
    coefficients = np.zeros(standardized_train.shape[1], dtype=np.float64)
    l1_penalty = alpha * l1_ratio
    l2_penalty = alpha * (1.0 - l1_ratio)
    column_norms = np.mean(np.square(standardized_train), axis=0) + l2_penalty

    for _ in range(100):
        old = coefficients.copy()
        for feature_index in range(standardized_train.shape[1]):
            residual = centered_y - standardized_train @ coefficients
            residual += standardized_train[:, feature_index] * coefficients[feature_index]
            rho = float(np.dot(standardized_train[:, feature_index], residual) / n_samples)
            coefficients[feature_index] = _soft_threshold(rho, l1_penalty) / column_norms[feature_index]
        if np.max(np.abs(coefficients - old)) < 1.0e-8:
            break
    return standardized_validation @ coefficients + y_mean, coefficients


def _fit_pca_projection(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    *,
    max_components: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only PCA for hand-feature baselines and transform train/validation rows."""
    standardized_train, standardized_validation, _, _ = _standardized_train_validation(
        train_x, validation_x
    )
    component_count = min(max_components, standardized_train.shape[1], max(len(standardized_train) - 1, 1))
    _, _, vt = np.linalg.svd(standardized_train, full_matrices=False)
    components = vt[:component_count].T
    return standardized_train @ components, standardized_validation @ components


def _ridge_residualize(
    train_control_x: np.ndarray,
    validation_control_x: np.ndarray,
    train_values: np.ndarray,
    validation_values: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove train-fit linear control-feature predictions from one vector or matrix."""
    train = np.asarray(train_values, dtype=np.float64)
    validation = np.asarray(validation_values, dtype=np.float64)
    if train.ndim == 1:
        predicted_train, _ = _ridge_predict(train_control_x, train, train_control_x, alpha=alpha)
        predicted_validation, _ = _ridge_predict(
            train_control_x, train, validation_control_x, alpha=alpha
        )
        return train - predicted_train, validation - predicted_validation

    train_residuals: list[np.ndarray] = []
    validation_residuals: list[np.ndarray] = []
    for column_index in range(train.shape[1]):
        train_residual, validation_residual = _ridge_residualize(
            train_control_x,
            validation_control_x,
            train[:, column_index],
            validation[:, column_index],
            alpha=alpha,
        )
        train_residuals.append(train_residual)
        validation_residuals.append(validation_residual)
    return np.column_stack(train_residuals), np.column_stack(validation_residuals)


def _incremental_residual_predictions(
    model_name: str,
    hand_train_x: np.ndarray,
    z_train_x: np.ndarray,
    train_y: np.ndarray,
    hand_validation_x: np.ndarray,
    z_validation_x: np.ndarray,
    *,
    hand_alpha: float,
    residual_alpha: float,
    hand_l1_ratio: float | None = None,
    residual_l1_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the outer-train hand model and train-residual z model without validation labels.

    Returns the hand-only validation prediction, predicted validation residual,
    and their reconstructed sum in the model's target space.
    """
    hand_train_prediction, _ = _predict_regression_head(
        model_name,
        hand_train_x,
        train_y,
        hand_train_x,
        alpha=hand_alpha,
        l1_ratio=hand_l1_ratio,
    )
    hand_validation_prediction, _ = _predict_regression_head(
        model_name,
        hand_train_x,
        train_y,
        hand_validation_x,
        alpha=hand_alpha,
        l1_ratio=hand_l1_ratio,
    )
    training_residual = train_y - hand_train_prediction
    residual_validation_prediction, _ = _predict_regression_head(
        model_name,
        z_train_x,
        training_residual,
        z_validation_x,
        alpha=residual_alpha,
        l1_ratio=residual_l1_ratio,
    )
    return (
        hand_validation_prediction,
        residual_validation_prediction,
        hand_validation_prediction + residual_validation_prediction,
    )


def _incremental_comparison_metrics(
    actual: np.ndarray,
    hand_prediction: np.ndarray,
    hand_plus_z_prediction: np.ndarray,
) -> dict[str, float]:
    """Compare matched hand-only and incremental predictions in original target units."""
    hand_metrics = _regression_metrics(actual, hand_prediction)
    incremental_metrics = _regression_metrics(actual, hand_plus_z_prediction)
    hand_rmse = float(hand_metrics["rmse"])
    return {
        "hand_only_rmse": hand_rmse,
        "hand_plus_z_rmse": float(incremental_metrics["rmse"]),
        "delta_rmse": float(incremental_metrics["rmse"]) - hand_rmse,
        "rmse_ratio_vs_hand": (
            float(incremental_metrics["rmse"]) / hand_rmse if hand_rmse > 0.0 else 0.0
        ),
        "hand_only_mae": float(hand_metrics["mae"]),
        "hand_plus_z_mae": float(incremental_metrics["mae"]),
        "delta_mae": float(incremental_metrics["mae"]) - float(hand_metrics["mae"]),
        "hand_only_r2": float(hand_metrics["r2"]),
        "hand_plus_z_r2": float(incremental_metrics["r2"]),
    }


def _predict_regression_head(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch one simple regression head without adding an external ML dependency."""
    if model_name == "ridge":
        return _ridge_predict(train_x, train_y, validation_x, alpha=alpha)
    if model_name == "huber":
        return _huber_predict(train_x, train_y, validation_x, alpha=alpha)
    if model_name == "elastic_net":
        if l1_ratio is None:
            raise ValueError("Elastic Net requires l1_ratio.")
        return _elastic_net_predict(train_x, train_y, validation_x, alpha=alpha, l1_ratio=l1_ratio)
    raise ValueError(f"Unknown regression head: {model_name}")


def _normalise_alpha_grid(alphas: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Validate and de-duplicate a positive regularization grid."""
    unique = tuple(sorted({float(alpha) for alpha in alphas}))
    if not unique:
        raise ValueError("At least one alpha is required.")
    if any(alpha <= 0.0 for alpha in unique):
        raise ValueError("All alphas must be positive.")
    return unique


def _fallback_alpha(alpha_grid: tuple[float, ...]) -> float:
    """Return the conventional fixed ridge alpha when it exists, otherwise the middle grid value."""
    if 1.0 in alpha_grid:
        return 1.0
    return alpha_grid[len(alpha_grid) // 2]


def _purged_training_mask(
    train_dates: np.ndarray,
    *,
    validation_start: pd.Timestamp,
    horizon: int,
    calendar_dates: np.ndarray,
) -> np.ndarray:
    """Keep rows whose h-date future label interval ends before validation starts."""
    dates = pd.DatetimeIndex(pd.to_datetime(train_dates))
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_dates)).sort_values().unique()
    calendar_positions = {pd.Timestamp(date): index for index, date in enumerate(calendar)}
    validation_position = calendar_positions[pd.Timestamp(validation_start)]
    train_positions = np.asarray([calendar_positions[pd.Timestamp(date)] for date in dates])
    return train_positions + horizon < validation_position


def _expanding_time_folds(
    train_dates: np.ndarray,
    *,
    horizon: int,
    calendar_dates: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    """Build exactly three expanding chronological folds with target-horizon purging.

    The available training dates are divided into four contiguous blocks. Each fold
    validates on the next block and trains on the preceding prefix, after excluding
    rows whose h-date future target interval reaches the validation block.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(train_dates))
    unique_dates = dates.sort_values().unique()
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_dates)).sort_values().unique()
    calendar_positions = {pd.Timestamp(date): index for index, date in enumerate(calendar)}
    date_positions = np.asarray([calendar_positions[pd.Timestamp(date)] for date in unique_dates])
    # Leave enough initial history that even the first purged fold retains observations.
    eligible_starts = np.flatnonzero(date_positions > date_positions[0] + horizon + 19)
    if not len(eligible_starts):
        return []
    first_validation_index = int(eligible_starts[0])
    validation_blocks = [block for block in np.array_split(unique_dates[first_validation_index:], 3) if len(block)]
    if len(validation_blocks) != 3:
        return []
    folds: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []
    for fold_index in range(3):
        validation_dates = pd.DatetimeIndex(validation_blocks[fold_index])
        validation_start = pd.Timestamp(validation_dates[0])
        validation_mask = dates.isin(validation_dates)
        # A target observed at position t covers the next h dates, so require t + h < validation start.
        train_mask = _purged_training_mask(
            train_dates,
            validation_start=validation_start,
            horizon=horizon,
            calendar_dates=calendar_dates,
        )
        if train_mask.sum() < 2 or validation_mask.sum() < 1:
            return []
        folds.append(
            (
                np.asarray(train_mask),
                np.asarray(validation_mask),
                {
                    "fold": fold_index + 1,
                    "inner_train_count": int(train_mask.sum()),
                    "inner_validation_count": int(validation_mask.sum()),
                    "inner_train_end": str(pd.Timestamp(dates[train_mask].max()).date()),
                    "inner_validation_start": str(validation_start.date()),
                    "inner_validation_end": str(pd.Timestamp(validation_dates[-1]).date()),
                    "purge_horizon_dates": horizon,
                },
            )
        )
    return folds


def _select_model_parameters(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_dates: np.ndarray,
    alpha_grid: tuple[float, ...],
    *,
    horizon: int,
    calendar_dates: np.ndarray,
    l1_ratios: tuple[float, ...] = (),
    include_debug_diagnostics: bool = False,
) -> dict[str, object]:
    """Select one head's parameters and retain full grid diagnostics only on request."""
    fallback = _fallback_alpha(alpha_grid)
    fallback_ratio = l1_ratios[len(l1_ratios) // 2] if l1_ratios else None
    folds = _expanding_time_folds(train_dates, horizon=horizon, calendar_dates=calendar_dates)
    if not folds:
        return {
            "selected_alpha": fallback,
            **({"selected_l1_ratio": fallback_ratio} if model_name == "elastic_net" else {}),
            "inner_validation_score": None,
            "selected_inner_fold_scores": [],
            "selected_at_grid_boundary": fallback in {alpha_grid[0], alpha_grid[-1]},
            **(
                {
                    "selection_status": "insufficient_history_fallback",
                    "candidate_diagnostics": [],
                }
                if include_debug_diagnostics
                else {}
            ),
        }

    candidates = [(candidate, None) for candidate in alpha_grid]
    if model_name == "elastic_net":
        candidates = [(candidate, ratio) for candidate in alpha_grid for ratio in l1_ratios]
    diagnostics: list[dict[str, object]] = []
    best = (fallback, fallback_ratio)
    best_score = np.inf
    best_fold_scores: list[dict[str, object]] = []
    for candidate_alpha, candidate_ratio in candidates:
        fold_scores: list[dict[str, object]] = []
        for train_mask, validation_mask, fold_metadata in folds:
            if model_name == "logistic":
                predicted, _ = _logistic_predict(
                    train_x[train_mask], train_y[train_mask], train_x[validation_mask], alpha=candidate_alpha
                )
                score = float(_classification_metrics(train_y[validation_mask], predicted)["log_loss"])
            else:
                predicted, _ = _predict_regression_head(
                    model_name,
                    train_x[train_mask],
                    train_y[train_mask],
                    train_x[validation_mask],
                    alpha=candidate_alpha,
                    l1_ratio=candidate_ratio,
                )
                score = float(_regression_metrics(train_y[validation_mask], predicted)["rmse"])
            fold_scores.append({**fold_metadata, "score": score})
        mean_score = float(np.mean([float(fold["score"]) for fold in fold_scores]))
        if include_debug_diagnostics:
            diagnostics.append(
                {
                    "alpha": candidate_alpha,
                    **({"l1_ratio": candidate_ratio} if candidate_ratio is not None else {}),
                    "mean_validation_score": mean_score,
                    "fold_scores": fold_scores,
                }
            )
        if mean_score < best_score:
            best_score = mean_score
            best = (candidate_alpha, candidate_ratio)
            best_fold_scores = fold_scores
    return {
        "selected_alpha": best[0],
        **({"selected_l1_ratio": best[1]} if model_name == "elastic_net" else {}),
        "inner_validation_score": best_score if np.isfinite(best_score) else None,
        "selected_inner_fold_scores": best_fold_scores,
        "selected_at_grid_boundary": (
            best[0] in {alpha_grid[0], alpha_grid[-1]}
            or (model_name == "elastic_net" and best[1] in {l1_ratios[0], l1_ratios[-1]})
        ),
        **(
            {
                "selection_status": "selected",
                "candidate_diagnostics": diagnostics,
            }
            if include_debug_diagnostics
            else {}
        ),
    }


def _correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return a finite Pearson correlation, using zero for constant series."""
    if actual.std() == 0.0 or predicted.std() == 0.0:
        return 0.0
    return float(np.corrcoef(actual, predicted)[0, 1])


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Compute regression, rank-correlation, distribution, and calibration diagnostics."""
    finite = np.isfinite(actual) & np.isfinite(predicted)
    finite_actual = actual[finite]
    finite_predicted = predicted[finite]
    if not len(finite_actual):
        raise ValueError("Cannot score predictions without any finite actual/prediction pairs.")

    residual = finite_actual - finite_predicted
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum(np.square(finite_actual - finite_actual.mean())))
    r2 = 1.0 - float(np.sum(np.square(residual))) / denominator if denominator > 0.0 else 0.0
    actual_std = float(finite_actual.std())
    prediction_std = float(finite_predicted.std())
    actual_ranks = pd.Series(finite_actual).rank(method="average").to_numpy()
    prediction_ranks = pd.Series(finite_predicted).rank(method="average").to_numpy()
    return {
        "scored_count": int(len(finite_actual)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": r2,
        "pearson_correlation": _correlation(finite_actual, finite_predicted),
        "spearman_correlation": _correlation(actual_ranks, prediction_ranks),
        "actual_mean": float(finite_actual.mean()),
        "actual_std": actual_std,
        "prediction_mean": float(finite_predicted.mean()),
        "prediction_std": prediction_std,
        "bias": float(np.mean(finite_predicted - finite_actual)),
        "std_ratio": prediction_std / actual_std if actual_std > 0.0 else 0.0,
    }


def _roc_auc(actual: np.ndarray, probability: np.ndarray) -> float:
    """Calculate binary ROC-AUC from ranks, returning 0.5 when the label is degenerate."""
    positives = actual == 1
    negatives = actual == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if positive_count == 0 or negative_count == 0:
        return 0.5
    ranks = pd.Series(probability).rank(method="average").to_numpy(dtype=np.float64)
    positive_rank_sum = float(ranks[positives].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def _pr_auc(actual: np.ndarray, probability: np.ndarray) -> float:
    """Calculate average precision for binary labels sorted by predicted probability."""
    positives = actual == 1
    positive_count = int(positives.sum())
    if positive_count == 0:
        return 0.0
    order = np.argsort(-probability)
    sorted_actual = actual[order]
    cumulative_positive = np.cumsum(sorted_actual)
    precision = cumulative_positive / (np.arange(len(sorted_actual)) + 1)
    return float((precision * sorted_actual).sum() / positive_count)


def _classification_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    """Compute binary classification diagnostics for regime-label probes."""
    finite = np.isfinite(actual) & np.isfinite(probability)
    finite_actual = actual[finite].astype(np.int64)
    finite_probability = np.clip(probability[finite], 1.0e-12, 1.0 - 1.0e-12)
    if not len(finite_actual):
        raise ValueError("Cannot score classification predictions without finite rows.")

    predicted_label = finite_probability >= 0.5
    positives = finite_actual == 1
    negatives = finite_actual == 0
    positive_accuracy = float((predicted_label[positives] == 1).mean()) if positives.any() else 0.0
    negative_accuracy = float((predicted_label[negatives] == 0).mean()) if negatives.any() else 0.0
    return {
        "scored_count": int(len(finite_actual)),
        "accuracy": float((predicted_label == finite_actual).mean()),
        "balanced_accuracy": (positive_accuracy + negative_accuracy) / 2.0,
        "roc_auc": float(_roc_auc(finite_actual, finite_probability)),
        "pr_auc": float(_pr_auc(finite_actual, finite_probability)),
        "brier_score": float(np.mean(np.square(finite_probability - finite_actual))),
        "log_loss": float(
            -np.mean(
                finite_actual * np.log(finite_probability)
                + (1 - finite_actual) * np.log(1.0 - finite_probability)
            )
        ),
        "class_prevalence": float(finite_actual.mean()),
    }


def _moving_block_bootstrap_indices(
    observation_count: int,
    block_length: int,
    sample_count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Draw fixed-seed moving-block samples while preserving order inside every block."""
    if observation_count <= 0 or block_length <= 0 or sample_count <= 0:
        raise ValueError("Bootstrap observation, block, and sample counts must be positive.")
    effective_block_length = min(block_length, observation_count)
    blocks_per_sample = int(np.ceil(observation_count / effective_block_length))
    maximum_start = observation_count - effective_block_length
    rng = np.random.default_rng(seed)
    samples = np.empty((sample_count, observation_count), dtype=np.int64)
    offsets = np.arange(effective_block_length, dtype=np.int64)
    for sample_index in range(sample_count):
        starts = rng.integers(0, maximum_start + 1, size=blocks_per_sample)
        samples[sample_index] = np.concatenate([start + offsets for start in starts])[:observation_count]
    return samples


def _paired_moving_block_bootstrap(
    actual: np.ndarray,
    predicted: np.ndarray,
    hand_prediction: np.ndarray,
    *,
    task_type: str,
    block_length: int,
    sample_count: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    window_ids: np.ndarray | None = None,
) -> dict[str, object]:
    """Bootstrap aligned candidate/hand predictions, without allowing blocks across windows."""
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    hand_values = np.asarray(hand_prediction, dtype=np.float64)
    if not (actual_values.ndim == predicted_values.ndim == hand_values.ndim == 1):
        raise ValueError("Paired bootstrap inputs must be one-dimensional.")
    if not (len(actual_values) == len(predicted_values) == len(hand_values)):
        raise ValueError("Paired bootstrap inputs must have identical aligned lengths.")
    if window_ids is None:
        windows = np.zeros(len(actual_values), dtype=np.int64)
    else:
        windows = np.asarray(window_ids)
        if windows.ndim != 1 or len(windows) != len(actual_values):
            raise ValueError("Bootstrap window IDs must align with prediction rows.")
    finite = np.isfinite(actual_values) & np.isfinite(predicted_values) & np.isfinite(hand_values)
    actual_values = actual_values[finite]
    predicted_values = predicted_values[finite]
    hand_values = hand_values[finite]
    windows = windows[finite]
    if not len(actual_values):
        raise ValueError("Paired bootstrap requires at least one finite aligned row.")

    # Resample every validation window independently so a moving block never
    # crosses an outer-window boundary in the pooled stability interval.
    segment_positions = [np.flatnonzero(windows == window) for window in pd.unique(windows)]
    segment_samples = [
        _moving_block_bootstrap_indices(
            len(positions), block_length, sample_count, seed=seed + segment_index
        )
        for segment_index, positions in enumerate(segment_positions)
    ]
    sampled_positions = [
        np.concatenate(
            [positions[indices[sample_index]] for positions, indices in zip(segment_positions, segment_samples)]
        )
        for sample_index in range(sample_count)
    ]

    if task_type == "regression":
        candidate_rmse = float(np.sqrt(np.mean(np.square(actual_values - predicted_values))))
        hand_rmse = float(np.sqrt(np.mean(np.square(actual_values - hand_values))))
        point_estimates = {
            "rmse_difference_vs_hand": candidate_rmse - hand_rmse,
            "pearson_correlation": _correlation(actual_values, predicted_values),
        }
        draws = {
            "rmse_difference_vs_hand": np.asarray(
                [
                    np.sqrt(np.mean(np.square(actual_values[index] - predicted_values[index])))
                    - np.sqrt(np.mean(np.square(actual_values[index] - hand_values[index])))
                    for index in sampled_positions
                ]
            ),
            "pearson_correlation": np.asarray(
                [_correlation(actual_values[index], predicted_values[index]) for index in sampled_positions]
            ),
        }
    elif task_type == "classification":
        clipped_prediction = np.clip(predicted_values, 1.0e-12, 1.0 - 1.0e-12)
        clipped_hand = np.clip(hand_values, 1.0e-12, 1.0 - 1.0e-12)
        point_estimates = {
            "roc_auc": float(_roc_auc(actual_values.astype(np.int64), clipped_prediction)),
            "brier_score_difference_vs_hand": float(
                np.mean(np.square(clipped_prediction - actual_values))
                - np.mean(np.square(clipped_hand - actual_values))
            ),
        }
        draws = {
            "roc_auc": np.asarray(
                [
                    _roc_auc(actual_values[index].astype(np.int64), clipped_prediction[index])
                    for index in sampled_positions
                ]
            ),
            "brier_score_difference_vs_hand": np.asarray(
                [
                    np.mean(np.square(clipped_prediction[index] - actual_values[index]))
                    - np.mean(np.square(clipped_hand[index] - actual_values[index]))
                    for index in sampled_positions
                ]
            ),
        }
    else:
        raise ValueError(f"Unsupported bootstrap task type: {task_type!r}")

    return {
        "sample_count": sample_count,
        "block_length": block_length,
        "seed": seed,
        "point_estimates": point_estimates,
        "confidence_intervals_95": {
            name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            for name, values in draws.items()
        },
    }


def _logistic_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic L2 logistic head using train-only standardization."""
    standardized_train, standardized_validation, _, _ = _standardized_train_validation(
        train_x, validation_x
    )
    coefficients = np.zeros(standardized_train.shape[1], dtype=np.float64)
    intercept = float(np.log(np.clip(train_y.mean(), 1.0e-6, 1.0 - 1.0e-6) / np.clip(1.0 - train_y.mean(), 1.0e-6, 1.0)))
    learning_rate = 0.2
    for _ in range(400):
        logits = np.clip(intercept + standardized_train @ coefficients, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = probability - train_y
        intercept -= learning_rate * float(error.mean())
        coefficients -= learning_rate * (
            standardized_train.T @ error / len(train_y) + alpha * coefficients
        )
    validation_logits = np.clip(intercept + standardized_validation @ coefficients, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-validation_logits)), coefficients


def _target_constraint(target_name: str) -> tuple[float | None, float | None, str | None]:
    """Return the legal prediction interval for target families with known physical bounds."""
    if "log_realized_vol" in target_name:
        return None, None, None
    if "log_drawdown_magnitude" in target_name:
        return 0.0, None, "log_drawdown_magnitude_below_zero"
    if "max_drawdown_magnitude" in target_name or "drawdown_magnitude" in target_name:
        return 0.0, None, "drawdown_magnitude_below_zero"
    if "max_drawdown" in target_name:
        return None, 0.0, "max_drawdown_above_zero"
    if "realized_vol" in target_name:
        return 0.0, None, "realized_volatility_below_zero"
    if "probability" in target_name or target_name.endswith("_prob"):
        return 0.0, 1.0, "probability_outside_zero_one"
    # Current trend scores are return / volatility and are not bounded.
    return None, None, None


def _invalid_predictions(
    target_name: str,
    predicted: np.ndarray,
    *,
    score_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag non-finite predictions and enforce physical bounds only in supported spaces."""
    if score_space == "original":
        lower, upper, constraint_reason = _target_constraint(target_name)
    elif score_space == "probability":
        lower, upper, constraint_reason = 0.0, 1.0, "probability_outside_zero_one"
    else:
        lower, upper, constraint_reason = None, None, None
    invalid = ~np.isfinite(predicted)
    reasons = np.full(len(predicted), "", dtype=object)
    reasons[invalid] = "non_finite_prediction"
    constrained_invalid = np.zeros(len(predicted), dtype=bool)
    if lower is not None:
        constrained_invalid |= predicted < lower
    if upper is not None:
        constrained_invalid |= predicted > upper
    constrained_invalid &= np.isfinite(predicted)
    invalid |= constrained_invalid
    if constraint_reason is not None:
        reasons[constrained_invalid] = constraint_reason
    return invalid, reasons


def _score_predictions(
    target_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    score_space: str,
    baseline_metrics: dict[str, float | int] | None,
) -> dict[str, object]:
    """Score one predictor and attach invalidity and baseline-ratio diagnostics."""
    metrics: dict[str, object] = _regression_metrics(actual, predicted)
    invalid, _ = _invalid_predictions(target_name, predicted, score_space=score_space)
    metrics["invalid_prediction_count"] = int(invalid.sum())
    metrics["invalid_prediction_rate"] = float(invalid.mean())
    if baseline_metrics is None:
        metrics["rmse_ratio_vs_baseline"] = 1.0
        metrics["mae_ratio_vs_baseline"] = 1.0
    else:
        baseline_rmse = float(baseline_metrics["rmse"])
        baseline_mae = float(baseline_metrics["mae"])
        metrics["rmse_ratio_vs_baseline"] = (
            float(metrics["rmse"]) / baseline_rmse if baseline_rmse > 0.0 else 0.0
        )
        metrics["mae_ratio_vs_baseline"] = (
            float(metrics["mae"]) / baseline_mae if baseline_mae > 0.0 else 0.0
        )
    return metrics


def _target_horizon(target_name: str) -> int | None:
    """Extract a trailing horizon such as 21 from a canonical target name."""
    match = re.search(r"_(\d+)d(?:__.*)?$", target_name)
    return int(match.group(1)) if match else None


def _trailing_proxy_column(raw_target: str) -> str | None:
    """Map one future target to the matching past-only trailing proxy column."""
    horizon = _target_horizon(raw_target)
    if horizon is None:
        return None
    if "future_return" in raw_target:
        return f"baseline__trailing_return_{horizon}d"
    if "future_realized_vol" in raw_target:
        return f"baseline__trailing_realized_vol_{horizon}d"
    if "future_max_drawdown" in raw_target:
        return f"baseline__trailing_max_drawdown_{horizon}d"
    if "future_trend_score" in raw_target:
        return f"baseline__current_trend_score_{horizon}d"
    return None


def _feature_families(
    target_train: pd.DataFrame,
    target_validation: pd.DataFrame,
    *,
    z_columns: list[str],
    baseline_feature_columns: list[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build Phase 3 feature-family matrices from a fold without leaking validation labels."""
    families: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            Z_ONLY_FAMILY,
            target_train[z_columns].to_numpy(dtype=np.float64),
            target_validation[z_columns].to_numpy(dtype=np.float64),
        )
    ]
    if baseline_feature_columns:
        hand_train = target_train[baseline_feature_columns].to_numpy(dtype=np.float64)
        hand_validation = target_validation[baseline_feature_columns].to_numpy(dtype=np.float64)
        families.append((HAND_FEATURE_FAMILY, hand_train, hand_validation))
        try:
            pca_train, pca_validation = _fit_pca_projection(hand_train, hand_validation)
            families.append((HAND_PCA_FAMILY, pca_train, pca_validation))
        except ValueError:
            pass
        families.append(
            (
                HAND_PLUS_Z_FAMILY,
                target_train[[*baseline_feature_columns, *z_columns]].to_numpy(dtype=np.float64),
                target_validation[[*baseline_feature_columns, *z_columns]].to_numpy(dtype=np.float64),
            )
        )
    return families


def _classification_label_frame(
    train_values: np.ndarray,
    validation_values: np.ndarray,
    raw_target: str,
) -> list[dict[str, object]]:
    """Construct train-thresholded binary regime labels for one raw target."""
    finite_train = train_values[np.isfinite(train_values)]
    if len(finite_train) < 3:
        return []
    labels: list[dict[str, object]] = []
    horizon = _target_horizon(raw_target)
    suffix = f"_{horizon}d" if horizon is not None else ""

    if "future_realized_vol" in raw_target:
        threshold = float(np.quantile(finite_train, CLASSIFICATION_QUANTILE))
        labels.append(
            {
                "classification_label": f"high_vol{suffix}",
                "threshold": threshold,
                "positive_rule": f"{raw_target} >= train_q{CLASSIFICATION_QUANTILE}",
                "train_y": (train_values >= threshold).astype(np.float64),
                "validation_y": (validation_values >= threshold).astype(np.float64),
            }
        )
    elif "future_max_drawdown" in raw_target:
        train_magnitude = np.maximum(-finite_train, 0.0)
        threshold = float(np.quantile(train_magnitude, CLASSIFICATION_QUANTILE))
        labels.append(
            {
                "classification_label": f"severe_drawdown{suffix}",
                "threshold": threshold,
                "positive_rule": f"-{raw_target} >= train_q{CLASSIFICATION_QUANTILE}",
                "train_y": (np.maximum(-train_values, 0.0) >= threshold).astype(np.float64),
                "validation_y": (np.maximum(-validation_values, 0.0) >= threshold).astype(np.float64),
            }
        )
    elif "future_return" in raw_target:
        labels.append(
            {
                "classification_label": f"positive_return{suffix}",
                "threshold": 0.0,
                "positive_rule": f"{raw_target} > 0",
                "train_y": (train_values > 0.0).astype(np.float64),
                "validation_y": (validation_values > 0.0).astype(np.float64),
            }
        )
    elif "future_trend_score" in raw_target:
        strong_threshold = float(np.quantile(finite_train, CLASSIFICATION_QUANTILE))
        weak_threshold = float(np.quantile(np.abs(finite_train), 1.0 - CLASSIFICATION_QUANTILE))
        labels.extend(
            [
                {
                    "classification_label": f"strong_trend{suffix}",
                    "threshold": strong_threshold,
                    "positive_rule": f"{raw_target} >= train_q{CLASSIFICATION_QUANTILE}",
                    "train_y": (train_values >= strong_threshold).astype(np.float64),
                    "validation_y": (validation_values >= strong_threshold).astype(np.float64),
                },
                {
                    "classification_label": f"weak_or_chop{suffix}",
                    "threshold": weak_threshold,
                    "positive_rule": f"abs({raw_target}) <= train_q{1.0 - CLASSIFICATION_QUANTILE}",
                    "train_y": (np.abs(train_values) <= weak_threshold).astype(np.float64),
                    "validation_y": (np.abs(validation_values) <= weak_threshold).astype(np.float64),
                },
            ]
        )
    return labels


# ============================================================================
# FROZEN RIDGE PROBES
# ============================================================================


def _bootstrap_reliability_summary(
    predictions: pd.DataFrame,
    *,
    representation_variant: str,
    sample_count: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    """Build per-window paired CIs and cross-window stability for representation probes.

    Every candidate is compared with the same model head trained on hand features.
    Candidate and hand rows are joined one-to-one by date before any resampling.
    Hand-only and hand-PCA rows are comparators, not evaluated representation
    combinations, so they are excluded from the combination count.
    """
    model_rows = predictions.loc[
        predictions["predictor_kind"].eq("model")
        & ~predictions["feature_family"].isin({HAND_FEATURE_FAMILY, HAND_PCA_FAMILY})
    ].copy()
    per_window: list[dict[str, object]] = []
    aligned_by_combination: dict[tuple[str, ...], list[pd.DataFrame]] = {}
    group_columns = [
        "validation_window_name",
        "raw_target",
        "target",
        "task_type",
        "score_space",
        "model_name",
        "feature_family",
        "predictor_name",
    ]
    for keys, candidate in model_rows.groupby(group_columns, sort=True):
        (
            window_name,
            raw_target,
            target,
            task_type,
            score_space,
            model_name,
            feature_family,
            predictor_name,
        ) = (str(value) for value in keys)
        hand_predictor_name = f"{model_name}__{HAND_FEATURE_FAMILY}"
        hand = predictions.loc[
            predictions["validation_window_name"].eq(window_name)
            & predictions["target"].eq(target)
            & predictions["task_type"].eq(task_type)
            & predictions["score_space"].eq(score_space)
            & predictions["predictor_name"].eq(hand_predictor_name),
            ["date", "actual", "prediction"],
        ]
        if hand.empty:
            continue
        aligned = candidate[["date", "actual", "prediction"]].merge(
            hand,
            on="date",
            how="inner",
            suffixes=("_candidate", "_hand"),
            validate="one_to_one",
        ).sort_values("date")
        if aligned.empty or not np.allclose(
            aligned["actual_candidate"], aligned["actual_hand"], equal_nan=True
        ):
            raise ValueError(
                f"Candidate and hand predictions are not target-aligned for {predictor_name!r}."
            )
        horizon = _target_horizon(raw_target)
        if horizon is None:
            raise ValueError(f"Bootstrap target has no parseable horizon: {raw_target!r}")
        bootstrap = _paired_moving_block_bootstrap(
            aligned["actual_candidate"].to_numpy(dtype=np.float64),
            aligned["prediction_candidate"].to_numpy(dtype=np.float64),
            aligned["prediction_hand"].to_numpy(dtype=np.float64),
            task_type=task_type,
            block_length=horizon,
            sample_count=sample_count,
            seed=seed,
        )
        identity = {
            "validation_window_name": window_name,
            "raw_target": raw_target,
            "target": target,
            "task_type": task_type,
            "score_space": score_space,
            "model_name": model_name,
            "feature_family": feature_family,
            "predictor_name": predictor_name,
            "representation_variant": representation_variant,
            "comparison_predictor": hand_predictor_name,
            "aligned_row_count": int(len(aligned)),
        }
        per_window.append({**identity, **bootstrap})
        combination_key = (
            raw_target,
            target,
            task_type,
            score_space,
            model_name,
            feature_family,
            predictor_name,
        )
        aligned_by_combination.setdefault(combination_key, []).append(
            aligned.assign(validation_window_name=window_name)
        )

    stability: list[dict[str, object]] = []
    for combination_key, windows in sorted(aligned_by_combination.items()):
        raw_target, target, task_type, score_space, model_name, feature_family, predictor_name = combination_key
        pooled = pd.concat(windows, ignore_index=True).sort_values(
            ["validation_window_name", "date"]
        )
        horizon = _target_horizon(raw_target)
        if horizon is None:
            continue
        pooled_bootstrap = _paired_moving_block_bootstrap(
            pooled["actual_candidate"].to_numpy(dtype=np.float64),
            pooled["prediction_candidate"].to_numpy(dtype=np.float64),
            pooled["prediction_hand"].to_numpy(dtype=np.float64),
            task_type=task_type,
            block_length=horizon,
            sample_count=sample_count,
            seed=seed,
            window_ids=pooled["validation_window_name"].to_numpy(),
        )
        matching_windows = [
            row
            for row in per_window
            if all(
                row[name] == value
                for name, value in {
                    "raw_target": raw_target,
                    "target": target,
                    "task_type": task_type,
                    "score_space": score_space,
                    "model_name": model_name,
                    "feature_family": feature_family,
                    "predictor_name": predictor_name,
                }.items()
            )
        ]
        if task_type == "regression":
            primary_metric = "rmse_difference_vs_hand"
            correlation_signs = [
                {
                    "validation_window_name": str(row["validation_window_name"]),
                    "sign": (
                        "positive"
                        if float(row["point_estimates"]["pearson_correlation"]) > 0.0
                        else "negative"
                        if float(row["point_estimates"]["pearson_correlation"]) < 0.0
                        else "zero"
                    ),
                }
                for row in matching_windows
            ]
        else:
            primary_metric = "brier_score_difference_vs_hand"
            correlation_signs = []
        primary_values = [float(row["point_estimates"][primary_metric]) for row in matching_windows]
        stability.append(
            {
                "raw_target": raw_target,
                "target": target,
                "task_type": task_type,
                "score_space": score_space,
                "model_name": model_name,
                "feature_family": feature_family,
                "predictor_name": predictor_name,
                "representation_variant": representation_variant,
                "comparison_predictor": f"{model_name}__{HAND_FEATURE_FAMILY}",
                "window_count": len(matching_windows),
                "correlation_signs": correlation_signs,
                "windows_beating_hand": int(sum(value < 0.0 for value in primary_values)),
                "primary_metric": primary_metric,
                "worst_window_metric": float(max(primary_values)),
                "median_metric": float(np.median(primary_values)),
                "bootstrap_interval_95": pooled_bootstrap["confidence_intervals_95"][primary_metric],
                "pooled_bootstrap": pooled_bootstrap,
            }
        )
    return per_window, stability, len(stability)


def _window_summaries(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize target-level result rows without mixing target scales."""
    summaries: list[dict[str, object]] = []
    result_frame = pd.DataFrame(results)
    result_frame = result_frame.loc[result_frame["task_type"].eq("regression")]
    if result_frame.empty:
        return summaries
    group_columns = [
        "validation_window_name",
        "target_transform",
        "score_space",
        "predictor_name",
    ]
    for keys, frame in result_frame.groupby(group_columns, sort=True):
        window_name, target_transform, score_space, predictor_name = keys
        summaries.append(
            {
                "validation_window_name": str(window_name),
                "target_transform": str(target_transform),
                "score_space": str(score_space),
                "predictor_name": str(predictor_name),
                "predictor_kind": str(frame["predictor_kind"].iloc[0]),
                "targets_scored": int(len(frame)),
                "median_r2": float(frame["r2"].median()),
                "median_rmse_ratio_vs_baseline": float(
                    frame["rmse_ratio_vs_baseline"].median()
                ),
                "median_mae_ratio_vs_baseline": float(
                    frame["mae_ratio_vs_baseline"].median()
                ),
                "mean_pearson_correlation": float(frame["pearson_correlation"].mean()),
                "mean_spearman_correlation": float(frame["spearman_correlation"].mean()),
                "invalid_prediction_count": int(frame["invalid_prediction_count"].sum()),
                "invalid_prediction_rate": float(
                    frame["invalid_prediction_count"].sum() / frame["validation_count"].sum()
                ),
                "inverse_prediction_clip_count": int(
                    frame["inverse_prediction_clip_count"].sum()
                ),
                "inverse_prediction_clip_rate": float(
                    frame["inverse_prediction_clip_count"].sum()
                    / frame["validation_count"].sum()
                ),
            }
        )
    return summaries


def _regression_final_summary(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize each regression model against the strongest simple same-window baseline."""
    frame = pd.DataFrame(results)
    if frame.empty or "task_type" not in frame:
        return []
    frame = frame.loc[frame["task_type"].eq("regression")].copy()
    if frame.empty:
        return []

    baseline_families = {"constant", "trailing_proxy", HAND_FEATURE_FAMILY, HAND_PCA_FAMILY}
    baseline_rows = frame.loc[frame["feature_family"].isin(baseline_families)]
    strongest: dict[tuple[str, str, str], float] = {}
    for keys, group in baseline_rows.groupby(["validation_window_name", "target", "score_space"]):
        strongest[(str(keys[0]), str(keys[1]), str(keys[2]))] = float(group["rmse"].min())

    summaries: list[dict[str, object]] = []
    group_columns = ["target", "score_space", "model_name", "feature_family"]
    for keys, group in frame.groupby(group_columns, sort=True):
        target, score_space, model_name, feature_family = (str(value) for value in keys)
        ratios: list[float] = []
        beaten_windows = 0
        for row in group.itertuples(index=False):
            baseline_rmse = strongest.get(
                (str(row.validation_window_name), str(row.target), str(row.score_space))
            )
            if baseline_rmse is None or baseline_rmse <= 0.0:
                continue
            ratio = float(row.rmse) / baseline_rmse
            ratios.append(ratio)
            if ratio < 1.0:
                beaten_windows += 1
        window_count = int(group["validation_window_name"].nunique())
        mean_correlation = float(group["pearson_correlation"].mean())
        median_ratio = float(np.median(ratios)) if ratios else float("nan")
        invalid_rate = float(
            group["invalid_prediction_count"].sum() / group["validation_count"].sum()
        )
        summaries.append(
            {
                "task_type": "regression",
                "target": target,
                "target_family": _target_family(target),
                "horizon_days": _target_horizon(target),
                "score_space": score_space,
                "model_name": model_name,
                "feature_family": feature_family,
                "window_count": window_count,
                "mean_r2": float(group["r2"].mean()),
                "median_r2": float(group["r2"].median()),
                "worst_window_r2": float(group["r2"].min()),
                "mean_pearson_correlation": mean_correlation,
                "mean_spearman_correlation": float(group["spearman_correlation"].mean()),
                "correlation_stability_positive_windows": int(
                    (group["pearson_correlation"] > 0.0).sum()
                ),
                "windows_beating_strongest_baseline": beaten_windows,
                "median_rmse_ratio_vs_strongest_baseline": median_ratio,
                "invalid_prediction_rate": invalid_rate,
                "gate": _regression_gate(
                    window_count=window_count,
                    beaten_windows=beaten_windows,
                    mean_r2=float(group["r2"].mean()),
                    median_ratio=median_ratio,
                    mean_correlation=mean_correlation,
                    invalid_rate=invalid_rate,
                ),
            }
        )
    return summaries


def _classification_final_summary(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize classification heads against the class-prior baseline."""
    frame = pd.DataFrame(results)
    if frame.empty or "task_type" not in frame:
        return []
    frame = frame.loc[frame["task_type"].eq("classification")].copy()
    if frame.empty:
        return []

    prior: dict[tuple[str, str], float] = {}
    prior_rows = frame.loc[frame["predictor_name"].eq("class_prior")]
    for keys, group in prior_rows.groupby(["validation_window_name", "classification_label"]):
        prior[(str(keys[0]), str(keys[1]))] = float(group["brier_score"].iloc[0])

    summaries: list[dict[str, object]] = []
    group_columns = ["classification_label", "model_name", "feature_family"]
    for keys, group in frame.groupby(group_columns, sort=True):
        label, model_name, feature_family = (str(value) for value in keys)
        ratios: list[float] = []
        beaten_windows = 0
        for row in group.itertuples(index=False):
            baseline_brier = prior.get((str(row.validation_window_name), str(row.classification_label)))
            if baseline_brier is None or baseline_brier <= 0.0:
                continue
            ratio = float(row.brier_score) / baseline_brier
            ratios.append(ratio)
            if ratio < 1.0:
                beaten_windows += 1
        window_count = int(group["validation_window_name"].nunique())
        median_ratio = float(np.median(ratios)) if ratios else float("nan")
        summaries.append(
            {
                "task_type": "classification",
                "classification_label": label,
                "model_name": model_name,
                "feature_family": feature_family,
                "window_count": window_count,
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_roc_auc": float(group["roc_auc"].mean()),
                "mean_pr_auc": float(group["pr_auc"].mean()),
                "mean_brier_score": float(group["brier_score"].mean()),
                "windows_beating_class_prior": beaten_windows,
                "median_brier_ratio_vs_class_prior": median_ratio,
                "gate": _classification_gate(
                    window_count=window_count,
                    beaten_windows=beaten_windows,
                    mean_roc_auc=float(group["roc_auc"].mean()),
                    median_ratio=median_ratio,
                ),
            }
        )
    return summaries


def _target_family(target_name: str) -> str:
    """Collapse target variants into a stable family label for report summaries."""
    if "future_realized_vol" in target_name:
        return "future_realized_vol"
    if "future_max_drawdown" in target_name:
        return "future_max_drawdown"
    if "future_return" in target_name:
        return "future_return"
    if "future_trend_score" in target_name:
        return "future_trend_score"
    return "unknown"


def _regression_gate(
    *,
    window_count: int,
    beaten_windows: int,
    mean_r2: float,
    median_ratio: float,
    mean_correlation: float,
    invalid_rate: float,
) -> str:
    """Assign a conservative Phase 4 pass/fail label to one regression summary."""
    required_windows = min(2, window_count)
    if (
        beaten_windows >= required_windows
        and mean_correlation > 0.0
        and np.isfinite(median_ratio)
        and median_ratio < 1.0
        and invalid_rate <= 0.01
    ):
        return "promising"
    if mean_r2 < -0.1 and beaten_windows == 0:
        return "failure"
    return "weak"


def _classification_gate(
    *,
    window_count: int,
    beaten_windows: int,
    mean_roc_auc: float,
    median_ratio: float,
) -> str:
    """Assign a conservative Phase 4 pass/fail label to one classification summary."""
    required_windows = min(2, window_count)
    if beaten_windows >= required_windows and mean_roc_auc > 0.55 and median_ratio < 1.0:
        return "promising"
    if beaten_windows == 0 and mean_roc_auc <= 0.5:
        return "failure"
    return "weak"


def _gate_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    """Count pass/fail gate labels using JSON-native integer values."""
    counts: dict[str, int] = {}
    for row in rows:
        gate = str(row["gate"])
        counts[gate] = counts.get(gate, 0) + 1
    return counts


def _append_scored_result(
    *,
    results: list[dict[str, object]],
    prediction_rows: list[pd.DataFrame],
    common: dict[str, object],
    target_validation: pd.DataFrame,
    score_space: str,
    score_target_name: str,
    predictor_name: str,
    predictor_kind: str,
    model_name: str,
    feature_family: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline_metrics: dict[str, object] | None,
    inverse_clipped: np.ndarray | None = None,
) -> dict[str, object]:
    """Score and store one predictor/result block."""
    metrics = _score_predictions(
        score_target_name,
        actual,
        predicted,
        score_space=score_space,
        baseline_metrics=baseline_metrics,
    )
    if inverse_clipped is None:
        inverse_clipped = np.zeros(len(predicted), dtype=bool)
    metrics["inverse_prediction_clip_count"] = int(inverse_clipped.sum())
    metrics["inverse_prediction_clip_rate"] = float(inverse_clipped.mean())
    results.append(
        {
            **common,
            "task_type": "regression",
            "score_space": score_space,
            "score_target_name": score_target_name,
            "predictor_name": predictor_name,
            "predictor_kind": predictor_kind,
            "model_name": model_name,
            "feature_family": feature_family,
            **metrics,
        }
    )
    invalid, invalid_reasons = _invalid_predictions(
        score_target_name, predicted, score_space=score_space
    )
    prediction_rows.append(
        pd.DataFrame(
            {
                "date": target_validation["date"].to_numpy(),
                "validation_window_name": common["validation_window_name"],
                "raw_target": common["raw_target"],
                "target": common["target"],
                "target_transform": common["target_transform"],
                "task_type": "regression",
                "score_space": score_space,
                "score_target_name": score_target_name,
                "horizon_days": common["horizon_days"],
                "predictor_name": predictor_name,
                "predictor_kind": predictor_kind,
                "model_name": model_name,
                "feature_family": feature_family,
                "comparison_baseline": common["comparison_baseline"],
                "selected_alpha": common["selected_alpha"],
                "alpha_selection": common["alpha_selection"],
                "actual": actual,
                "prediction": predicted,
                "invalid_prediction": invalid,
                "invalid_prediction_reason": invalid_reasons,
                "inverse_prediction_clipped": inverse_clipped,
            }
        )
    )
    return metrics


def _append_classification_result(
    *,
    results: list[dict[str, object]],
    prediction_rows: list[pd.DataFrame],
    common: dict[str, object],
    target_validation: pd.DataFrame,
    classification_label: str,
    positive_rule: str,
    threshold: float,
    predictor_name: str,
    predictor_kind: str,
    model_name: str,
    feature_family: str,
    actual: np.ndarray,
    probability: np.ndarray,
    baseline_metrics: dict[str, object] | None,
) -> dict[str, object]:
    """Score and store one binary regime-label prediction block."""
    metrics: dict[str, object] = _classification_metrics(actual, probability)
    if baseline_metrics is None:
        metrics["brier_ratio_vs_baseline"] = 1.0
        metrics["log_loss_ratio_vs_baseline"] = 1.0
    else:
        baseline_brier = float(baseline_metrics["brier_score"])
        baseline_log_loss = float(baseline_metrics["log_loss"])
        metrics["brier_ratio_vs_baseline"] = (
            float(metrics["brier_score"]) / baseline_brier if baseline_brier > 0.0 else 0.0
        )
        metrics["log_loss_ratio_vs_baseline"] = (
            float(metrics["log_loss"]) / baseline_log_loss if baseline_log_loss > 0.0 else 0.0
        )
    results.append(
        {
            **common,
            "task_type": "classification",
            "classification_label": classification_label,
            "positive_rule": positive_rule,
            "threshold": threshold,
            "score_space": "probability",
            "score_target_name": classification_label,
            "predictor_name": predictor_name,
            "predictor_kind": predictor_kind,
            "model_name": model_name,
            "feature_family": feature_family,
            **metrics,
        }
    )
    prediction_rows.append(
        pd.DataFrame(
            {
                "date": target_validation["date"].to_numpy(),
                "validation_window_name": common["validation_window_name"],
                "raw_target": common["raw_target"],
                "target": classification_label,
                "target_transform": "binary_label",
                "task_type": "classification",
                "score_space": "probability",
                "score_target_name": classification_label,
                "horizon_days": common["horizon_days"],
                "predictor_name": predictor_name,
                "predictor_kind": predictor_kind,
                "model_name": model_name,
                "feature_family": feature_family,
                "comparison_baseline": common["comparison_baseline"],
                "selected_alpha": common["selected_alpha"],
                "alpha_selection": common["alpha_selection"],
                "actual": actual,
                "prediction": probability,
                "invalid_prediction": ~np.isfinite(probability),
                "invalid_prediction_reason": np.where(
                    np.isfinite(probability), "", "non_finite_prediction"
                ),
                "inverse_prediction_clipped": False,
            }
        )
    )
    return metrics


def run_frozen_probes(
    embedding_artifact: Path | None = None,
    target_artifact: Path | None = None,
    *,
    probe_dataset_artifact: Path | None = None,
    output_root: Path = Path("runs/probes"),
    ridge_alphas: tuple[float, ...] | list[float] = DEFAULT_RIDGE_ALPHAS,
    huber_alphas: tuple[float, ...] | list[float] = DEFAULT_HUBER_ALPHAS,
    elastic_net_alphas: tuple[float, ...] | list[float] = DEFAULT_ELASTIC_NET_ALPHAS,
    elastic_net_l1_ratios: tuple[float, ...] | list[float] = DEFAULT_ELASTIC_NET_L1_RATIOS,
    logistic_alphas: tuple[float, ...] | list[float] = DEFAULT_LOGISTIC_ALPHAS,
    representation_variant: str | None = None,
    include_debug_diagnostics: bool = False,
) -> Path:
    """Run frozen probes, omitting large grid and coefficient diagnostics by default."""
    model_alpha_grids = {
        "ridge": _normalise_alpha_grid(ridge_alphas),
        "huber": _normalise_alpha_grid(huber_alphas),
        "elastic_net": _normalise_alpha_grid(elastic_net_alphas),
        "logistic": _normalise_alpha_grid(logistic_alphas),
    }
    l1_ratios = tuple(sorted({float(value) for value in elastic_net_l1_ratios}))
    if not l1_ratios or any(value <= 0.0 or value > 1.0 for value in l1_ratios):
        raise ValueError("Elastic Net l1 ratios must be in (0, 1].")
    alpha_selection = "three_fold_expanding_purged"

    if probe_dataset_artifact is not None:
        if embedding_artifact is not None or target_artifact is not None:
            raise ValueError("Use either a probe dataset or embedding/target artifacts, not both.")
        probe_dataset, metadata = load_probe_dataset(probe_dataset_artifact)
        source_database = metadata["source_database_sha256"]
        target_columns = [str(name) for name in metadata["target_columns"]]
        z_columns = [str(name) for name in metadata["z_columns"]]
        baseline_feature_columns = [str(name) for name in metadata.get("baseline_feature_columns", [])]
        embedding_source = metadata["embedding_artifact"]
        target_source = metadata["target_artifact"]
        checkpoint_id = metadata["checkpoint_id"]
        checkpoint_step = metadata["checkpoint_step"]
        representation_source = metadata["representation_source"]
        dataset_variant = str(metadata["representation_variant"])
        if representation_variant is not None and representation_variant != dataset_variant:
            raise ValueError(
                f"Probe dataset contains {dataset_variant!r}, not requested variant {representation_variant!r}."
            )
        representation_variant = dataset_variant
        resolved_representation_config = metadata["resolved_representation_config"]
    else:
        if embedding_artifact is None or target_artifact is None:
            raise ValueError("Embedding and target artifacts are both required.")
        probe_dataset, metadata = assemble_probe_dataset(
            embedding_artifact, target_artifact, representation_variant=representation_variant
        )
        source_database = metadata["source_database_sha256"]
        target_columns = [str(name) for name in metadata["target_columns"]]
        z_columns = [str(name) for name in metadata["z_columns"]]
        baseline_feature_columns = [str(name) for name in metadata.get("baseline_feature_columns", [])]
        embedding_source = str(embedding_artifact.resolve())
        target_source = str(target_artifact.resolve())
        embedding_manifest = metadata["embedding_manifest"]
        checkpoint_id = embedding_manifest["checkpoint_id"]
        checkpoint_step = embedding_manifest["checkpoint_step"]
        representation_source = metadata["representation_source"]
        representation_variant = metadata["representation_variant"]
        resolved_representation_config = embedding_manifest["resolved_representation_config"]

    representation_diagnostics: dict[str, object] = {}
    diagnostics_path = Path(embedding_source) / "diagnostics.json"
    if diagnostics_path.is_file():
        representation_report = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        representation_diagnostics = dict(
            representation_report.get("validation_rank_diagnostics", {})
        )

    target_specs = target_transform_specs(target_columns)
    transformed_target_names = [spec.target for spec in target_specs]

    validation = probe_dataset.loc[probe_dataset["split"].eq("validation")].copy()
    train = probe_dataset.loc[probe_dataset["split"].eq("train")].copy()
    calendar_dates = probe_dataset["date"].sort_values().unique()
    windows = sorted(str(name) for name in validation["validation_window_name"].unique() if name)

    results: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    coefficients_by_fold: list[dict[str, object]] = []
    parameter_selection_by_fold: list[dict[str, object]] = []
    incremental_comparisons: list[dict[str, object]] = []
    for window_name in windows:
        fold_validation = validation.loc[validation["validation_window_name"].eq(window_name)]
        fold_start = fold_validation["date"].min()
        fold_train = train.loc[train["date"] < fold_start]
        if fold_train.empty:
            raise ValueError(f"Validation window {window_name} has no prior train embeddings.")

        for spec in target_specs:
            available_column = f"target_available__{spec.raw_target}"
            target_validation = fold_validation.loc[fold_validation[available_column]]
            if target_validation.empty:
                continue
            horizon = _target_horizon(spec.raw_target)
            if horizon is None:
                raise ValueError(f"Target has no parseable horizon for purging: {spec.raw_target}")
            outer_purge_mask = _purged_training_mask(
                fold_train["date"].to_numpy(),
                validation_start=pd.Timestamp(fold_start),
                horizon=horizon,
                calendar_dates=calendar_dates,
            )
            target_train = fold_train.loc[fold_train[available_column] & outer_purge_mask]
            if target_train.empty:
                continue

            raw_train_y = target_train[spec.raw_target].to_numpy(dtype=np.float64)
            transformed_train_y = transform_target_values(raw_train_y, spec)
            finite_train = np.isfinite(transformed_train_y)
            if not finite_train.any():
                continue
            raw_train_y = raw_train_y[finite_train]
            transformed_train_y = transformed_train_y[finite_train]
            target_train = target_train.loc[finite_train]
            train_dates = target_train["date"].to_numpy()

            raw_actual = target_validation[spec.raw_target].to_numpy(dtype=np.float64)
            transformed_actual = transform_target_values(raw_actual, spec)
            finite_validation = np.isfinite(raw_actual) & np.isfinite(transformed_actual)
            if not finite_validation.any():
                continue
            raw_actual = raw_actual[finite_validation]
            transformed_actual = transformed_actual[finite_validation]
            target_validation = target_validation.loc[finite_validation]

            common = {
                "validation_window_name": window_name,
                "raw_target": spec.raw_target,
                "target": spec.target,
                "target_transform": spec.transform,
                "horizon_days": horizon,
                "outer_purge_horizon_dates": horizon,
                "train_start": str(target_train["date"].min().date()),
                "train_end": str(target_train["date"].max().date()),
                "validation_start": str(target_validation["date"].min().date()),
                "validation_end": str(target_validation["date"].max().date()),
                "train_count": int(len(target_train)),
                "validation_count": int(len(target_validation)),
                "comparison_baseline": TRAIN_MEAN_BASELINE,
                "alpha_selection": alpha_selection,
            }

            transformed_baseline = np.full_like(transformed_actual, transformed_train_y.mean())
            raw_baseline, baseline_inverse_clipped = inverse_transform_predictions(
                transformed_baseline, spec
            )
            proxy_column = _trailing_proxy_column(spec.raw_target)
            proxy_available = proxy_column in target_validation.columns if proxy_column else False
            transformed_proxy: np.ndarray | None = None
            raw_proxy: np.ndarray | None = None
            if proxy_available and proxy_column is not None:
                raw_proxy = target_validation[proxy_column].to_numpy(dtype=np.float64)
                transformed_proxy = transform_target_values(raw_proxy, spec)

            # Raw targets are scored once. Transformed targets are scored both in
            # model space and after inverse-mapping back to the original target units.
            score_blocks: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
            if spec.transform == "raw":
                score_blocks.append(
                    (
                        "original",
                        spec.raw_target,
                        raw_actual,
                        transformed_baseline,
                        np.zeros(len(raw_actual), dtype=bool),
                    )
                )
            else:
                score_blocks.append(
                    (
                        "transformed",
                        spec.target,
                        transformed_actual,
                        transformed_baseline,
                        np.zeros(len(transformed_actual), dtype=bool),
                    )
                )
                score_blocks.append(
                    (
                        "original",
                        spec.raw_target,
                        raw_actual,
                        raw_baseline,
                        baseline_inverse_clipped,
                    )
                )

            for (
                score_space,
                score_target_name,
                actual,
                baseline,
                baseline_clip_mask,
            ) in score_blocks:
                baseline_metrics = _append_scored_result(
                    results=results,
                    prediction_rows=prediction_rows,
                    common={**common, "selected_alpha": None},
                    target_validation=target_validation,
                    score_space=score_space,
                    score_target_name=score_target_name,
                    predictor_name=TRAIN_MEAN_BASELINE,
                    predictor_kind="baseline",
                    model_name=TRAIN_MEAN_BASELINE,
                    feature_family="constant",
                    actual=actual,
                    predicted=baseline,
                    baseline_metrics=None,
                    inverse_clipped=(
                        baseline_inverse_clipped
                        if score_space == "original" and spec.transform != "raw"
                        else baseline_clip_mask
                    ),
                )

                if transformed_proxy is not None and raw_proxy is not None:
                    proxy_prediction = (
                        raw_proxy if score_space == "original" else transformed_proxy
                    )
                    finite_proxy = np.where(np.isfinite(proxy_prediction), proxy_prediction, baseline)
                    _append_scored_result(
                        results=results,
                        prediction_rows=prediction_rows,
                        common={**common, "selected_alpha": None},
                        target_validation=target_validation,
                        score_space=score_space,
                        score_target_name=score_target_name,
                        predictor_name=TRAILING_TARGET_PROXY_BASELINE,
                        predictor_kind="baseline",
                        model_name=TRAILING_TARGET_PROXY_BASELINE,
                        feature_family="trailing_proxy",
                        actual=actual,
                        predicted=finite_proxy,
                        baseline_metrics=baseline_metrics,
                    )

            feature_family_matrices = _feature_families(
                target_train,
                target_validation,
                z_columns=z_columns,
                baseline_feature_columns=baseline_feature_columns,
            )
            if baseline_feature_columns:
                hand_train_x = target_train[baseline_feature_columns].to_numpy(dtype=np.float64)
                hand_validation_x = target_validation[baseline_feature_columns].to_numpy(dtype=np.float64)
                z_train_x = target_train[z_columns].to_numpy(dtype=np.float64)
                z_validation_x = target_validation[z_columns].to_numpy(dtype=np.float64)
                residual_alpha = _fallback_alpha(model_alpha_grids["ridge"])
                try:
                    residual_z_train_x, residual_z_validation_x = _ridge_residualize(
                        hand_train_x,
                        hand_validation_x,
                        z_train_x,
                        z_validation_x,
                        alpha=residual_alpha,
                    )
                    feature_family_matrices.append(
                        (
                            FEATURE_RESIDUALIZED_Z_FAMILY,
                            residual_z_train_x,
                            residual_z_validation_x,
                        )
                    )
                except ValueError:
                    pass

                for model_name in REGRESSION_HEADS:
                    model_l1_ratios = l1_ratios if model_name == "elastic_net" else ()
                    hand_selection = _select_model_parameters(
                        model_name,
                        hand_train_x,
                        transformed_train_y,
                        train_dates,
                        model_alpha_grids[model_name],
                        horizon=horizon,
                        calendar_dates=calendar_dates,
                        l1_ratios=model_l1_ratios,
                        include_debug_diagnostics=include_debug_diagnostics,
                    )
                    hand_alpha = float(hand_selection["selected_alpha"])
                    hand_l1_ratio = hand_selection.get("selected_l1_ratio")
                    hand_train_prediction, _ = _predict_regression_head(
                        model_name,
                        hand_train_x,
                        transformed_train_y,
                        hand_train_x,
                        alpha=hand_alpha,
                        l1_ratio=float(hand_l1_ratio) if hand_l1_ratio is not None else None,
                    )
                    residual_train_y = transformed_train_y - hand_train_prediction
                    residual_selection = _select_model_parameters(
                        model_name,
                        z_train_x,
                        residual_train_y,
                        train_dates,
                        model_alpha_grids[model_name],
                        horizon=horizon,
                        calendar_dates=calendar_dates,
                        l1_ratios=model_l1_ratios,
                        include_debug_diagnostics=include_debug_diagnostics,
                    )
                    residual_alpha = float(residual_selection["selected_alpha"])
                    residual_l1_ratio = residual_selection.get("selected_l1_ratio")
                    parameter_selection_by_fold.append(
                        {
                            "validation_window_name": window_name,
                            "raw_target": spec.raw_target,
                            "target": spec.target,
                            "target_transform": spec.transform,
                            "model_name": model_name,
                            "feature_family": HAND_FEATURE_FAMILY,
                            "selection_stage": "hand_target_model",
                            "selection_method": alpha_selection,
                            **hand_selection,
                        }
                    )
                    parameter_selection_by_fold.append(
                        {
                            "validation_window_name": window_name,
                            "raw_target": spec.raw_target,
                            "target": spec.target,
                            "target_transform": spec.transform,
                            "model_name": model_name,
                            "feature_family": HAND_PLUS_RESIDUAL_Z_FAMILY,
                            "selection_stage": "training_residual_model",
                            "selection_method": alpha_selection,
                            **residual_selection,
                        }
                    )
                    hand_prediction, residual_prediction, hand_plus_z_prediction = (
                        _incremental_residual_predictions(
                            model_name,
                            hand_train_x,
                            z_train_x,
                            transformed_train_y,
                            hand_validation_x,
                            z_validation_x,
                            hand_alpha=hand_alpha,
                            residual_alpha=residual_alpha,
                            hand_l1_ratio=(
                                float(hand_l1_ratio) if hand_l1_ratio is not None else None
                            ),
                            residual_l1_ratio=(
                                float(residual_l1_ratio)
                                if residual_l1_ratio is not None
                                else None
                            ),
                        )
                    )
                    if spec.transform == "raw":
                        original_hand_prediction = hand_prediction
                        original_incremental_prediction = hand_plus_z_prediction
                        incremental_clipped = np.zeros(len(raw_actual), dtype=bool)
                    else:
                        original_hand_prediction, _ = inverse_transform_predictions(
                            hand_prediction, spec
                        )
                        original_incremental_prediction, incremental_clipped = (
                            inverse_transform_predictions(hand_plus_z_prediction, spec)
                        )
                    comparison = _incremental_comparison_metrics(
                        raw_actual,
                        original_hand_prediction,
                        original_incremental_prediction,
                    )
                    incremental_comparisons.append(
                        {
                            "validation_window_name": window_name,
                            "raw_target": spec.raw_target,
                            "model_name": model_name,
                            "score_space": "original",
                            "hand_feature_family": HAND_FEATURE_FAMILY,
                            "incremental_feature_family": HAND_PLUS_RESIDUAL_Z_FAMILY,
                            "training_residual_min": float(residual_train_y.min()),
                            "training_residual_max": float(residual_train_y.max()),
                            "reconstruction_max_abs_error": float(
                                np.max(np.abs(hand_plus_z_prediction - hand_prediction - residual_prediction))
                            ),
                            **comparison,
                        }
                    )
                    hand_metrics = _score_predictions(
                        spec.raw_target,
                        raw_actual,
                        original_hand_prediction,
                        score_space="original",
                        baseline_metrics=None,
                    )
                    predictor_name = f"{model_name}__{HAND_PLUS_RESIDUAL_Z_FAMILY}"
                    _append_scored_result(
                        results=results,
                        prediction_rows=prediction_rows,
                        common={
                            **common,
                            "selected_alpha": residual_alpha,
                            **(
                                {"selected_l1_ratio": residual_l1_ratio}
                                if model_name == "elastic_net"
                                else {}
                            ),
                            "inner_validation_score": residual_selection["inner_validation_score"],
                            "selected_inner_fold_scores": residual_selection[
                                "selected_inner_fold_scores"
                            ],
                            "selected_at_grid_boundary": residual_selection[
                                "selected_at_grid_boundary"
                            ],
                        },
                        target_validation=target_validation,
                        score_space="original",
                        score_target_name=spec.raw_target,
                        predictor_name=predictor_name,
                        predictor_kind="model",
                        model_name=model_name,
                        feature_family=HAND_PLUS_RESIDUAL_Z_FAMILY,
                        actual=raw_actual,
                        predicted=original_incremental_prediction,
                        baseline_metrics=hand_metrics,
                        inverse_clipped=incremental_clipped,
                    )

            for feature_family, train_x, validation_x in feature_family_matrices:
                for model_name in REGRESSION_HEADS:
                    selection = _select_model_parameters(
                        model_name,
                        train_x,
                        transformed_train_y,
                        train_dates,
                        model_alpha_grids[model_name],
                        horizon=horizon,
                        calendar_dates=calendar_dates,
                        l1_ratios=l1_ratios if model_name == "elastic_net" else (),
                        include_debug_diagnostics=include_debug_diagnostics,
                    )
                    selected_alpha = float(selection["selected_alpha"])
                    selected_l1_ratio = selection.get("selected_l1_ratio")
                    parameter_selection_by_fold.append(
                        {
                            "validation_window_name": window_name,
                            "raw_target": spec.raw_target,
                            "target": spec.target,
                            "target_transform": spec.transform,
                            "model_name": model_name,
                            "feature_family": feature_family,
                            "selection_method": alpha_selection,
                            **selection,
                        }
                    )
                    transformed_predicted, coefficients = _predict_regression_head(
                        model_name,
                        train_x,
                        transformed_train_y,
                        validation_x,
                        alpha=selected_alpha,
                        l1_ratio=float(selected_l1_ratio) if selected_l1_ratio is not None else None,
                    )
                    raw_predicted, model_inverse_clipped = inverse_transform_predictions(
                        transformed_predicted, spec
                    )
                    for score_space, score_target_name, actual, baseline, _ in score_blocks:
                        predicted = (
                            raw_predicted
                            if score_space == "original" and spec.transform != "raw"
                            else transformed_predicted
                        )
                        clip_mask = (
                            model_inverse_clipped
                            if score_space == "original" and spec.transform != "raw"
                            else np.zeros(len(predicted), dtype=bool)
                        )
                        baseline_metrics = _score_predictions(
                            score_target_name,
                            actual,
                            baseline,
                            score_space=score_space,
                            baseline_metrics=None,
                        )
                        predictor_name = f"{model_name}__{feature_family}"
                        _append_scored_result(
                            results=results,
                            prediction_rows=prediction_rows,
                            common={
                                **common,
                                "selected_alpha": selected_alpha,
                                **({"selected_l1_ratio": selected_l1_ratio} if model_name == "elastic_net" else {}),
                                "inner_validation_score": selection["inner_validation_score"],
                                "selected_inner_fold_scores": selection[
                                    "selected_inner_fold_scores"
                                ],
                                "selected_at_grid_boundary": selection[
                                    "selected_at_grid_boundary"
                                ],
                            },
                            target_validation=target_validation,
                            score_space=score_space,
                            score_target_name=score_target_name,
                            predictor_name=predictor_name,
                            predictor_kind="model",
                            model_name=model_name,
                            feature_family=feature_family,
                            actual=actual,
                            predicted=predicted,
                            baseline_metrics=baseline_metrics,
                            inverse_clipped=clip_mask,
                        )
                    if include_debug_diagnostics:
                        coefficients_by_fold.append(
                            {
                                "validation_window_name": window_name,
                                "raw_target": spec.raw_target,
                                "target": spec.target,
                                "target_transform": spec.transform,
                                "predictor_name": f"{model_name}__{feature_family}",
                                "model_name": model_name,
                                "feature_family": feature_family,
                                "selected_alpha": selected_alpha,
                                **(
                                    {"selected_l1_ratio": selected_l1_ratio}
                                    if model_name == "elastic_net"
                                    else {}
                                ),
                                "coefficients": coefficients.tolist(),
                            }
                        )

            if spec.transform == "raw":
                for label_spec in _classification_label_frame(raw_train_y, raw_actual, spec.raw_target):
                    label_train_y = np.asarray(label_spec["train_y"], dtype=np.float64)
                    label_validation_y = np.asarray(label_spec["validation_y"], dtype=np.float64)
                    if len(np.unique(label_train_y)) < 2:
                        continue
                    prior_probability = np.full_like(label_validation_y, label_train_y.mean())
                    classification_common = {
                        **common,
                        "target": str(label_spec["classification_label"]),
                        "target_transform": "binary_label",
                        "selected_alpha": None,
                    }
                    baseline_metrics = _append_classification_result(
                        results=results,
                        prediction_rows=prediction_rows,
                        common=classification_common,
                        target_validation=target_validation,
                        classification_label=str(label_spec["classification_label"]),
                        positive_rule=str(label_spec["positive_rule"]),
                        threshold=float(label_spec["threshold"]),
                        predictor_name="class_prior",
                        predictor_kind="baseline",
                        model_name="class_prior",
                        feature_family="constant",
                        actual=label_validation_y,
                        probability=prior_probability,
                        baseline_metrics=None,
                    )
                    for feature_family, train_x, validation_x in _feature_families(
                        target_train,
                        target_validation,
                        z_columns=z_columns,
                        baseline_feature_columns=baseline_feature_columns,
                    ):
                        selection = _select_model_parameters(
                            "logistic",
                            train_x,
                            label_train_y,
                            train_dates,
                            model_alpha_grids["logistic"],
                            horizon=horizon,
                            calendar_dates=calendar_dates,
                            include_debug_diagnostics=include_debug_diagnostics,
                        )
                        selected_alpha = float(selection["selected_alpha"])
                        parameter_selection_by_fold.append(
                            {
                                "validation_window_name": window_name,
                                "raw_target": spec.raw_target,
                                "target": str(label_spec["classification_label"]),
                                "target_transform": "binary_label",
                                "model_name": "logistic",
                                "feature_family": feature_family,
                                "selection_method": alpha_selection,
                                **selection,
                            }
                        )
                        probability, coefficients = _logistic_predict(
                            train_x,
                            label_train_y,
                            validation_x,
                            alpha=selected_alpha,
                        )
                        predictor_name = f"logistic__{feature_family}"
                        _append_classification_result(
                            results=results,
                            prediction_rows=prediction_rows,
                            common={
                                **classification_common,
                                "selected_alpha": selected_alpha,
                                "inner_validation_score": selection["inner_validation_score"],
                                "selected_inner_fold_scores": selection[
                                    "selected_inner_fold_scores"
                                ],
                                "selected_at_grid_boundary": selection[
                                    "selected_at_grid_boundary"
                                ],
                            },
                            target_validation=target_validation,
                            classification_label=str(label_spec["classification_label"]),
                            positive_rule=str(label_spec["positive_rule"]),
                            threshold=float(label_spec["threshold"]),
                            predictor_name=predictor_name,
                            predictor_kind="model",
                            model_name="logistic",
                            feature_family=feature_family,
                            actual=label_validation_y,
                            probability=probability,
                            baseline_metrics=baseline_metrics,
                        )
                        if include_debug_diagnostics:
                            coefficients_by_fold.append(
                                {
                                    "validation_window_name": window_name,
                                    "raw_target": spec.raw_target,
                                    "target": str(label_spec["classification_label"]),
                                    "target_transform": "binary_label",
                                    "predictor_name": predictor_name,
                                    "model_name": "logistic",
                                    "feature_family": feature_family,
                                    "selected_alpha": selected_alpha,
                                    "coefficients": coefficients.tolist(),
                                }
                            )

    if not prediction_rows:
        raise RuntimeError("No finite probe folds were available.")
    predictions = pd.concat(prediction_rows, ignore_index=True)

    aggregate: list[dict[str, object]] = []
    group_columns = ["target", "score_space"]
    for (target_name, score_space), target_predictions in predictions.groupby(
        group_columns, sort=True
    ):
        if target_predictions.empty:
            continue
        baseline_frame = target_predictions.loc[
            target_predictions["predictor_name"].eq(TRAIN_MEAN_BASELINE)
        ]
        if baseline_frame.empty:
            continue
        score_target_name = str(target_predictions["score_target_name"].iloc[0])
        baseline_metrics = _score_predictions(
            score_target_name,
            baseline_frame["actual"].to_numpy(dtype=np.float64),
            baseline_frame["prediction"].to_numpy(dtype=np.float64),
            score_space=str(score_space),
            baseline_metrics=None,
        )
        baseline_clip_rate = float(baseline_frame["inverse_prediction_clipped"].mean())
        baseline_metrics["inverse_prediction_clip_count"] = int(
            baseline_frame["inverse_prediction_clipped"].sum()
        )
        baseline_metrics["inverse_prediction_clip_rate"] = baseline_clip_rate
        for predictor_name, frame in target_predictions.groupby("predictor_name", sort=True):
            metrics = _score_predictions(
                score_target_name,
                frame["actual"].to_numpy(dtype=np.float64),
                frame["prediction"].to_numpy(dtype=np.float64),
                score_space=str(score_space),
                baseline_metrics=None if predictor_name == TRAIN_MEAN_BASELINE else baseline_metrics,
            )
            metrics["inverse_prediction_clip_count"] = int(
                frame["inverse_prediction_clipped"].sum()
            )
            metrics["inverse_prediction_clip_rate"] = float(
                frame["inverse_prediction_clipped"].mean()
            )
            aggregate.append(
                {
                    "raw_target": str(frame["raw_target"].iloc[0]),
                    "target": str(target_name),
                    "target_transform": str(frame["target_transform"].iloc[0]),
                    "score_space": str(score_space),
                    "score_target_name": score_target_name,
                    "horizon_days": _target_horizon(str(frame["raw_target"].iloc[0])),
                    "predictor_name": str(predictor_name),
                    "predictor_kind": str(frame["predictor_kind"].iloc[0]),
                    "model_name": str(frame["model_name"].iloc[0]),
                    "feature_family": str(frame["feature_family"].iloc[0]),
                    "comparison_baseline": TRAIN_MEAN_BASELINE,
                    "validation_count": int(len(frame)),
                    **metrics,
                }
            )

    run_id = hashlib.sha256(
        (
            f"{source_database}|{alpha_selection}|{json.dumps(model_alpha_grids, sort_keys=True)}|"
            f"{','.join(str(value) for value in l1_ratios)}|"
            f"{','.join(transformed_target_names)}|{representation_variant}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    regression_summary = _regression_final_summary(results)
    classification_summary = _classification_final_summary(results)
    bootstrap_by_window, stability_summary, evaluated_combination_count = (
        _bootstrap_reliability_summary(
            predictions,
            representation_variant=representation_variant,
        )
    )
    destination, temporary = artifact_destination(output_root, run_id)
    try:
        probe_dataset.to_parquet(
            temporary / "probe_dataset.parquet", index=False, compression="zstd"
        )
        predictions.to_parquet(
            temporary / "predictions.parquet", index=False, compression="zstd"
        )
        report = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_id": checkpoint_id,
            "checkpoint_step": checkpoint_step,
            "representation_source": representation_source,
            "representation_variant": representation_variant,
            "resolved_probe_config": {
                "alpha_selection": alpha_selection,
                "ridge_alphas": list(model_alpha_grids["ridge"]),
                "huber_alphas": list(model_alpha_grids["huber"]),
                "elastic_net_alphas": list(model_alpha_grids["elastic_net"]),
                "elastic_net_l1_ratios": list(l1_ratios),
                "logistic_alphas": list(model_alpha_grids["logistic"]),
                "inner_fold_count": 3,
                "horizon_purge_enabled": True,
                "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
                "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
                "bootstrap_block_length": "target_horizon",
                "include_debug_diagnostics": include_debug_diagnostics,
            },
            "resolved_representation_config": resolved_representation_config,
            "alpha_selection": alpha_selection,
            "target_transform_epsilon": TARGET_TRANSFORM_EPSILON,
            "probe_dataset_artifact": (
                str(probe_dataset_artifact.resolve()) if probe_dataset_artifact else None
            ),
            "embedding_artifact": embedding_source,
            "target_artifact": target_source,
            "raw_target_columns": target_columns,
            "target_columns": target_columns,
            "transformed_target_columns": transformed_target_names,
            "target_transform_specs": [asdict(spec) for spec in target_specs],
            "validation_windows": metadata["validation_windows"],
            "z_columns": z_columns,
            "baseline_feature_columns": baseline_feature_columns,
            "feature_families": [
                Z_ONLY_FAMILY,
                HAND_FEATURE_FAMILY,
                HAND_PCA_FAMILY,
                HAND_PLUS_Z_FAMILY,
                HAND_PLUS_RESIDUAL_Z_FAMILY,
                FEATURE_RESIDUALIZED_Z_FAMILY,
            ],
            "baseline_families": [
                TRAIN_MEAN_BASELINE,
                TRAILING_TARGET_PROXY_BASELINE,
                HAND_FEATURE_FAMILY,
                HAND_PCA_FAMILY,
            ],
            "regression_heads": list(REGRESSION_HEADS),
            "classification_heads": ["logistic"],
            "results": results,
            "aggregate_out_of_fold": aggregate,
            "window_summaries": _window_summaries(results),
            "final_regression_summary": regression_summary,
            "final_classification_summary": classification_summary,
            "bootstrap_by_window": bootstrap_by_window,
            "stability_summary": stability_summary,
            "evaluated_target_model_variant_combination_count": evaluated_combination_count,
            "multiple_comparison_correction_applied": False,
            "pass_fail_gate_counts": {
                "regression": _gate_counts(regression_summary),
                "classification": _gate_counts(classification_summary),
            },
            "parameter_selection_by_fold": parameter_selection_by_fold,
            "incremental_hand_plus_z_comparisons": incremental_comparisons,
            "representation_diagnostics": representation_diagnostics,
            "targets_joined_into_pretraining_artifact": False,
            "phase2_notes": {
                "raw_targets_are_still_scored": True,
                "transformed_targets_are_scored_in_model_space": True,
                "transformed_targets_are_inverse_scored_in_original_space": True,
                "inverse_prediction_clip_count_is_diagnostic": True,
                "incremental_residuals_fit_on_outer_training_rows_only": True,
                "oracle_validation_recalibration_included": False,
            },
            "phase3_notes": {
                "train_mean_baseline_is_retained": True,
                "trailing_target_proxy_baseline_enabled": True,
                "hand_market_feature_baselines_enabled": bool(baseline_feature_columns),
                "classification_labels_use_train_only_thresholds": True,
                "neural_probe_heads_included": False,
            },
            "phase4_notes": {
                "hand_plus_residual_z_enabled": bool(baseline_feature_columns),
                "feature_residualized_z_only_enabled": bool(baseline_feature_columns),
                "residualization_controls": "past-only baseline features",
                "residualization_fit_scope": "outer training dates only",
                "raw_pooled_state_variants_included": representation_variant == "pooled_raw_256",
                "raw_pooled_state_variants_note": (
                    "The selected representation is stored in z_* columns; pooled_raw_256 preserves "
                    "the unprojected pooled state."
                ),
            },
        }
        if include_debug_diagnostics:
            report["coefficients_by_fold"] = coefficients_by_fold
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (temporary / "summary.md").write_text(
            build_summary_markdown(report), encoding="utf-8"
        )
        publish_artifact(temporary, destination)
    except Exception:
        clean_temporary_artifact(temporary)
        raise
    print(f"Built frozen probe report: {destination}")
    return destination


# ============================================================================
# COMMAND-LINE ENTRY POINTS
# ============================================================================


def export_targets_main() -> None:
    """Run the separate canonical probe-target export CLI."""
    parser = argparse.ArgumentParser(description="Export separate FI-JEPA probe targets.")
    parser.add_argument(
        "--database", type=Path, default=Path("data/processed/market_data.duckdb")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/probe_targets"))
    parser.add_argument(
        "--name",
        help="Readable artifact directory name. Defaults to '<database_stem>_targets'.",
    )
    parser.add_argument(
        "--market-symbol",
        default="ETF_SPY",
        help="Market proxy symbol used for past-only trailing baseline features.",
    )
    args = parser.parse_args()
    export_probe_targets(
        args.database,
        output_root=args.output_root,
        name=args.name,
        market_symbol=args.market_symbol,
    )


def build_probe_dataset_main() -> None:
    """Build one reusable joined probe dataset from separate source artifacts."""
    parser = argparse.ArgumentParser(description="Build a reusable FI-JEPA probe dataset.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/probe_targets"))
    parser.add_argument(
        "--representation-variant",
        help="Explicit exported representation variant to place in this probe dataset.",
    )
    parser.add_argument(
        "--name",
        help="Readable artifact directory name. Defaults to '<embedding_artifact>_probe_dataset'.",
    )
    args = parser.parse_args()
    build_probe_dataset(
        args.embeddings,
        args.targets,
        output_root=args.output_root,
        name=args.name,
        representation_variant=args.representation_variant,
    )


def run_probes_main() -> None:
    """Run leakage-safe frozen probes from source artifacts or one reusable dataset."""
    parser = argparse.ArgumentParser(description="Run FI-JEPA frozen linear probes.")
    parser.add_argument("--probe-dataset", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs/probes"))
    parser.add_argument(
        "--representation-variant",
        help="Run one explicit representation variant from a multi-variant embedding artifact.",
    )
    parser.add_argument("--ridge-alphas", default=",".join(str(value) for value in DEFAULT_RIDGE_ALPHAS))
    parser.add_argument("--huber-alphas", default=",".join(str(value) for value in DEFAULT_HUBER_ALPHAS))
    parser.add_argument("--elastic-net-alphas", default=",".join(str(value) for value in DEFAULT_ELASTIC_NET_ALPHAS))
    parser.add_argument(
        "--elastic-net-l1-ratios", default=",".join(str(value) for value in DEFAULT_ELASTIC_NET_L1_RATIOS)
    )
    parser.add_argument("--logistic-alphas", default=",".join(str(value) for value in DEFAULT_LOGISTIC_ALPHAS))
    parser.add_argument(
        "--include-debug-diagnostics",
        action="store_true",
        help="Include full parameter-grid candidates and coefficients in report.json.",
    )

    args = parser.parse_args()
    if args.probe_dataset is None and (args.embeddings is None or args.targets is None):
        parser.error("provide --probe-dataset or both --embeddings and --targets")
    if args.probe_dataset is not None and (
        args.embeddings is not None or args.targets is not None
    ):
        parser.error("--probe-dataset cannot be combined with --embeddings or --targets")
    run_frozen_probes(
        args.embeddings,
        args.targets,
        probe_dataset_artifact=args.probe_dataset,
        output_root=args.output_root,
        ridge_alphas=tuple(float(value) for value in args.ridge_alphas.split(",") if value.strip()),
        huber_alphas=tuple(float(value) for value in args.huber_alphas.split(",") if value.strip()),
        elastic_net_alphas=tuple(float(value) for value in args.elastic_net_alphas.split(",") if value.strip()),
        elastic_net_l1_ratios=tuple(
            float(value) for value in args.elastic_net_l1_ratios.split(",") if value.strip()
        ),
        logistic_alphas=tuple(float(value) for value in args.logistic_alphas.split(",") if value.strip()),
        representation_variant=args.representation_variant,
        include_debug_diagnostics=args.include_debug_diagnostics,
    )
