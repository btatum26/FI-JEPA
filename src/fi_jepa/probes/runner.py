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
from fi_jepa.probes.targets import export_probe_targets
from fi_jepa.probes.target_transforms import (
    TARGET_TRANSFORM_EPSILON,
    inverse_transform_predictions,
    target_transform_specs,
    transform_target_values,
)

DEFAULT_RIDGE_ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
TRAIN_MEAN_BASELINE = "train_mean"
TRAILING_TARGET_PROXY_BASELINE = "trailing_target_proxy"
Z_ONLY_FAMILY = "z_only"
HAND_FEATURE_FAMILY = "hand_market_features"
HAND_PCA_FAMILY = "hand_market_pca"
HAND_PLUS_Z_FAMILY = "hand_market_features_plus_z"
TARGET_RESIDUALIZED_Z_FAMILY = "target_residualized_z_only"
FEATURE_RESIDUALIZED_Z_FAMILY = "feature_residualized_z_only"
REGRESSION_HEADS = ("ridge", "huber", "elastic_net")
CLASSIFICATION_QUANTILE = 0.8


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

    gram = standardized_train.T @ standardized_train

    coefficients = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
        standardized_train.T @ centered_y,
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
        coefficients = np.linalg.solve(
            standardized_train.T @ weighted_x + alpha * np.eye(standardized_train.shape[1]),
            weighted_x.T @ centered_y,
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


def _predict_regression_head(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch one simple regression head without adding an external ML dependency."""
    if model_name == "ridge":
        return _ridge_predict(train_x, train_y, validation_x, alpha=alpha)
    if model_name == "huber":
        return _huber_predict(train_x, train_y, validation_x, alpha=alpha)
    if model_name == "elastic_net":
        return _elastic_net_predict(train_x, train_y, validation_x, alpha=alpha)
    raise ValueError(f"Unknown regression head: {model_name}")


def _normalise_alpha_grid(alphas: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Validate and de-duplicate a ridge alpha grid while preserving sort order."""
    unique = tuple(sorted({float(alpha) for alpha in alphas}))
    if not unique:
        raise ValueError("At least one ridge alpha is required.")
    if any(alpha <= 0.0 for alpha in unique):
        raise ValueError("All ridge alphas must be positive.")
    return unique


def _fallback_alpha(alpha_grid: tuple[float, ...]) -> float:
    """Return the conventional fixed ridge alpha when it exists, otherwise the middle grid value."""
    if 1.0 in alpha_grid:
        return 1.0
    return alpha_grid[len(alpha_grid) // 2]


def _select_ridge_alpha(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_dates: np.ndarray,
    alpha_grid: tuple[float, ...],
) -> tuple[float, list[dict[str, object]]]:
    """Choose ridge alpha with a tail inner-validation split inside the outer train period."""
    fallback = _fallback_alpha(alpha_grid)
    dates = pd.to_datetime(train_dates)
    unique_dates = pd.Index(dates).sort_values().unique()
    if len(unique_dates) < 4 or len(train_y) < 6:
        return fallback, []

    split_index = min(max(int(len(unique_dates) * 0.8), 1), len(unique_dates) - 1)
    validation_start = unique_dates[split_index]
    inner_train = dates < validation_start
    inner_validation = dates >= validation_start
    if inner_train.sum() < 2 or inner_validation.sum() < 1:
        return fallback, []

    diagnostics: list[dict[str, object]] = []
    best_alpha = fallback
    best_rmse = np.inf
    for candidate_alpha in alpha_grid:
        predicted, _ = _ridge_predict(
            train_x[inner_train],
            train_y[inner_train],
            train_x[inner_validation],
            alpha=candidate_alpha,
        )
        metrics = _regression_metrics(train_y[inner_validation], predicted)
        rmse = float(metrics["rmse"])
        diagnostics.append(
            {
                "alpha": candidate_alpha,
                "inner_train_count": int(inner_train.sum()),
                "inner_validation_count": int(inner_validation.sum()),
                "inner_validation_start": str(pd.Timestamp(validation_start).date()),
                "inner_validation_rmse": rmse,
                "inner_validation_r2": float(metrics["r2"]),
                "inner_validation_pearson_correlation": float(metrics["pearson_correlation"]),
            }
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = candidate_alpha
    return best_alpha, diagnostics


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
    penalty = alpha / max(len(train_y), 1)
    for _ in range(400):
        logits = np.clip(intercept + standardized_train @ coefficients, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = probability - train_y
        intercept -= learning_rate * float(error.mean())
        coefficients -= learning_rate * (
            standardized_train.T @ error / len(train_y) + penalty * coefficients
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
    target_name: str, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flag non-finite and physically invalid predictions for one target family."""
    lower, upper, constraint_reason = _target_constraint(target_name)
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


def _validation_recalibration(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    """Fit diagnostic-only validation labels on predictions to expose scale/intercept failure."""
    finite = np.isfinite(actual) & np.isfinite(predicted)
    finite_actual = actual[finite]
    finite_predicted = predicted[finite]
    design = np.column_stack([np.ones(len(finite_predicted)), finite_predicted])
    intercept, slope = np.linalg.lstsq(design, finite_actual, rcond=None)[0]
    recalibrated = intercept + slope * finite_predicted
    return {
        "uses_validation_labels": True,
        "intercept": float(intercept),
        "slope": float(slope),
        "metrics": _regression_metrics(finite_actual, recalibrated),
    }


def _score_predictions(
    target_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    baseline_metrics: dict[str, float | int] | None,
) -> dict[str, object]:
    """Score one predictor and attach invalidity, baseline-ratio, and recalibration diagnostics."""
    metrics: dict[str, object] = _regression_metrics(actual, predicted)
    invalid, _ = _invalid_predictions(target_name, predicted)
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
    metrics["validation_recalibration"] = _validation_recalibration(actual, predicted)
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
        score_target_name, actual, predicted, baseline_metrics=baseline_metrics
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
    invalid, invalid_reasons = _invalid_predictions(score_target_name, predicted)
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
    alpha: float | None = None,
    alphas: tuple[float, ...] | list[float] = DEFAULT_RIDGE_ALPHAS,
) -> Path:
    """Run leakage-safe walk-forward ridge probes from source or reusable dataset artifacts."""
    alpha_grid = _normalise_alpha_grid(alphas)
    if alpha is not None and alpha <= 0.0:
        raise ValueError("Ridge alpha must be positive.")
    alpha_selection = "fixed" if alpha is not None else "inner_walk_forward"

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
    else:
        if embedding_artifact is None or target_artifact is None:
            raise ValueError("Embedding and target artifacts are both required.")
        probe_dataset, metadata = assemble_probe_dataset(embedding_artifact, target_artifact)
        source_database = metadata["source_database_sha256"]
        target_columns = [str(name) for name in metadata["target_columns"]]
        z_columns = [str(name) for name in metadata["z_columns"]]
        baseline_feature_columns = [str(name) for name in metadata.get("baseline_feature_columns", [])]
        embedding_source = str(embedding_artifact.resolve())
        target_source = str(target_artifact.resolve())

    target_specs = target_transform_specs(target_columns)
    transformed_target_names = [spec.target for spec in target_specs]

    validation = probe_dataset.loc[probe_dataset["split"].eq("validation")].copy()
    train = probe_dataset.loc[probe_dataset["split"].eq("train")].copy()
    windows = sorted(str(name) for name in validation["validation_window_name"].unique() if name)

    results: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    coefficients_by_fold: list[dict[str, object]] = []
    alpha_selection_by_fold: list[dict[str, object]] = []
    for window_name in windows:
        fold_validation = validation.loc[validation["validation_window_name"].eq(window_name)]
        fold_start = fold_validation["date"].min()
        fold_train = train.loc[train["date"] < fold_start]
        if fold_train.empty:
            raise ValueError(f"Validation window {window_name} has no prior train embeddings.")

        for spec in target_specs:
            available_column = f"target_available__{spec.raw_target}"
            target_train = fold_train.loc[fold_train[available_column]]
            target_validation = fold_validation.loc[fold_validation[available_column]]
            if target_train.empty or target_validation.empty:
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
                "horizon_days": _target_horizon(spec.raw_target),
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
                        "raw",
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
                        "raw",
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
                        if score_space == "raw" and spec.transform != "raw"
                        else baseline_clip_mask
                    ),
                )

                if transformed_proxy is not None and raw_proxy is not None:
                    proxy_prediction = raw_proxy if score_space == "raw" else transformed_proxy
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
                residual_alpha = _fallback_alpha(alpha_grid)
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

                residual_train_y, residual_actual = _ridge_residualize(
                    hand_train_x,
                    hand_validation_x,
                    transformed_train_y,
                    transformed_actual,
                    alpha=residual_alpha,
                )
                target_residual_name = f"{spec.target}__target_residual"
                target_residual_transform = f"{spec.transform}_target_residualized"
                target_residual_baseline = np.full_like(residual_actual, residual_train_y.mean())
                target_residual_common = {
                    **common,
                    "target": target_residual_name,
                    "target_transform": target_residual_transform,
                    "selected_alpha": None,
                }
                target_residual_baseline_metrics = _append_scored_result(
                    results=results,
                    prediction_rows=prediction_rows,
                    common=target_residual_common,
                    target_validation=target_validation,
                    score_space="residual",
                    score_target_name=target_residual_name,
                    predictor_name=TRAIN_MEAN_BASELINE,
                    predictor_kind="baseline",
                    model_name=TRAIN_MEAN_BASELINE,
                    feature_family="constant",
                    actual=residual_actual,
                    predicted=target_residual_baseline,
                    baseline_metrics=None,
                )
                selected_alpha = float(alpha) if alpha is not None else None
                alpha_diagnostics: list[dict[str, object]] = []
                if selected_alpha is None:
                    selected_alpha, alpha_diagnostics = _select_ridge_alpha(
                        z_train_x, residual_train_y, train_dates, alpha_grid
                    )
                alpha_selection_by_fold.append(
                    {
                        "validation_window_name": window_name,
                        "raw_target": spec.raw_target,
                        "target": target_residual_name,
                        "target_transform": target_residual_transform,
                        "feature_family": TARGET_RESIDUALIZED_Z_FAMILY,
                        "selected_alpha": selected_alpha,
                        "selection_method": alpha_selection,
                        "candidate_diagnostics": alpha_diagnostics,
                    }
                )
                for model_name in REGRESSION_HEADS:
                    residual_predicted, coefficients = _predict_regression_head(
                        model_name,
                        z_train_x,
                        residual_train_y,
                        z_validation_x,
                        alpha=selected_alpha,
                    )
                    predictor_name = f"{model_name}__{TARGET_RESIDUALIZED_Z_FAMILY}"
                    _append_scored_result(
                        results=results,
                        prediction_rows=prediction_rows,
                        common={
                            **target_residual_common,
                            "selected_alpha": selected_alpha,
                        },
                        target_validation=target_validation,
                        score_space="residual",
                        score_target_name=target_residual_name,
                        predictor_name=predictor_name,
                        predictor_kind="model",
                        model_name=model_name,
                        feature_family=TARGET_RESIDUALIZED_Z_FAMILY,
                        actual=residual_actual,
                        predicted=residual_predicted,
                        baseline_metrics=target_residual_baseline_metrics,
                    )
                    coefficients_by_fold.append(
                        {
                            "validation_window_name": window_name,
                            "raw_target": spec.raw_target,
                            "target": target_residual_name,
                            "target_transform": target_residual_transform,
                            "predictor_name": predictor_name,
                            "model_name": model_name,
                            "feature_family": TARGET_RESIDUALIZED_Z_FAMILY,
                            "selected_alpha": selected_alpha,
                            "coefficients": coefficients.tolist(),
                        }
                    )

            for feature_family, train_x, validation_x in feature_family_matrices:
                selected_alpha = float(alpha) if alpha is not None else None
                alpha_diagnostics: list[dict[str, object]] = []
                if selected_alpha is None:
                    selected_alpha, alpha_diagnostics = _select_ridge_alpha(
                        train_x, transformed_train_y, train_dates, alpha_grid
                    )
                alpha_selection_by_fold.append(
                    {
                        "validation_window_name": window_name,
                        "raw_target": spec.raw_target,
                        "target": spec.target,
                        "target_transform": spec.transform,
                        "feature_family": feature_family,
                        "selected_alpha": selected_alpha,
                        "selection_method": alpha_selection,
                        "candidate_diagnostics": alpha_diagnostics,
                    }
                )
                for model_name in REGRESSION_HEADS:
                    transformed_predicted, coefficients = _predict_regression_head(
                        model_name,
                        train_x,
                        transformed_train_y,
                        validation_x,
                        alpha=selected_alpha,
                    )
                    raw_predicted, model_inverse_clipped = inverse_transform_predictions(
                        transformed_predicted, spec
                    )
                    for score_space, score_target_name, actual, baseline, _ in score_blocks:
                        predicted = raw_predicted if score_space == "raw" and spec.transform != "raw" else transformed_predicted
                        clip_mask = (
                            model_inverse_clipped
                            if score_space == "raw" and spec.transform != "raw"
                            else np.zeros(len(predicted), dtype=bool)
                        )
                        baseline_metrics = _score_predictions(
                            score_target_name,
                            actual,
                            baseline,
                            baseline_metrics=None,
                        )
                        predictor_name = f"{model_name}__{feature_family}"
                        _append_scored_result(
                            results=results,
                            prediction_rows=prediction_rows,
                            common={**common, "selected_alpha": selected_alpha},
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
                        "selected_alpha": float(alpha) if alpha is not None else _fallback_alpha(alpha_grid),
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
                        probability, coefficients = _logistic_predict(
                            train_x,
                            label_train_y,
                            validation_x,
                            alpha=float(classification_common["selected_alpha"]),
                        )
                        predictor_name = f"logistic__{feature_family}"
                        _append_classification_result(
                            results=results,
                            prediction_rows=prediction_rows,
                            common=classification_common,
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
                        coefficients_by_fold.append(
                            {
                                "validation_window_name": window_name,
                                "raw_target": spec.raw_target,
                                "target": str(label_spec["classification_label"]),
                                "target_transform": "binary_label",
                                "predictor_name": predictor_name,
                                "model_name": "logistic",
                                "feature_family": feature_family,
                                "selected_alpha": classification_common["selected_alpha"],
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
            f"{source_database}|{alpha_selection}|{alpha}|{','.join(str(value) for value in alpha_grid)}|"
            f"{','.join(transformed_target_names)}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    regression_summary = _regression_final_summary(results)
    classification_summary = _classification_final_summary(results)
    destination, temporary = artifact_destination(output_root, run_id)
    try:
        probe_dataset.to_parquet(
            temporary / "probe_dataset.parquet", index=False, compression="zstd"
        )
        predictions.to_parquet(
            temporary / "predictions.parquet", index=False, compression="zstd"
        )
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "fixed_alpha": alpha,
            "alpha_selection": alpha_selection,
            "alpha_grid": list(alpha_grid),
            "target_transform_epsilon": TARGET_TRANSFORM_EPSILON,
            "probe_dataset_artifact": (
                str(probe_dataset_artifact.resolve()) if probe_dataset_artifact else None
            ),
            "embedding_artifact": embedding_source,
            "target_artifact": target_source,
            "source_database_sha256": source_database,
            "raw_target_columns": target_columns,
            "target_columns": target_columns,
            "transformed_target_columns": transformed_target_names,
            "target_transform_specs": [asdict(spec) for spec in target_specs],
            "z_columns": z_columns,
            "baseline_feature_columns": baseline_feature_columns,
            "feature_families": [
                Z_ONLY_FAMILY,
                HAND_FEATURE_FAMILY,
                HAND_PCA_FAMILY,
                HAND_PLUS_Z_FAMILY,
                TARGET_RESIDUALIZED_Z_FAMILY,
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
            "pass_fail_gate_counts": {
                "regression": _gate_counts(regression_summary),
                "classification": _gate_counts(classification_summary),
            },
            "coefficients_by_fold": coefficients_by_fold,
            "alpha_selection_by_fold": alpha_selection_by_fold,
            "recalibration_is_diagnostic_only": True,
            "targets_joined_into_pretraining_artifact": False,
            "phase2_notes": {
                "raw_targets_are_still_scored": True,
                "transformed_targets_are_scored_in_model_space": True,
                "transformed_targets_are_inverse_scored_in_raw_space": True,
                "inverse_prediction_clip_count_is_diagnostic": True,
            },
            "phase3_notes": {
                "train_mean_baseline_is_retained": True,
                "trailing_target_proxy_baseline_enabled": True,
                "hand_market_feature_baselines_enabled": bool(baseline_feature_columns),
                "classification_labels_use_train_only_thresholds": True,
                "neural_probe_heads_included": False,
            },
            "phase4_notes": {
                "target_residualized_z_only_enabled": bool(baseline_feature_columns),
                "feature_residualized_z_only_enabled": bool(baseline_feature_columns),
                "residualization_controls": "past-only baseline features",
                "residualization_fit_scope": "outer training dates only",
                "raw_pooled_state_variants_included": False,
                "raw_pooled_state_variants_note": (
                    "Current probe dataset contains exported z_* coordinates only; raw pooled-state "
                    "variants require an evaluation export contract change."
                ),
            },
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
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
        "--name",
        help="Readable artifact directory name. Defaults to '<embedding_artifact>_probe_dataset'.",
    )
    args = parser.parse_args()
    build_probe_dataset(args.embeddings, args.targets, output_root=args.output_root, name=args.name)


def run_probes_main() -> None:
    """Run leakage-safe frozen probes from source artifacts or one reusable dataset."""
    parser = argparse.ArgumentParser(description="Run FI-JEPA frozen linear probes.")
    parser.add_argument("--probe-dataset", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs/probes"))
    parser.add_argument(
        "--alpha",
        type=float,
        help="Use one fixed ridge alpha instead of inner walk-forward alpha selection.",
    )
    parser.add_argument(
        "--alphas",
        default="0.0001,0.001,0.01,0.1,1,10,100,1000,10000",
        help="Comma-separated ridge alpha grid used when --alpha is omitted.",
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
        alpha=args.alpha,
        alphas=tuple(float(value) for value in args.alphas.split(",") if value.strip()),
    )
