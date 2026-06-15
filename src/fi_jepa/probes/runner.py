from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

PROBE_REPORT_VERSION = 3
TRAIN_MEAN_BASELINE = "train_mean"
TARGET_TRANSFORM_EPSILON = 1.0e-12


# ============================================================================
# TARGET TRANSFORMS
# ============================================================================


@dataclass(frozen=True)
class TargetTransformSpec:
    """One leakage-safe target variant for a raw future target column."""

    raw_target: str
    target: str
    transform: str
    description: str
    inverse_clips_to_target_support: bool


def _target_transform_specs(
    target_columns: list[str], *, mode: str = "phase2"
) -> list[TargetTransformSpec]:
    """Return target variants to probe.

    Phase 2 keeps the original raw target probes and adds target-space variants for
    constrained/skewed targets. All transforms are deterministic and fitted only from
    the target values available inside each outer training fold.
    """
    if mode not in {"raw", "phase2"}:
        raise ValueError("target transform mode must be 'raw' or 'phase2'.")

    specs: list[TargetTransformSpec] = []
    for target_name in target_columns:
        specs.append(
            TargetTransformSpec(
                raw_target=target_name,
                target=target_name,
                transform="raw",
                description="Original target values.",
                inverse_clips_to_target_support=False,
            )
        )
        if mode == "raw":
            continue

        if "future_realized_vol" in target_name:
            specs.append(
                TargetTransformSpec(
                    raw_target=target_name,
                    target=f"{target_name}__log_realized_vol",
                    transform="log_realized_vol",
                    description="log(max(realized_volatility, epsilon)); inverse is exp(prediction).",
                    inverse_clips_to_target_support=False,
                )
            )
        elif "future_max_drawdown" in target_name:
            specs.append(
                TargetTransformSpec(
                    raw_target=target_name,
                    target=f"{target_name}__drawdown_magnitude",
                    transform="drawdown_magnitude",
                    description="clip(-max_drawdown, 0, inf); inverse is -clip(prediction, 0, inf).",
                    inverse_clips_to_target_support=True,
                )
            )
            specs.append(
                TargetTransformSpec(
                    raw_target=target_name,
                    target=f"{target_name}__log_drawdown_magnitude",
                    transform="log_drawdown_magnitude",
                    description="log1p(clip(-max_drawdown, 0, inf)); inverse is -expm1(clip(prediction, 0, inf)).",
                    inverse_clips_to_target_support=True,
                )
            )
    return specs


def _transform_target_values(raw_values: np.ndarray, spec: TargetTransformSpec) -> np.ndarray:
    """Map raw target values into the model-fitting target space."""
    values = np.asarray(raw_values, dtype=np.float64)
    if spec.transform == "raw":
        return values
    if spec.transform == "log_realized_vol":
        return np.log(np.maximum(values, TARGET_TRANSFORM_EPSILON))
    if spec.transform == "drawdown_magnitude":
        return np.maximum(-values, 0.0)
    if spec.transform == "log_drawdown_magnitude":
        return np.log1p(np.maximum(-values, 0.0))
    raise ValueError(f"Unknown target transform: {spec.transform}")


def _inverse_transform_predictions(
    transformed_predictions: np.ndarray, spec: TargetTransformSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Map model-space predictions back to the original raw target units.

    Returns the raw-space prediction and a boolean mask indicating whether the inverse
    had to clip the prediction to stay inside the transformed target support.
    """
    predicted = np.asarray(transformed_predictions, dtype=np.float64)
    clipped = np.zeros(len(predicted), dtype=bool)
    if spec.transform == "raw":
        return predicted, clipped
    if spec.transform == "log_realized_vol":
        return np.exp(predicted), clipped
    if spec.transform == "drawdown_magnitude":
        clipped = np.isfinite(predicted) & (predicted < 0.0)
        magnitude = np.maximum(predicted, 0.0)
        return -magnitude, clipped
    if spec.transform == "log_drawdown_magnitude":
        clipped = np.isfinite(predicted) & (predicted < 0.0)
        log_magnitude = np.maximum(predicted, 0.0)
        return -np.expm1(log_magnitude), clipped
    raise ValueError(f"Unknown target transform: {spec.transform}")


# ============================================================================
# REGRESSION AND DIAGNOSTIC METRICS
# ============================================================================


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one train-standardized ridge model and predict validation rows."""
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    standardized_train = (train_x - x_mean) / x_std
    standardized_validation = (validation_x - x_mean) / x_std
    y_mean = float(train_y.mean())
    centered_y = train_y - y_mean
    gram = standardized_train.T @ standardized_train
    coefficients = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
        standardized_train.T @ centered_y,
    )
    return standardized_validation @ coefficients + y_mean, coefficients


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


# ============================================================================
# FROZEN RIDGE PROBES
# ============================================================================


def _window_summaries(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize target-level result rows without mixing target scales."""
    summaries: list[dict[str, object]] = []
    result_frame = pd.DataFrame(results)
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
            "score_space": score_space,
            "score_target_name": score_target_name,
            "predictor_name": predictor_name,
            "predictor_kind": predictor_kind,
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
                "score_space": score_space,
                "score_target_name": score_target_name,
                "horizon_days": common["horizon_days"],
                "predictor_name": predictor_name,
                "predictor_kind": predictor_kind,
                "comparison_baseline": TRAIN_MEAN_BASELINE,
                "actual": actual,
                "prediction": predicted,
                "invalid_prediction": invalid,
                "invalid_prediction_reason": invalid_reasons,
                "inverse_prediction_clipped": inverse_clipped,
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
    alpha: float = 1.0,
    target_transform_mode: str = "phase2",
) -> Path:
    """Run leakage-safe walk-forward ridge probes from source or reusable dataset artifacts."""
    if alpha <= 0.0:
        raise ValueError("Ridge alpha must be positive.")
    if target_transform_mode not in {"raw", "phase2"}:
        raise ValueError("target_transform_mode must be 'raw' or 'phase2'.")

    if probe_dataset_artifact is not None:
        if embedding_artifact is not None or target_artifact is not None:
            raise ValueError("Use either a probe dataset or embedding/target artifacts, not both.")
        probe_dataset, metadata = load_probe_dataset(probe_dataset_artifact)
        source_database = metadata["source_database_sha256"]
        target_columns = [str(name) for name in metadata["target_columns"]]
        z_columns = [str(name) for name in metadata["z_columns"]]
        embedding_source = metadata["embedding_artifact"]
        target_source = metadata["target_artifact"]
    else:
        if embedding_artifact is None or target_artifact is None:
            raise ValueError("Embedding and target artifacts are both required.")
        probe_dataset, metadata = assemble_probe_dataset(embedding_artifact, target_artifact)
        source_database = metadata["source_database_sha256"]
        target_columns = [str(name) for name in metadata["target_columns"]]
        z_columns = [str(name) for name in metadata["z_columns"]]
        embedding_source = str(embedding_artifact.resolve())
        target_source = str(target_artifact.resolve())

    target_specs = _target_transform_specs(target_columns, mode=target_transform_mode)
    transformed_target_names = [spec.target for spec in target_specs]

    validation = probe_dataset.loc[probe_dataset["split"].eq("validation")].copy()
    train = probe_dataset.loc[probe_dataset["split"].eq("train")].copy()
    windows = sorted(str(name) for name in validation["validation_window_name"].unique() if name)

    results: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    coefficients_by_fold: list[dict[str, object]] = []
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

            train_x = target_train[z_columns].to_numpy(dtype=np.float64)
            raw_train_y = target_train[spec.raw_target].to_numpy(dtype=np.float64)
            transformed_train_y = _transform_target_values(raw_train_y, spec)
            finite_train = np.isfinite(transformed_train_y)
            if not finite_train.any():
                continue
            train_x = train_x[finite_train]
            transformed_train_y = transformed_train_y[finite_train]
            target_train = target_train.loc[finite_train]

            validation_x = target_validation[z_columns].to_numpy(dtype=np.float64)
            raw_actual = target_validation[spec.raw_target].to_numpy(dtype=np.float64)
            transformed_actual = _transform_target_values(raw_actual, spec)
            finite_validation = np.isfinite(raw_actual) & np.isfinite(transformed_actual)
            if not finite_validation.any():
                continue
            validation_x = validation_x[finite_validation]
            raw_actual = raw_actual[finite_validation]
            transformed_actual = transformed_actual[finite_validation]
            target_validation = target_validation.loc[finite_validation]

            transformed_predicted, coefficients = _ridge_predict(
                train_x, transformed_train_y, validation_x, alpha=alpha
            )
            transformed_baseline = np.full_like(
                transformed_actual, transformed_train_y.mean()
            )
            raw_predicted, ridge_inverse_clipped = _inverse_transform_predictions(
                transformed_predicted, spec
            )
            raw_baseline, baseline_inverse_clipped = _inverse_transform_predictions(
                transformed_baseline, spec
            )

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
            }

            # Raw targets are scored once. Transformed targets are scored both in
            # model space and after inverse-mapping back to the original target units.
            score_blocks: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            if spec.transform == "raw":
                score_blocks.append(
                    (
                        "raw",
                        spec.raw_target,
                        raw_actual,
                        transformed_predicted,
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
                        transformed_predicted,
                        transformed_baseline,
                        np.zeros(len(transformed_actual), dtype=bool),
                    )
                )
                score_blocks.append(
                    (
                        "raw",
                        spec.raw_target,
                        raw_actual,
                        raw_predicted,
                        raw_baseline,
                        ridge_inverse_clipped,
                    )
                )

            for (
                score_space,
                score_target_name,
                actual,
                predicted,
                baseline,
                ridge_clip_mask,
            ) in score_blocks:
                if score_space == "raw" and spec.transform != "raw":
                    baseline_clip_mask = baseline_inverse_clipped
                else:
                    baseline_clip_mask = np.zeros(len(baseline), dtype=bool)

                baseline_metrics = _append_scored_result(
                    results=results,
                    prediction_rows=prediction_rows,
                    common=common,
                    target_validation=target_validation,
                    score_space=score_space,
                    score_target_name=score_target_name,
                    predictor_name=TRAIN_MEAN_BASELINE,
                    predictor_kind="baseline",
                    actual=actual,
                    predicted=baseline,
                    baseline_metrics=None,
                    inverse_clipped=baseline_clip_mask,
                )
                _append_scored_result(
                    results=results,
                    prediction_rows=prediction_rows,
                    common=common,
                    target_validation=target_validation,
                    score_space=score_space,
                    score_target_name=score_target_name,
                    predictor_name="ridge",
                    predictor_kind="model",
                    actual=actual,
                    predicted=predicted,
                    baseline_metrics=baseline_metrics,
                    inverse_clipped=ridge_clip_mask,
                )

            coefficients_by_fold.append(
                {
                    "validation_window_name": window_name,
                    "raw_target": spec.raw_target,
                    "target": spec.target,
                    "target_transform": spec.transform,
                    "predictor_name": "ridge",
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
                    "comparison_baseline": TRAIN_MEAN_BASELINE,
                    "validation_count": int(len(frame)),
                    **metrics,
                }
            )

    run_id = hashlib.sha256(
        f"{source_database}|{alpha}|{target_transform_mode}|{PROBE_REPORT_VERSION}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    destination, temporary = artifact_destination(output_root, run_id)
    try:
        probe_dataset.to_parquet(
            temporary / "probe_dataset.parquet", index=False, compression="zstd"
        )
        predictions.to_parquet(
            temporary / "predictions.parquet", index=False, compression="zstd"
        )
        report = {
            "format_version": PROBE_REPORT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "alpha": alpha,
            "target_transform_mode": target_transform_mode,
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
            "results": results,
            "aggregate_out_of_fold": aggregate,
            "window_summaries": _window_summaries(results),
            "coefficients_by_fold": coefficients_by_fold,
            "recalibration_is_diagnostic_only": True,
            "targets_joined_into_pretraining_artifact": False,
            "phase2_notes": {
                "raw_targets_are_still_scored": True,
                "transformed_targets_are_scored_in_model_space": True,
                "transformed_targets_are_inverse_scored_in_raw_space": True,
                "inverse_prediction_clip_count_is_diagnostic": True,
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
    args = parser.parse_args()
    export_probe_targets(args.database, output_root=args.output_root)


def build_probe_dataset_main() -> None:
    """Build one reusable joined probe dataset from separate source artifacts."""
    parser = argparse.ArgumentParser(description="Build a reusable FI-JEPA probe dataset.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/probe_datasets"))
    args = parser.parse_args()
    build_probe_dataset(args.embeddings, args.targets, output_root=args.output_root)


def run_probes_main() -> None:
    """Run leakage-safe frozen probes from source artifacts or one reusable dataset."""
    parser = argparse.ArgumentParser(description="Run FI-JEPA frozen linear probes.")
    parser.add_argument("--probe-dataset", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs/probes"))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--target-transforms",
        choices=("phase2", "raw"),
        default="phase2",
        help="Use phase2 transformed target variants, or raw-only compatibility mode.",
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
        target_transform_mode=args.target_transforms,
    )
