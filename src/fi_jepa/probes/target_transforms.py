from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


def target_transform_specs(target_columns: list[str]) -> list[TargetTransformSpec]:
    """Return raw plus Phase 2 transformed target variants for ridge probes."""
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


def transform_target_values(raw_values: np.ndarray, spec: TargetTransformSpec) -> np.ndarray:
    """Map raw target values into the target space used for model fitting."""
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


def inverse_transform_predictions(
    transformed_predictions: np.ndarray, spec: TargetTransformSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Map model-space predictions back to raw target units and report support clipping."""
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
