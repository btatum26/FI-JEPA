from __future__ import annotations

import torch
from torch.nn import functional as F


# ============================================================================
# REPRESENTATION REGULARIZATION
# ============================================================================


def pooled_variance_covariance_loss(
    states: torch.Tensor,
    *,
    variance_floor: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weak batch-level anti-collapse losses for pooled online states.

    The variance term floors the mean feature standard deviation rather than
    flooring every feature independently. This intentionally permits inactive
    dimensions and only penalizes a nearly non-varying representation as a
    whole. The covariance term averages each feature's summed squared
    off-diagonal covariance to discourage redundant dimensions without
    constraining their individual variances.

    Statistics are computed in float32 for mixed-precision stability. A batch
    with fewer than two samples returns differentiable zeros because it cannot
    estimate variance or covariance.

    Args:
        states: One pooled online representation per sample, shaped ``[B, D]``.
        variance_floor: Minimum desired mean feature standard deviation.
        epsilon: Positive numerical stabilizer inside the standard deviation.

    Returns:
        ``(variance_loss, covariance_loss, mean_feature_std)`` scalar tensors.
    """
    if states.ndim != 2:
        raise ValueError(f"states must have shape [B, D]; got {tuple(states.shape)}.")
    if states.shape[1] <= 0:
        raise ValueError("states must have a positive feature dimension.")
    if variance_floor < 0.0:
        raise ValueError("variance_floor cannot be negative.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    values = states.float()
    if values.shape[0] < 2:
        zero = values.sum() * 0.0
        return zero, zero, zero

    # [B, D] -> [B, D]. Batch centering isolates sample-to-sample variation.
    centered = values - values.mean(dim=0, keepdim=True)
    feature_std = torch.sqrt(centered.var(dim=0, unbiased=True) + epsilon)  # [D].
    mean_feature_std = feature_std.mean()
    variance_loss = F.relu(mean_feature_std.new_tensor(variance_floor) - mean_feature_std)

    # [D, B] @ [B, D] -> [D, D]. Only redundant cross-feature movement is penalized.
    covariance = centered.transpose(0, 1) @ centered / (values.shape[0] - 1)
    squared_covariance = covariance.square()
    off_diagonal_sum = squared_covariance.sum() - squared_covariance.diagonal().sum()
    covariance_loss = off_diagonal_sum / states.shape[1]
    return variance_loss, covariance_loss, mean_feature_std
