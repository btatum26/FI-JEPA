from __future__ import annotations

import pytest
import torch

from fi_jepa.losses import pooled_variance_covariance_loss


# ============================================================================
# REPRESENTATION REGULARIZATION
# ============================================================================


def test_pooled_variance_loss_penalizes_constant_batch() -> None:
    states = torch.ones(4, 8, requires_grad=True)

    variance_loss, covariance_loss, mean_feature_std = pooled_variance_covariance_loss(
        states,
        variance_floor=0.1,
        epsilon=0.0001,
    )

    assert mean_feature_std.item() == pytest.approx(0.01)
    assert variance_loss.item() == pytest.approx(0.09)
    assert covariance_loss.item() == pytest.approx(0.0)
    (variance_loss + covariance_loss).backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()


def test_aggregate_variance_floor_does_not_require_every_dimension_to_vary() -> None:
    states = torch.zeros(4, 4)
    states[:, 0] = torch.tensor([-1.0, -1.0, 1.0, 1.0])

    variance_loss, covariance_loss, mean_feature_std = pooled_variance_covariance_loss(
        states,
        variance_floor=0.1,
        epsilon=0.0001,
    )

    assert mean_feature_std.item() > 0.1
    assert variance_loss.item() == pytest.approx(0.0)
    assert covariance_loss.item() == pytest.approx(0.0)


def test_covariance_loss_penalizes_redundant_dimensions() -> None:
    states = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ]
    )

    _, covariance_loss, _ = pooled_variance_covariance_loss(
        states,
        variance_floor=0.0,
        epsilon=0.0001,
    )

    assert covariance_loss.item() > 0.0


def test_single_sample_batch_returns_differentiable_zeros() -> None:
    states = torch.randn(1, 4, requires_grad=True)

    variance_loss, covariance_loss, mean_feature_std = pooled_variance_covariance_loss(
        states,
        variance_floor=0.1,
        epsilon=0.0001,
    )

    assert variance_loss.item() == pytest.approx(0.0)
    assert covariance_loss.item() == pytest.approx(0.0)
    assert mean_feature_std.item() == pytest.approx(0.0)
    (variance_loss + covariance_loss).backward()
    assert states.grad is not None
