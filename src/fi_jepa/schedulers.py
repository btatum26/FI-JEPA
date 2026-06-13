from __future__ import annotations
import math
from torch.optim import AdamW


class WarmupCosineLRSchedule:
    """Warm one AdamW learning rate linearly, then decay it with cosine.

    The schedule is indexed by successful optimizer steps. Calls beyond the
    originally planned run clamp to ``min_lr``, which keeps replayed batches
    after a basic epoch resume from extending the cosine curve.
    """

    def __init__(
        self,
        optimizer: AdamW,
        *,
        base_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ):
        if not 0 <= warmup_steps < total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps).")
        if not 0.0 <= min_lr <= base_lr:
            raise ValueError("Learning rates must satisfy 0 <= min_lr <= base_lr.")
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.last_step = -1

    def value_at(self, step: int) -> float:
        """Return the clamped learning rate for a zero-based optimizer step."""
        if step < 0:
            raise ValueError("Schedule step cannot be negative.")
        if step >= self.total_steps:
            return self.min_lr
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)

        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 1:
            return self.min_lr
        progress = float(step - self.warmup_steps) / float(decay_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def apply(self, step: int, *, commit: bool) -> float:
        """Apply one step's LR, optionally recording that the step succeeded."""
        value = self.value_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        if commit:
            self.last_step = step
        return value

    def state_dict(self) -> dict[str, int | float]:
        """Return the complete schedule state stored in checkpoints."""
        return {
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict[str, int | float]) -> None:
        """Restore state while rejecting a schedule with different bounds."""
        expected = self.state_dict()
        for name in ("base_lr", "min_lr", "warmup_steps", "total_steps"):
            if state[name] != expected[name]:
                raise ValueError(f"Checkpoint LR schedule disagrees on {name}.")
        self.last_step = int(state["last_step"])


class LinearEMAMomentumSchedule:
    """Increase target-encoder EMA momentum linearly by optimizer step."""

    def __init__(self, *, start: float, end: float, total_steps: int):
        if not 0.0 <= start <= end <= 1.0:
            raise ValueError("EMA momentum must satisfy 0 <= start <= end <= 1.")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.last_step = -1

    def value_at(self, step: int) -> float:
        """Return momentum for a zero-based step, clamped at the final value."""
        if step < 0:
            raise ValueError("Schedule step cannot be negative.")
        if self.total_steps == 1 or step >= self.total_steps - 1:
            return self.end
        progress = float(step) / float(self.total_steps - 1)
        return self.start + (self.end - self.start) * progress

    def commit(self, step: int) -> float:
        """Record one successful EMA update and return its momentum."""
        value = self.value_at(step)
        self.last_step = step
        return value

    def state_dict(self) -> dict[str, int | float]:
        """Return the complete schedule state stored in checkpoints."""
        return {
            "start": self.start,
            "end": self.end,
            "total_steps": self.total_steps,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict[str, int | float]) -> None:
        """Restore state while rejecting a schedule with different bounds."""
        expected = self.state_dict()
        for name in ("start", "end", "total_steps"):
            if state[name] != expected[name]:
                raise ValueError(f"Checkpoint EMA schedule disagrees on {name}.")
        self.last_step = int(state["last_step"])