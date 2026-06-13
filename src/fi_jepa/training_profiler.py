from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import sys
import threading
import traceback
from typing import Any

import torch


TimingRecord = dict[str, float | int]


# ============================================================================
# PYTHON STACK SAMPLING
# ============================================================================


class PythonStackSampler:
    """Sample the constructing thread's Python stack into folded-stack format.

    PyTorch's Kineto stack export is empty on some Windows builds. This sampler
    records the main training thread independently, producing a non-empty
    ``cpu_stacks.txt`` that can be loaded by folded-stack/flame-graph tools.
    Dataloader worker processes remain outside this parent-process sampler.
    """

    def __init__(self, output_path: Path, *, interval_seconds: float = 0.01):
        if interval_seconds <= 0.0:
            raise ValueError("Python stack sampling interval must be positive.")
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.target_thread_id = threading.get_ident()
        self.samples: Counter[str] = Counter()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._sample_until_stopped,
            name="fi-jepa-python-stack-sampler",
            daemon=True,
        )

    def start(self) -> None:
        """Start sampling the training thread."""
        self.thread.start()

    def stop(self) -> None:
        """Stop sampling and write folded stacks ordered by sample count."""
        self.stop_event.set()
        self.thread.join()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{stack} {count}"
            for stack, count in self.samples.most_common()
        ]
        self.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _sample_until_stopped(self) -> None:
        """Collect complete Python stacks until the owning training run stops."""
        while not self.stop_event.is_set():
            frame = sys._current_frames().get(self.target_thread_id)
            if frame is not None:
                stack = ";".join(
                    (
                        f"{entry.name} "
                        f"[{entry.filename.replace(chr(92), '/')}:{entry.lineno}]"
                    ).replace(";", "_")
                    for entry in traceback.extract_stack(frame)
                )
                if stack:
                    self.samples[stack] += 1
            self.stop_event.wait(self.interval_seconds)


# ============================================================================
# PROFILER CONSTRUCTION
# ============================================================================


def profile_range(name: str, *, enabled: bool) -> Any:
    """Return a named profiler range without adding instrumentation to normal runs."""
    return torch.profiler.record_function(name) if enabled else nullcontext()


def build_training_profiler(
    output_dir: Path,
    device: torch.device,
    *,
    wait_steps: int,
    warmup_steps: int,
    active_steps: int,
    python_stacks: bool,
) -> torch.profiler.profile:
    """Build a scheduled profiler that writes traces and device/CPU summaries.

    The TensorBoard trace handler emits a ``*.pt.trace.json`` Chrome trace. The
    callback writes a compact device-first table plus an exhaustive CPU table.
    Optional stack capture writes a folded-stack file suitable for flame graphs.
    """
    trace_handler = torch.profiler.tensorboard_trace_handler(
        str(output_dir),
        worker_name="training",
    )

    def write_trace(profiler: torch.profiler.profile) -> None:
        trace_handler(profiler)
        key_averages = profiler.key_averages()
        sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        summary = key_averages.table(sort_by=sort_by, row_limit=100)
        cpu_summary = key_averages.table(sort_by="self_cpu_time_total", row_limit=-1)
        (output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
        (output_dir / "cpu_summary.txt").write_text(cpu_summary + "\n", encoding="utf-8")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1,
        ),
        on_trace_ready=write_trace,
        record_shapes=True,
        profile_memory=True,
        with_stack=python_stacks,
        with_flops=True,
        acc_events=True,
    )


# ============================================================================
# RUNTIME TIMING SUMMARIES
# ============================================================================


def write_runtime_timing_summary(
    path: Path,
    warmup_records: list[TimingRecord],
    boundary_records: list[TimingRecord],
) -> None:
    """Write readable per-epoch wall-clock timings for work outside model steps."""
    lines = [
        "EPOCH WARM-UP",
        "epoch  dataset_epoch_update_s  dataloader_iterator_startup_s  total_s",
    ]
    lines.extend(
        (
            f"{int(record['epoch']) + 1:5d}  "
            f"{float(record['dataset_epoch_update_seconds']):22.3f}  "
            f"{float(record['dataloader_iterator_startup_seconds']):29.3f}  "
            f"{float(record['total_seconds']):7.3f}"
        )
        for record in warmup_records
    )
    lines.extend(
        [
            "",
            "EPOCH BOUNDARY",
            "epoch  validation_s  representation_evaluation_s  best_checkpoint_s  latest_checkpoint_s  total_s",
        ]
    )
    lines.extend(
        (
            f"{int(record['epoch']) + 1:5d}  "
            f"{float(record['validation_seconds']):12.3f}  "
            f"{float(record['representation_evaluation_seconds']):27.3f}  "
            f"{float(record['best_checkpoint_seconds']):17.3f}  "
            f"{float(record['latest_checkpoint_seconds']):19.3f}  "
            f"{float(record['total_seconds']):7.3f}"
        )
        for record in boundary_records
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_runtime_timing_to_profile_summary(profile_summary: Path, runtime_summary: Path) -> None:
    """Append wall-clock epoch timings beneath the operator-level profiler table."""
    with profile_summary.open("a", encoding="utf-8") as file:
        file.write("\nWALL-CLOCK RUNTIME SECTIONS\n\n")
        file.write(runtime_summary.read_text(encoding="utf-8"))
