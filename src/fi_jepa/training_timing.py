from __future__ import annotations

from pathlib import Path


TimingRecord = dict[str, float | int]


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
