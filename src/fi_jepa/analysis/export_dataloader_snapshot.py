from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
import pandas as pd
import torch

from fi_jepa.dataloader import DensePanelStore, FIJepaDataConfig, build_fi_jepa_dataloader
from fi_jepa.dataloader.panel_store import Split

matplotlib.use("Agg")
from matplotlib import pyplot as plt

InputGroup = Literal["asset", "market", "macro"]


# ============================================================================
# BATCH FLATTENING
# ============================================================================


def _tensor_array(batch: dict[str, object], name: str) -> np.ndarray:
    """Converts a torch tensor to a NumPy array."""
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {name!r} must be a tensor, got {type(value).__name__}.")
    return value.detach().cpu().numpy()


def _feature_rows(
    batch: dict[str, object],
    store: DensePanelStore,
    group: InputGroup,
    *,
    sample_count: int,
    asset_count: int,
) -> pd.DataFrame:
    """Flatten one model input stream while retaining every corresponding validity mask.

    The emitted rows preserve the exact normalized values supplied to the model,
    including zero-filled invalid values. ``feature_valid`` and
    ``stream_row_valid`` must be inspected alongside ``value`` to distinguish a
    real zero from a missing value represented by zero.
    """
    values = _tensor_array(batch, f"{group}_patches")[:sample_count]
    feature_valid = _tensor_array(batch, f"{group}_feature_mask_patched")[:sample_count]
    feature_names = store.feature_names[group]

    if group == "asset":
        # [B, P, L, A, F] -> one row per exact scalar supplied to the asset tokenizer.
        values = values[:, :, :, :asset_count, :]
        feature_valid = feature_valid[:, :, :, :asset_count, :]
        stream_row_valid = _tensor_array(batch, "valid_asset_mask_patched")[
            :sample_count, :, :, :asset_count
        ]
        patch_asset_valid = _tensor_array(batch, "patch_asset_mask")[
            :sample_count, :, :asset_count
        ]
        selected_asset_ids = _tensor_array(batch, "asset_ids")[:sample_count, :asset_count]

        samples, patches, days, assets, features = values.shape
        frame = pd.DataFrame(
            {
                "input_group": group,
                "sample_index": np.repeat(np.arange(samples), patches * days * assets * features),
                "patch_index": np.tile(
                    np.repeat(np.arange(patches), days * assets * features), samples
                ),
                "day_in_patch": np.tile(
                    np.repeat(np.arange(days), assets * features), samples * patches
                ),
                "asset_slot": np.tile(
                    np.repeat(np.arange(assets), features), samples * patches * days
                ),
                "feature_index": np.tile(
                    np.arange(features), samples * patches * days * assets
                ),
                "value": values.reshape(-1),
                "feature_valid": feature_valid.reshape(-1),
                "stream_row_valid": np.broadcast_to(
                    stream_row_valid[..., None], values.shape
                ).reshape(-1),
                "patch_asset_valid": np.broadcast_to(
                    patch_asset_valid[:, :, None, :, None], values.shape
                ).reshape(-1),
                "asset_id": np.broadcast_to(
                    selected_asset_ids[:, None, None, :, None], values.shape
                ).reshape(-1),
            }
        )
        frame["symbol"] = store.assets[frame["asset_id"].to_numpy(dtype=np.int64)]
    else:
        # [B, P, L, F] -> one row per exact scalar supplied to a date-stream tokenizer.
        stream_row_valid = _tensor_array(batch, f"valid_{group}_date_mask_patched")[
            :sample_count
        ]
        samples, patches, days, features = values.shape
        frame = pd.DataFrame(
            {
                "input_group": group,
                "sample_index": np.repeat(np.arange(samples), patches * days * features),
                "patch_index": np.tile(
                    np.repeat(np.arange(patches), days * features), samples
                ),
                "day_in_patch": np.tile(
                    np.repeat(np.arange(days), features), samples * patches
                ),
                "asset_slot": pd.array([pd.NA] * values.size, dtype="Int64"),
                "feature_index": np.tile(np.arange(features), samples * patches * days),
                "value": values.reshape(-1),
                "feature_valid": feature_valid.reshape(-1),
                "stream_row_valid": np.broadcast_to(
                    stream_row_valid[..., None], values.shape
                ).reshape(-1),
                "patch_asset_valid": pd.array([pd.NA] * values.size, dtype="boolean"),
                "asset_id": pd.array([pd.NA] * values.size, dtype="Int64"),
                "symbol": pd.array([pd.NA] * values.size, dtype="string"),
            }
        )

    frame["feature_name"] = np.asarray(feature_names, dtype=object)[
        frame["feature_index"].to_numpy(dtype=np.int64)
    ]
    return frame


def batch_to_snapshot_frame(
    batch: dict[str, object],
    store: DensePanelStore,
    config: FIJepaDataConfig,
    *,
    sample_limit: int = 1,
    asset_limit: int = 16,
) -> pd.DataFrame:
    """Flatten a limited slice of one real dataloader batch into an inspectable table.

    ``sample_limit`` and ``asset_limit`` use zero to mean all values present in
    the batch. Limiting assets affects only asset-stream rows; market and macro
    rows remain complete for every included sample.
    """
    asset_patches = _tensor_array(batch, "asset_patches")
    batch_sample_count = int(asset_patches.shape[0])
    batch_asset_count = int(asset_patches.shape[3])
    sample_count = batch_sample_count if sample_limit == 0 else min(sample_limit, batch_sample_count)
    asset_count = batch_asset_count if asset_limit == 0 else min(asset_limit, batch_asset_count)
    if sample_count <= 0 or asset_count <= 0:
        raise ValueError("sample_limit and asset_limit must be zero or positive.")

    frames = [
        _feature_rows(
            batch,
            store,
            group,
            sample_count=sample_count,
            asset_count=asset_count,
        )
        for group in ("asset", "market", "macro")
    ]
    frame = pd.concat(frames, ignore_index=True)

    # Map flattened rows back to the exact lookback dates and per-sample request metadata.
    sample_indices = frame["sample_index"].to_numpy(dtype=np.int64)
    patch_indices = frame["patch_index"].to_numpy(dtype=np.int64)
    endpoint_indices = _tensor_array(batch, "sample_date_idx")[:sample_count]
    frame["date_idx"] = (
        endpoint_indices[sample_indices]
        - config.lookback_days
        + 1
        + patch_indices * config.patch_len
        + frame["day_in_patch"].to_numpy(dtype=np.int64)
    )
    frame["date"] = pd.to_datetime(store.dates[frame["date_idx"].to_numpy(dtype=np.int64)])

    patch_context = _tensor_array(batch, "patch_context_mask")[:sample_count]
    patch_target_eligible = _tensor_array(batch, "patch_target_eligible")[:sample_count]
    jepa_context = _tensor_array(batch, "jepa_context_mask")[:sample_count]
    jepa_target = _tensor_array(batch, "jepa_target_mask")[:sample_count]
    target_ids = _tensor_array(batch, "target_patch_ids")[:sample_count]
    target_id_mask = _tensor_array(batch, "target_patch_id_mask")[:sample_count]
    target_patch_rank = np.full(patch_context.shape, -1, dtype=np.int64)
    for sample_index in range(sample_count):
        enabled_ids = target_ids[sample_index, target_id_mask[sample_index]]
        target_patch_rank[sample_index, enabled_ids] = np.arange(len(enabled_ids), dtype=np.int64)

    frame["patch_context_valid"] = patch_context[sample_indices, patch_indices]
    frame["patch_target_eligible"] = patch_target_eligible[sample_indices, patch_indices]
    frame["jepa_context"] = jepa_context[sample_indices, patch_indices]
    frame["jepa_target"] = jepa_target[sample_indices, patch_indices]
    ranks = target_patch_rank[sample_indices, patch_indices]
    frame["target_patch_rank"] = pd.array(
        np.where(ranks >= 0, ranks, pd.NA),
        dtype="Int64",
    )

    metadata = pd.DataFrame(
        {
            "sample_index": np.arange(sample_count),
            "sample_date_idx": endpoint_indices,
            "sample_date": pd.to_datetime(batch["sample_date"][:sample_count]),
            "split": batch["split_label"][:sample_count],
            "validation_window_name": batch["validation_window_name"][:sample_count],
            "asset_view": batch["asset_view"][:sample_count],
            "view_index": _tensor_array(batch, "view_index")[:sample_count],
            "request_seed": np.asarray(batch["request_seed"][:sample_count], dtype=np.uint64),
            "k_assets": batch["k_assets"][:sample_count],
            "n_endpoint_valid_assets": batch["n_endpoint_valid_assets"][:sample_count],
            "target_eligible_patch_count": _tensor_array(batch, "target_eligible_patch_count")[
                :sample_count
            ],
            "batch_sample_count": batch_sample_count,
            "snapshot_sample_count": sample_count,
            "batch_asset_count": batch_asset_count,
            "snapshot_asset_count": asset_count,
        }
    )
    return frame.merge(metadata, on="sample_index", how="left", validate="many_to_one")


# ============================================================================
# SNAPSHOT EXPORT
# ============================================================================


def export_dataloader_snapshot(
    config_path: Path,
    output_path: Path,
    *,
    split: Split = "train",
    sample_limit: int = 1,
    asset_limit: int = 16,
    shuffle: bool | None = None,
    train_epoch: int = 0,
) -> Path:
    """Export one actual collated JEPA batch slice as a long-format parquet file."""
    if train_epoch < 0:
        raise ValueError("train_epoch must be zero or positive.")
    config = FIJepaDataConfig.from_yaml(config_path)
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    try:
        loader = build_fi_jepa_dataloader(config, split, store=store, shuffle=shuffle)
        if split == "train":
            loader.dataset.set_epoch(train_epoch)
        batch = next(iter(loader))
        snapshot = batch_to_snapshot_frame(
            batch,
            store,
            config,
            sample_limit=sample_limit,
            asset_limit=asset_limit,
        )
        snapshot["request_epoch"] = train_epoch if split == "train" else 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_parquet(output_path, index=False, compression="zstd")
    finally:
        store.close()

    print(
        f"Wrote dataloader snapshot: {output_path} "
        f"({len(snapshot):,} scalar model-input rows)"
    )
    return output_path


# ============================================================================
# MASK HISTOGRAM EXPORT
# ============================================================================


def export_jepa_mask_histogram(
    config_path: Path,
    output_path: Path,
    *,
    batch_limit: int = 0,
    train_epoch: int = 0,
    example_limit: int = 64,
) -> Path:
    """Export JEPA mask histograms and example target-mask layouts.

    Three record types are emitted:

    - ``patch_position`` reports how often each temporal patch is selected.
    - ``selected_patch_count`` reports the distribution of selected patches per sample.
    - ``target_mask_example`` retains complete target masks for heatmap panels.

    ``batch_limit=0`` scans each complete split. Positive limits are useful for
    faster exploratory comparisons while still consuming real collated batches.
    """
    if batch_limit < 0 or train_epoch < 0 or example_limit <= 0:
        raise ValueError("batch_limit and train_epoch cannot be negative; example_limit must be positive.")

    config = FIJepaDataConfig.from_yaml(config_path)
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    rows: list[dict[str, object]] = []
    try:
        for split in ("train", "validation"):
            loader = build_fi_jepa_dataloader(config, split, store=store)
            if split == "train":
                loader.dataset.set_epoch(train_epoch)
            position_counts = {
                "jepa_context_mask": np.zeros(config.num_patches, dtype=np.int64),
                "jepa_target_mask": np.zeros(config.num_patches, dtype=np.int64),
            }
            selected_count_histograms = {
                "jepa_context_mask": np.zeros(config.num_patches + 1, dtype=np.int64),
                "jepa_target_mask": np.zeros(config.num_patches + 1, dtype=np.int64),
            }
            target_examples: list[np.ndarray] = []
            sample_count = 0
            batch_count = 0

            for batch_index, batch in enumerate(loader):
                if batch_limit and batch_index >= batch_limit:
                    break
                batch_count += 1
                for mask_name in ("jepa_context_mask", "jepa_target_mask"):
                    mask = _tensor_array(batch, mask_name).astype(bool, copy=False)
                    position_counts[mask_name] += mask.sum(axis=0, dtype=np.int64)
                    selected_counts = mask.sum(axis=1, dtype=np.int64)
                    selected_count_histograms[mask_name] += np.bincount(
                        selected_counts,
                        minlength=config.num_patches + 1,
                    )
                remaining_examples = example_limit - len(target_examples)
                if remaining_examples > 0:
                    target_mask = _tensor_array(batch, "jepa_target_mask").astype(bool, copy=False)
                    target_examples.extend(target_mask[:remaining_examples].copy())
                sample_count += int(_tensor_array(batch, "jepa_context_mask").shape[0])

            if sample_count == 0:
                raise RuntimeError(f"No {split} samples were available for mask histograms.")

            for mask_name in ("jepa_context_mask", "jepa_target_mask"):
                for patch_index, count in enumerate(position_counts[mask_name]):
                    rows.append(
                        {
                            "split": split,
                            "histogram_type": "patch_position",
                            "mask_name": mask_name,
                            "bin": patch_index,
                            "count": int(count),
                            "fraction": float(count / sample_count),
                            "sample_count": sample_count,
                            "batch_count": batch_count,
                            "request_epoch": train_epoch if split == "train" else 0,
                            "example_index": -1,
                        }
                    )
                for selected_patch_count, count in enumerate(
                    selected_count_histograms[mask_name]
                ):
                    rows.append(
                        {
                            "split": split,
                            "histogram_type": "selected_patch_count",
                            "mask_name": mask_name,
                            "bin": selected_patch_count,
                            "count": int(count),
                            "fraction": float(count / sample_count),
                            "sample_count": sample_count,
                            "batch_count": batch_count,
                            "request_epoch": train_epoch if split == "train" else 0,
                            "example_index": -1,
                        }
                    )
            for example_index, target_mask in enumerate(target_examples):
                for patch_index, selected in enumerate(target_mask):
                    rows.append(
                        {
                            "split": split,
                            "histogram_type": "target_mask_example",
                            "mask_name": "jepa_target_mask",
                            "bin": patch_index,
                            "count": int(selected),
                            "fraction": float(selected),
                            "sample_count": sample_count,
                            "batch_count": batch_count,
                            "request_epoch": train_epoch if split == "train" else 0,
                            "example_index": example_index,
                        }
                    )
    finally:
        store.close()

    histogram = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    histogram.to_parquet(output_path, index=False, compression="zstd")
    print(f"Wrote JEPA mask histogram: {output_path} ({len(histogram):,} histogram rows)")
    return output_path


# ============================================================================
# DISTRIBUTION CHARTS
# ============================================================================


def collect_model_input_value_samples(
    config_path: Path,
    *,
    batch_limit: int = 8,
    max_values_per_group: int = 200_000,
    train_epoch: int = 0,
) -> pd.DataFrame:
    """Sample valid scalar inputs from real train and validation batches.

    Values are included only when both their feature mask and stream-row mask
    are enabled. Each batch contributes a bounded random sample per input group,
    then each split/group is reduced to ``max_values_per_group`` values. This
    keeps the chart representative without materializing hundreds of millions
    of repeated lookback-window scalars.
    """
    if batch_limit <= 0 or max_values_per_group <= 0 or train_epoch < 0:
        raise ValueError("batch_limit and max_values_per_group must be positive; train_epoch cannot be negative.")

    config = FIJepaDataConfig.from_yaml(config_path)
    store = DensePanelStore(config.artifact_path, cache_root=config.cache_root)
    samples: dict[tuple[str, str], list[np.ndarray]] = {
        (split, group): []
        for split in ("train", "validation")
        for group in ("asset", "market", "macro")
    }
    try:
        for split in ("train", "validation"):
            loader = build_fi_jepa_dataloader(config, split, store=store)
            if split == "train":
                loader.dataset.set_epoch(train_epoch)
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, train_epoch, 7 if split == "train" else 11]))
            per_batch_limit = max(1, max_values_per_group // batch_limit)

            for batch_index, batch in enumerate(loader):
                print(f"Collecting {split} values: batch {batch_index + 1}", end="\r")
                
                if batch_index >= batch_limit:
                    break
                for group in ("asset", "market", "macro"):
                    values = _tensor_array(batch, f"{group}_patches")
                    feature_valid = _tensor_array(batch, f"{group}_feature_mask_patched")
                    if group == "asset":
                        row_valid = _tensor_array(batch, "valid_asset_mask_patched")[..., None]
                    else:
                        row_valid = _tensor_array(batch, f"valid_{group}_date_mask_patched")[..., None]
                        
                    # collect all valid finite values
                    valid_values = values[feature_valid & row_valid]
                    valid_values = valid_values[np.isfinite(valid_values)]
                    
                    # if there are too many values, randomly select a subset for this batch
                    if len(valid_values) > per_batch_limit:
                        selected = rng.choice(len(valid_values), size=per_batch_limit, replace=False)
                        valid_values = valid_values[selected]
                        
                    samples[(split, group)].append(valid_values.astype(np.float32, copy=False))
    finally:
        store.close()

    rows: list[pd.DataFrame] = []
    # split: [train, validation], group: [asset, market, macro], values are lists of arrays from each batch
    for (split, group), chunks in samples.items():
        values = np.concatenate(chunks)
        if len(values) > max_values_per_group:
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, train_epoch, len(group), len(split)]))
            values = values[rng.choice(len(values), size=max_values_per_group, replace=False)]
        rows.append(pd.DataFrame({"split": split, "input_group": group, "value": values}))
        
    print(f"rows: {[len(row) for row in rows]}")  # Debugging output to check the number of values collected per split/group
    return pd.concat(rows, ignore_index=True)


def plot_model_input_value_distribution(samples: pd.DataFrame, output_path: Path) -> Path:
    """Plot overlaid train-versus-validation distributions of valid model inputs."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    colors = {"train": "#1f77b4", "validation": "#d62728"}
    for axis, group in zip(axes, ("asset", "market", "macro"), strict=True):
        group_values = samples.loc[samples["input_group"].eq(group), "value"].to_numpy()
        lower, upper = np.quantile(group_values, [0.005, 0.995])
        bins = np.linspace(lower, upper, 81)
        for split in ("train", "validation"):
            values = samples.loc[
                samples["input_group"].eq(group) & samples["split"].eq(split),
                "value",
            ].to_numpy()
            clipped = values[(values >= lower) & (values <= upper)]
            axis.hist(
                clipped,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=colors[split],
                label=f"{split} (n={len(values):,})",
            )
        axis.set_title(f"{group.capitalize()} input values")
        axis.set_xlabel(f"Normalized value (central 99%, {lower:.2f} to {upper:.2f})")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("FI-JEPA Valid Scalar Inputs Served by Train and Validation Loaders", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Wrote model-input value distribution chart: {output_path}")
    return output_path


def plot_jepa_mask_distribution(histogram: pd.DataFrame, output_path: Path) -> Path:
    """Plot aggregate mask distributions and per-sample target-block layouts."""
    figure, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    colors = {"train": "#1f77b4", "validation": "#d62728"}
    mask_titles = {
        "jepa_context_mask": "Context",
        "jepa_target_mask": "Target",
    }
    for column, mask_name in enumerate(("jepa_context_mask", "jepa_target_mask")):
        position_axis = axes[0, column]
        count_axis = axes[1, column]
        for split in ("train", "validation"):
            position = histogram.loc[
                histogram["histogram_type"].eq("patch_position")
                & histogram["mask_name"].eq(mask_name)
                & histogram["split"].eq(split)
            ]
            selected_counts = histogram.loc[
                histogram["histogram_type"].eq("selected_patch_count")
                & histogram["mask_name"].eq(mask_name)
                & histogram["split"].eq(split)
            ]
            position_axis.plot(
                position["bin"],
                position["fraction"],
                linewidth=2,
                color=colors[split],
                label=split,
            )
            count_axis.step(
                selected_counts["bin"],
                selected_counts["fraction"],
                where="mid",
                linewidth=2,
                color=colors[split],
                label=split,
            )

        position_axis.set_title(f"{mask_titles[mask_name]} usage by patch position")
        position_axis.set_xlabel("Patch index, oldest to newest")
        position_axis.set_ylabel("Fraction of samples using patch")
        position_axis.set_ylim(0.0, 1.02)
        position_axis.grid(alpha=0.25)
        position_axis.legend()

        count_axis.set_title(f"{mask_titles[mask_name]} patches selected per sample")
        count_axis.set_xlabel("Selected patch count")
        count_axis.set_ylabel("Fraction of samples")
        count_axis.set_ylim(bottom=0.0)
        count_axis.grid(alpha=0.25)
        count_axis.legend()

    for column, split in enumerate(("train", "validation")):
        heatmap_axis = axes[2, column]
        examples = histogram.loc[
            histogram["histogram_type"].eq("target_mask_example")
            & histogram["split"].eq(split)
        ]
        matrix = examples.pivot(
            index="example_index",
            columns="bin",
            values="fraction",
        ).sort_index(axis=1).to_numpy()
        first_target = np.argmax(matrix, axis=1)
        matrix = matrix[np.argsort(first_target, kind="stable")]
        heatmap_axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
        heatmap_axis.set_title(f"{split.capitalize()} target-block layouts ({len(matrix)} samples)")
        heatmap_axis.set_xlabel("Patch index, oldest to newest")
        heatmap_axis.set_ylabel("Samples, sorted by first target")

    figure.suptitle("FI-JEPA Train vs Validation Mask Distribution", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Wrote JEPA mask distribution chart: {output_path}")
    return output_path


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse the dataloader snapshot CLI."""
    parser = argparse.ArgumentParser(
        description="Flatten one actual FI-JEPA dataloader batch slice into parquet."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dataloader.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/dataloader_snapshot.parquet"))
    parser.add_argument(
        "--mask-histogram-output",
        type=Path,
        default=Path("runs/dataloader_mask_histogram.parquet"),
    )
    parser.add_argument(
        "--value-chart-output",
        type=Path,
        default=Path("runs/dataloader_value_distribution.png"),
    )
    parser.add_argument(
        "--mask-chart-output",
        type=Path,
        default=Path("runs/dataloader_mask_distribution.png"),
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument(
        "--train-epoch",
        type=int,
        default=0,
        help="Training epoch encoded into deterministic random-K views and JEPA masks.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=1,
        help="Samples from the collated batch to export; use 0 for the entire batch.",
    )
    parser.add_argument(
        "--asset-limit",
        type=int,
        default=16,
        help="Selected asset slots to export per sample; use 0 for every served asset.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override runtime split shuffling. By default train shuffles and validation does not.",
    )
    parser.add_argument(
        "--histogram-batch-limit",
        type=int,
        default=0,
        help="Batches per split used for mask histograms; use 0 for each complete split.",
    )
    parser.add_argument(
        "--mask-example-limit",
        type=int,
        default=64,
        help="Complete target masks per split displayed in the mask-layout heatmaps.",
    )
    parser.add_argument(
        "--value-chart-batch-limit",
        type=int,
        default=8,
        help="Real batches per split sampled for the train-versus-validation value chart.",
    )
    parser.add_argument(
        "--value-chart-max-values-per-group",
        type=int,
        default=200_000,
        help="Maximum sampled scalar values per split and input group.",
    )
    parser.add_argument(
        "--skip-mask-histogram",
        action="store_true",
        help="Skip the mask histogram parquet and chart.",
    )
    parser.add_argument(
        "--skip-value-chart",
        action="store_true",
        help="Skip the train-versus-validation model-input value chart.",
    )
    return parser.parse_args()


def main() -> None:
    """Export one inspectable snapshot from the live dataloader path."""
    args = parse_args()
    export_dataloader_snapshot(
        args.config,
        args.output,
        split=args.split,
        sample_limit=args.sample_limit,
        asset_limit=args.asset_limit,
        shuffle=args.shuffle,
        train_epoch=args.train_epoch,
    )
    if not args.skip_mask_histogram:
        mask_histogram_path = export_jepa_mask_histogram(
            args.config,
            args.mask_histogram_output,
            batch_limit=args.histogram_batch_limit,
            train_epoch=args.train_epoch,
            example_limit=args.mask_example_limit,
        )
        plot_jepa_mask_distribution(
            pd.read_parquet(mask_histogram_path),
            args.mask_chart_output,
        )
    if not args.skip_value_chart:
        value_samples = collect_model_input_value_samples(
            args.config,
            batch_limit=args.value_chart_batch_limit,
            max_values_per_group=args.value_chart_max_values_per_group,
            train_epoch=args.train_epoch,
        )
        plot_model_input_value_distribution(value_samples, args.value_chart_output)


if __name__ == "__main__":
    main()
