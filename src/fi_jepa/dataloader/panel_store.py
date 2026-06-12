from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

Split = Literal["train", "validation", "diagnostic"]

REQUIRED_ARTIFACT_FILES = {
    "manifest.json",
    "config_resolved.yaml",
    "dates.parquet",
    "assets.parquet",
    "feature_manifest.parquet",
    "normalization.parquet",
    "quality_report.json",
    "train_asset_features.parquet",
    "train_market_features.parquet",
    "train_macro_features.parquet",
    "validation_asset_features.parquet",
    "validation_market_features.parquet",
    "validation_macro_features.parquet",
}
FORBIDDEN_FEATURE_PATTERN = re.compile(r"future_|target|label", re.IGNORECASE)
CACHE_FORMAT_VERSION = 1
CACHE_MANIFEST_NAME = "cache_manifest.json"
CACHE_ARRAY_NAMES = (
    "asset_x",
    "asset_feature_mask",
    "valid_asset",
    "market_x",
    "market_feature_mask",
    "valid_market_date",
    "macro_x",
    "macro_feature_mask",
    "valid_macro_date",
)
METADATA_ARRAY_NAMES = (
    "train_permission",
    "validation_permission",
    "protected_holdout",
    "sample_eligible",
    "validation_sample",
)


# ============================================================================
# FROZEN PANEL STORE
# ============================================================================


class FrozenPanelStore:
    """Load one immutable sparse artifact through a worker-safe dense cache.

    Sparse facts are streamed once into persistent ``.npy`` memmaps indexed by
    the artifact's contiguous date and asset IDs. The completed cache is opened
    read-only in the parent and reopened after worker-process deserialization,
    avoiding dense-array copies under spawn. Every window slice reapplies the
    requested split's date permissions, so retaining both fact sets in one
    cache cannot expose protected validation facts to training samples.
    """

    def __init__(
        self,
        artifact_path: Path | str,
        *,
        cache_root: Path | str | None = None,
    ):
        self.artifact_path = Path(artifact_path)
        self._validate_required_files()
        self.manifest = json.loads(
            (self.artifact_path / "manifest.json").read_text(encoding="utf-8")
        )
        self.dataset_version = str(
            self.manifest.get("build_id")
            or self.manifest.get("artifact_name")
            or self.manifest.get("name")
            or self.artifact_path.name
        )

        # Manifests define all dense array axes and split permissions.
        self.dates = pd.read_parquet(self.artifact_path / "dates.parquet")
        self.assets = pd.read_parquet(self.artifact_path / "assets.parquet")
        self.feature_manifest = pd.read_parquet(self.artifact_path / "feature_manifest.parquet")
        self.resolved_config = yaml.safe_load(
            (self.artifact_path / "config_resolved.yaml").read_text(encoding="utf-8")
        )
        self._validate_manifests()

        # Feature-manifest order, rather than an architecture document, is the
        # source of truth for tensor dimensions and feature positions.
        self.feature_names = {
            group: (
                self.feature_manifest.loc[self.feature_manifest["input_group"].eq(group)]
                .sort_values("feature_index")["feature_name"]
                .tolist()
            )
            for group in ("asset", "market", "macro")
        }
        self.date_count = len(self.dates)
        self.asset_count = len(self.assets)

        # Permissions are reapplied when slicing windows. Holding both fact
        # sets in one cache therefore does not grant cross-split visibility.
        self.train_permission = self.dates["train_fact_allowed"].to_numpy(dtype=bool)
        self.validation_permission = self.dates["validation_fact_allowed"].to_numpy(dtype=bool)
        self.protected_holdout = self.dates["protected_holdout"].to_numpy(dtype=bool)
        self.sample_eligible = self.dates["sample_eligible"].to_numpy(dtype=bool)
        self.validation_sample = self.dates["validation_sample"].to_numpy(dtype=bool)
        self.date_values = self.dates["date"].tolist()
        self._mark_metadata_arrays_read_only()

        default_cache_root = self.artifact_path.parent / ".frozen_panel_store_cache"
        self.cache_root = Path(cache_root or default_cache_root).resolve()
        if self.cache_root.is_relative_to(self.artifact_path.resolve()):
            raise ValueError("cache_root must be outside the immutable artifact directory.")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._cache_key = self._build_cache_key()
        self.cache_path = (
            self.cache_root / f"{self.artifact_path.name}_{self._cache_key[:20]}"
        ).resolve()
        self._ensure_cache()
        self._open_cache_arrays()

    def __getstate__(self) -> dict[str, object]:
        """Serialize store metadata without copying any dense mapped arrays."""
        state = self.__dict__.copy()
        for name in CACHE_ARRAY_NAMES:
            state.pop(name, None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore metadata and reopen the completed cache read-only."""
        self.__dict__.update(state)
        self._mark_metadata_arrays_read_only()
        self._open_cache_arrays()

    def _mark_metadata_arrays_read_only(self) -> None:
        """Prevent mutation of the small split-permission arrays held in memory."""
        for name in METADATA_ARRAY_NAMES:
            array = getattr(self, name)
            array.setflags(write=False)

    def _validate_required_files(self) -> None:
        """Require the complete frozen-artifact contract before loading data."""
        missing = sorted(
            name for name in REQUIRED_ARTIFACT_FILES if not (self.artifact_path / name).is_file()
        )
        if missing:
            raise FileNotFoundError(f"Frozen artifact is missing required files: {missing}")

    def _validate_manifests(self) -> None:
        """Validate manifest schemas, contiguous IDs, and input-only features.

        Dense indexing relies on ordered contiguous date, asset, and per-group
        feature IDs. Feature names are also screened here so future or target
        columns fail before any fact file is loaded into model-facing arrays.
        """
        required_date_columns = {
            "date_idx",
            "date",
            "sample_eligible",
            "validation_sample",
            "protected_holdout",
            "train_fact_allowed",
            "validation_fact_allowed",
        }
        missing_dates = sorted(required_date_columns - set(self.dates.columns))
        if missing_dates:
            raise ValueError(f"dates.parquet is missing columns: {missing_dates}")
        if self.dates["date_idx"].tolist() != list(range(len(self.dates))):
            raise ValueError("dates.parquet date_idx must be contiguous and ordered.")
        if not self.dates["date"].is_monotonic_increasing:
            raise ValueError("dates.parquet dates must be ordered.")

        required_asset_columns = {"asset_id", "symbol", "trainable"}
        missing_assets = sorted(required_asset_columns - set(self.assets.columns))
        if missing_assets:
            raise ValueError(f"assets.parquet is missing columns: {missing_assets}")
        if self.assets["asset_id"].tolist() != list(range(len(self.assets))):
            raise ValueError("assets.parquet asset_id must be contiguous and ordered.")

        required_feature_columns = {"feature_name", "feature_index", "input_group", "dtype"}
        missing_features = sorted(required_feature_columns - set(self.feature_manifest.columns))
        if missing_features:
            raise ValueError(f"feature_manifest.parquet is missing columns: {missing_features}")
        if set(self.feature_manifest["input_group"]) != {"asset", "market", "macro"}:
            raise ValueError("Feature manifest must contain asset, market, and macro groups.")

        forbidden = (
            self.feature_manifest["feature_name"]
            .astype(str)
            .str.contains(FORBIDDEN_FEATURE_PATTERN)
        )
        if forbidden.any():
            names = self.feature_manifest.loc[forbidden, "feature_name"].tolist()
            raise ValueError(f"Forbidden target-like features in artifact: {names}")
        for group, frame in self.feature_manifest.groupby("input_group"):
            indices = frame.sort_values("feature_index")["feature_index"].tolist()
            if indices != list(range(len(indices))):
                raise ValueError(f"{group} feature indices must be contiguous from zero.")

    def _cache_array_specs(self) -> dict[str, dict[str, object]]:
        """Return the authoritative shape and dtype contract for cached arrays."""
        asset_dim = len(self.feature_names["asset"])
        market_dim = len(self.feature_names["market"])
        macro_dim = len(self.feature_names["macro"])
        return {
            "asset_x": {
                "shape": [self.date_count, self.asset_count, asset_dim],
                "dtype": "float32",
            },
            "asset_feature_mask": {
                "shape": [self.date_count, self.asset_count, asset_dim],
                "dtype": "bool",
            },
            "valid_asset": {
                "shape": [self.date_count, self.asset_count],
                "dtype": "bool",
            },
            "market_x": {
                "shape": [self.date_count, market_dim],
                "dtype": "float32",
            },
            "market_feature_mask": {
                "shape": [self.date_count, market_dim],
                "dtype": "bool",
            },
            "valid_market_date": {"shape": [self.date_count], "dtype": "bool"},
            "macro_x": {
                "shape": [self.date_count, macro_dim],
                "dtype": "float32",
            },
            "macro_feature_mask": {
                "shape": [self.date_count, macro_dim],
                "dtype": "bool",
            },
            "valid_macro_date": {"shape": [self.date_count], "dtype": "bool"},
        }

    def _build_cache_key(self) -> str:
        """Hash the cache format, artifact identity, manifest, and file metadata."""
        files = {}
        for name in sorted(REQUIRED_ARTIFACT_FILES):
            stat = (self.artifact_path / name).stat()
            files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        identity = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "artifact_path": str(self.artifact_path.resolve()),
            "artifact_manifest": self.manifest,
            "required_files": files,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validate_cache(self, cache_path: Path) -> bool:
        """Return whether one published cache exactly matches the expected contract."""
        try:
            cache_manifest = json.loads(
                (cache_path / CACHE_MANIFEST_NAME).read_text(encoding="utf-8")
            )
            if cache_manifest != {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "cache_key": self._cache_key,
                "arrays": self._cache_array_specs(),
            }:
                return False
            for name, spec in self._cache_array_specs().items():
                array = np.load(cache_path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                try:
                    if list(array.shape) != spec["shape"]:
                        return False
                    if array.dtype != np.dtype(str(spec["dtype"])):
                        return False
                finally:
                    self._close_memmap(array)
        except (
            AttributeError,
            EOFError,
            FileNotFoundError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return False
        return True

    def _ensure_cache(self) -> None:
        """Build and atomically publish the dense cache when no valid cache exists."""
        if self._validate_cache(self.cache_path):
            return
        if self.cache_path.exists():
            if self.cache_path.is_dir():
                shutil.rmtree(self.cache_path)
            else:
                self.cache_path.unlink()

        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{self.cache_path.name}.tmp-", dir=self.cache_root)
        )
        try:
            self._build_cache(temporary_path)
            try:
                temporary_path.rename(self.cache_path)
            except OSError:
                # Another process may have published the same immutable cache
                # while this process was building its private temporary copy.
                if not self._validate_cache(self.cache_path):
                    raise
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)

        if not self._validate_cache(self.cache_path):
            raise RuntimeError(f"Failed to publish a valid FrozenPanelStore cache: {self.cache_path}")

    def _build_cache(self, cache_path: Path) -> None:
        """Stream validated sparse facts into writable maps, then mark completion."""
        self._allocate_cache_arrays(cache_path)
        try:
            # Train and validation facts are date-disjoint, so both can occupy
            # one dense cache without overwriting one another.
            for split in ("train", "validation"):
                self._load_asset_facts(split)
                self._load_date_facts(split, "market")
                self._load_date_facts(split, "macro")
            self._flush_cache_arrays()
        finally:
            self._close_cache_arrays()

        cache_manifest = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_key": self._cache_key,
            "arrays": self._cache_array_specs(),
        }
        (cache_path / CACHE_MANIFEST_NAME).write_text(
            json.dumps(cache_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _allocate_cache_arrays(self, cache_path: Path) -> None:
        """Allocate zero-filled writable memmaps and in-memory overlap trackers.

        Asset arrays use ``[date, asset, feature]``. Market and macro arrays use
        ``[date, feature]``. Zeros are storage fill values only; paired boolean
        masks preserve all missingness semantics.
        """
        for name, spec in self._cache_array_specs().items():
            array = np.lib.format.open_memmap(
                cache_path / f"{name}.npy",
                mode="w+",
                dtype=np.dtype(str(spec["dtype"])),
                shape=tuple(spec["shape"]),
            )
            array[...] = 0
            setattr(self, name, array)

        # Separate row-presence masks detect split overlap even when valid_date is false.
        self._market_row_present = np.zeros(self.date_count, dtype=bool)
        self._macro_row_present = np.zeros(self.date_count, dtype=bool)

    def _flush_cache_arrays(self) -> None:
        """Flush every writable map before the cache directory is published."""
        for name in CACHE_ARRAY_NAMES:
            array = getattr(self, name)
            if isinstance(array, np.memmap):
                array.flush()

    @staticmethod
    def _close_memmap(array: np.ndarray) -> None:
        """Close one NumPy memmap's underlying file mapping when present."""
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()

    def _close_cache_arrays(self) -> None:
        """Close and remove live mapped-array attributes."""
        for name in CACHE_ARRAY_NAMES:
            array = getattr(self, name, None)
            if array is not None:
                self._close_memmap(array)
                delattr(self, name)
        self.__dict__.pop("_market_row_present", None)
        self.__dict__.pop("_macro_row_present", None)

    def _open_cache_arrays(self) -> None:
        """Open every completed dense array read-only in the current process."""
        if not self._validate_cache(self.cache_path):
            raise RuntimeError(f"FrozenPanelStore cache is invalid: {self.cache_path}")
        for name in CACHE_ARRAY_NAMES:
            array = np.load(self.cache_path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            setattr(self, name, array)

    def _validate_fact_schema(self, path: Path, group: str) -> None:
        """Validate one sparse fact file against its feature-manifest group."""
        columns = set(pq.read_schema(path).names)

        # Every value must retain its paired validity column. Zero alone cannot
        # distinguish a real observation from a missing-value fill.
        required = {"date", "date_idx", *self.feature_names[group]}
        required.update(f"{name}__valid" for name in self.feature_names[group])
        required.add("valid_asset" if group == "asset" else "valid_date")
        if group == "asset":
            required.add("asset_id")

        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        forbidden = sorted(name for name in columns if FORBIDDEN_FEATURE_PATTERN.search(name))
        if forbidden:
            raise ValueError(f"{path.name} contains forbidden target-like columns: {forbidden}")

    def _load_asset_facts(self, split: str) -> None:
        """Stream sparse asset facts directly into final dense array positions.

        Parquet row groups are processed in bounded batches. Each sparse
        ``(date_idx, asset_id)`` row writes directly to ``[date, asset, feature]``
        and duplicate or overlapping keys are rejected before assignment.
        """
        path = self.artifact_path / f"{split}_asset_features.parquet"
        self._validate_fact_schema(path, "asset")
        features = self.feature_names["asset"]
        columns = ["date_idx", "asset_id", "valid_asset", *features]
        columns.extend(f"{name}__valid" for name in features)

        # Avoid materializing the full sparse asset panel as an intermediate frame.
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
            frame = batch.to_pandas()
            if frame.duplicated(["date_idx", "asset_id"]).any():
                raise ValueError(f"{path.name} contains duplicate (date_idx, asset_id) rows.")
            date_ids = frame["date_idx"].to_numpy(dtype=np.int64)
            asset_ids = frame["asset_id"].to_numpy(dtype=np.int64)
            if (
                (date_ids < 0).any()
                or (date_ids >= self.date_count).any()
                or (asset_ids < 0).any()
                or (asset_ids >= self.asset_count).any()
            ):
                raise ValueError(f"{path.name} contains out-of-range fact keys.")
            if self.valid_asset[date_ids, asset_ids].any():
                raise ValueError(f"{path.name} overlaps existing asset facts.")

            # Advanced indexing writes sparse rows into final [date, asset, feature] slots.
            self.asset_x[date_ids, asset_ids] = frame[features].to_numpy(dtype=np.float32)
            self.asset_feature_mask[date_ids, asset_ids] = frame[
                [f"{name}__valid" for name in features]
            ].to_numpy(dtype=bool)
            self.valid_asset[date_ids, asset_ids] = frame["valid_asset"].to_numpy(dtype=bool)

    def _load_date_facts(self, split: str, group: Literal["market", "macro"]) -> None:
        """Stream one date-level feature group into the shared date spine.

        Market and macro files are dense within a split but remain keyed by
        ``date_idx``. Loading through those IDs preserves alignment across split
        files and allows explicit duplicate/overlap checks.
        """
        path = self.artifact_path / f"{split}_{group}_features.parquet"
        self._validate_fact_schema(path, group)

        features = self.feature_names[group]
        columns = ["date_idx", "valid_date", *features]
        columns.extend(f"{name}__valid" for name in features)
        values = self.market_x if group == "market" else self.macro_x
        feature_mask = self.market_feature_mask if group == "market" else self.macro_feature_mask
        valid_date = self.valid_market_date if group == "market" else self.valid_macro_date
        row_present = self._market_row_present if group == "market" else self._macro_row_present

        for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
            frame = batch.to_pandas()
            if frame.duplicated(["date_idx"]).any():
                raise ValueError(f"{path.name} contains duplicate date_idx rows.")
            date_ids = frame["date_idx"].to_numpy(dtype=np.int64)
            if (date_ids < 0).any() or (date_ids >= self.date_count).any():
                raise ValueError(f"{path.name} contains out-of-range date_idx values.")
            if row_present[date_ids].any():
                raise ValueError(f"{path.name} overlaps existing {group} facts.")

            values[date_ids] = frame[features].to_numpy(dtype=np.float32)
            feature_mask[date_ids] = frame[[f"{name}__valid" for name in features]].to_numpy(
                dtype=bool
            )
            valid_date[date_ids] = frame["valid_date"].to_numpy(dtype=bool)
            row_present[date_ids] = True

    def permission_for(self, split: Split) -> np.ndarray:
        """Return the date-level fact permission mask for a requested split."""
        return self.train_permission if split == "train" else self.validation_permission

    def sample_indices_for(self, split: Split) -> np.ndarray:
        """Return candidate sample endpoints for training or evaluation."""
        if split == "train":
            return np.flatnonzero(self.sample_eligible & self.train_permission)
        return np.flatnonzero(self.validation_sample)

    def endpoint_asset_ids(self, date_idx: int, split: Split) -> np.ndarray:
        """Return endpoint-valid assets only when the split may read the date."""
        if not self.permission_for(split)[date_idx]:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.valid_asset[date_idx]).astype(np.int64)

    def window_masks(
        self,
        sample_date_idx: int,
        asset_ids: np.ndarray,
        split: Split,
        lookback_days: int,
    ) -> dict[str, np.ndarray]:
        """Build permission-filtered daily masks for one lookback window.

        Early-history windows are left-padded to the requested fixed length.
        Padded asset IDs use ``-1`` externally and gather harmless asset zero
        internally, after which slot and split permissions clear placeholder
        validity. No feature-value arrays are copied by this method.
        """
        asset_ids = np.asarray(asset_ids, dtype=np.int64)
        asset_slot_mask = asset_ids >= 0
        safe_asset_ids = np.where(asset_slot_mask, asset_ids, 0)

        # Fixed destination arrays make all downstream samples shape-stable.
        valid_asset_mask = np.zeros((lookback_days, len(asset_ids)), dtype=bool)
        valid_market_date_mask = np.zeros(lookback_days, dtype=bool)
        valid_macro_date_mask = np.zeros(lookback_days, dtype=bool)
        holdout_date_mask = np.zeros(lookback_days, dtype=bool)
        padded_date_mask = np.ones(lookback_days, dtype=bool)
        date_indices = np.full(lookback_days, -1, dtype=np.int64)

        source_start = max(0, sample_date_idx - lookback_days + 1)
        source_stop = sample_date_idx + 1
        destination_start = lookback_days - (source_stop - source_start)
        destination = slice(destination_start, lookback_days)
        source = slice(source_start, source_stop)
        permission = self.permission_for(split)[source]

        # Gather real source dates into the right side of the padded destination.
        valid_asset_mask[destination] = self.valid_asset[source][:, safe_asset_ids]
        valid_asset_mask[destination] &= permission[:, None] & asset_slot_mask[None, :]
        valid_market_date_mask[destination] = self.valid_market_date[source] & permission
        valid_macro_date_mask[destination] = self.valid_macro_date[source] & permission
        holdout_date_mask[destination] = self.protected_holdout[source]
        padded_date_mask[destination] = False
        date_indices[destination] = np.arange(source_start, source_stop, dtype=np.int64)

        # A day is usable context when any stream has split-permitted data.
        valid_date_mask = (
            valid_market_date_mask | valid_macro_date_mask | valid_asset_mask.any(axis=1)
        )
        return {
            "asset_slot_mask": asset_slot_mask,
            "valid_asset_mask": valid_asset_mask,
            "valid_market_date_mask": valid_market_date_mask,
            "valid_macro_date_mask": valid_macro_date_mask,
            "valid_date_mask": valid_date_mask,
            "holdout_date_mask": holdout_date_mask,
            "padded_date_mask": padded_date_mask,
            "date_indices": date_indices,
        }

    def window(
        self,
        sample_date_idx: int,
        asset_ids: np.ndarray,
        split: Split,
        lookback_days: int,
    ) -> dict[str, np.ndarray]:
        """Reconstruct a zero-filled dense window with every validity mask.

        Values are gathered only for real source dates. Split permissions and
        padded-asset masks are then applied to both values and feature-validity
        masks, ensuring inaccessible values remain exactly zero.
        """
        masks = self.window_masks(sample_date_idx, asset_ids, split, lookback_days)
        asset_ids = np.asarray(asset_ids, dtype=np.int64)
        safe_asset_ids = np.where(masks["asset_slot_mask"], asset_ids, 0)

        # [W, A, F_asset].
        asset_x = np.zeros((lookback_days, len(asset_ids), self.asset_x.shape[2]), np.float32)
        asset_feature_mask = np.zeros_like(asset_x, dtype=bool)
        # [W, F_market].
        market_x = np.zeros((lookback_days, self.market_x.shape[1]), np.float32)
        market_feature_mask = np.zeros_like(market_x, dtype=bool)
        # [W, F_macro].
        macro_x = np.zeros((lookback_days, self.macro_x.shape[1]), np.float32)
        macro_feature_mask = np.zeros_like(macro_x, dtype=bool)

        real_slots = masks["date_indices"] >= 0
        source_ids = masks["date_indices"][real_slots]
        permission = self.permission_for(split)[source_ids]

        # Gather only real dates, then clear values and masks the split cannot consume.
        asset_x[real_slots] = self.asset_x[source_ids][:, safe_asset_ids]
        asset_feature_mask[real_slots] = self.asset_feature_mask[source_ids][:, safe_asset_ids]
        asset_x[real_slots] *= permission[:, None, None] & masks["asset_slot_mask"][None, :, None]
        asset_feature_mask[real_slots] &= (
            permission[:, None, None] & masks["asset_slot_mask"][None, :, None]
        )
        market_x[real_slots] = self.market_x[source_ids] * permission[:, None]
        market_feature_mask[real_slots] = self.market_feature_mask[source_ids] & permission[:, None]
        macro_x[real_slots] = self.macro_x[source_ids] * permission[:, None]
        macro_feature_mask[real_slots] = self.macro_feature_mask[source_ids] & permission[:, None]
        return {
            **masks,
            "asset_x": asset_x,
            "asset_feature_mask": asset_feature_mask,
            "market_x": market_x,
            "market_feature_mask": market_feature_mask,
            "macro_x": macro_x,
            "macro_feature_mask": macro_feature_mask,
        }
