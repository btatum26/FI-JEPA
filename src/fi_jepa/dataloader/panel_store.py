from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from fi_jepa.dataloader.validation import (
    validate_build_id,
    validate_cache,
    validate_cache_root,
    validate_fact_schema,
    validate_required_artifact_files,
    validate_source_manifests,
)

Split = Literal["train", "validation"]

CACHE_FORMAT_VERSION = 1
SOURCE_HASH_FILES = ("manifest.json", "config_resolved.yaml", "feature_manifest.parquet")
SPARSE_FACT_FILES = tuple(
    f"{split}_{group}_features.parquet"
    for split in ("train", "validation")
    for group in ("asset", "market", "macro")
)
REQUIRED_ARTIFACT_FILES = {
    *SOURCE_HASH_FILES,
    *SPARSE_FACT_FILES,
    "dates.parquet",
    "assets.parquet",
}
PANEL_ARRAY_NAMES = tuple(
    f"{split}_{name}"
    for split, names in (
        (
            "train",
            (
                "asset_x",
                "asset_feature_mask",
                "valid_asset_mask",
                "market_x",
                "market_feature_mask",
                "valid_market_date",
                "macro_x",
                "macro_feature_mask",
                "valid_macro_date",
                "target_date_mask",
            ),
        ),
        (
            "validation",
            (
                "asset_x",
                "asset_feature_mask",
                "valid_asset_mask",
                "market_x",
                "market_feature_mask",
                "valid_market_date",
                "macro_x",
                "macro_feature_mask",
                "valid_macro_date",
            ),
        ),
    )
    for name in names
)
CACHE_ARRAY_NAMES = ("dates", "assets", *PANEL_ARRAY_NAMES)
CACHE_METADATA_FILES = (
    "config_resolved.yaml",
    "feature_manifest.parquet",
    "train_request_index.parquet",
    "validation_request_index.parquet",
)


# ============================================================================
# DENSE PANEL STORE
# ============================================================================


class DensePanelStore:
    """Open an immutable, split-specific dense panel cache.

    Parent-process construction validates or builds the cache before workers
    exist. Spawned workers receive only metadata and the completed cache path;
    deserialization reopens every NumPy array read-only and never validates,
    repairs, deletes, or publishes cache files.
    """

    def __init__(
        self,
        artifact_path: Path | str,
        *,
        cache_root: Path | str = Path("data/cache/dense_panel"),
    ):
        self.artifact_path = Path(artifact_path).resolve()
        self.cache_root = Path(cache_root).resolve()
        validate_cache_root(self.artifact_path, self.cache_root)

        # validate the build_id matches the artifact
        validate_required_artifact_files(self.artifact_path, REQUIRED_ARTIFACT_FILES)
        self.manifest = json.loads((self.artifact_path / "manifest.json").read_text(encoding="utf-8"))
        self.dataset_version = validate_build_id(self.manifest)
        self.cache_path = (self.cache_root / f"{self.dataset_version}_v{CACHE_FORMAT_VERSION}").resolve()
        self._source_identity = self._build_source_identity()

        self.cache_root.mkdir(parents=True, exist_ok=True)
        print(f"Dense panel cache: checking {self.cache_path}")
        if self._validate_cache(self.cache_path):
            print(f"Dense panel cache: reusing {self.cache_path}")
        else:
            self._load_source_metadata()
            self._expected_manifest = self._build_expected_cache_manifest()
            self._rebuild_cache()

        self.feature_manifest = pd.read_parquet(self.cache_path / "feature_manifest.parquet")
        self.resolved_config = yaml.safe_load((self.cache_path / "config_resolved.yaml").read_text(encoding="utf-8"))
        self.feature_names = {
            group: (
                self.feature_manifest.loc[self.feature_manifest["input_group"].eq(group)]
                .sort_values("feature_index")["feature_name"]
                .tolist()
            )
            for group in ("asset", "market", "macro")
        }
        self._load_request_indexes()
        self._open_cache_arrays()
        self.date_count = len(self.dates)
        self.asset_count = len(self.assets)

        # Source DataFrames exist only during a cache rebuild.
        if hasattr(self, "_source_dates"):
            del self._source_dates
            del self._source_assets

    def __getstate__(self) -> dict[str, object]:
        """Serialize metadata without copying mapped dense-panel arrays."""
        state = self.__dict__.copy()
        for name in CACHE_ARRAY_NAMES:
            state.pop(name, None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Reopen an already-published cache without worker-side mutation."""
        self.__dict__.update(state)
        self._open_cache_arrays()

    def request_index_for(self, split: Split) -> pd.DataFrame:
        """Return the immutable request index for one split."""
        return self.train_request_index if split == "train" else self.validation_request_index

    def endpoint_asset_ids(self, sample_date_idx: int, split: Split) -> np.ndarray:
        """Return asset IDs with a valid observation at one split endpoint."""
        valid = getattr(self, f"{split}_valid_asset_mask")
        return np.flatnonzero(valid[sample_date_idx]).astype(np.int64)

    def arrays_for(self, split: Split) -> dict[str, np.ndarray]:
        """Return one split's read-only panel arrays under runtime field names."""
        names = (
            "asset_x",
            "asset_feature_mask",
            "valid_asset_mask",
            "market_x",
            "market_feature_mask",
            "valid_market_date",
            "macro_x",
            "macro_feature_mask",
            "valid_macro_date",
        )
        arrays = {name: getattr(self, f"{split}_{name}") for name in names}
        if split == "train":
            arrays["target_date_mask"] = self.train_target_date_mask
        return arrays

    def close(self) -> None:
        """Close every read-only cache mapping owned by this process."""
        for name in CACHE_ARRAY_NAMES:
            array = getattr(self, name, None)
            if array is not None:
                self._close_memmap(array)
                delattr(self, name)

    # ============================================================================
    # SOURCE VALIDATION AND CACHE IDENTITY
    # ============================================================================

    def _load_source_metadata(self) -> None:
        """Parse and validate source metadata only when the dense cache must be rebuilt."""
        self._source_dates = pd.read_parquet(self.artifact_path / "dates.parquet")
        self._source_assets = pd.read_parquet(self.artifact_path / "assets.parquet")
        self.feature_manifest = pd.read_parquet(self.artifact_path / "feature_manifest.parquet")
        self.resolved_config = yaml.safe_load((self.artifact_path / "config_resolved.yaml").read_text(encoding="utf-8"))
        validate_source_manifests(self._source_dates, self._source_assets, self.feature_manifest)
        self.feature_names = {
            group: (
                self.feature_manifest.loc[self.feature_manifest["input_group"].eq(group)]
                .sort_values("feature_index")["feature_name"]
                .tolist()
            )
            for group in ("asset", "market", "macro")
        }
        self.date_count = len(self._source_dates)
        self.asset_count = len(self._source_assets)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Return the SHA-256 digest of one source-contract file."""
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _array_specs(self) -> dict[str, dict[str, object]]:
        """Return the exact required shape and dtype of every cache array."""
        d = self.date_count
        n = self.asset_count
        fa = len(self.feature_names["asset"])
        fm = len(self.feature_names["market"])
        fx = len(self.feature_names["macro"])
        date_values = pd.to_datetime(self._source_dates["date"]).to_numpy(dtype="datetime64[D]")
        asset_values = np.asarray(self._source_assets["symbol"].astype(str).tolist(), dtype=str)
        common = {
            "asset_x": ([d, n, fa], "float32"),
            "asset_feature_mask": ([d, n, fa], "bool"),
            "valid_asset_mask": ([d, n], "bool"),
            "market_x": ([d, fm], "float32"),
            "market_feature_mask": ([d, fm], "bool"),
            "valid_market_date": ([d], "bool"),
            "macro_x": ([d, fx], "float32"),
            "macro_feature_mask": ([d, fx], "bool"),
            "valid_macro_date": ([d], "bool"),
        }
        specs = {
            "dates": {"shape": [d], "dtype": date_values.dtype.str},
            "assets": {"shape": [n], "dtype": asset_values.dtype.str},
        }
        for split in ("train", "validation"):
            for name, (shape, dtype) in common.items():
                specs[f"{split}_{name}"] = {"shape": shape, "dtype": dtype}
        specs["train_target_date_mask"] = {"shape": [d], "dtype": "bool"}
        return specs

    def _build_source_identity(self) -> dict[str, object]:
        """Fingerprint every source input without parsing source Parquets."""
        fact_stats = {}
        for name in SPARSE_FACT_FILES:
            stat = (self.artifact_path / name).stat()
            fact_stats[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "source_artifact_id": self.dataset_version,
            "source_manifest_sha256": self._sha256(self.artifact_path / "manifest.json"),
            "source_config_sha256": self._sha256(self.artifact_path / "config_resolved.yaml"),
            "source_feature_manifest_sha256": self._sha256(self.artifact_path / "feature_manifest.parquet"),
            "source_dates_sha256": self._sha256(self.artifact_path / "dates.parquet"),
            "source_assets_sha256": self._sha256(self.artifact_path / "assets.parquet"),
            "source_sparse_fact_files": fact_stats,
        }

    def _build_expected_cache_manifest(self) -> dict[str, object]:
        """Build the strict manifest required for a newly rebuilt cache."""
        specs = self._array_specs()
        return {
            **self._source_identity,
            "array_shapes": {name: spec["shape"] for name, spec in specs.items()},
            "array_dtypes": {name: spec["dtype"] for name, spec in specs.items()},
        }

    def _validate_cache(self, cache_path: Path) -> bool:
        """Return whether a published cache exactly matches the source contract."""
        return validate_cache(
            cache_path,
            self._source_identity,
            CACHE_ARRAY_NAMES,
            CACHE_METADATA_FILES,
        )

    # ============================================================================
    # CACHE BUILD AND WORKER-SAFE OPEN
    # ============================================================================

    def _rebuild_cache(self) -> None:
        """Build and atomically publish a cache already known to be stale or missing."""
        print(f"Dense panel cache: rebuilding {self.cache_path}")
        if self.cache_path.exists():
            if self.cache_path.is_dir():
                shutil.rmtree(self.cache_path)
            else:
                self.cache_path.unlink()
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.cache_path.name}.tmp-", dir=self.cache_root)
        )
        try:
            self._build_cache(temporary)
            try:
                temporary.rename(self.cache_path)
                print(f"Dense panel cache: published {self.cache_path}")
            except OSError:
                if not self._validate_cache(self.cache_path):
                    raise
                print(f"Dense panel cache: reusing concurrently published {self.cache_path}")
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        if not self._validate_cache(self.cache_path):
            raise RuntimeError(f"Failed to publish a valid dense panel cache: {self.cache_path}")

    def _build_cache(self, cache_path: Path) -> None:
        """Write split-specific panels and request indexes, then manifest last."""
        self._allocate_cache_arrays(cache_path)
        try:
            for split in ("train", "validation"):
                self._load_asset_facts(split)
                self._load_date_facts(split, "market")
                self._load_date_facts(split, "macro")
            self.train_target_date_mask[:] = self._source_dates[
                "train_fact_allowed"
            ].to_numpy(dtype=bool)
            self._flush_cache_arrays()
            self._write_request_indexes(cache_path)
        finally:
            self._close_cache_arrays()

        shutil.copy2(self.artifact_path / "config_resolved.yaml", cache_path / "config_resolved.yaml",)
        shutil.copy2(self.artifact_path / "feature_manifest.parquet", cache_path / "feature_manifest.parquet",)
        (cache_path / "manifest.json").write_text(
            json.dumps(self._expected_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _allocate_cache_arrays(self, cache_path: Path) -> None:
        """Allocate zero-filled arrays using the final cache contract."""
        date_values = pd.to_datetime(self._source_dates["date"]).to_numpy(dtype="datetime64[D]")
        asset_values = np.asarray(self._source_assets["symbol"].astype(str).tolist(), dtype=str)
        np.save(cache_path / "dates.npy", date_values, allow_pickle=False)
        np.save(cache_path / "assets.npy", asset_values, allow_pickle=False)
        for name, spec in self._array_specs().items():
            if name in {"dates", "assets"}:
                continue
            array = np.lib.format.open_memmap(
                cache_path / f"{name}.npy",
                mode="w+",
                dtype=np.dtype(str(spec["dtype"])),
                shape=tuple(spec["shape"]),
            )
            array[...] = 0
            setattr(self, name, array)

    def _flush_cache_arrays(self) -> None:
        """Flush every writable panel map before publishing metadata."""
        for name in PANEL_ARRAY_NAMES:
            array = getattr(self, name)
            if isinstance(array, np.memmap):
                array.flush()

    @staticmethod
    def _close_memmap(array: np.ndarray) -> None:
        """Close a NumPy memmap's underlying file mapping when present."""
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()

    def _close_cache_arrays(self) -> None:
        """Close live mapped arrays during cache construction."""
        for name in PANEL_ARRAY_NAMES:
            array = getattr(self, name, None)
            if array is not None:
                self._close_memmap(array)
                delattr(self, name)

    def _open_cache_arrays(self) -> None:
        """Open all completed arrays read-only in the current process."""
        # loads each dense .npy array as read-only
        for name in CACHE_ARRAY_NAMES:
            setattr(self, name, np.load(self.cache_path / f"{name}.npy", mmap_mode="r", allow_pickle=False))

    def _load_request_indexes(self) -> None:
        """Load small request tables once for dataset construction and workers."""
        self.train_request_index = pd.read_parquet(self.cache_path / "train_request_index.parquet")
        self.validation_request_index = pd.read_parquet(self.cache_path / "validation_request_index.parquet")

    # ============================================================================
    # SPARSE FACT IMPORT
    # ============================================================================

    def _load_asset_facts(self, split: Split) -> None:
        """Stream sparse asset rows into one split-specific dense panel."""
        path = self.artifact_path / f"{split}_asset_features.parquet"
        validate_fact_schema(path, "asset", self.feature_names["asset"])
        features = self.feature_names["asset"]
        columns = ["date_idx", "asset_id", "valid_asset", *features]
        columns.extend(f"{name}__valid" for name in features)
        values = getattr(self, f"{split}_asset_x")
        feature_mask = getattr(self, f"{split}_asset_feature_mask")
        valid_mask = getattr(self, f"{split}_valid_asset_mask")
        present = np.zeros((self.date_count, self.asset_count), dtype=bool)

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
            if present[date_ids, asset_ids].any():
                raise ValueError(f"{path.name} contains duplicate (date_idx, asset_id) rows.")
            present[date_ids, asset_ids] = True
            values[date_ids, asset_ids] = frame[features].to_numpy(dtype=np.float32)
            feature_mask[date_ids, asset_ids] = frame[
                [f"{name}__valid" for name in features]
            ].to_numpy(dtype=bool)
            valid_mask[date_ids, asset_ids] = frame["valid_asset"].to_numpy(dtype=bool)

    def _load_date_facts(self, split: Split, group: Literal["market", "macro"]) -> None:
        """Stream sparse date rows into one split-specific dense stream."""
        path = self.artifact_path / f"{split}_{group}_features.parquet"
        validate_fact_schema(path, group, self.feature_names[group])
        features = self.feature_names[group]
        columns = ["date_idx", "valid_date", *features]
        columns.extend(f"{name}__valid" for name in features)
        values = getattr(self, f"{split}_{group}_x")
        feature_mask = getattr(self, f"{split}_{group}_feature_mask")
        valid_date = getattr(self, f"{split}_valid_{group}_date")
        present = np.zeros(self.date_count, dtype=bool)

        for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
            frame = batch.to_pandas()
            if frame.duplicated(["date_idx"]).any():
                raise ValueError(f"{path.name} contains duplicate date_idx rows.")
            date_ids = frame["date_idx"].to_numpy(dtype=np.int64)
            if (date_ids < 0).any() or (date_ids >= self.date_count).any():
                raise ValueError(f"{path.name} contains out-of-range date_idx values.")
            if present[date_ids].any():
                raise ValueError(f"{path.name} contains duplicate date_idx rows.")
            present[date_ids] = True
            values[date_ids] = frame[features].to_numpy(dtype=np.float32)
            feature_mask[date_ids] = frame[
                [f"{name}__valid" for name in features]
            ].to_numpy(dtype=bool)
            valid_date[date_ids] = frame["valid_date"].to_numpy(dtype=bool)

    def _write_request_indexes(self, cache_path: Path) -> None:
        """Persist only artifact-stable request metadata."""
        window_names = (
            self._source_dates["validation_window_name"]
            if "validation_window_name" in self._source_dates
            else pd.Series([""] * self.date_count)
        )
        for split, selector in (
            (
                "train",
                self._source_dates["sample_eligible"]
                & self._source_dates["train_fact_allowed"],
            ),
            ("validation", self._source_dates["validation_sample"]),
        ):
            date_ids = self._source_dates.loc[selector, "date_idx"].to_numpy(dtype=np.int64)
            valid_assets = getattr(self, f"{split}_valid_asset_mask")
            frame = pd.DataFrame(
                {
                    "sample_date_idx": date_ids,
                    "sample_date": self._source_dates.loc[selector, "date"].tolist(),
                    "n_endpoint_valid_assets": valid_assets[date_ids].sum(axis=1).astype(np.int32),
                    "validation_window_name": window_names.loc[selector]
                    .fillna("")
                    .astype(str)
                    .tolist(),
                }
            )
            frame.to_parquet(
                cache_path / f"{split}_request_index.parquet",
                index=False,
                compression="zstd",
            )
