from __future__ import annotations

from fnmatch import fnmatch
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yaml

OAS_PATTERNS = ("high_yield_oas_", "corporate_oas_", "hy_minus_corporate_oas")
TARGET_PATTERNS = ("future_", "target", "label")


# ============================================================================
# CONFIGURATION AND IDENTIFIERS
# ============================================================================


def load_model_dataset_config(config_path: Path) -> dict[str, object]:
    """Load and minimally validate the sparse model-dataset configuration."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    required = {"dataset_name", "source_database", "output_root", "dates", "splits", "assets"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Model dataset config is missing required keys: {missing}")
    if config["normalization"]["method"] != "train_fold_robust_zscore":
        raise ValueError("Only train_fold_robust_zscore normalization is supported.")
    return config


def quote_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def quote_literal(value: str | Path) -> str:
    return f"'{str(value).replace(chr(39), chr(39) * 2)}'"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_id_for(config: dict[str, object], source_sha256: str) -> str:
    """Return the deterministic immutable identifier for a source/config pair."""
    build_payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{source_sha256}:{build_payload}".encode()).hexdigest()[:16]


def artifact_name_for(build_id: str, created_at: datetime) -> str:
    """Return a UTC timestamp-prefixed directory name that sorts by creation time."""
    timestamp = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{build_id}"


def find_existing_artifact(output_root: Path, build_id: str) -> Path | None:
    """Find a dated artifact by build ID, migrating a legacy hash-only directory."""
    dated = sorted(path for path in output_root.glob(f"*_{build_id}") if path.is_dir())
    if dated:
        return dated[0]

    legacy = output_root / build_id
    if not legacy.is_dir():
        return None
    created_at = datetime.fromtimestamp(legacy.stat().st_ctime, tz=timezone.utc)
    destination = output_root / artifact_name_for(build_id, created_at)
    if destination.exists():
        raise FileExistsError(f"Cannot migrate legacy artifact; destination exists: {destination}")
    legacy.rename(destination)
    return destination


def resolve_feature_manifest(config: dict[str, object]) -> pd.DataFrame:
    """Resolve ordered feature groups, families, sources, and transforms."""
    rows: list[dict[str, object]] = []
    transforms = config["normalization"].get("transforms", {})
    for input_group in ("asset", "market", "macro"):
        feature_index = 0
        for family in config["features"][input_group]:
            for name in family["names"]:
                transform = "none"
                for candidate, patterns in transforms.items():
                    if any(fnmatch(name, pattern) for pattern in patterns):
                        transform = candidate
                rows.append(
                    {
                        "feature_name": name,
                        "feature_index": feature_index,
                        "input_group": input_group,
                        "feature_family": family["feature_family"],
                        "series_source": family["series_source"],
                        "dtype": "float32",
                        "normalized": True,
                        "normalization_method": config["normalization"]["method"],
                        "transform": transform,
                    }
                )
                feature_index += 1
    manifest = pd.DataFrame(rows)
    duplicates = manifest.loc[manifest.duplicated(["input_group", "feature_name"], keep=False)]
    if not duplicates.empty:
        raise ValueError(
            "Duplicate configured features: "
            f"{duplicates[['input_group', 'feature_name']].to_dict('records')}"
        )
    return manifest
