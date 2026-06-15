from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import duckdb

from fi_jepa.probes.artifacts import (
    artifact_destination,
    clean_temporary_artifact,
    file_sha256,
    publish_artifact,
)

PROBE_TARGET_SCHEMA_VERSION = 1


# ============================================================================
# IMMUTABLE PROBE-TARGET EXPORT
# ============================================================================


def export_probe_targets(
    database_path: Path,
    *,
    output_root: Path = Path("data/probe_targets"),
) -> Path:
    """Export canonical future targets into a separate immutable probe artifact."""
    database_path = database_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Canonical database does not exist: {database_path}")
    source_sha256 = file_sha256(database_path)
    with duckdb.connect(str(database_path), read_only=True) as connection:
        targets = connection.execute("SELECT * FROM targets ORDER BY date").fetchdf()
    if targets.empty:
        raise RuntimeError("Canonical targets table is empty.")
    target_columns = sorted(name for name in targets.columns if name.startswith("future_"))
    if not target_columns:
        raise ValueError("Canonical targets table contains no future_ target columns.")

    artifact_id = hashlib.sha256(
        f"{source_sha256}|{','.join(target_columns)}|{PROBE_TARGET_SCHEMA_VERSION}".encode("utf-8")
    ).hexdigest()[:16]
    destination, temporary = artifact_destination(output_root, artifact_id)
    try:
        targets.to_parquet(temporary / "targets.parquet", index=False, compression="zstd")
        manifest = {
            "format_version": PROBE_TARGET_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_database": str(database_path),
            "source_database_sha256": source_sha256,
            "target_columns": target_columns,
            "row_count": int(len(targets)),
            "encoder_features_included": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        publish_artifact(temporary, destination)
    except Exception:
        clean_temporary_artifact(temporary)
        raise
    print(f"Built probe-target export: {destination}")
    return destination
