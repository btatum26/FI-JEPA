from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid


# ============================================================================
# SHARED ARTIFACT UTILITIES
# ============================================================================


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(artifact: Path) -> dict[str, object]:
    """Read one artifact manifest and fail with a useful path if it is missing."""
    manifest_path = artifact / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest does not exist: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def artifact_destination(output_root: Path, artifact_id: str) -> tuple[Path, Path]:
    """Create an immutable artifact's temporary directory and return both paths."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output_root / f"{timestamp}_{artifact_id}"
    temporary = output_root / f".tmp-{artifact_id}-{uuid.uuid4().hex}"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    return destination, temporary


def readable_artifact_destination(output_root: Path, name: str) -> tuple[Path, Path]:
    """Create a readable artifact destination and a temporary sibling directory."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    if not safe_name:
        raise ValueError("Artifact name must contain at least one letter or number.")
    destination = output_root / safe_name
    if destination.exists():
        raise FileExistsError(f"Artifact already exists: {destination}")
    temporary = output_root / f".tmp-{safe_name}-{uuid.uuid4().hex}"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    return destination, temporary


def publish_artifact(temporary: Path, destination: Path) -> None:
    """Atomically publish an artifact, cleaning its temporary directory on failure."""
    try:
        temporary.replace(destination)
    except Exception:
        for path in temporary.glob("*"):
            path.unlink()
        temporary.rmdir()
        raise


def clean_temporary_artifact(temporary: Path) -> None:
    """Remove an unpublished flat artifact directory after a failed write."""
    if temporary.exists():
        for path in temporary.glob("*"):
            path.unlink()
        temporary.rmdir()
