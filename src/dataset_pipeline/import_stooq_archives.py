from __future__ import annotations

import argparse
from pathlib import Path
import shutil

DATA = Path("data")

# ============================================================================
# ARCHIVE PRESERVATION
# ============================================================================


def preserve_archives(source_archives: list[Path], raw_dir: Path) -> list[Path]:
    """Copy source ZIP snapshots into immutable raw storage when needed.

    Existing raw archives with the same name and byte size are retained rather
    than recopied. The separately generated SHA-256 snapshot manifest provides
    stronger provenance after preservation.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    preserved: list[Path] = []
    for source in source_archives:
        destination = raw_dir / source.name
        if not destination.exists() or destination.stat().st_size != source.stat().st_size:
            shutil.copy2(source, destination)
        preserved.append(destination)
    return preserved


# ============================================================================
# ARCHIVE IMPORT PIPELINE
# ============================================================================


def import_archives(source_archives: list[Path]) -> None:
    """Preserve Stooq snapshots without creating duplicate normalized datasets.

    The canonical builder reads ZIP members directly into
    ``data/processed/market_data.duckdb``. This command only places immutable
    source archives under raw storage.
    """
    raw_dir = DATA / "raw" / "stooq" / "bulk_archives"
    archives = preserve_archives(source_archives, raw_dir)
    print(f"Preserved {len(archives)} raw Stooq archives.")


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Stooq bulk daily-text archives.")
    parser.add_argument("archives", type=Path, nargs="+", help="Paths to Stooq ZIP archives.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import_archives(args.archives)


if __name__ == "__main__":
    main()
