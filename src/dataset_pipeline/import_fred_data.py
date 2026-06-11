from __future__ import annotations

from pathlib import Path

from dataset_pipeline.fred_loader import load_configured_fred_data

DATA = Path("data")


# ============================================================================
# FRED IMPORT PIPELINE
# ============================================================================


def import_fred_data() -> None:
    """Download missing raw snapshots for every enabled FRED feature series."""
    api_key = None
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("FRED_API_KEY="):
                api_key = line.partition("=")[2].strip().strip("'\"")
                break

    macro_data, series = load_configured_fred_data(
        raw_dir=DATA / "raw" / "fred",
        features_path=Path("configs/features.yaml"),
        api_key=api_key,
        download_missing=True,
    )
    print(
        f"Imported {len(series)} FRED series with {len(macro_data):,} normalized observations."
    )


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================


def main() -> None:
    import_fred_data()


if __name__ == "__main__":
    main()
