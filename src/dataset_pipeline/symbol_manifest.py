from __future__ import annotations

import hashlib

import pandas as pd


# ============================================================================
# STABLE INSTRUMENT IDENTIFIERS
# ============================================================================


def make_instrument_id(source: str, source_symbol: str, first_date: str | None) -> str:
    """Create a deterministic source-specific instrument identifier."""
    key = f"{source}|{source_symbol}|{first_date or 'unknown'}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# ============================================================================
# SYMBOL MANIFEST CONSTRUCTION
# ============================================================================


def build_symbol_manifest(price_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Combine observed price coverage with supplied instrument metadata.

    Coverage dates and row counts come from the normalized price table rather
    than external claims. Missing survivorship and point-in-time fields receive
    conservative defaults so unknown status is never promoted to verified.
    """
    coverage = (
        price_df.groupby(["symbol", "source", "source_symbol"], as_index=False)
        .agg(
            first_available_date=("date", "min"),
            last_available_date=("date", "max"),
            n_rows=("date", "size"),
        )
    )
    manifest = coverage.merge(metadata, on=["symbol", "source", "source_symbol"], how="left")
    manifest["instrument_id"] = [
        make_instrument_id(source, source_symbol, str(first_date))
        for source, source_symbol, first_date in zip(
            manifest["source"],
            manifest["source_symbol"],
            manifest["first_available_date"],
            strict=True,
        )
    ]
    if "survivorship_status" not in manifest:
        manifest["survivorship_status"] = "unknown"
    else:
        manifest["survivorship_status"] = manifest["survivorship_status"].fillna("unknown")
    if "point_in_time_valid" not in manifest:
        manifest["point_in_time_valid"] = False
    else:
        manifest["point_in_time_valid"] = manifest["point_in_time_valid"].fillna(False)
    return manifest
