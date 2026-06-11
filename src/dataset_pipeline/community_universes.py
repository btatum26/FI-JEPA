from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================================
# COMMUNITY UNIVERSE NORMALIZATION
# ============================================================================


def ticker_to_stooq_symbol(ticker: str) -> str:
    """Map a Wikipedia-style US ticker to Stooq's US source-symbol convention."""
    return f"{ticker.strip().lower().replace('.', '-')}.us"


def load_community_universes(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize current-constituent snapshots and historical change logs.

    These community files are useful provenance but are not treated as an
    institutional point-in-time membership source. Current rows are explicitly
    marked as survivorship-biased backfills, while change rows retain their
    effective dates without being promoted into reconstructed membership.
    """
    root = Path(root)
    current_frames: list[pd.DataFrame] = []
    change_frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*_current.csv")):
        universe_name = path.stem.removesuffix("_current")
        frame = pd.read_csv(path, dtype=str).fillna("")
        frame["universe_name"] = universe_name
        frame["source_file"] = str(path)
        frame["source_symbol"] = frame["symbol"].map(ticker_to_stooq_symbol)
        frame["universe_type"] = "current_constituents_backfilled"
        frame["survivorship_bias"] = "high"
        frame["point_in_time_membership"] = False
        current_frames.append(frame)

    for path in sorted(root.glob("*_changes.csv")):
        universe_name = path.stem.removesuffix("_changes")
        frame = pd.read_csv(path, dtype=str).fillna("")
        frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce").dt.date
        frame["universe_name"] = universe_name
        frame["source_file"] = str(path)
        change_frames.append(frame)

    if not current_frames:
        raise ValueError(f"No *_current.csv community universes found under {root}")
    current = pd.concat(current_frames, ignore_index=True)
    changes = pd.concat(change_frames, ignore_index=True) if change_frames else pd.DataFrame()
    return current, changes


def build_sp500_symbols(current: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Build Stooq symbol definitions and metadata from the current S&P 500 snapshot."""
    sp500 = current.loc[current["universe_name"].eq("sp500")].copy()
    if sp500.empty:
        raise ValueError("Community universes do not contain an sp500_current snapshot")

    symbols = [
        {
            "canonical_symbol": f"STOCK_US_{row.symbol.replace('.', '_')}",
            "source_symbol": row.source_symbol,
            "asset_type": "stock",
            "exchange": None,
            "currency": "USD",
        }
        for row in sp500.itertuples(index=False)
    ]
    metadata = sp500.rename(columns={"symbol": "ticker"})[
        [
            "ticker",
            "name",
            "source_symbol",
            "universe_name",
            "universe_type",
            "survivorship_bias",
            "point_in_time_membership",
            "source_file",
        ]
    ].copy()
    metadata["symbol"] = metadata["ticker"].map(lambda value: f"STOCK_US_{value.replace('.', '_')}")
    metadata["sector"] = pd.NA
    metadata["sector_metadata_status"] = "not_provided_by_community_snapshot"
    return symbols, metadata
