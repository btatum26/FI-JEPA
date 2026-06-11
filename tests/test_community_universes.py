from __future__ import annotations

import pandas as pd

from dataset_pipeline.community_universes import (
    build_sp500_symbols,
    load_community_universes,
    ticker_to_stooq_symbol,
)


def test_ticker_to_stooq_symbol_handles_class_shares() -> None:
    assert ticker_to_stooq_symbol("BRK.B") == "brk-b.us"


def test_load_community_universes_and_build_sp500_symbols(tmp_path) -> None:
    pd.DataFrame([{"symbol": "AAPL", "name": "Apple"}]).to_csv(
        tmp_path / "sp500_current.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "effective_date": "January 1, 2020",
                "added_symbol": "AAPL",
                "added_name": "Apple",
                "removed_symbol": "",
                "removed_name": "",
            }
        ]
    ).to_csv(tmp_path / "sp500_changes.csv", index=False)

    current, changes = load_community_universes(tmp_path)
    symbols, metadata = build_sp500_symbols(current)

    assert current.loc[0, "survivorship_bias"] == "high"
    assert changes.loc[0, "effective_date"].isoformat() == "2020-01-01"
    assert symbols[0]["canonical_symbol"] == "STOCK_US_AAPL"
    assert metadata.loc[0, "sector_metadata_status"] == "not_provided_by_community_snapshot"
