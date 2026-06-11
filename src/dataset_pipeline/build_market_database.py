from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dataset_pipeline.alignment import align_prices_to_calendar
from dataset_pipeline.calendar import infer_trading_calendar
from dataset_pipeline.checks import (
    check_feature_leakage,
    check_no_duplicate_price_rows,
    check_ohlc_sanity,
)
from dataset_pipeline.community_universes import (
    build_sp500_symbols,
    load_community_universes,
)
from dataset_pipeline.cross_sectional_features import build_cross_sectional_features
from dataset_pipeline.database_io import write_market_database
from dataset_pipeline.fred_loader import load_configured_fred_data
from dataset_pipeline.macro_features import build_macro_features
from dataset_pipeline.market_features import add_asset_features
from dataset_pipeline.stooq_loader import StooqArchiveLoader, StooqSymbol
from dataset_pipeline.symbol_manifest import build_symbol_manifest
from dataset_pipeline.targets import build_market_targets

DATA = Path("data")
DATABASE_PATH = DATA / "processed" / "market_data.duckdb"
ARCHIVES = [
    DATA / "raw" / "stooq" / "bulk_archives" / "d_us_txt.zip",
    DATA / "raw" / "stooq" / "bulk_archives" / "d_world_txt.zip",
]

BASE_MARKET_SYMBOLS = [
    StooqSymbol("IDX_SPX", "^spx", "index"),
    StooqSymbol("IDX_NDX", "^ndx", "index"),
    StooqSymbol("IDX_RUT", "^rut", "index"),
    StooqSymbol("IDX_DJI", "^dji", "index"),
    *[
        StooqSymbol(f"ETF_{ticker}", f"{ticker.lower()}.us", "etf")
        for ticker in (
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "TLT",
            "IEF",
            "SHY",
            "AGG",
            "GLD",
            "USO",
            "UUP",
            "EFA",
            "EEM",
            "HYG",
            "LQD",
            "XLB",
            "XLC",
            "XLE",
            "XLF",
            "XLI",
            "XLK",
            "XLP",
            "XLRE",
            "XLU",
            "XLV",
            "XLY",
        )
    ],
]


# ============================================================================
# CANONICAL DATASET BUILD
# ============================================================================


def _metadata_for_base_symbols(symbols: list[StooqSymbol]) -> pd.DataFrame:
    """Create conservative metadata for the static market proxy universe."""
    return pd.DataFrame(
        [
            {
                "symbol": symbol.canonical_symbol,
                "source": "stooq",
                "source_symbol": symbol.source_symbol,
                "asset_type": symbol.asset_type,
                "exchange": symbol.exchange,
                "currency": symbol.currency,
                "survivorship_status": "unknown",
                "point_in_time_valid": False,
                "universe_name": "market_assets",
                "universe_type": "static_proxy",
                "survivorship_bias": "low_for_market_state_not_point_in_time",
            }
            for symbol in symbols
        ]
    )


def build_market_database() -> None:
    """Build the single canonical FI-JEPA market database.

    Past-only encoder inputs are split by their natural grain. ``features``
    contains one market-state row per date, while ``ticker_features`` contains
    one row per date and symbol. Future targets remain physically separate.
    The current S&P 500 cross-section is intentionally marked as
    survivorship-biased.
    """
    missing_archives = [str(path) for path in ARCHIVES if not path.exists()]
    if missing_archives:
        raise FileNotFoundError(f"Missing required Stooq archives: {missing_archives}")

    current, changes = load_community_universes(DATA / "community_universes")
    stock_definitions, stock_metadata = build_sp500_symbols(current)
    requested_symbols = BASE_MARKET_SYMBOLS + [
        StooqSymbol(**definition) for definition in stock_definitions
    ]

    loader = StooqArchiveLoader(ARCHIVES)
    frames: dict[str, pd.DataFrame] = {}
    unavailable: list[str] = []
    for index, symbol in enumerate(requested_symbols, start=1):
        try:
            frames[symbol.canonical_symbol] = loader.load_symbol(symbol)
        except KeyError:
            unavailable.append(symbol.source_symbol)
        if index % 50 == 0 or index == len(requested_symbols):
            print(f"Loaded {index:,}/{len(requested_symbols):,} requested symbols.")

    available_symbols = [
        symbol for symbol in requested_symbols if symbol.canonical_symbol in frames
    ]
    prices = pd.concat(
        [frames[symbol.canonical_symbol] for symbol in available_symbols],
        ignore_index=True,
    )
    check_no_duplicate_price_rows(prices)
    bad_ohlc = check_ohlc_sanity(prices)

    available_base_symbols = [
        symbol for symbol in BASE_MARKET_SYMBOLS if symbol.canonical_symbol in frames
    ]
    base_metadata = _metadata_for_base_symbols(available_base_symbols)
    stock_metadata = stock_metadata.loc[stock_metadata["symbol"].isin(frames)].copy()
    stock_metadata["source"] = "stooq"
    stock_metadata["asset_type"] = "stock"
    stock_metadata["exchange"] = pd.NA
    stock_metadata["currency"] = "USD"
    stock_metadata["survivorship_status"] = "active_current"
    stock_metadata["point_in_time_valid"] = False
    metadata = pd.concat([base_metadata, stock_metadata], ignore_index=True)

    symbol_manifest = build_symbol_manifest(prices, metadata)
    trading_calendar = infer_trading_calendar(prices, reference_symbol=None)
    aligned = align_prices_to_calendar(prices, trading_calendar, symbol_manifest)
    daily_panel = add_asset_features(aligned)
    check_feature_leakage(daily_panel)
    market_features = build_cross_sectional_features(
        daily_panel.loc[daily_panel["asset_type"].eq("stock")]
    )
    check_feature_leakage(market_features)
    macro_data, fred_series = load_configured_fred_data(
        DATA / "raw" / "fred",
        Path("configs/features.yaml"),
    )
    macro_features = build_macro_features(macro_data, trading_calendar, fred_series)
    check_feature_leakage(macro_features)
    targets = build_market_targets(daily_panel)

    build_timestamp = pd.Timestamp.now(tz="UTC")
    manifest = {
        "dataset_name": "fi_jepa_market_data",
        "database_path": str(DATABASE_PATH),
        "universe_type": "current_constituents_backfilled_plus_static_market_proxies",
        "survivorship_bias": "high",
        "point_in_time_membership": False,
        "price_source": "stooq_bulk_archives",
        "symbol_count": int(len(symbol_manifest)),
        "price_row_count": int(len(prices)),
        "ticker_feature_row_count": int(len(daily_panel)),
        "feature_row_count": int(len(market_features)),
        "macro_series_count": int(len(fred_series)),
        "macro_observation_count": int(len(macro_data)),
        "bad_ohlc_row_count": int(len(bad_ohlc)),
        "target_row_count": int(len(targets)),
        "first_date": str(trading_calendar["date"].min()),
        "last_date": str(trading_calendar["date"].max()),
        "raw_price_first_date": str(prices["date"].min()),
        "raw_price_last_date": str(prices["date"].max()),
        "build_timestamp": str(build_timestamp),
        "source_snapshot_date": str(prices["download_timestamp"].max()),
        "unavailable_source_symbols": unavailable,
    }
    build_metadata = pd.DataFrame(
        [{**manifest, "unavailable_source_symbols": json.dumps(unavailable)}]
    )

    write_market_database(
        DATABASE_PATH,
        tables={
            "ticker_features": daily_panel,
            "market_features": market_features,
            "macro_features": macro_features,
            "targets": targets,
            "symbol_manifest": symbol_manifest,
            "trading_calendar": trading_calendar,
            "community_current_constituents": current,
            "community_changes": changes,
            "build_metadata": build_metadata,
        },
        derived_tables={
            "features": """
                SELECT
                    market.date,
                    market.* EXCLUDE (
                        date,
                        max_source_date_used,
                        available_asof,
                        uses_future_data
                    ),
                    macro.* EXCLUDE (
                        date,
                        max_source_date_used,
                        available_asof,
                        uses_future_data
                    ),
                    greatest(
                        market.max_source_date_used,
                        macro.max_source_date_used
                    ) AS max_source_date_used,
                    greatest(
                        market.available_asof,
                        macro.available_asof
                    ) AS available_asof,
                    market.uses_future_data OR macro.uses_future_data AS uses_future_data
                FROM market_features AS market
                INNER JOIN macro_features AS macro USING (date)
            """
        },
        drop_tables=("market_features", "macro_features"),
        validation_queries={
            "feature_leakage": """
                SELECT count(*)
                FROM features
                WHERE coalesce(uses_future_data, false)
                   OR max_source_date_used > date
                   OR available_asof > date
            """,
            "target_columns_in_features": """
                SELECT count(*)
                FROM information_schema.columns
                WHERE table_name IN ('features', 'ticker_features')
                  AND (
                      column_name LIKE 'future_%'
                      OR column_name LIKE '%target%'
                      OR column_name LIKE '%label%'
                )
            """,
            "duplicate_date_feature_rows": """
                SELECT count(*)
                FROM (
                    SELECT date
                    FROM features
                    GROUP BY date
                    HAVING count(*) > 1
                )
            """,
            "ticker_feature_leakage": """
                SELECT count(*)
                FROM ticker_features
                WHERE coalesce(uses_future_data, false)
                   OR max_source_date_used > date
                   OR available_asof > date
            """,
            "duplicate_ticker_feature_rows": """
                SELECT count(*)
                FROM (
                    SELECT date, symbol
                    FROM ticker_features
                    GROUP BY date, symbol
                    HAVING count(*) > 1
                )
            """,
        },
    )

    manifests_dir = DATA / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"Built {DATABASE_PATH}: {len(symbol_manifest):,} symbols, "
        f"{len(market_features):,} date feature rows, "
        f"{len(daily_panel):,} ticker feature rows, "
        f"{len(unavailable):,} unavailable symbols."
    )


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical FI-JEPA market database.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    build_market_database()


if __name__ == "__main__":
    main()
