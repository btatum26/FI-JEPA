from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from dataset_pipeline.alignment import align_prices_to_calendar
from dataset_pipeline.calendar import infer_trading_calendar
from dataset_pipeline.checks import check_feature_leakage, check_ohlc_sanity
from dataset_pipeline.database_io import write_market_database
from dataset_pipeline.fred_loader import FredSeries
from dataset_pipeline.macro_features import build_macro_features
from dataset_pipeline.market_features import add_asset_features
from dataset_pipeline.targets import build_market_targets


def sample_prices() -> pd.DataFrame:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(180)]
    rows = []
    for symbol, offset in [("ETF_SPY", 0.0), ("ETF_QQQ", 10.0)]:
        for index, day in enumerate(dates):
            close = 100.0 + offset + index * 0.25
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1_000_000.0,
                    "asset_type": "etf",
                }
            )
    return pd.DataFrame(rows)


def test_features_are_past_only_and_targets_are_future_only() -> None:
    prices = sample_prices()
    features = add_asset_features(prices)
    check_feature_leakage(features)
    features["valid_observation"] = True
    targets = build_market_targets(features, horizons=(21,))

    assert not features["uses_future_data"].any()
    assert targets["uses_future_data"].all()
    assert np.isclose(targets.loc[0, "future_return_21d"], np.log(105.25 / 100.0))
    assert targets["future_return_21d"].tail(21).isna().all()


def test_calendar_alignment_marks_missing_expected() -> None:
    prices = sample_prices()
    qqq_missing_date = prices.loc[prices["symbol"].eq("ETF_QQQ"), "date"].iloc[30]
    prices = prices.loc[
        ~((prices["symbol"] == "ETF_QQQ") & (prices["date"] == qqq_missing_date))
    ]
    manifest = (
        prices.groupby("symbol", as_index=False)
        .agg(first_available_date=("date", "min"), last_available_date=("date", "max"))
        .assign(survivorship_status="unknown")
    )
    calendar = infer_trading_calendar(prices)
    aligned = align_prices_to_calendar(prices, calendar, manifest)
    row = aligned.loc[
        (aligned["symbol"] == "ETF_QQQ") & (aligned["date"] == qqq_missing_date)
    ].iloc[0]
    assert row["observation_status"] == "missing_expected"
    assert not row["valid_observation"]


def test_all_symbol_calendar_preserves_dates_before_reference_symbol_history() -> None:
    prices = sample_prices()
    earlier_date = date(1990, 1, 2)
    earlier_row = prices.loc[prices["symbol"].eq("ETF_QQQ")].iloc[[0]].copy()
    earlier_row["date"] = earlier_date
    prices = pd.concat([prices, earlier_row], ignore_index=True)

    calendar = infer_trading_calendar(prices, reference_symbol=None)

    assert calendar["date"].min() == earlier_date
    assert calendar["calendar_name"].eq("inferred_from_all_symbols").all()


def test_ohlc_sanity_returns_bad_rows() -> None:
    prices = sample_prices().head(2).copy()
    prices.loc[prices.index[0], "high"] = 1.0
    assert len(check_ohlc_sanity(prices)) == 1


def test_macro_features_respect_asof_dates() -> None:
    calendar = pd.DataFrame(
        {
            "date": [date(2020, 1, day) for day in range(1, 11)],
            "is_trading_day": True,
        }
    )
    definitions = [
        FredSeries(series_id=series_id, name=name)
        for series_id, name in [
            ("DGS3MO", "treasury_3m"),
            ("DGS2", "treasury_2y"),
            ("DGS10", "treasury_10y"),
            ("DGS30", "treasury_30y"),
            ("BAMLH0A0HYM2", "high_yield_oas"),
            ("BAMLC0A0CM", "corporate_oas"),
        ]
    ]
    rows = []
    for index, definition in enumerate(definitions):
        rows.extend(
            [
                {
                    "date": date(2020, 1, 1),
                    "asof_date": date(2020, 1, 3),
                    "series_id": definition.series_id,
                    "value": float(index),
                },
                {
                    "date": date(2020, 1, 7),
                    "asof_date": date(2020, 1, 7),
                    "series_id": definition.series_id,
                    "value": pd.NA,
                },
                {
                    "date": date(2020, 1, 6),
                    "asof_date": date(2020, 1, 8),
                    "series_id": definition.series_id,
                    "value": float(index + 10),
                },
            ]
        )

    features = build_macro_features(pd.DataFrame(rows), calendar, definitions)

    assert features.loc[features["date"] == date(2020, 1, 2), "treasury_10y_level"].isna().all()
    assert features.loc[features["date"] == date(2020, 1, 3), "treasury_10y_level"].iloc[0] == 2
    assert features.loc[features["date"] == date(2020, 1, 7), "treasury_10y_level"].iloc[0] == 2
    assert features.loc[features["date"] == date(2020, 1, 8), "treasury_10y_level"].iloc[0] == 12
    check_feature_leakage(features)


def test_macro_features_preload_history_before_first_output_date() -> None:
    calendar = pd.DataFrame(
        {
            "date": [date(2020, 4, 1), date(2020, 4, 2)],
            "is_trading_day": True,
        }
    )
    definitions = [
        FredSeries(series_id=series_id, name=name)
        for series_id, name in [
            ("DGS3MO", "treasury_3m"),
            ("DGS2", "treasury_2y"),
            ("DGS10", "treasury_10y"),
            ("DGS30", "treasury_30y"),
            ("BAMLH0A0HYM2", "high_yield_oas"),
            ("BAMLC0A0CM", "corporate_oas"),
        ]
    ]
    rows = []
    for index, definition in enumerate(definitions):
        for day_index, observation_date in enumerate(
            pd.bdate_range(end="2020-04-02", periods=70)
        ):
            rows.append(
                {
                    "date": observation_date.date(),
                    "asof_date": observation_date.date(),
                    "series_id": definition.series_id,
                    "value": float(index + day_index),
                }
            )

    features = build_macro_features(pd.DataFrame(rows), calendar, definitions)

    assert features["date"].tolist() == [date(2020, 4, 1), date(2020, 4, 2)]
    assert features.loc[0, "treasury_10y_change_63d"] == 63.0
    check_feature_leakage(features)


def test_market_database_materializes_separate_date_and_ticker_features(tmp_path) -> None:
    database_path = tmp_path / "market_data.duckdb"
    ticker_features = pd.DataFrame(
        {"date": [date(2020, 1, 1)], "symbol": ["ETF_SPY"], "close": [100.0]}
    )
    market = pd.DataFrame({"date": [date(2020, 1, 1)], "breadth_1d": [0.6]})
    targets = pd.DataFrame(
        {"date": [date(2020, 1, 1)], "symbol": ["ETF_SPY"], "future_return_21d": [0.1]}
    )

    write_market_database(
        database_path,
        tables={
            "ticker_features": ticker_features,
            "market_features": market,
            "targets": targets,
        },
        derived_tables={
            "features": """
                SELECT *
                FROM market_features
            """
        },
        drop_tables=("market_features",),
        validation_queries={
            "target_columns_in_features": """
                SELECT count(*)
                FROM information_schema.columns
                WHERE table_name = 'features' AND column_name LIKE 'future_%'
            """
        },
    )

    with duckdb.connect(str(database_path), read_only=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        features = connection.execute("SELECT * FROM features").fetchdf()
        feature_columns = {
            row[0] for row in connection.execute("DESCRIBE features").fetchall()
        }

    assert tables == {"features", "targets", "ticker_features"}
    assert features.loc[0, "breadth_1d"] == 0.6
    assert "symbol" not in feature_columns
    assert "close" not in feature_columns
    assert "future_return_21d" not in feature_columns


def test_database_validation_rejects_target_columns_in_features(tmp_path) -> None:
    database_path = tmp_path / "market_data.duckdb"
    features = pd.DataFrame(
        {
            "date": [date(2020, 1, 1)],
            "symbol": ["ETF_SPY"],
            "future_return_21d": [0.1],
        }
    )

    with np.testing.assert_raises(AssertionError):
        write_market_database(
            database_path,
            tables={"features": features},
            validation_queries={
                "target_columns_in_features": """
                    SELECT count(*)
                    FROM information_schema.columns
                    WHERE table_name = 'features' AND column_name LIKE 'future_%'
                """
            },
        )
