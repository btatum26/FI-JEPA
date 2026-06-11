from __future__ import annotations

import pandas as pd


# ============================================================================
# PRICE VALIDATION
# ============================================================================


def check_no_duplicate_price_rows(prices: pd.DataFrame) -> None:
    duplicates = int(prices.duplicated(["date", "symbol"]).sum())
    if duplicates:
        raise AssertionError(f"Duplicate date-symbol price rows: {duplicates}")


def check_ohlc_sanity(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.loc[
        (prices["high"] < prices[["open", "close", "low"]].max(axis=1))
        | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))
        | (prices["close"] <= 0)
    ].copy()


# ============================================================================
# LEAKAGE VALIDATION
# ============================================================================


def check_feature_leakage(features: pd.DataFrame) -> None:
    """Reject encoder features that claim or demonstrate future availability.

    The check covers explicit future-data flags and both important temporal
    boundaries: the latest source observation used and the date the feature
    became available. Either boundary extending beyond the feature date is
    leakage.
    """
    if "uses_future_data" in features:
        future_rows = int(features["uses_future_data"].fillna(False).sum())
        if future_rows:
            raise AssertionError(f"Encoder features contain future-data rows: {future_rows}")
    if {"max_source_date_used", "date"}.issubset(features.columns):
        bad = pd.to_datetime(features["max_source_date_used"]) > pd.to_datetime(features["date"])
        if bad.any():
            raise AssertionError(f"Feature rows use data after t: {int(bad.sum())}")
    if {"available_asof", "date"}.issubset(features.columns):
        bad = pd.to_datetime(features["available_asof"]) > pd.to_datetime(features["date"])
        if bad.any():
            raise AssertionError(f"Feature rows available after t: {int(bad.sum())}")


def check_train_scaler_dates(train_end: object, scaler_fit_end: object) -> None:
    if pd.Timestamp(scaler_fit_end) > pd.Timestamp(train_end):
        raise AssertionError("Scaler was fit beyond train_end.")
