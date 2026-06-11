from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# ASSET-LEVEL ENCODER FEATURES
# ============================================================================


def add_asset_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 21, 63, 126),
) -> pd.DataFrame:
    """Add past-only rolling price and liquidity features for each instrument.

    Rows are sorted within symbol before any differences or rolling windows are
    calculated. Every generated feature uses observations through the current
    row only. The returned leakage metadata therefore marks ``date`` as both
    the latest source date and the availability date.

    Partial rolling windows are allowed after at least half the requested
    horizon, with a minimum of five observations. This improves early-history
    coverage but means those early values are not full-window estimates.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    df["log_close"] = np.log(df["close"])
    df["return_1d"] = df.groupby("symbol")["log_close"].diff()
    df["dollar_volume"] = df["close"] * df["volume"]

    for horizon in windows:
        grouped = df.groupby("symbol", group_keys=False)
        df[f"return_{horizon}d"] = grouped["log_close"].diff(horizon)
        min_periods = max(5, horizon // 2)
        df[f"realized_vol_{horizon}d"] = (
            grouped["return_1d"]
            .rolling(horizon, min_periods=min_periods)
            .std()
            .reset_index(level=0, drop=True)
            * np.sqrt(252)
        )
        rolling_max = (
            grouped["close"]
            .rolling(horizon, min_periods=min_periods)
            .max()
            .reset_index(level=0, drop=True)
        )
        df[f"drawdown_{horizon}d"] = df["close"] / rolling_max - 1.0
        sma = (
            grouped["close"]
            .rolling(horizon, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"ma_distance_{horizon}d"] = df["close"] / sma - 1.0
        abs_path = (
            grouped["return_1d"]
            .rolling(horizon, min_periods=min_periods)
            .apply(lambda values: np.nansum(np.abs(values)), raw=True)
            .reset_index(level=0, drop=True)
        )
        df[f"path_efficiency_{horizon}d"] = (
            df[f"return_{horizon}d"].abs() / (abs_path + 1e-12)
        )

    df["max_source_date_used"] = df["date"]
    df["available_asof"] = df["date"]
    df["uses_future_data"] = False
    return df.drop(columns=["log_close"])
