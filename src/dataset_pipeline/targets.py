from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# FORWARD TARGET PRIMITIVES
# ============================================================================


def future_realized_volatility(market: pd.DataFrame, horizon: int) -> pd.Series:
    """Calculate annualized volatility strictly over returns from t+1 to t+h."""
    return (
        market["return_1d"]
        .shift(-1)
        .rolling(horizon, min_periods=horizon)
        .std()
        .shift(-(horizon - 1))
        * np.sqrt(252)
    )


def future_return(market: pd.DataFrame, horizon: int) -> pd.Series:
    """Calculate the close-to-close log return from t to t+h."""
    return np.log(market["close"].shift(-horizon) / market["close"])


def future_max_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    """Calculate the worst drawdown encountered from t+1 through t+h.

    Each path begins at the current close so a future decline is measured
    against either that starting value or a later future peak. Rows without a
    complete forward window are returned as missing.
    """
    values: list[float] = []
    prices = close.to_numpy(dtype=float)
    for index in range(len(prices)):
        end = index + horizon + 1
        if end > len(prices) or not np.isfinite(prices[index]):
            values.append(np.nan)
            continue
        path = prices[index:end]
        running_max = np.maximum.accumulate(path)
        values.append(float(np.nanmin((path / running_max - 1.0)[1:])))
    return pd.Series(values, index=close.index)


# ============================================================================
# MARKET TARGET TABLE
# ============================================================================


def build_market_targets(
    panel: pd.DataFrame,
    market_symbol: str = "ETF_SPY",
    horizons: tuple[int, ...] = (21, 63, 126),
) -> pd.DataFrame:
    """Build downstream market targets that are excluded from encoder inputs.

    The selected market proxy is restricted to valid observations and sorted
    before targets are calculated. Every target uses future information by
    design, and the returned table is explicitly marked ``uses_future_data``.
    """
    market = (
        panel.loc[(panel["symbol"] == market_symbol) & panel["valid_observation"]]
        .sort_values("date")
        .copy()
        .reset_index(drop=True)
    )
    out = market[["date"]].copy()
    out["symbol"] = market_symbol
    for horizon in horizons:
        out[f"future_return_{horizon}d"] = future_return(market, horizon)
        out[f"future_realized_vol_{horizon}d"] = future_realized_volatility(market, horizon)
        out[f"future_max_drawdown_{horizon}d"] = future_max_drawdown(market["close"], horizon)
        out[f"future_trend_score_{horizon}d"] = out[f"future_return_{horizon}d"] / (
            out[f"future_realized_vol_{horizon}d"] + 1e-12
        )
    out["uses_future_data"] = True
    return out
