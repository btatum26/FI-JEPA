from __future__ import annotations

import pandas as pd


# ============================================================================
# SYMBOL-CALENDAR ALIGNMENT
# ============================================================================


def align_prices_to_calendar(
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Align prices to a complete trading-date by symbol grid.

    Missing rows are classified using manifest coverage boundaries. Dates
    before first coverage are marked ``not_listed_yet``; dates after last
    coverage remain ``after_last_observed`` unless verified metadata explicitly
    identifies the instrument as delisted. Gaps inside the coverage window are
    treated as expected-but-missing observations.
    """
    dates = calendar.loc[calendar["is_trading_day"], "date"].sort_values().unique()
    symbols = manifest["symbol"].sort_values().unique()
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    aligned = prices.set_index(["date", "symbol"]).reindex(idx).reset_index()
    aligned = aligned.merge(
        manifest[["symbol", "first_available_date", "last_available_date", "survivorship_status"]],
        on="symbol",
        how="left",
    )
    for column in ("source", "source_symbol", "asset_type", "exchange", "currency"):
        if column in manifest:
            values = manifest.set_index("symbol")[column]
            aligned[column] = aligned[column].fillna(aligned["symbol"].map(values))
    aligned["observation_status"] = "ok"
    aligned.loc[aligned["close"].isna(), "observation_status"] = "missing_expected"
    aligned.loc[
        aligned["date"] < aligned["first_available_date"], "observation_status"
    ] = "not_listed_yet"
    aligned.loc[
        aligned["date"] > aligned["last_available_date"], "observation_status"
    ] = "after_last_observed"
    aligned.loc[
        (aligned["date"] > aligned["last_available_date"])
        & (aligned["survivorship_status"] == "confirmed_delisted"),
        "observation_status",
    ] = "confirmed_delisted"
    aligned["valid_observation"] = aligned["observation_status"].eq("ok")
    return aligned
