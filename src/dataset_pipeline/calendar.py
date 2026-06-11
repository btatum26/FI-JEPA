from __future__ import annotations

import pandas as pd


# ============================================================================
# TRADING CALENDAR CONSTRUCTION
# ============================================================================


def infer_trading_calendar(
    price_df: pd.DataFrame,
    reference_symbol: str | None = "ETF_SPY",
) -> pd.DataFrame:
    """Infer expected trading dates from valid closes.

    When ``reference_symbol`` is provided, the resulting calendar follows that
    instrument and is suitable for detecting missing observations against its
    trading history. When it is ``None``, the calendar is the union of all
    observed instrument dates so alignment preserves every loaded price row.
    Both modes avoid inventing dates from a generic weekday calendar.
    """
    valid = price_df["close"].notna()
    if reference_symbol is None:
        ref = price_df.loc[valid, ["date"]].drop_duplicates()
        calendar_name = "inferred_from_all_symbols"
    else:
        ref = price_df.loc[
            valid & price_df["symbol"].eq(reference_symbol),
            ["date"],
        ].drop_duplicates()
        calendar_name = f"inferred_from_{reference_symbol}"
    if ref.empty:
        scope = "any symbol" if reference_symbol is None else f"reference symbol {reference_symbol}"
        raise ValueError(f"No valid close rows for {scope}")

    cal = ref.sort_values("date").copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["is_trading_day"] = True
    cal["calendar_name"] = calendar_name
    cal["year"] = cal["date"].dt.year
    cal["month"] = cal["date"].dt.month
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["quarter"] = cal["date"].dt.quarter
    cal["date"] = cal["date"].dt.date
    return cal.reset_index(drop=True)
