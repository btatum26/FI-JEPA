from __future__ import annotations

import pandas as pd


# ============================================================================
# CROSS-SECTIONAL ENCODER FEATURES
# ============================================================================


def build_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid instrument observations into daily market-state features.

    Only rows explicitly marked as valid observations participate. The
    resulting dispersion, breadth, and moving-average breadth values are
    equal-weighted and known at the close of the same date.
    """
    valid = panel.loc[panel["valid_observation"]].copy()

    def iqr(values: pd.Series) -> float:
        return float(values.quantile(0.75) - values.quantile(0.25))

    out = (
        valid.groupby("date")
        .agg(
            xs_dispersion_1d=("return_1d", "std"),
            xs_iqr_1d=("return_1d", iqr),
            breadth_1d=("return_1d", lambda values: float((values > 0).mean())),
            n_assets=("symbol", "nunique"),
        )
        .reset_index()
    )
    if "ma_distance_63d" in valid.columns:
        above = (
            valid.groupby("date")["ma_distance_63d"]
            .apply(lambda values: float((values > 0).mean()))
            .rename("pct_above_ma_63d")
            .reset_index()
        )
        out = out.merge(above, on="date", how="left")
    out["max_source_date_used"] = out["date"]
    out["available_asof"] = out["date"]
    out["uses_future_data"] = False
    return out
