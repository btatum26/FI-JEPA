from __future__ import annotations

import pandas as pd

from dataset_pipeline.fred_loader import FredSeries


# ============================================================================
# MACRO ENCODER FEATURES
# ============================================================================


def build_macro_features(
    macro_data: pd.DataFrame,
    calendar: pd.DataFrame,
    series_definitions: list[FredSeries],
    change_windows: tuple[int, ...] = (1, 5, 21, 63),
) -> pd.DataFrame:
    """Align available FRED observations and derive date-level features.

    Each series is joined using its ``asof_date`` rather than observation date.
    Values are forward-filled only after they become available. This preserves
    the configured release-lag boundary and prevents future releases from
    entering earlier encoder rows.

    Before calculating changes, the function prepends enough business dates to
    cover the largest requested window. The warmup rows use source observations
    from before the canonical calendar begins and are removed from the returned
    frame. This maximizes first-date macro feature coverage without extending
    the published dataset or fabricating ticker-price history.
    """
    requested_dates = pd.DatetimeIndex(
        pd.to_datetime(
            calendar.loc[calendar["is_trading_day"], "date"].sort_values().unique()
        )
    ).as_unit("ns")
    if requested_dates.empty:
        raise ValueError("Trading calendar contains no output dates.")

    warmup_rows = max(change_windows, default=0)
    if warmup_rows:
        warmup_dates = pd.bdate_range(
            end=requested_dates.min() - pd.offsets.BusinessDay(),
            periods=warmup_rows,
        )
        calculation_dates = warmup_dates.as_unit("ns").union(requested_dates).sort_values()
    else:
        calculation_dates = requested_dates

    out = pd.DataFrame({"date": calculation_dates})
    max_source_dates: list[pd.Series] = []
    available_dates: list[pd.Series] = []

    for series in series_definitions:
        observations = macro_data.loc[
            macro_data["series_id"].eq(series.series_id),
            ["date", "asof_date", "value"],
        ].dropna(subset=["asof_date", "value"])
        observations = observations.sort_values(["asof_date", "date"]).copy()
        observations["date"] = pd.to_datetime(observations["date"]).astype("datetime64[ns]")
        observations["asof_date"] = pd.to_datetime(observations["asof_date"]).astype(
            "datetime64[ns]"
        )
        observations = observations.drop_duplicates("asof_date", keep="last")
        aligned = pd.merge_asof(
            out[["date"]],
            observations,
            left_on="date",
            right_on="asof_date",
            direction="backward",
        )
        level_name = f"{series.name}_level"
        out[level_name] = aligned["value"]
        for window in change_windows:
            out[f"{series.name}_change_{window}d"] = out[level_name].diff(window)
        max_source_dates.append(aligned["date_y"])
        available_dates.append(aligned["asof_date"])

    out["yield_curve_10y_2y"] = out["treasury_10y_level"] - out["treasury_2y_level"]
    out["yield_curve_10y_3m"] = out["treasury_10y_level"] - out["treasury_3m_level"]
    out["yield_curve_30y_10y"] = out["treasury_30y_level"] - out["treasury_10y_level"]
    out["hy_minus_corporate_oas"] = (
        out["high_yield_oas_level"] - out["corporate_oas_level"]
    )
    out["max_source_date_used"] = pd.concat(max_source_dates, axis=1).max(axis=1)
    out["available_asof"] = pd.concat(available_dates, axis=1).max(axis=1)
    out["uses_future_data"] = False
    out = out.loc[out["date"].isin(requested_dates)].reset_index(drop=True)
    out["date"] = out["date"].dt.date
    out["max_source_date_used"] = pd.to_datetime(out["max_source_date_used"]).dt.date
    out["available_asof"] = pd.to_datetime(out["available_asof"]).dt.date
    return out
