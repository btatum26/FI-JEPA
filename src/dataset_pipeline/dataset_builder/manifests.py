from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd


# ============================================================================
# DATE AND ASSET MANIFESTS
# ============================================================================


def build_date_manifest(
    dates: pd.Series,
    *,
    sample_dates: pd.Series | None = None,
    context_start: str,
    sample_start: str,
    sample_end: str | None,
    lookback_days: int,
    max_forward_horizon: int,
    validation_windows: list[dict[str, str]],
) -> pd.DataFrame:
    """Build explicit, disjoint validation-protection flags on a trading-date spine."""
    frame = pd.DataFrame({"date": pd.to_datetime(dates).sort_values().drop_duplicates()})
    frame = frame.loc[frame["date"] >= pd.Timestamp(context_start)].reset_index(drop=True)
    if sample_end is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(sample_end)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No dates remain after applying the configured date range.")

    n_dates = len(frame)
    sample_calendar = (
        set(pd.to_datetime(sample_dates).tolist())
        if sample_dates is not None
        else set(frame["date"].tolist())
    )
    sample_eligible = (
        frame["date"].isin(sample_calendar) & frame["date"].ge(pd.Timestamp(sample_start))
    ).to_numpy()
    if sample_end is not None:
        sample_eligible &= frame["date"].le(pd.Timestamp(sample_end)).to_numpy()
    validation = np.zeros(n_dates, dtype=bool)
    input_lookback = np.zeros(n_dates, dtype=bool)
    forward_target = np.zeros(n_dates, dtype=bool)
    names: list[set[str]] = [set() for _ in range(n_dates)]

    for window in validation_windows:
        in_window = (
            frame["date"].between(pd.Timestamp(window["start"]), pd.Timestamp(window["end"]))
            & sample_eligible
        ).to_numpy()
        positions = np.flatnonzero(in_window)
        if positions.size == 0:
            raise ValueError(f"Validation window contains no sample dates: {window['name']}")
        first, last = int(positions[0]), int(positions[-1])
        input_start = max(0, first - max(lookback_days - 1, 0))
        forward_end = min(n_dates, last + 1 + max_forward_horizon)
        validation[positions] = True
        input_lookback[input_start:first] = True
        forward_target[last + 1 : forward_end] = True
        for index in range(input_start, forward_end):
            names[index].add(window["name"])

    input_lookback &= ~validation
    forward_target &= ~validation & ~input_lookback
    protected = validation | input_lookback | forward_target
    frame.insert(0, "date_idx", np.arange(n_dates, dtype=np.int32))
    frame["sample_eligible"] = sample_eligible
    frame["validation_sample"] = validation
    frame["protected_input_lookback"] = input_lookback
    frame["protected_forward_target"] = forward_target
    frame["protected_holdout"] = protected
    frame["train_fact_allowed"] = ~protected
    frame["validation_fact_allowed"] = protected
    frame["validation_window_name"] = [
        "|".join(sorted(values)) if values else pd.NA for values in names
    ]
    frame["date"] = frame["date"].dt.date
    return frame


def build_asset_manifest(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, object],
) -> pd.DataFrame:
    """Resolve selected assets and mark eligibility from allowed train facts."""
    assets = connection.execute(
        """
        SELECT symbol, asset_type, first_available_date, last_available_date
        FROM symbol_manifest
        ORDER BY symbol
        """
    ).fetchdf()
    selection = config["assets"]
    include_types = set(selection.get("include_asset_types") or [])
    include_symbols = set(selection.get("include_symbols") or [])
    exclude_symbols = set(selection.get("exclude_symbols") or [])
    if include_types:
        assets = assets.loc[assets["asset_type"].isin(include_types)]
    if include_symbols:
        assets = assets.loc[assets["symbol"].isin(include_symbols)]
    assets = assets.loc[~assets["symbol"].isin(exclude_symbols)].copy()
    if assets.empty:
        raise ValueError("Asset selection produced no assets.")

    counts = connection.execute(
        """
        SELECT ticker.symbol, count(*) AS valid_train_observations
        FROM ticker_features AS ticker
        INNER JOIN date_manifest AS dates USING (date)
        WHERE dates.train_fact_allowed AND ticker.valid_observation
        GROUP BY ticker.symbol
        """
    ).fetchdf()
    assets = assets.merge(counts, on="symbol", how="left")
    assets["valid_train_observations"] = (
        assets["valid_train_observations"].fillna(0).astype("int64")
    )
    minimum = int(selection["minimum_train_observations"])
    assets["trainable"] = assets["valid_train_observations"].ge(minimum)
    assets["exclusion_reason"] = pd.Series(pd.NA, index=assets.index, dtype="string")
    assets.loc[~assets["trainable"], "exclusion_reason"] = (
        f"fewer_than_{minimum}_valid_train_observations"
    )
    assets = assets.sort_values("symbol").reset_index(drop=True)
    assets.insert(0, "asset_id", np.arange(len(assets), dtype=np.int32))
    return assets
