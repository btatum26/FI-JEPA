from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import pandas as pd
import requests
import yaml


# ============================================================================
# SERIES DEFINITIONS
# ============================================================================


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    name: str
    release_lag_assumption: str = "same_day"
    lag_business_days: int = 0
    frequency: str = "unknown"
    point_in_time_available: bool = False
    revised_data_flag: bool | None = None


# ============================================================================
# FRED DOWNLOADS AND NORMALIZATION
# ============================================================================


class FredLoader:
    def __init__(self, raw_dir: Path, api_key: str | None = None):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    def download_series(self, series: FredSeries) -> Path:
        """Download and cache the official FRED observation response."""
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY is required for official FRED API use.")
        out = self.raw_dir / f"{series.series_id}.json"
        if out.exists() and out.stat().st_size > 0:
            return out

        response = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series.series_id, "api_key": self.api_key, "file_type": "json"},
            timeout=30,
        )
        response.raise_for_status()
        out.write_text(response.text, encoding="utf-8")
        return out

    def load_series(self, path: Path, series: FredSeries) -> pd.DataFrame:
        """Normalize cached FRED observations and retain leakage metadata.

        Standard FRED observations are generally revised rather than
        point-in-time values. The output explicitly records that limitation so
        downstream feature builders cannot silently treat them as vintage data.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        obs = pd.DataFrame(payload["observations"])
        obs["date"] = pd.to_datetime(obs["date"]).dt.date
        obs["value"] = pd.to_numeric(obs["value"].replace(".", pd.NA), errors="coerce")

        df = obs[["date", "value"]].copy()
        df["series_id"] = series.series_id
        df["source"] = "fred"
        df["frequency"] = series.frequency
        df["release_lag_assumption"] = series.release_lag_assumption
        df["asof_date"] = (
            pd.to_datetime(df["date"]) + pd.offsets.BusinessDay(series.lag_business_days)
        ).dt.date
        df["point_in_time_available"] = series.point_in_time_available
        df["revised_data_flag"] = (
            not series.point_in_time_available
            if series.revised_data_flag is None
            else series.revised_data_flag
        )
        return df[
            [
                "date",
                "series_id",
                "value",
                "source",
                "frequency",
                "release_lag_assumption",
                "asof_date",
                "point_in_time_available",
                "revised_data_flag",
            ]
        ]


# ============================================================================
# CONFIG-DRIVEN FRED DATA
# ============================================================================


def load_configured_fred_series(features_path: Path) -> list[FredSeries]:
    """Load enabled FRED series definitions from the feature configuration.

    Daily and slow macro groups are handled independently. A disabled group is
    excluded entirely, which keeps revised slow macro observations out of the
    first-model dataset until their availability policy is intentionally
    enabled.
    """
    config = yaml.safe_load(Path(features_path).read_text(encoding="utf-8"))
    series: list[FredSeries] = []
    for group_name in ("macro_daily", "macro_slow"):
        group = config["feature_groups"][group_name]
        if not group.get("enabled", False):
            continue
        for name, definition in group["series"].items():
            if definition.get("source") != "fred":
                continue
            lag_business_days = int(definition.get("lag_business_days", 0))
            lag_policy = definition.get(
                "lag_policy",
                group.get("default_lag_policy", "same_day"),
            )
            series.append(
                FredSeries(
                    series_id=definition["series_id"],
                    name=name,
                    release_lag_assumption=lag_policy,
                    lag_business_days=lag_business_days,
                    frequency="daily" if group_name == "macro_daily" else "slow",
                    revised_data_flag=bool(definition.get("revised_data_flag", True)),
                )
            )
    return series


def load_configured_fred_data(
    raw_dir: Path,
    features_path: Path,
    api_key: str | None = None,
    download_missing: bool = False,
) -> tuple[pd.DataFrame, list[FredSeries]]:
    """Load the enabled FRED dataset, optionally downloading missing snapshots."""
    loader = FredLoader(raw_dir, api_key=api_key)
    series_definitions = load_configured_fred_series(features_path)
    frames: list[pd.DataFrame] = []
    for series in series_definitions:
        path = loader.raw_dir / f"{series.series_id}.json"
        if not path.exists():
            if not download_missing:
                raise FileNotFoundError(
                    f"Missing raw FRED snapshot {path}. Run `uv run import-fred-data` first."
                )
            path = loader.download_series(series)
        frames.append(loader.load_series(path, series))
    return pd.concat(frames, ignore_index=True), series_definitions
