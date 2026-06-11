from __future__ import annotations

import json
from datetime import date

from dataset_pipeline.fred_loader import FredLoader, FredSeries


def test_fred_loader_applies_release_lag_and_revised_flag(tmp_path) -> None:
    path = tmp_path / "UNRATE.json"
    path.write_text(
        json.dumps({"observations": [{"date": "2020-01-03", "value": "3.5"}]}),
        encoding="utf-8",
    )
    series = FredSeries(
        series_id="UNRATE",
        name="unemployment",
        release_lag_assumption="conservative_release_lag",
        lag_business_days=5,
        frequency="slow",
        revised_data_flag=True,
    )

    frame = FredLoader(tmp_path, api_key="unused").load_series(path, series)

    assert frame.loc[0, "date"] == date(2020, 1, 3)
    assert frame.loc[0, "asof_date"] == date(2020, 1, 10)
    assert frame.loc[0, "frequency"] == "slow"
    assert bool(frame.loc[0, "revised_data_flag"])
