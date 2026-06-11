from __future__ import annotations

from zipfile import ZipFile

from dataset_pipeline.stooq_loader import StooqArchiveLoader, StooqSymbol


def test_stooq_archive_loader_reads_bulk_text(tmp_path) -> None:
    archive = tmp_path / "daily.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            "data/daily/us/nyse etfs/spy.us.txt",
            "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
            "SPY.US,D,20200102,000000,100,102,99,101,1000,0\n",
        )

    loader = StooqArchiveLoader([archive])
    inventory = loader.inventory()
    prices = loader.load_symbol(StooqSymbol("ETF_SPY", "spy.us", "etf"))

    assert inventory.loc[0, "source_symbol"] == "spy.us"
    assert prices.loc[0, "symbol"] == "ETF_SPY"
    assert prices.loc[0, "close"] == 101
