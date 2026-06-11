from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
from zipfile import ZipFile

import pandas as pd
import requests


# ============================================================================
# SYMBOL DEFINITIONS
# ============================================================================


@dataclass(frozen=True)
class StooqSymbol:
    canonical_symbol: str
    source_symbol: str
    asset_type: str
    exchange: str | None = None
    currency: str | None = "USD"


# ============================================================================
# SINGLE-SYMBOL DOWNLOADS
# ============================================================================


class StooqLoader:
    def __init__(self, raw_dir: Path, sleep_seconds: float = 0.5):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = sleep_seconds

    def download_symbol_csv(self, source_symbol: str, interval: str = "d") -> Path:
        """Download and cache a Stooq single-symbol CSV."""
        safe_name = source_symbol.replace("^", "idx_").replace(".", "_")
        out = self.raw_dir / f"{safe_name}_{interval}.csv"
        if out.exists() and out.stat().st_size > 0:
            return out

        response = requests.get(
            "https://stooq.com/q/d/l/",
            params={"s": source_symbol, "i": interval},
            timeout=30,
        )
        response.raise_for_status()
        text = response.text.strip()
        if not text or "No data" in text or len(text.splitlines()) < 2:
            raise ValueError(f"No usable Stooq data for {source_symbol}")

        out.write_text(text, encoding="utf-8")
        time.sleep(self.sleep_seconds)
        return out

    def load_csv(
        self,
        path: Path,
        symbol: StooqSymbol,
        source: str = "stooq",
        adjusted_flag: bool = False,
    ) -> pd.DataFrame:
        """Normalize a Stooq single-symbol CSV into the canonical price schema.

        The source CSV is preserved separately. This method adds the canonical
        research symbol and provenance fields needed to trace every normalized
        row back to the downloaded raw file.
        """
        df = pd.read_csv(path)
        df.columns = [column.lower().strip() for column in df.columns]
        expected = {"date", "open", "high", "low", "close", "volume"}
        if missing := expected - set(df.columns):
            raise ValueError(f"{path} missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["symbol"] = symbol.canonical_symbol
        df["source_symbol"] = symbol.source_symbol
        df["source"] = source
        df["adjusted_flag"] = adjusted_flag
        df["currency"] = symbol.currency
        df["exchange"] = symbol.exchange
        df["asset_type"] = symbol.asset_type
        df["download_timestamp"] = pd.Timestamp.now(tz="UTC")
        df["raw_file"] = str(path)
        df["quality_flag"] = "ok"
        cols = [
            "date",
            "symbol",
            "source_symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
            "adjusted_flag",
            "currency",
            "exchange",
            "asset_type",
            "download_timestamp",
            "raw_file",
            "quality_flag",
        ]
        return df[cols].sort_values(["symbol", "date"]).reset_index(drop=True)


# ============================================================================
# BULK ARCHIVE INGESTION
# ============================================================================


class StooqArchiveLoader:
    """Read Stooq bulk daily-text archives without extracting them."""

    def __init__(self, archives: list[Path]):
        self.archives = [Path(archive) for archive in archives]
        self._member_index: dict[str, list[tuple[Path, str]]] | None = None

    def _build_member_index(self) -> dict[str, list[tuple[Path, str]]]:
        """Index archive members once for efficient multi-symbol dataset builds."""
        if self._member_index is None:
            index: dict[str, list[tuple[Path, str]]] = {}
            for archive in self.archives:
                with ZipFile(archive) as zip_file:
                    for member in zip_file.namelist():
                        if member.lower().endswith(".txt"):
                            source_symbol = Path(member).stem.lower()
                            index.setdefault(source_symbol, []).append((archive, member))
            self._member_index = index
        return self._member_index

    def inventory(self) -> pd.DataFrame:
        """Describe every daily text member in the configured archives.

        Inventory rows retain the archive and member path instead of assuming
        that a ticker uniquely identifies an instrument across all markets.
        """
        rows: list[dict[str, object]] = []
        for archive in self.archives:
            with ZipFile(archive) as zip_file:
                for info in zip_file.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".txt"):
                        continue
                    path_parts = Path(info.filename).parts
                    rows.append(
                        {
                            "archive": archive.name,
                            "member_path": info.filename,
                            "source_symbol": Path(info.filename).stem.lower(),
                            "market": path_parts[2] if len(path_parts) > 2 else None,
                            "category": path_parts[3] if len(path_parts) > 3 else None,
                            "compressed_size": info.compress_size,
                            "uncompressed_size": info.file_size,
                        }
                    )
        return pd.DataFrame(rows)

    def find_member(self, source_symbol: str) -> tuple[Path, str]:
        """Locate exactly one archive member for a Stooq source symbol."""
        matches = self._build_member_index().get(source_symbol.lower(), [])
        if not matches:
            raise KeyError(f"{source_symbol} not found in Stooq archives")
        if len(matches) > 1:
            raise ValueError(f"Multiple archive members found for {source_symbol}: {matches}")
        return matches[0]

    def load_symbol(self, symbol: StooqSymbol, adjusted_flag: bool = False) -> pd.DataFrame:
        """Read one archived symbol and normalize it without extracting the ZIP.

        Stooq bulk files use angle-bracketed column names, compact YYYYMMDD
        dates, and ``VOL`` rather than ``volume``. The returned frame matches
        the same canonical schema as :meth:`StooqLoader.load_csv`.
        """
        archive, member = self.find_member(symbol.source_symbol)
        with ZipFile(archive) as zip_file:
            raw = pd.read_csv(BytesIO(zip_file.read(member)))

        raw.columns = [column.lower().strip("<>") for column in raw.columns]
        raw = raw.rename(columns={"vol": "volume"})
        expected = {"date", "open", "high", "low", "close", "volume"}
        if missing := expected - set(raw.columns):
            raise ValueError(f"{archive.name}:{member} missing columns: {missing}")

        raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d").dt.date
        raw["symbol"] = symbol.canonical_symbol
        raw["source_symbol"] = symbol.source_symbol
        raw["source"] = "stooq"
        raw["adjusted_flag"] = adjusted_flag
        raw["currency"] = symbol.currency
        raw["exchange"] = symbol.exchange
        raw["asset_type"] = symbol.asset_type
        raw["download_timestamp"] = pd.Timestamp.fromtimestamp(archive.stat().st_mtime, tz="UTC")
        raw["raw_file"] = f"{archive.name}:{member}"
        raw["quality_flag"] = "ok"
        columns = [
            "date",
            "symbol",
            "source_symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
            "adjusted_flag",
            "currency",
            "exchange",
            "asset_type",
            "download_timestamp",
            "raw_file",
            "quality_flag",
        ]
        return raw[columns].sort_values("date").reset_index(drop=True)
