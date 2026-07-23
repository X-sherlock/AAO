from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from asset_allocation.data_download.base import (
    MarketDataProvider,
    normalize_ohlcv_frame,
)
from asset_allocation.exceptions import DataValidationError


class LocalFileProvider(MarketDataProvider):
    name = "local_file"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch(
        self,
        assets: tuple[str, ...],
        start_date: str | None,
        end_date: str | None,
        **parameters: Any,
    ) -> pd.DataFrame:
        if not self.path.is_file():
            raise FileNotFoundError(f"local OHLCV file does not exist: {self.path}")
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(self.path)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(self.path)
        else:
            raise ValueError("local_file supports only CSV and Parquet")
        canonical = normalize_ohlcv_frame(frame)
        requested = set(assets)
        found = set(canonical["asset_code"])
        missing = sorted(requested - found)
        if missing:
            raise DataValidationError(
                "local file is missing required assets: " + ", ".join(missing)
            )
        canonical = canonical[canonical["asset_code"].isin(assets)]
        if start_date:
            canonical = canonical[canonical["date"] >= pd.Timestamp(start_date)]
        if end_date:
            canonical = canonical[canonical["date"] <= pd.Timestamp(end_date)]
        if canonical.empty:
            raise DataValidationError("local file has no rows in requested date range")
        return canonical.reset_index(drop=True)
