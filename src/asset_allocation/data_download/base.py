from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from asset_allocation.data_download.synthetic import OHLCVData
from asset_allocation.exceptions import DataValidationError


OHLCV_COLUMNS = (
    "date",
    "asset_code",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "currency",
    "source",
)
NUMERIC_COLUMNS = ("open", "high", "low", "close", "adjusted_close", "volume")


class MarketDataProvider(ABC):
    name: str

    @abstractmethod
    def fetch(
        self,
        assets: tuple[str, ...],
        start_date: str | None,
        end_date: str | None,
        **parameters: Any,
    ) -> pd.DataFrame:
        """Return the canonical long-table OHLCV protocol."""


def normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "asset": "asset_code",
            "ticker": "asset_code",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
    ).copy()
    missing = [column for column in OHLCV_COLUMNS if column not in renamed.columns]
    if missing:
        raise DataValidationError(
            "provider output is missing required fields: " + ", ".join(missing)
        )
    result = renamed.loc[:, OHLCV_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.tz_localize(None)
    result["asset_code"] = result["asset_code"].astype(str).str.upper()
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise")
    result["currency"] = result["currency"].astype(str).str.upper()
    result["source"] = result["source"].astype(str)
    return result.sort_values(["asset_code", "date"], kind="stable").reset_index(drop=True)


def ohlcv_to_frame(data: OHLCVData, source: str = "synthetic") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    adjusted = np.asarray(data.adjusted_close)
    for t, dt in enumerate(data.dates):
        for i, asset in enumerate(data.assets):
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "asset_code": asset,
                    "open": data.open[t, i],
                    "high": data.high[t, i],
                    "low": data.low[t, i],
                    "close": data.close[t, i],
                    "adjusted_close": adjusted[t, i],
                    "volume": data.volume[t, i],
                    "currency": data.currencies[i],
                    "source": data.sources[i] if data.sources else source,
                }
            )
    return normalize_ohlcv_frame(pd.DataFrame(rows))


def frame_to_ohlcv(
    frame: pd.DataFrame,
    assets: tuple[str, ...] | None = None,
    require_complete_calendar: bool = True,
) -> OHLCVData:
    canonical = normalize_ohlcv_frame(frame)
    ordered_assets = assets or tuple(canonical["asset_code"].drop_duplicates())
    found = set(canonical["asset_code"])
    missing_assets = [asset for asset in ordered_assets if asset not in found]
    if missing_assets:
        raise DataValidationError(
            "required assets are missing: " + ", ".join(missing_assets)
        )
    canonical = canonical[canonical["asset_code"].isin(ordered_assets)]
    common_dates = None
    for asset in ordered_assets:
        dates = set(canonical.loc[canonical["asset_code"] == asset, "date"])
        common_dates = dates if common_dates is None else common_dates & dates
    dates = sorted(common_dates or ())
    if not dates:
        raise DataValidationError("assets have no common trading dates")
    if require_complete_calendar:
        union_dates = set(canonical["date"])
        lost = len(union_dates) - len(dates)
        if lost:
            # Alignment is explicit and never fills a missing raw price.
            canonical = canonical[canonical["date"].isin(dates)]
    matrices: dict[str, np.ndarray] = {}
    for column in NUMERIC_COLUMNS:
        pivot = canonical.pivot(index="date", columns="asset_code", values=column)
        pivot = pivot.reindex(index=dates, columns=ordered_assets)
        if pivot.isna().any().any():
            raise DataValidationError(
                f"{column} is incomplete on the common aligned calendar"
            )
        matrices[column] = pivot.to_numpy(dtype=np.float64)
    currencies = tuple(
        str(
            canonical.loc[canonical["asset_code"] == asset, "currency"].iloc[0]
        )
        for asset in ordered_assets
    )
    sources = tuple(
        str(canonical.loc[canonical["asset_code"] == asset, "source"].iloc[0])
        for asset in ordered_assets
    )
    return OHLCVData(
        dates=np.asarray(dates, dtype="datetime64[D]"),
        assets=tuple(ordered_assets),
        open=matrices["open"],
        high=matrices["high"],
        low=matrices["low"],
        close=matrices["close"],
        volume=matrices["volume"],
        adjusted_close=matrices["adjusted_close"],
        currencies=currencies,
        sources=sources,
    )
