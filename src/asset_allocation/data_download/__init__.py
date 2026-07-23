"""Data acquisition and deterministic fixture generation."""
from .base import (
    MarketDataProvider,
    OHLCV_COLUMNS,
    frame_to_ohlcv,
    normalize_ohlcv_frame,
    ohlcv_to_frame,
)
from .local_file import LocalFileProvider
from .synthetic import OHLCVData, SyntheticProvider, make_synthetic_ohlcv

__all__ = [
    "LocalFileProvider",
    "MarketDataProvider",
    "OHLCVData",
    "OHLCV_COLUMNS",
    "SyntheticProvider",
    "YFinanceProvider",
    "frame_to_ohlcv",
    "make_synthetic_ohlcv",
    "normalize_ohlcv_frame",
    "ohlcv_to_frame",
]


def __getattr__(name: str):
    if name == "YFinanceProvider":
        from .real_ohlcv import YFinanceProvider

        return YFinanceProvider
    raise AttributeError(name)
