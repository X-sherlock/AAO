from __future__ import annotations

import numpy as np

from asset_allocation.data_download.synthetic import OHLCVData
from asset_allocation.exceptions import DataValidationError


def validate_ohlcv(data: OHLCVData) -> None:
    t, n = data.shape
    if t < 2 or n < 2:
        raise DataValidationError("OHLCV data must contain at least 2 dates and assets")
    if len(data.dates) != t or len(data.assets) != n:
        raise DataValidationError("date or asset dimensions do not match close")
    if np.any(np.diff(data.dates).astype("timedelta64[D]") <= np.timedelta64(0, "D")):
        raise DataValidationError("dates must be strictly increasing")
    for name in ("open", "high", "low", "close", "adjusted_close", "volume"):
        values = np.asarray(getattr(data, name))
        if values.shape != (t, n):
            raise DataValidationError(f"{name} shape does not match close")
        if not np.all(np.isfinite(values)):
            raise DataValidationError(f"{name} contains non-finite values")
        if name == "volume" and np.any(values < 0.0):
            raise DataValidationError("volume must be non-negative")
        if name != "volume" and np.any(values <= 0.0):
            raise DataValidationError(f"{name} must be strictly positive")
    if np.any(data.high < np.maximum(data.open, data.close)):
        raise DataValidationError("high is below open or close")
    if np.any(data.low > np.minimum(data.open, data.close)):
        raise DataValidationError("low is above open or close")


def temporal_split(
    n_rows: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[slice, slice, slice]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    if train_end < 1 or validation_end >= n_rows:
        raise ValueError("not enough rows for disjoint train/validation/test splits")
    return (
        slice(0, train_end),
        slice(train_end, validation_end),
        slice(validation_end, n_rows),
    )
