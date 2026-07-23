from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_rebalance_indices(dates: np.ndarray) -> np.ndarray:
    """Return each calendar week's last observed trading row without future filling."""
    frame = pd.DataFrame(
        {"date": pd.to_datetime(dates), "index": np.arange(len(dates), dtype=int)}
    )
    indices = (
        frame.assign(week=frame["date"].dt.to_period("W-FRI"))
        .groupby("week", sort=True)["index"]
        .max()
        .to_numpy(dtype=int)
    )
    return indices


def holding_period_returns(
    adjusted_close: np.ndarray, decision_indices: np.ndarray
) -> np.ndarray:
    prices = np.asarray(adjusted_close, dtype=np.float64)[decision_indices]
    result = np.full_like(prices, np.nan)
    result[1:] = prices[1:] / prices[:-1] - 1.0
    return result
