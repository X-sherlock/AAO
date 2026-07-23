from dataclasses import replace

import numpy as np
import pytest

from asset_allocation.data_download.synthetic import make_synthetic_ohlcv
from asset_allocation.data_validation import temporal_split, validate_ohlcv
from asset_allocation.exceptions import DataValidationError
from asset_allocation.feature_engineering import build_features, simple_returns


ASSETS = ("SPY", "IEF", "HYG", "GLD")


def test_returns_and_temporal_splits():
    close = np.array([[100.0, 50.0], [101.0, 49.0], [102.01, 49.0]])
    returns = simple_returns(close)
    np.testing.assert_allclose(returns[1], [0.01, -0.02])
    np.testing.assert_allclose(returns[2], [0.01, 0.0])
    train, validation, test = temporal_split(100)
    assert (train.stop, validation.start, validation.stop, test.start) == (60, 60, 80, 80)


def test_feature_rows_do_not_read_the_future():
    data = make_synthetic_ohlcv(ASSETS, periods=80, seed=11)
    original = build_features(data, windows=(5, 10), ewma_decay=0.94)
    changed_close = data.close.copy()
    changed_close[41:] *= 10.0
    changed = build_features(
        replace(data, close=changed_close), windows=(5, 10), ewma_decay=0.94
    )
    np.testing.assert_allclose(original.values[40], changed.values[40])
    assert not np.allclose(original.values[41], changed.values[41])


def test_validation_rejects_non_increasing_dates():
    data = make_synthetic_ohlcv(ASSETS, periods=40, seed=3)
    dates = data.dates.copy()
    dates[10] = dates[9]
    with pytest.raises(DataValidationError, match="strictly increasing"):
        validate_ohlcv(replace(data, dates=dates))
