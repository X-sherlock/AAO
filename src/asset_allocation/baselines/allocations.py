from __future__ import annotations

import numpy as np


def equal_weight(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def inverse_volatility(covariance: np.ndarray) -> np.ndarray:
    volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-16))
    inverse = 1.0 / volatility
    return inverse / inverse.sum()


def minimum_variance(covariance: np.ndarray) -> np.ndarray:
    inverse = np.linalg.pinv(covariance)
    ones = np.ones(len(covariance))
    weights = inverse @ ones
    if float(weights.sum()) <= 0.0:
        return equal_weight(len(covariance))
    weights /= weights.sum()
    weights = np.maximum(weights, 0.0)
    return weights / weights.sum()
