from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostBreakdown:
    total: float
    base: float
    liquidity: float
    volatility: float


def transaction_cost(
    target_weights: np.ndarray,
    drift_weights: np.ndarray,
    liquidity_proxy: np.ndarray,
    volatility: np.ndarray,
    base_bps: float,
    liquidity_coefficient: float,
    volatility_coefficient: float,
) -> CostBreakdown:
    change = np.asarray(target_weights) - np.asarray(drift_weights)
    absolute = np.abs(change)
    base = float((base_bps / 10_000.0) * absolute.sum())
    liquidity = float(
        liquidity_coefficient * np.sum(np.maximum(liquidity_proxy, 0.0) * absolute)
    )
    volatility_cost = float(
        volatility_coefficient * np.sum(np.maximum(volatility, 0.0) * change**2)
    )
    total = base + liquidity + volatility_cost
    if not np.isfinite(total) or total < 0.0:
        raise ValueError("transaction cost must be finite and non-negative")
    return CostBreakdown(total, base, liquidity, volatility_cost)
