from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class AnchorEvaluation:
    weights: np.ndarray
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    annual_turnover: float
    score: float


def enumerate_feasible_anchors(
    lower: np.ndarray,
    upper: np.ndarray,
    step: float,
    max_candidates: int = 1_000_000,
) -> Iterator[np.ndarray]:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if step <= 0.0 or not np.isclose(1.0 / step, round(1.0 / step)):
        raise ValueError("step must be a positive divisor of one")
    units = int(round(1.0 / step))
    low_units = np.ceil(lower / step - 1e-12).astype(int)
    high_units = np.floor(upper / step + 1e-12).astype(int)
    emitted = 0

    def recurse(index: int, remaining: int, chosen: list[int]) -> Iterator[np.ndarray]:
        nonlocal emitted
        if index == len(lower) - 1:
            if low_units[index] <= remaining <= high_units[index]:
                emitted += 1
                if emitted > max_candidates:
                    raise RuntimeError("anchor candidate count exceeds configured limit")
                yield np.asarray(chosen + [remaining], dtype=np.float64) * step
            return
        minimum_rest = int(low_units[index + 1 :].sum())
        maximum_rest = int(high_units[index + 1 :].sum())
        start = max(low_units[index], remaining - maximum_rest)
        stop = min(high_units[index], remaining - minimum_rest)
        for value in range(start, stop + 1):
            yield from recurse(index + 1, remaining - value, chosen + [value])

    yield from recurse(0, units, [])


def evaluate_anchor(
    weights: np.ndarray,
    returns: np.ndarray,
    rebalance_every: int = 21,
    periods_per_year: int = 252,
) -> AnchorEvaluation:
    weights = np.asarray(weights, dtype=np.float64)
    current = weights.copy()
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    turnover = 0.0
    daily: list[float] = []
    for t, asset_return in enumerate(np.asarray(returns)):
        portfolio_return = float(current @ asset_return)
        daily.append(portfolio_return)
        wealth *= 1.0 + portfolio_return
        peak = max(peak, wealth)
        max_drawdown = max(max_drawdown, 1.0 - wealth / peak)
        current = current * (1.0 + asset_return) / (1.0 + portfolio_return)
        if (t + 1) % rebalance_every == 0:
            turnover += float(np.abs(weights - current).sum())
            current = weights.copy()
    daily_array = np.asarray(daily)
    years = len(daily_array) / periods_per_year
    annual_return = wealth ** (1.0 / years) - 1.0 if years > 0 else 0.0
    annual_volatility = float(daily_array.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    annual_turnover = turnover / years if years > 0 else 0.0
    score = sharpe - 0.25 * max_drawdown - 0.01 * annual_turnover
    return AnchorEvaluation(
        weights=weights,
        annual_return=float(annual_return),
        annual_volatility=annual_volatility,
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
        annual_turnover=float(annual_turnover),
        score=float(score),
    )
