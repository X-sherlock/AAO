from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PerformanceMetrics:
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    cvar_95: float
    annual_turnover: float
    total_cost: float
    cost_to_gross_gain: float | None
    average_anchor_deviation: float
    constraint_violations: int

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def performance_metrics(
    net_returns: np.ndarray,
    gross_returns: np.ndarray,
    turnover: np.ndarray,
    costs: np.ndarray,
    anchor_deviation: np.ndarray,
    constraint_violations: int,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    net = np.asarray(net_returns, dtype=np.float64)
    gross = np.asarray(gross_returns, dtype=np.float64)
    if len(net) == 0 or np.any(net <= -1.0):
        raise ValueError("net return series must be non-empty and greater than -1")
    wealth = np.cumprod(1.0 + net)
    years = len(net) / periods_per_year
    annual_return = float(wealth[-1] ** (1.0 / years) - 1.0)
    annual_volatility = float(net.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    downside = net[net < 0.0]
    downside_volatility = (
        float(np.sqrt(np.mean(downside**2)) * np.sqrt(periods_per_year))
        if len(downside)
        else 0.0
    )
    sortino = annual_return / downside_volatility if downside_volatility > 0 else 0.0
    peak = np.maximum.accumulate(wealth)
    max_drawdown = float(np.max(1.0 - wealth / peak))
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0
    cutoff = np.quantile(net, 0.05)
    cvar = float(-np.mean(net[net <= cutoff]))
    annual_turnover = float(np.sum(turnover) / years)
    total_cost = float(np.sum(costs))
    gross_gain = float(np.prod(1.0 + gross) - 1.0)
    cost_to_gross = total_cost / gross_gain if gross_gain > 0 else None
    return PerformanceMetrics(
        annual_return,
        annual_volatility,
        float(sharpe),
        float(sortino),
        max_drawdown,
        float(calmar),
        cvar,
        annual_turnover,
        total_cost,
        float(cost_to_gross) if cost_to_gross is not None else None,
        float(np.mean(anchor_deviation)),
        int(constraint_violations),
    )
