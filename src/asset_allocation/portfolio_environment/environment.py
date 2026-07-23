from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asset_allocation.config import ExperimentConfig
from asset_allocation.constraint_projection import ProjectionResult, project_weights
from asset_allocation.transaction_cost import CostBreakdown, transaction_cost


def drift_weights(weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    gross_return = float(np.asarray(weights) @ np.asarray(asset_returns))
    denominator = 1.0 + gross_return
    if denominator <= 0.0:
        raise ValueError("portfolio gross wealth must remain positive")
    drifted = np.asarray(weights) * (1.0 + np.asarray(asset_returns)) / denominator
    if not np.isclose(drifted.sum(), 1.0, atol=1e-10):
        raise ValueError("drifted weights do not sum to one")
    return drifted


@dataclass(frozen=True)
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    target_raw: np.ndarray
    target_projected: np.ndarray
    trade: np.ndarray
    cost: CostBreakdown
    projection: ProjectionResult
    gross_return: float
    net_return: float
    wealth: float
    peak: float
    drawdown: float


class PortfolioEnvironment:
    """A deterministic daily environment with explicit t -> action -> t+1 order."""

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        covariances: np.ndarray,
        liquidity_proxy: np.ndarray,
        config: ExperimentConfig,
        start_index: int,
        end_index: int,
        lambda_drawdown: float = 1.0,
        lambda_deviation: float = 0.01,
        lambda_turnover: float = 0.001,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float64)
        self.returns = np.asarray(returns, dtype=np.float64)
        self.covariances = np.asarray(covariances, dtype=np.float64)
        self.liquidity_proxy = np.asarray(liquidity_proxy, dtype=np.float64)
        self.config = config
        self.start_index = int(start_index)
        self.end_index = int(end_index)
        self.lambda_drawdown = float(lambda_drawdown)
        self.lambda_deviation = float(lambda_deviation)
        self.lambda_turnover = float(lambda_turnover)
        if not (0 <= self.start_index <= self.end_index < len(self.features) - 1):
            raise ValueError("environment indices must leave one next-period return")
        if not np.all(np.isfinite(self.features[self.start_index : self.end_index + 1])):
            raise ValueError("environment feature window contains non-finite values")
        self.reset()

    @property
    def state_dim(self) -> int:
        return self.features.shape[1] + len(self.config.assets) + 4

    def reset(self) -> np.ndarray:
        self.index = self.start_index
        self.current_weights = self.config.base_weights.copy()
        self.wealth = 1.0
        self.peak = 1.0
        self.drawdown = 0.0
        self.last_transaction_cost = 0.0
        return self._state()

    def _state(self) -> np.ndarray:
        return np.concatenate(
            [
                self.features[self.index],
                self.current_weights,
                np.array(
                    [
                        self.wealth,
                        self.peak,
                        self.drawdown,
                        self.last_transaction_cost,
                    ]
                ),
            ]
        )

    def step(self, direction: np.ndarray, alpha: float) -> StepResult:
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != self.config.base_weights.shape:
            raise ValueError("direction shape does not match assets")
        if abs(float(direction.sum())) > 1e-6:
            raise ValueError("direction must sum to zero")
        l1_norm = float(np.abs(direction).sum())
        if l1_norm > 0.0 and abs(l1_norm - 1.0) > 1e-6:
            raise ValueError("non-zero direction must have L1 norm one")
        if not 0.0 <= alpha <= self.config.alpha_max:
            raise ValueError("alpha is outside configured bounds")
        raw = self.config.base_weights + alpha * direction
        covariance = self.covariances[self.index]
        projection = project_weights(
            raw_weights=raw,
            base_weights=self.config.base_weights,
            drift_weights=self.current_weights,
            weight_min=self.config.weight_min,
            weight_max=self.config.weight_max,
            max_deviation=self.config.max_deviation,
            turnover_max=self.config.turnover_max,
            covariance=covariance,
            risk_max=self.config.risk_max_daily,
            tolerance=self.config.projection.tolerance,
            max_iterations=self.config.projection.max_iterations,
        )
        target = projection.weights
        volatility = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        cost = transaction_cost(
            target,
            self.current_weights,
            self.liquidity_proxy[self.index],
            volatility,
            self.config.cost.base_bps,
            self.config.cost.liquidity_coefficient,
            self.config.cost.volatility_coefficient,
        )
        next_returns = self.returns[self.index + 1]
        gross_return = float(target @ next_returns)
        net_growth = 1.0 + gross_return - cost.total
        if net_growth <= 0.0:
            raise ValueError("transaction costs and returns made net wealth non-positive")
        old_drawdown = self.drawdown
        self.wealth *= net_growth
        self.peak = max(self.peak, self.wealth)
        self.drawdown = 1.0 - self.wealth / self.peak
        incremental_drawdown = max(self.drawdown - old_drawdown, 0.0)
        deviation_penalty = float(np.sum((target - self.config.base_weights) ** 2))
        reward = (
            float(np.log(net_growth))
            - self.lambda_drawdown * incremental_drawdown
            - self.lambda_deviation * deviation_penalty
            - self.lambda_turnover * projection.turnover
        )
        trade = target - self.current_weights
        self.last_transaction_cost = cost.total
        self.current_weights = drift_weights(target, next_returns)
        done = self.index >= self.end_index
        if not done:
            self.index += 1
        result_state = self._state()
        return StepResult(
            state=result_state,
            reward=reward,
            done=done,
            target_raw=raw,
            target_projected=target,
            trade=trade,
            cost=cost,
            projection=projection,
            gross_return=gross_return,
            net_return=net_growth - 1.0,
            wealth=self.wealth,
            peak=self.peak,
            drawdown=self.drawdown,
        )
