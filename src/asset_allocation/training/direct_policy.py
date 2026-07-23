from __future__ import annotations

import numpy as np

from asset_allocation.constraint_projection import ProjectionResult, project_weights


def one_step_direct_optimization(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    base_weights: np.ndarray,
    drift_weights: np.ndarray,
    weight_min: np.ndarray,
    weight_max: np.ndarray,
    max_deviation: np.ndarray,
    turnover_max: float,
    risk_max: float,
    risk_aversion: float = 10.0,
    anchor_penalty: float = 1.0,
    learning_rate: float = 0.05,
    iterations: int = 100,
) -> ProjectionResult:
    """Projected gradient baseline for the required non-RL comparison."""
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    result: ProjectionResult | None = None
    for _ in range(iterations):
        gradient = (
            -np.asarray(expected_returns)
            + 2.0 * risk_aversion * np.asarray(covariance) @ weights
            + 2.0 * anchor_penalty * (weights - base_weights)
        )
        result = project_weights(
            weights - learning_rate * gradient,
            base_weights,
            drift_weights,
            weight_min,
            weight_max,
            max_deviation,
            turnover_max,
            covariance,
            risk_max,
        )
        weights = result.weights
    assert result is not None
    return result
