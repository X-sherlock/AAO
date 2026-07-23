import numpy as np

from asset_allocation.config import (
    CostConfig,
    ExperimentConfig,
    ProjectionConfig,
)
from asset_allocation.portfolio_environment import PortfolioEnvironment, drift_weights
from asset_allocation.transaction_cost import transaction_cost


def small_config(base_bps=0.0):
    return ExperimentConfig(
        seed=1,
        assets=("A", "B"),
        base_weights=np.array([0.6, 0.4]),
        alpha_max=0.2,
        weight_min=np.array([0.0, 0.0]),
        weight_max=np.array([1.0, 1.0]),
        max_deviation=np.array([0.5, 0.5]),
        turnover_max=2.0,
        risk_max_daily=1.0,
        feature_windows=(2,),
        ewma_decay=0.9,
        cost=CostConfig(base_bps, 0.0, 0.0),
        projection=ProjectionConfig(1e-10, 1000),
    )


def test_drift_weights_matches_hand_calculation():
    weights = np.array([0.6, 0.4])
    returns = np.array([0.10, -0.05])
    expected = np.array([0.6 * 1.10, 0.4 * 0.95]) / (1.0 + 0.6 * 0.10 - 0.4 * 0.05)
    np.testing.assert_allclose(drift_weights(weights, returns), expected)


def test_transaction_cost_is_non_negative_and_decomposed():
    result = transaction_cost(
        np.array([0.7, 0.3]),
        np.array([0.6, 0.4]),
        np.array([0.2, 0.4]),
        np.array([0.01, 0.02]),
        base_bps=5,
        liquidity_coefficient=0.001,
        volatility_coefficient=0.5,
    )
    assert result.total == result.base + result.liquidity + result.volatility
    assert result.total > 0.0


def test_environment_one_step_matches_hand_calculation():
    config = small_config()
    features = np.zeros((4, 3))
    returns = np.array(
        [[np.nan, np.nan], [0.10, -0.05], [0.0, 0.0], [0.0, 0.0]]
    )
    covariance = np.repeat((np.eye(2) * 0.0001)[None, :, :], 4, axis=0)
    liquidity = np.zeros((4, 2))
    environment = PortfolioEnvironment(
        features, returns, covariance, liquidity, config, start_index=0, end_index=0
    )
    transition = environment.step(np.zeros(2), alpha=0.0)
    expected_gross = 0.6 * 0.10 + 0.4 * -0.05
    assert transition.done
    assert transition.gross_return == expected_gross
    assert transition.cost.total == 0.0
    assert transition.wealth == 1.0 + expected_gross
    assert transition.drawdown == 0.0


def test_environment_is_reproducible_for_same_actions():
    config = small_config(base_bps=2.0)
    features = np.zeros((5, 2))
    returns = np.array(
        [[np.nan, np.nan], [0.01, -0.01], [0.02, 0.0], [-0.01, 0.01], [0.0, 0.0]]
    )
    covariance = np.repeat((np.eye(2) * 0.0001)[None, :, :], 5, axis=0)
    liquidity = np.zeros((5, 2))
    outputs = []
    for _ in range(2):
        env = PortfolioEnvironment(
            features, returns, covariance, liquidity, config, start_index=0, end_index=2
        )
        episode = []
        while True:
            transition = env.step(np.array([0.5, -0.5]), alpha=0.1)
            episode.append((transition.wealth, transition.drawdown))
            if transition.done:
                break
        outputs.append(episode)
    np.testing.assert_allclose(outputs[0], outputs[1])
