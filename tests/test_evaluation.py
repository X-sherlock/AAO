import numpy as np

from asset_allocation.config import CostConfig, ExperimentConfig, ProjectionConfig
from asset_allocation.evaluation import evaluate_strategic_anchor
from asset_allocation.portfolio_environment import PortfolioEnvironment


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=7,
        assets=("A", "B"),
        base_weights=np.array([0.6, 0.4]),
        alpha_max=0.2,
        weight_min=np.zeros(2),
        weight_max=np.ones(2),
        max_deviation=np.ones(2),
        turnover_max=2.0,
        risk_max_daily=1.0,
        feature_windows=(2,),
        ewma_decay=0.9,
        cost=CostConfig(0.0, 0.0, 0.0),
        projection=ProjectionConfig(1e-10, 1000),
    )


def test_strategic_anchor_evaluation_returns_auditable_metrics():
    dates = np.arange(
        np.datetime64("2024-01-05"),
        np.datetime64("2024-02-02"),
        np.timedelta64(7, "D"),
    )
    returns = np.array(
        [
            [np.nan, np.nan],
            [0.10, 0.00],
            [0.00, 0.10],
            [-0.10, 0.00],
        ]
    )
    covariance = np.repeat((np.eye(2) * 0.0001)[None, :, :], 4, axis=0)
    environment = PortfolioEnvironment(
        features=np.zeros((4, 3)),
        returns=returns,
        covariances=covariance,
        liquidity_proxy=np.zeros((4, 2)),
        config=_config(),
        start_index=0,
        end_index=2,
    )

    evaluation = evaluate_strategic_anchor(
        environment, dates, periods_per_year=52
    )

    assert evaluation["summary"]["periods"] == 3
    assert evaluation["summary"]["return_start"] == "2024-01-12"
    assert evaluation["summary"]["return_end"] == "2024-01-26"
    assert evaluation["summary"]["metrics"]["constraint_violations"] == 0
    assert np.isclose(
        evaluation["summary"]["final_wealth"], np.prod([1.06, 1.04, 0.94])
    )
    assert len(evaluation["timeseries"]) == 3
    np.testing.assert_allclose(
        evaluation["timeseries"][0]["target_weights"], [0.6, 0.4]
    )
