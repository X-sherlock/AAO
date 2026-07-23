import numpy as np

from asset_allocation.base_allocation import enumerate_feasible_anchors, evaluate_anchor
from asset_allocation.training.direct_policy import one_step_direct_optimization


def test_anchor_enumeration_and_evaluation():
    candidates = list(
        enumerate_feasible_anchors(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([1.0, 1.0, 1.0]),
            step=0.5,
        )
    )
    assert len(candidates) == 6
    assert all(np.isclose(candidate.sum(), 1.0) for candidate in candidates)
    returns = np.tile(np.array([[0.001, 0.0, -0.001]]), (30, 1))
    evaluation = evaluate_anchor(candidates[0], returns, rebalance_every=5)
    assert np.isfinite(evaluation.score)


def test_direct_optimizer_is_feasible():
    base = np.array([0.5, 0.3, 0.2])
    covariance = np.diag([0.0004, 0.0002, 0.0001])
    result = one_step_direct_optimization(
        expected_returns=np.array([0.001, 0.0005, 0.0002]),
        covariance=covariance,
        base_weights=base,
        drift_weights=base,
        weight_min=np.zeros(3),
        weight_max=np.ones(3),
        max_deviation=np.full(3, 0.2),
        turnover_max=0.3,
        risk_max=0.02,
        iterations=5,
    )
    assert result.max_violation < 1e-8
    assert np.isclose(result.weights.sum(), 1.0)
