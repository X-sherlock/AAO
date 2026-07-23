import numpy as np
import pytest

from asset_allocation.constraint_projection import project_weights
from asset_allocation.exceptions import InfeasibleProjectionError


def test_projection_enforces_all_hard_constraints():
    base = np.array([0.5, 0.3, 0.2])
    drift = np.array([0.48, 0.32, 0.20])
    covariance = np.array(
        [[0.0004, 0.00005, 0.0], [0.00005, 0.0002, 0.0], [0.0, 0.0, 0.0001]]
    )
    result = project_weights(
        raw_weights=np.array([0.8, 0.1, 0.1]),
        base_weights=base,
        drift_weights=drift,
        weight_min=np.array([0.1, 0.1, 0.1]),
        weight_max=np.array([0.7, 0.6, 0.5]),
        max_deviation=np.array([0.15, 0.15, 0.15]),
        turnover_max=0.20,
        covariance=covariance,
        risk_max=0.018,
    )
    assert abs(result.weights.sum() - 1.0) < 1e-8
    assert np.all(result.weights >= np.array([0.35, 0.15, 0.10]) - 1e-8)
    assert np.all(result.weights <= np.array([0.65, 0.45, 0.35]) + 1e-8)
    assert result.turnover <= 0.20 + 1e-8
    assert result.risk <= 0.018 + 1e-8
    assert result.max_violation <= 1e-8


def test_projection_reports_infeasible_simplex():
    with pytest.raises(InfeasibleProjectionError, match="simplex"):
        project_weights(
            raw_weights=np.array([0.5, 0.5]),
            base_weights=np.array([0.5, 0.5]),
            drift_weights=np.array([0.5, 0.5]),
            weight_min=np.array([0.6, 0.6]),
            weight_max=np.array([1.0, 1.0]),
            max_deviation=np.array([1.0, 1.0]),
            turnover_max=2.0,
            covariance=np.eye(2) * 0.01,
            risk_max=1.0,
        )
