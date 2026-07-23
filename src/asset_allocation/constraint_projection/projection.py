from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asset_allocation.exceptions import InfeasibleProjectionError


@dataclass(frozen=True)
class ProjectionResult:
    weights: np.ndarray
    iterations: int
    distance: float
    risk: float
    turnover: float
    max_violation: float


def _bounded_simplex(
    vector: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: float = 1.0,
) -> np.ndarray:
    if lower.sum() > total + 1e-12 or upper.sum() < total - 1e-12:
        raise InfeasibleProjectionError("box bounds do not intersect the simplex")
    low = float(np.min(vector - upper))
    high = float(np.max(vector - lower))
    for _ in range(100):
        midpoint = (low + high) / 2.0
        candidate = np.clip(vector - midpoint, lower, upper)
        if candidate.sum() > total:
            low = midpoint
        else:
            high = midpoint
    result = np.clip(vector - (low + high) / 2.0, lower, upper)
    residual = total - result.sum()
    if abs(residual) > 1e-12:
        free = (result > lower + 1e-12) & (result < upper - 1e-12)
        if np.any(free):
            result[free] += residual / free.sum()
    return result


def _l1_ball(vector: np.ndarray, radius: float) -> np.ndarray:
    if radius < 0.0:
        raise InfeasibleProjectionError("turnover radius cannot be negative")
    absolute = np.abs(vector)
    if absolute.sum() <= radius:
        return vector.copy()
    sorted_values = np.sort(absolute)[::-1]
    cumulative = np.cumsum(sorted_values)
    indices = np.nonzero(
        sorted_values * np.arange(1, len(vector) + 1) > cumulative - radius
    )[0]
    rho = int(indices[-1])
    threshold = (cumulative[rho] - radius) / (rho + 1)
    return np.sign(vector) * np.maximum(absolute - threshold, 0.0)


def _risk_ellipsoid(
    vector: np.ndarray, covariance: np.ndarray, risk_max: float
) -> np.ndarray:
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    coordinates = eigenvectors.T @ vector
    risk_squared = float(np.sum(eigenvalues * coordinates**2))
    limit_squared = risk_max**2
    if risk_squared <= limit_squared:
        return vector.copy()
    low = 0.0
    high = 1.0

    def constrained_risk(multiplier: float) -> float:
        scaled = coordinates / (1.0 + multiplier * eigenvalues)
        return float(np.sum(eigenvalues * scaled**2))

    while constrained_risk(high) > limit_squared:
        high *= 2.0
        if high > 1e20:
            raise InfeasibleProjectionError("risk projection failed to bracket root")
    for _ in range(120):
        midpoint = (low + high) / 2.0
        if constrained_risk(midpoint) > limit_squared:
            low = midpoint
        else:
            high = midpoint
    scaled = coordinates / (1.0 + ((low + high) / 2.0) * eigenvalues)
    return eigenvectors @ scaled


def _violations(
    weights: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    drift_weights: np.ndarray,
    turnover_max: float,
    covariance: np.ndarray,
    risk_max: float,
) -> tuple[float, float, float]:
    turnover = float(np.abs(weights - drift_weights).sum())
    risk = float(np.sqrt(max(weights @ covariance @ weights, 0.0)))
    violation = max(
        abs(float(weights.sum()) - 1.0),
        float(np.max(np.maximum(lower - weights, 0.0))),
        float(np.max(np.maximum(weights - upper, 0.0))),
        max(turnover - turnover_max, 0.0),
        max(risk - risk_max, 0.0),
    )
    return violation, risk, turnover


def project_weights(
    raw_weights: np.ndarray,
    base_weights: np.ndarray,
    drift_weights: np.ndarray,
    weight_min: np.ndarray,
    weight_max: np.ndarray,
    max_deviation: np.ndarray,
    turnover_max: float,
    covariance: np.ndarray,
    risk_max: float,
    tolerance: float = 1e-9,
    max_iterations: int = 2000,
) -> ProjectionResult:
    """Euclidean projection onto the convex hard-constraint intersection."""
    raw = np.asarray(raw_weights, dtype=np.float64)
    base = np.asarray(base_weights, dtype=np.float64)
    drift = np.asarray(drift_weights, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    n = len(raw)
    if any(np.asarray(x).shape != (n,) for x in (base, drift, weight_min, weight_max, max_deviation)):
        raise ValueError("all weight vectors must share one-dimensional shape")
    if covariance.shape != (n, n):
        raise ValueError("covariance must have shape (n_assets, n_assets)")
    lower = np.maximum(weight_min, base - max_deviation)
    upper = np.minimum(weight_max, base + max_deviation)
    if np.any(lower > upper):
        raise InfeasibleProjectionError("asset and strategic deviation bounds conflict")

    projectors = (
        lambda x: _bounded_simplex(x, lower, upper),
        lambda x: drift + _l1_ball(x - drift, turnover_max),
        lambda x: _risk_ellipsoid(x, covariance, risk_max),
    )
    x = raw.copy()
    corrections = [np.zeros_like(x) for _ in projectors]
    for iteration in range(1, max_iterations + 1):
        previous = x.copy()
        for i, projector in enumerate(projectors):
            shifted = x + corrections[i]
            projected = projector(shifted)
            corrections[i] = shifted - projected
            x = projected
        violation, risk, turnover = _violations(
            x, lower, upper, drift, turnover_max, covariance, risk_max
        )
        if np.linalg.norm(x - previous) <= tolerance and violation <= tolerance * 10:
            return ProjectionResult(
                weights=x,
                iterations=iteration,
                distance=float(np.linalg.norm(x - raw)),
                risk=risk,
                turnover=turnover,
                max_violation=violation,
            )
    violation, risk, turnover = _violations(
        x, lower, upper, drift, turnover_max, covariance, risk_max
    )
    raise InfeasibleProjectionError(
        "hard-constraint projection did not converge: "
        f"max_violation={violation:.3e}, risk={risk:.6f}, turnover={turnover:.6f}"
    )
