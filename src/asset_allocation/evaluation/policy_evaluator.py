from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from asset_allocation.policy_models.direction_scale import direction_and_scale
from asset_allocation.portfolio_environment import PortfolioEnvironment

from .metrics import performance_metrics


PolicyAction = Callable[
    [np.ndarray, PortfolioEnvironment], tuple[np.ndarray, float]
]


def evaluate_policy(
    environment: PortfolioEnvironment,
    dates: np.ndarray,
    action: PolicyAction,
    periods_per_year: int = 52,
) -> dict[str, Any]:
    """Run one deterministic episode and return metrics plus an audit series."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    dates = np.asarray(dates)
    if len(dates) != len(environment.features):
        raise ValueError("dates and environment features must have equal length")

    state = environment.reset()
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    anchor_deviations: list[float] = []
    constraint_violations = 0
    series: list[dict[str, Any]] = []

    while True:
        decision_index = environment.index
        direction, alpha = action(state, environment)
        transition = environment.step(direction, alpha)
        gross_returns.append(transition.gross_return)
        net_returns.append(transition.net_return)
        turnovers.append(transition.projection.turnover)
        costs.append(transition.cost.total)
        anchor_deviation = float(
            np.abs(
                transition.target_projected - environment.config.base_weights
            ).sum()
        )
        anchor_deviations.append(anchor_deviation)
        if (
            transition.projection.max_violation
            > environment.config.projection.tolerance * 10
        ):
            constraint_violations += 1
        series.append(
            {
                "decision_date": str(dates[decision_index]),
                "return_period_end": str(dates[decision_index + 1]),
                "gross_return": transition.gross_return,
                "transaction_cost": transition.cost.total,
                "net_return": transition.net_return,
                "turnover": transition.projection.turnover,
                "anchor_deviation_l1": anchor_deviation,
                "wealth": transition.wealth,
                "drawdown": transition.drawdown,
                "target_weights": transition.target_projected.tolist(),
            }
        )
        state = transition.state
        if transition.done:
            break

    metrics = performance_metrics(
        np.asarray(net_returns),
        np.asarray(gross_returns),
        np.asarray(turnovers),
        np.asarray(costs),
        np.asarray(anchor_deviations),
        constraint_violations,
        periods_per_year=periods_per_year,
    )
    summary = {
        "decision_start": series[0]["decision_date"],
        "return_start": series[0]["return_period_end"],
        "return_end": series[-1]["return_period_end"],
        "periods": len(series),
        "final_wealth": series[-1]["wealth"],
        "metrics": metrics.as_dict(),
    }
    return {"summary": summary, "timeseries": series}


def evaluate_actor_critic(
    model: object,
    environment: PortfolioEnvironment,
    dates: np.ndarray,
    device: str,
    periods_per_year: int = 52,
) -> dict[str, Any]:
    """Evaluate the mean action of a trained Gaussian actor-critic."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to evaluate an actor-critic") from exc

    was_training = bool(getattr(model, "training", False))
    model.eval()

    def action(
        state: np.ndarray, current_environment: PortfolioEnvironment
    ) -> tuple[np.ndarray, float]:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            distribution, _ = model.distribution_and_value(tensor)
        latent = distribution.mean.detach().cpu().numpy()
        return direction_and_scale(
            latent[:-1],
            float(latent[-1]),
            current_environment.config.alpha_max,
        )

    try:
        return evaluate_policy(environment, dates, action, periods_per_year)
    finally:
        model.train(was_training)


def evaluate_strategic_anchor(
    environment: PortfolioEnvironment,
    dates: np.ndarray,
    periods_per_year: int = 52,
) -> dict[str, Any]:
    """Evaluate periodic projection back to the configured strategic anchor."""

    def action(
        _state: np.ndarray, current_environment: PortfolioEnvironment
    ) -> tuple[np.ndarray, float]:
        return np.zeros(len(current_environment.config.assets)), 0.0

    return evaluate_policy(environment, dates, action, periods_per_year)
