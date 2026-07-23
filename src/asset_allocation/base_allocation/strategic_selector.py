from __future__ import annotations

from typing import Any

import numpy as np

from asset_allocation.base_allocation.anchors import enumerate_feasible_anchors


def _backtest(
    target: np.ndarray,
    returns: np.ndarray,
    base_bps: float,
    periods_per_year: int,
) -> dict[str, float]:
    current = target.copy()
    wealth = peak = 1.0
    max_drawdown = turnover = cost_total = 0.0
    net_returns: list[float] = []
    for asset_returns in np.asarray(returns, dtype=np.float64):
        trade = float(np.abs(target - current).sum())
        cost = trade * base_bps / 10_000.0
        gross = float(target @ asset_returns)
        net = gross - cost
        wealth *= 1.0 + net
        peak = max(peak, wealth)
        max_drawdown = max(max_drawdown, 1.0 - wealth / peak)
        net_returns.append(net)
        turnover += trade
        cost_total += cost
        current = target * (1.0 + asset_returns) / (1.0 + gross)
    values = np.asarray(net_returns)
    years = len(values) / periods_per_year
    annual_return = wealth ** (1.0 / years) - 1.0 if years else 0.0
    annual_volatility = (
        float(values.std(ddof=1) * np.sqrt(periods_per_year))
        if len(values) > 1
        else 0.0
    )
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    return {
        "annual_return": float(annual_return),
        "annual_volatility": annual_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "turnover": float(turnover),
        "transaction_cost": float(cost_total),
    }


def select_strategic_anchor(
    assets: tuple[str, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    step: float,
    train_returns: np.ndarray,
    validation_returns: np.ndarray,
    base_bps: float = 5.0,
    near_optimal_tolerance: float = 0.02,
    periods_per_year: int = 52,
    max_candidates: int = 1_000_000,
) -> dict[str, Any]:
    """Enumerate every feasible anchor using train and validation only."""
    candidates: list[dict[str, Any]] = []
    weight_to_index: dict[tuple[int, ...], int] = {}
    for weights in enumerate_feasible_anchors(lower, upper, step, max_candidates):
        train = _backtest(weights, train_returns, base_bps, periods_per_year)
        validation = _backtest(weights, validation_returns, base_bps, periods_per_year)
        validation_score = (
            validation["sharpe"]
            - 0.25 * validation["max_drawdown"]
            - 0.01 * validation["turnover"]
        )
        simplicity = float(
            np.sum(np.isclose(weights, 0.0))
            + np.sum(np.isclose(np.mod(weights, 0.1), 0.0, atol=1e-10))
        )
        row: dict[str, Any] = {
            **{
                f"base_weight_{asset}": float(weight)
                for asset, weight in zip(assets, weights)
            },
            **validation,
            "train_sharpe": train["sharpe"],
            "validation_score": float(validation_score),
            "simplicity_score": simplicity,
            "neighbor_stability": 0.0,
        }
        key = tuple(np.rint(weights / step).astype(int))
        weight_to_index[key] = len(candidates)
        candidates.append(row)
    if not candidates:
        raise ValueError("anchor constraints have no feasible configurations")
    scores = np.asarray([row["validation_score"] for row in candidates])
    for key, index in weight_to_index.items():
        neighbor_scores: list[float] = []
        for left in range(len(key)):
            for right in range(len(key)):
                if left == right or key[left] == 0:
                    continue
                neighbor = list(key)
                neighbor[left] -= 1
                neighbor[right] += 1
                neighbor_index = weight_to_index.get(tuple(neighbor))
                if neighbor_index is not None:
                    neighbor_scores.append(float(scores[neighbor_index]))
        candidates[index]["neighbor_stability"] = (
            float(np.mean(neighbor_scores) - np.std(neighbor_scores))
            if neighbor_scores
            else float(scores[index])
        )
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    threshold = best_score - abs(best_score) * near_optimal_tolerance
    near_indices = [index for index, score in enumerate(scores) if score >= threshold]
    selected_index = max(
        near_indices,
        key=lambda index: (
            candidates[index]["simplicity_score"],
            candidates[index]["neighbor_stability"],
            candidates[index]["validation_score"],
        ),
    )
    return {
        "selection_scope": "training_and_validation_only",
        "test_period_used": False,
        "candidate_count": len(candidates),
        "near_optimal_tolerance": near_optimal_tolerance,
        "near_optimal_threshold": threshold,
        "near_optimal_count": len(near_indices),
        "absolute_best": candidates[best_index],
        "selected_anchor": candidates[selected_index],
        "performance_gap_to_absolute_best": best_score
        - float(candidates[selected_index]["validation_score"]),
        "simplicity_reason": (
            "selected the simplest grid allocation inside the validation "
            "near-optimal set, then preferred stronger neighbor stability"
        ),
        "candidates": candidates,
    }
