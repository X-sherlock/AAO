from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asset_allocation.base_allocation import evaluate_anchor
from asset_allocation.config import load_config
from asset_allocation.data_download.synthetic import freeze_fixture, make_synthetic_ohlcv
from asset_allocation.data_validation import temporal_split, validate_ohlcv
from asset_allocation.evaluation import performance_metrics
from asset_allocation.feature_engineering import build_features
from asset_allocation.policy_models import DirectionScalePolicy
from asset_allocation.portfolio_environment import PortfolioEnvironment
from asset_allocation.reporting import write_strict_json
from asset_allocation.training.direct_policy import one_step_direct_optimization


def _git_metadata() -> dict[str, str | bool]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(PROJECT_ROOT)],
                cwd=root,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        return {
            "revision": revision,
            "repository_root": root,
            "project_tracked": tracked,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "revision": "not-available",
            "repository_root": "not-available",
            "project_tracked": False,
        }


def _local_anchor_candidates(config) -> list[np.ndarray]:
    candidates = [config.base_weights.copy()]
    step = 0.05
    for source in range(len(config.assets)):
        for destination in range(len(config.assets)):
            if source == destination:
                continue
            candidate = config.base_weights.copy()
            candidate[source] -= step
            candidate[destination] += step
            if np.all(candidate >= config.weight_min - 1e-12) and np.all(
                candidate <= config.weight_max + 1e-12
            ):
                candidates.append(candidate)
    unique = {tuple(np.round(candidate, 12)): candidate for candidate in candidates}
    return list(unique.values())


def run(config_path: Path) -> dict:
    config = load_config(config_path)
    data = make_synthetic_ohlcv(config.assets, periods=260, seed=config.seed)
    validate_ohlcv(data)
    raw_path = PROJECT_ROOT / "data" / "raw" / "synthetic_ohlcv_fixture.csv"
    data_metadata_path = (
        PROJECT_ROOT / "data" / "metadata" / "synthetic_ohlcv_fixture.json"
    )
    freeze_fixture(data, raw_path, data_metadata_path)
    features = build_features(data, config.feature_windows, config.ewma_decay)
    train_slice, validation_slice, test_slice = temporal_split(len(data.dates))
    train_start = max(features.earliest_valid_index, train_slice.start or 0)
    train_values = features.values[train_start : train_slice.stop]
    feature_mean = np.mean(train_values, axis=0)
    feature_std = np.std(train_values, axis=0)
    feature_std = np.where(feature_std < 1e-12, 1.0, feature_std)
    normalized = (features.values - feature_mean) / feature_std

    processed_path = PROJECT_ROOT / "data" / "processed" / "features_fixture.npz"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_path,
        dates=data.dates,
        assets=np.asarray(data.assets),
        features=normalized,
        feature_names=np.asarray(features.names),
        returns=features.returns,
        covariance=features.covariance,
        sample_covariance=features.sample_covariance,
        shrinkage_covariance=features.shrinkage_covariance,
        correlation=features.correlation,
        normalization_mean=feature_mean,
        normalization_std=feature_std,
        normalization_fit_end=train_slice.stop - 1,
    )
    feature_metadata = {
        "synthetic_fixture_only": True,
        "causal_contract": "feature row t uses OHLCV rows <= t only",
        "decision_return_contract": "state[t] -> weights[t] -> returns[t+1]",
        "normalization_fit_range": [train_start, train_slice.stop - 1],
        "full_sample_statistics_used": False,
        "covariance_versions": ["sample", "diagonal_shrinkage_10_percent"],
        "correlation_matrix_saved": True,
        "earliest_valid_index": features.earliest_valid_index,
        "feature_names": list(features.names),
        "windows": list(config.feature_windows),
        "ewma_decay": config.ewma_decay,
    }
    feature_metadata_path = PROJECT_ROOT / "data" / "metadata" / "features_fixture.json"
    feature_metadata_path.write_text(
        json.dumps(feature_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    anchor_returns = features.returns[train_start : train_slice.stop]
    anchor_evaluations = [
        evaluate_anchor(candidate, anchor_returns)
        for candidate in _local_anchor_candidates(config)
    ]
    anchor_evaluations.sort(key=lambda item: item.score, reverse=True)
    best_score = anchor_evaluations[0].score
    epsilon = 0.05
    near_optimal = [item for item in anchor_evaluations if item.score >= best_score - epsilon]

    volume_percentile_indices = [
        features.names.index(f"volume_percentile_{asset}") for asset in config.assets
    ]
    liquidity_proxy = 1.0 - features.values[:, volume_percentile_indices]
    environment = PortfolioEnvironment(
        normalized,
        features.returns,
        features.covariance,
        liquidity_proxy,
        config,
        start_index=max(test_slice.start, features.earliest_valid_index),
        end_index=test_slice.stop - 2,
    )
    policy = DirectionScalePolicy(
        input_dim=environment.state_dim,
        n_assets=len(config.assets),
        hidden_dim=64,
        alpha_max=config.alpha_max,
        seed=config.seed,
    )
    state = environment.reset()
    net_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    deviations: list[float] = []
    violations = 0
    first_step = None
    alpha_series: list[float] = []
    while True:
        raw, direction, alpha = policy.raw_weights(state, config.base_weights)
        transition = environment.step(direction, alpha)
        if first_step is None:
            first_step = {
                "state_dimension": len(state),
                "direction": direction,
                "direction_sum": direction.sum(),
                "direction_l1": np.abs(direction).sum(),
                "alpha": alpha,
                "raw_target": raw,
                "projected_target": transition.target_projected,
                "trade": transition.trade,
                "transaction_cost": transition.cost.total,
                "next_gross_return": transition.gross_return,
                "next_net_return": transition.net_return,
                "next_wealth": transition.wealth,
                "next_peak": transition.peak,
                "next_drawdown": transition.drawdown,
                "projection_iterations": transition.projection.iterations,
                "projection_max_violation": transition.projection.max_violation,
            }
        net_returns.append(transition.net_return)
        gross_returns.append(transition.gross_return)
        turnovers.append(transition.projection.turnover)
        costs.append(transition.cost.total)
        deviations.append(
            float(np.linalg.norm(transition.target_projected - config.base_weights))
        )
        alpha_series.append(alpha)
        violations += int(transition.projection.max_violation > 1e-7)
        state = transition.state
        if transition.done:
            break

    metrics = performance_metrics(
        np.asarray(net_returns),
        np.asarray(gross_returns),
        np.asarray(turnovers),
        np.asarray(costs),
        np.asarray(deviations),
        violations,
    )
    direct = one_step_direct_optimization(
        expected_returns=np.zeros(len(config.assets)),
        covariance=features.covariance[test_slice.start],
        base_weights=config.base_weights,
        drift_weights=config.base_weights,
        weight_min=config.weight_min,
        weight_max=config.weight_max,
        max_deviation=config.max_deviation,
        turnover_max=config.turnover_max,
        risk_max=config.risk_max_daily,
        iterations=10,
    )
    report = {
        "status": "passed",
        "scope": "CPU mechanism feasibility only",
        "synthetic_fixture_only": True,
        "performance_interpretation_prohibited": True,
        "seed": config.seed,
        "git": _git_metadata(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "rows": len(data.dates),
            "assets": list(data.assets),
            "raw_fixture": raw_path.relative_to(PROJECT_ROOT),
            "processed_fixture": processed_path.relative_to(PROJECT_ROOT),
            "train": [train_slice.start, train_slice.stop],
            "validation": [validation_slice.start, validation_slice.stop],
            "test": [test_slice.start, test_slice.stop],
        },
        "anchor_screen": {
            "mode": "configured anchor plus one-step 5 percentage-point neighborhood",
            "final_exhaustive_screen": False,
            "candidate_count": len(anchor_evaluations),
            "epsilon": epsilon,
            "near_optimal_count": len(near_optimal),
            "best_score": best_score,
            "configured_anchor_score": next(
                item.score
                for item in anchor_evaluations
                if np.allclose(item.weights, config.base_weights)
            ),
        },
        "one_step_trace": first_step,
        "mechanism_smoke_metrics": metrics.as_dict(),
        "mean_alpha": float(np.mean(alpha_series)),
        "direct_optimizer": {
            "weights": direct.weights,
            "risk": direct.risk,
            "turnover": direct.turnover,
            "max_violation": direct.max_violation,
        },
        "not_run_locally": [
            "PPO parameter training",
            "multi-seed GPU experiments",
            "rolling retraining",
            "real OHLCV backtest",
            "full benchmark and ablation matrix",
            "stage-two external risk datasets",
        ],
    }
    report_dir = PROJECT_ROOT / "reports" / "feasibility"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "feasibility_report.json"
    write_strict_json(report_path, report)
    anchor_path = report_dir / "anchor_candidates.json"
    write_strict_json(
        anchor_path,
        [
            {
                "weights": item.weights,
                "annual_return": item.annual_return,
                "annual_volatility": item.annual_volatility,
                "sharpe": item.sharpe,
                "max_drawdown": item.max_drawdown,
                "annual_turnover": item.annual_turnover,
                "score": item.score,
                "near_optimal": item.score >= best_score - epsilon,
            }
            for item in anchor_evaluations
        ],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "feasibility.yaml",
    )
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "synthetic_fixture_only": report["synthetic_fixture_only"],
                "constraint_violations": report["mechanism_smoke_metrics"][
                    "constraint_violations"
                ],
                "report": str(
                    PROJECT_ROOT / "reports" / "feasibility" / "feasibility_report.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
