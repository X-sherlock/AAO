from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asset_allocation.config import load_config
from asset_allocation.data_download.synthetic import make_synthetic_ohlcv
from asset_allocation.data_validation import temporal_split, validate_ohlcv
from asset_allocation.feature_engineering import build_features
from asset_allocation.portfolio_environment import PortfolioEnvironment
from asset_allocation.training import PPOConfig, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO plumbing smoke train on synthetic data; not a backtest"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "feasibility.yaml",
    )
    parser.add_argument(
        "--ppo-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ppo_smoke.yaml",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-synthetic-fixture",
        action="store_true",
        help="Required acknowledgement that this run has no research meaning.",
    )
    args = parser.parse_args()
    if not args.allow_synthetic_fixture:
        parser.error("--allow-synthetic-fixture is required for the bundled smoke data")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            'PyTorch is missing. Install with: python -m pip install -e ".[train]"'
        ) from exc
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")

    config = load_config(args.config.resolve())
    ppo_raw = yaml.safe_load(args.ppo_config.resolve().read_text(encoding="utf-8"))
    ppo_config = PPOConfig(**ppo_raw)
    data = make_synthetic_ohlcv(config.assets, periods=260, seed=config.seed)
    validate_ohlcv(data)
    features = build_features(data, config.feature_windows, config.ewma_decay)
    train_slice, _, _ = temporal_split(len(data.dates))
    train_start = max(features.earliest_valid_index, train_slice.start or 0)
    training_values = features.values[train_start : train_slice.stop]
    mean = training_values.mean(axis=0)
    std = training_values.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    normalized = (features.values - mean) / std
    volume_indices = [
        features.names.index(f"volume_percentile_{asset}") for asset in config.assets
    ]
    liquidity = 1.0 - features.values[:, volume_indices]

    def environment_factory() -> PortfolioEnvironment:
        return PortfolioEnvironment(
            normalized,
            features.returns,
            features.covariance,
            liquidity,
            config,
            start_index=train_start,
            end_index=train_slice.stop - 2,
        )

    model, history = train_ppo(environment_factory, ppo_config, device=device)
    output_dir = PROJECT_ROOT / "reports" / "ppo_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ppo_smoke_state_dict.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "state_dim": environment_factory().state_dim,
            "action_dim": len(config.assets) + 1,
            "ppo_config": asdict(ppo_config),
            "synthetic_fixture_only": True,
        },
        checkpoint_path,
    )
    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "synthetic_fixture_only": True,
                "performance_interpretation_prohibited": True,
                "device": device,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "scope": "PPO plumbing smoke only",
                "device": device,
                "checkpoint": str(checkpoint_path),
                "history": str(history_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
