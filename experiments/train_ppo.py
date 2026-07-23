from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asset_allocation.config import load_config
from asset_allocation.data_download.synthetic import make_synthetic_ohlcv
from asset_allocation.data_processing import ProcessedMarketDataset, load_processed_dataset
from asset_allocation.data_validation import fit_train_only_scaler
from asset_allocation.evaluation import (
    evaluate_actor_critic,
    evaluate_strategic_anchor,
)
from asset_allocation.feature_engineering import build_features
from asset_allocation.portfolio_environment import PortfolioEnvironment
from asset_allocation.training import PPOConfig, train_ppo


def _position(dates: np.ndarray, value: str, side: str = "left") -> int:
    return int(
        np.searchsorted(
            dates.astype("datetime64[D]"), np.datetime64(value), side=side
        )
    )


def _synthetic_dataset(config, seed: int) -> ProcessedMarketDataset:
    raw = make_synthetic_ohlcv(config.assets, periods=520, seed=seed)
    built = build_features(raw, config.feature_windows, config.ewma_decay)
    valid = np.arange(built.earliest_valid_index, len(raw.dates))
    dates = raw.dates[valid]
    n = len(dates)
    train_end, validation_end = int(n * 0.6) - 1, int(n * 0.8) - 1
    date_text = lambda index: str(dates[index])
    split = {
        "split_id": "synthetic_fixed_v1",
        "train_start": date_text(0),
        "train_end": date_text(train_end),
        "validation_start": date_text(train_end + 1),
        "validation_end": date_text(validation_end),
        "test_start": date_text(validation_end + 1),
        "test_end": date_text(n - 1),
    }
    volume_indices = [
        built.names.index(f"volume_percentile_{asset}") for asset in config.assets
    ]
    return ProcessedMarketDataset(
        "synthetic_in_memory",
        dates,
        config.assets,
        built.names,
        built.values[valid],
        built.returns[valid],
        built.sample_covariance[valid],
        built.shrinkage_covariance[valid],
        built.correlation[valid],
        1.0 - built.values[valid][:, volume_indices],
        {"splits": [split]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO and produce deterministic train/validation/test results "
            "from frozen real or synthetic data"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing checkpoint/result in the output directory",
    )
    args = parser.parse_args()
    raw: dict[str, Any] = yaml.safe_load(
        args.config.resolve().read_text(encoding="utf-8")
    )
    portfolio = load_config(PROJECT_ROOT / raw["portfolio_config"])
    dataset_config = raw["dataset"]
    dataset_type = dataset_config["type"]
    training_raw = raw.get("training", {})
    seed = int(
        args.seed
        if args.seed is not None
        else training_raw.get("seed", portfolio.seed)
    )
    if dataset_type == "frozen_real":
        data_path = Path(dataset_config["data_path"])
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        data = load_processed_dataset(data_path)
        if data.dataset_id != dataset_config["dataset_id"]:
            raise ValueError("configured dataset_id does not match frozen metadata")
    elif dataset_type == "synthetic":
        data = _synthetic_dataset(portfolio, seed)
    else:
        raise ValueError("dataset.type must be frozen_real or synthetic")
    if data.assets != portfolio.assets:
        raise ValueError(
            "processed asset order does not match portfolio configuration"
        )
    split_id = raw["split"]["split_id"]
    try:
        split = next(
            item
            for item in data.split_manifest["splits"]
            if item["split_id"] == split_id
        )
    except StopIteration as exc:
        raise ValueError(
            f"split_id is not present in frozen manifest: {split_id}"
        ) from exc
    train_start = _position(data.dates, split["train_start"])
    train_end = _position(data.dates, split["train_end"], side="right") - 1
    scaler = fit_train_only_scaler(data.features, train_start, train_end)
    normalized = scaler.transform(data.features)
    ppo_raw = dict(training_raw.get("ppo", {}))
    ppo_raw["seed"] = seed
    if args.total_steps is not None:
        if args.total_steps <= 0:
            raise ValueError("--total-steps must be positive")
        ppo_raw["total_steps"] = args.total_steps
    ppo_config = PPOConfig(**ppo_raw)
    requested_device = args.device or training_raw.get("device", "auto")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            'PyTorch is missing. Install with: python -m pip install -e ".[train]"'
        ) from exc
    device = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    estimator = str(raw.get("risk", {}).get("estimator", "shrinkage"))
    periods_per_year = int(raw.get("evaluation", {}).get("periods_per_year", 52))
    configured_output = args.output_dir or Path(
        training_raw.get("output_dir", f"reports/{data.dataset_id}/{split_id}/ppo")
    )
    output_dir = Path(configured_output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    checkpoint = output_dir / "model_state_dict.pt"
    result_path = output_dir / "training_result.json"
    timeseries_path = output_dir / "evaluation_timeseries.json"
    existing_outputs = [
        path for path in (checkpoint, result_path, timeseries_path) if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        names = ", ".join(path.name for path in existing_outputs)
        raise SystemExit(
            f"output directory already contains results ({names}); choose a new "
            "--output-dir or pass --overwrite explicitly"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    def environment_factory(
        start_index: int = train_start, end_index: int = train_end - 1
    ) -> PortfolioEnvironment:
        return PortfolioEnvironment(
            normalized,
            data.returns,
            data.covariance(estimator),
            data.liquidity_proxy,
            portfolio,
            start_index=start_index,
            end_index=end_index,
        )

    model, history = train_ppo(environment_factory, ppo_config, device=device)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dataset_id": data.dataset_id,
            "split_id": split_id,
            "assets": list(data.assets),
            "feature_names": list(data.feature_names),
            "scaler": scaler.as_dict(),
            "ppo_config": asdict(ppo_config),
        },
        checkpoint,
    )

    evaluations: dict[str, dict[str, Any]] = {}
    evaluation_timeseries: dict[str, dict[str, Any]] = {}
    for segment in ("train", "validation", "test"):
        segment_start = _position(data.dates, split[f"{segment}_start"])
        segment_end = _position(
            data.dates, split[f"{segment}_end"], side="right"
        ) - 1
        if segment_end <= segment_start:
            raise ValueError(
                f"{segment} split must contain at least two decision dates"
            )
        ppo_evaluation = evaluate_actor_critic(
            model,
            environment_factory(segment_start, segment_end - 1),
            data.dates,
            device,
            periods_per_year,
        )
        anchor_evaluation = evaluate_strategic_anchor(
            environment_factory(segment_start, segment_end - 1),
            data.dates,
            periods_per_year,
        )
        evaluations[segment] = {
            "ppo": ppo_evaluation["summary"],
            "strategic_anchor": anchor_evaluation["summary"],
        }
        evaluation_timeseries[segment] = {
            "ppo": ppo_evaluation["timeseries"],
            "strategic_anchor": anchor_evaluation["timeseries"],
        }

    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "requested_device": requested_device,
        "resolved_device": device,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if device == "cuda" else None
        ),
    }
    result = {
        "scope": (
            "PPO training with deterministic train/validation/test evaluation; "
            "validation and test rows are never used for fitting"
        ),
        "dataset_type": dataset_type,
        "dataset_id": data.dataset_id,
        "split_id": split_id,
        "seed": seed,
        "runtime": runtime,
        "risk_estimator": estimator,
        "periods_per_year": periods_per_year,
        "ppo_config": asdict(ppo_config),
        "history": history,
        "evaluations": evaluations,
        "checkpoint": str(checkpoint),
        "evaluation_timeseries": str(timeseries_path),
    }
    timeseries_path.write_text(
        json.dumps(
            {
                "dataset_id": data.dataset_id,
                "split_id": split_id,
                "seed": seed,
                "series": evaluation_timeseries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
