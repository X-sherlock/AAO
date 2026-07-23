from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _runtime(device: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            'PyTorch is missing. Install with: python -m pip install -e ".[train]"'
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build or use docker/Dockerfile.gpu."
        )
    resolved = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "requested_device": device,
        "resolved_device": resolved,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if resolved == "cuda" else None
        ),
    }


def _aggregate(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    if not run_results:
        return aggregate
    for segment in ("train", "validation", "test"):
        aggregate[segment] = {}
        for strategy in ("ppo", "strategic_anchor"):
            summaries = [
                result["evaluations"][segment][strategy] for result in run_results
            ]
            fields = {
                "final_wealth": [item["final_wealth"] for item in summaries],
            }
            for metric in summaries[0]["metrics"]:
                values = [item["metrics"][metric] for item in summaries]
                if all(value is not None for value in values):
                    fields[metric] = values
            aggregate[segment][strategy] = {
                name: {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for name, values in fields.items()
            }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reproducible multi-seed PPO experiment matrix"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Reuse seed runs that already contain training_result.json",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow each seed run to replace existing outputs",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = raw.get("training", {})
    seeds = args.seeds or raw.get("experiment", {}).get("seeds") or [
        int(training.get("seed", 42))
    ]
    if len(set(seeds)) != len(seeds):
        raise ValueError("experiment seeds must be unique")
    if args.total_steps is not None and args.total_steps <= 0:
        raise ValueError("--total-steps must be positive")

    configured_output = args.output_dir or Path(
        training.get("output_dir", "reports/ppo")
    )
    output_root = Path(configured_output)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    runtime = _runtime(args.device)
    print(json.dumps({"runtime": runtime, "seeds": seeds}, indent=2))
    results: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        seed_output = output_root / f"seed_{seed}"
        result_path = seed_output / "training_result.json"
        if args.resume and result_path.exists():
            print(f"Reusing completed seed {seed}: {result_path}")
        else:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "train_ppo.py"),
                "--config",
                str(config_path),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--output-dir",
                str(seed_output),
            ]
            if args.total_steps is not None:
                command.extend(["--total-steps", str(args.total_steps)])
            if args.overwrite:
                command.append("--overwrite")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        if not result_path.exists():
            raise RuntimeError(f"seed run did not create its result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(result)
        runs.append(
            {
                "seed": seed,
                "output_dir": str(seed_output),
                "result": str(result_path),
            }
        )

    matrix_result = {
        "config": str(config_path),
        "runtime": runtime,
        "seeds": seeds,
        "total_steps_override": args.total_steps,
        "runs": runs,
        "aggregate": _aggregate(results),
    }
    matrix_path = output_root / "matrix_result.json"
    matrix_path.write_text(
        json.dumps(matrix_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(matrix_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
