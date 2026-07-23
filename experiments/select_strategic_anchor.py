from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asset_allocation.base_allocation import select_strategic_anchor
from asset_allocation.config import load_config
from asset_allocation.data_processing import load_processed_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate and select strategic anchors")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    data = load_processed_dataset(PROJECT_ROOT / raw["dataset"]["data_path"])
    config = load_config(PROJECT_ROOT / raw["portfolio_config"])
    split_id = raw["split"]["split_id"]
    split = next(
        item for item in data.split_manifest["splits"] if item["split_id"] == split_id
    )
    dates = data.dates
    locate = lambda value: int(np.searchsorted(dates, np.datetime64(value), side="left"))
    train = data.returns[locate(split["train_start"]) : locate(split["train_end"]) + 1]
    validation = data.returns[
        locate(split["validation_start"]) : locate(split["validation_end"]) + 1
    ]
    anchor_raw = raw.get("anchor", {})
    lower = np.asarray(
        anchor_raw.get("weight_min", config.weight_min), dtype=np.float64
    )
    upper = np.asarray(
        anchor_raw.get("weight_max", config.weight_max), dtype=np.float64
    )
    result = select_strategic_anchor(
        data.assets,
        lower,
        upper,
        float(anchor_raw.get("step", 0.05)),
        train,
        validation,
        base_bps=config.cost.base_bps,
        near_optimal_tolerance=float(
            anchor_raw.get("near_optimal_tolerance", 0.02)
        ),
    )
    output = PROJECT_ROOT / anchor_raw.get(
        "output", f"reports/{data.dataset_id}/strategic_anchors.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
