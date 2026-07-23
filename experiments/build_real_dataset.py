from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asset_allocation.data_processing import build_processed_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build causal processed features from an immutable raw dataset"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    output = build_processed_dataset(raw, PROJECT_ROOT)
    print(output)


if __name__ == "__main__":
    main()
