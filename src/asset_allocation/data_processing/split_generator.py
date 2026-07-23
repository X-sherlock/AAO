from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitRecord:
    split_id: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _record(split_id: str, dates: pd.DatetimeIndex, bounds: tuple[int, ...]) -> SplitRecord:
    a, b, c, d, e, f = bounds
    if not (a <= b < c <= d < e <= f < len(dates)):
        raise ValueError(f"split {split_id} has overlapping or invalid bounds")
    fmt = lambda i: dates[i].date().isoformat()
    return SplitRecord(split_id, fmt(a), fmt(b), fmt(c), fmt(d), fmt(e), fmt(f))


def generate_splits(
    dates: np.ndarray | pd.DatetimeIndex,
    config: dict[str, Any],
) -> list[SplitRecord]:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("split dates must be unique and increasing")
    mode = str(config.get("mode", "fixed"))
    if mode == "fixed":
        if all(key in config for key in (
            "train_start", "train_end", "validation_start",
            "validation_end", "test_start", "test_end"
        )):
            positions = []
            for key, side in (
                ("train_start", "left"), ("train_end", "right"),
                ("validation_start", "left"), ("validation_end", "right"),
                ("test_start", "left"), ("test_end", "right"),
            ):
                value = pd.Timestamp(config[key])
                pos = int(index.searchsorted(value, side=side))
                positions.append(pos if side == "left" else pos - 1)
            bounds = tuple(positions)
        else:
            n = len(index)
            train_end = int(n * float(config.get("train_fraction", 0.6))) - 1
            validation_end = train_end + int(n * float(config.get("validation_fraction", 0.2)))
            bounds = (0, train_end, train_end + 1, validation_end, validation_end + 1, n - 1)
        return [_record(str(config.get("split_id", "fixed_v1")), index, bounds)]
    train_size = int(config["train_periods"])
    validation_size = int(config["validation_periods"])
    test_size = int(config["test_periods"])
    step = int(config.get("step_periods", test_size))
    splits: list[SplitRecord] = []
    test_end = train_size + validation_size + test_size - 1
    iteration = 0
    while test_end < len(index):
        train_start = 0 if mode == "expanding" else iteration * step
        train_end = iteration * step + train_size - 1
        validation_start = train_end + 1
        validation_end = validation_start + validation_size - 1
        test_start = validation_end + 1
        current_test_end = test_start + test_size - 1
        if current_test_end >= len(index):
            break
        splits.append(
            _record(
                f"{mode}_{iteration + 1:03d}",
                index,
                (
                    train_start,
                    train_end,
                    validation_start,
                    validation_end,
                    test_start,
                    current_test_end,
                ),
            )
        )
        iteration += 1
        test_end = current_test_end + step
    if mode not in {"expanding", "rolling"}:
        raise ValueError(f"unsupported split mode: {mode}")
    if not splits:
        raise ValueError(f"not enough dates to generate {mode} splits")
    return splits
