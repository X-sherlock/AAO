from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class CostConfig:
    base_bps: float
    liquidity_coefficient: float
    volatility_coefficient: float


@dataclass(frozen=True)
class ProjectionConfig:
    tolerance: float
    max_iterations: int


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    assets: tuple[str, ...]
    base_weights: np.ndarray
    alpha_max: float
    weight_min: np.ndarray
    weight_max: np.ndarray
    max_deviation: np.ndarray
    turnover_max: float
    risk_max_daily: float
    feature_windows: tuple[int, ...]
    ewma_decay: float
    cost: CostConfig
    projection: ProjectionConfig

    def __post_init__(self) -> None:
        n = len(self.assets)
        vectors = (
            self.base_weights,
            self.weight_min,
            self.weight_max,
            self.max_deviation,
        )
        if any(np.asarray(v).shape != (n,) for v in vectors):
            raise ValueError("all weight vectors must match assets")
        if not np.isclose(self.base_weights.sum(), 1.0, atol=1e-12):
            raise ValueError("base_weights must sum to one")
        if np.any(self.weight_min > self.weight_max):
            raise ValueError("weight_min cannot exceed weight_max")
        if not 0.0 <= self.alpha_max:
            raise ValueError("alpha_max must be non-negative")
        if not 0.0 < self.ewma_decay < 1.0:
            raise ValueError("ewma_decay must be in (0, 1)")


def _array(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        seed=int(raw["seed"]),
        assets=tuple(raw["assets"]),
        base_weights=_array(raw["base_weights"]),
        alpha_max=float(raw["alpha_max"]),
        weight_min=_array(raw["weight_min"]),
        weight_max=_array(raw["weight_max"]),
        max_deviation=_array(raw["max_deviation"]),
        turnover_max=float(raw["turnover_max"]),
        risk_max_daily=float(raw["risk_max_daily"]),
        feature_windows=tuple(int(x) for x in raw["feature_windows"]),
        ewma_decay=float(raw["ewma_decay"]),
        cost=CostConfig(**raw["cost"]),
        projection=ProjectionConfig(**raw["projection"]),
    )
