from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainOnlyScaler:
    mean: np.ndarray
    scale: np.ndarray
    train_start: int
    train_end: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fit_index_range_inclusive": [self.train_start, self.train_end],
            "fit_scope": "training_only",
        }


def fit_train_only_scaler(
    values: np.ndarray, train_start: int, train_end: int
) -> TrainOnlyScaler:
    if not 0 <= train_start <= train_end < len(values):
        raise ValueError("invalid inclusive training range")
    training = np.asarray(values[train_start : train_end + 1], dtype=np.float64)
    if not np.all(np.isfinite(training)):
        raise ValueError("training features contain non-finite values")
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return TrainOnlyScaler(mean, scale, train_start, train_end)


def assert_causal_perturbation(
    original_values: np.ndarray,
    perturbed_values: np.ndarray,
    through_index: int,
    atol: float = 1e-12,
) -> None:
    np.testing.assert_allclose(
        original_values[: through_index + 1],
        perturbed_values[: through_index + 1],
        atol=atol,
        rtol=0.0,
        equal_nan=True,
    )
