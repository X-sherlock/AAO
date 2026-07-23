from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def direction_and_scale(
    direction_logits: np.ndarray,
    scale_logit: float,
    alpha_max: float,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, float]:
    logits = np.asarray(direction_logits, dtype=np.float64)
    centered = logits - logits.mean()
    norm = float(np.abs(centered).sum())
    if norm <= epsilon:
        direction = np.zeros_like(centered)
    else:
        direction = centered / norm
    clipped = float(np.clip(scale_logit, -40.0, 40.0))
    alpha = float(alpha_max / (1.0 + np.exp(-clipped)))
    return direction, alpha


@dataclass
class DirectionScalePolicy:
    """Small deterministic two-layer MLP used for CPU mechanism checks."""

    input_dim: int
    n_assets: int
    hidden_dim: int
    alpha_max: float
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        first_scale = np.sqrt(2.0 / max(self.input_dim, 1))
        hidden_scale = np.sqrt(2.0 / max(self.hidden_dim, 1))
        self.w1 = rng.normal(0.0, first_scale, (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = rng.normal(0.0, hidden_scale, (self.hidden_dim, self.hidden_dim))
        self.b2 = np.zeros(self.hidden_dim)
        self.w_direction = rng.normal(
            0.0, hidden_scale, (self.hidden_dim, self.n_assets)
        )
        self.b_direction = np.zeros(self.n_assets)
        self.w_scale = rng.normal(0.0, hidden_scale, self.hidden_dim)
        self.b_scale = 0.0

    def __call__(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.input_dim,):
            raise ValueError(f"state must have shape ({self.input_dim},)")
        hidden = np.tanh(state @ self.w1 + self.b1)
        hidden = np.tanh(hidden @ self.w2 + self.b2)
        logits = hidden @ self.w_direction + self.b_direction
        scale_logit = float(hidden @ self.w_scale + self.b_scale)
        return direction_and_scale(logits, scale_logit, self.alpha_max)

    def raw_weights(
        self, state: np.ndarray, base_weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        direction, alpha = self(state)
        return np.asarray(base_weights) + alpha * direction, direction, alpha
