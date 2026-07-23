from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from asset_allocation.exceptions import OptionalDependencyError
from asset_allocation.policy_models.direction_scale import direction_and_scale
from asset_allocation.portfolio_environment import PortfolioEnvironment


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 100_000
    rollout_steps: int = 512
    update_epochs: int = 10
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    hidden_dim: int = 64
    seed: int = 20260723


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise OptionalDependencyError(
            'PPO requires PyTorch; install with: python -m pip install -e ".[train]"'
        ) from exc
    return torch, nn


def build_actor_critic(
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    device: str = "cpu",
):
    torch, nn = _torch()

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
            self.critic = nn.Linear(hidden_dim, 1)

        def distribution_and_value(self, state):
            encoded = self.encoder(state)
            mean = self.actor_mean(encoded)
            std = torch.exp(self.actor_log_std).expand_as(mean)
            return (
                torch.distributions.Normal(mean, std),
                self.critic(encoded).squeeze(-1),
            )

    return ActorCritic().to(device)


def _gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_value = bootstrap_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * gae_lambda * nonterminal * last
        advantages[t] = last
    return advantages, advantages + values


def train_ppo(
    environment_factory: Callable[[], PortfolioEnvironment],
    config: PPOConfig,
    device: str = "cpu",
) -> tuple[object, list[dict[str, float]]]:
    """Train a compact Gaussian-latent PPO policy.

    The sampled latent action is transformed into a zero-sum, L1-normalized
    direction and bounded scale before the deterministic hard projection.
    """
    torch, nn = _torch()
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    environment = environment_factory()
    state_dim = environment.state_dim
    action_dim = len(environment.config.assets) + 1

    model = build_actor_critic(state_dim, action_dim, config.hidden_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    state = environment.reset()
    completed_steps = 0
    history: list[dict[str, float]] = []

    while completed_steps < config.total_steps:
        states: list[np.ndarray] = []
        latents: list[np.ndarray] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        dones: list[float] = []
        values: list[float] = []
        rollout_length = min(config.rollout_steps, config.total_steps - completed_steps)
        for _ in range(rollout_length):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                distribution, value = model.distribution_and_value(state_tensor)
                latent = distribution.sample()
                log_prob = distribution.log_prob(latent).sum()
            latent_numpy = latent.cpu().numpy()
            direction, alpha = direction_and_scale(
                latent_numpy[:-1],
                float(latent_numpy[-1]),
                environment.config.alpha_max,
            )
            transition = environment.step(direction, alpha)
            states.append(state)
            latents.append(latent_numpy)
            log_probs.append(float(log_prob.cpu()))
            rewards.append(transition.reward)
            dones.append(float(transition.done))
            values.append(float(value.cpu()))
            state = transition.state
            if transition.done:
                state = environment.reset()
        completed_steps += rollout_length
        with torch.no_grad():
            _, bootstrap = model.distribution_and_value(
                torch.as_tensor(state, dtype=torch.float32, device=device)
            )
        advantages, returns = _gae(
            np.asarray(rewards),
            np.asarray(dones),
            np.asarray(values),
            float(bootstrap.cpu()),
            config.gamma,
            config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        state_batch = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        latent_batch = torch.as_tensor(np.asarray(latents), dtype=torch.float32, device=device)
        old_log_batch = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
        advantage_batch = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        return_batch = torch.as_tensor(returns, dtype=torch.float32, device=device)
        indices = np.arange(rollout_length)
        losses: list[float] = []
        for _ in range(config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_length, config.minibatch_size):
                selected = indices[start : start + config.minibatch_size]
                distribution, predicted_value = model.distribution_and_value(
                    state_batch[selected]
                )
                new_log = distribution.log_prob(latent_batch[selected]).sum(dim=-1)
                ratio = torch.exp(new_log - old_log_batch[selected])
                unclipped = ratio * advantage_batch[selected]
                clipped = torch.clamp(
                    ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                ) * advantage_batch[selected]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.mean(
                    (predicted_value - return_batch[selected]) ** 2
                )
                entropy = distribution.entropy().sum(dim=-1).mean()
                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "steps": float(completed_steps),
                "mean_reward": float(np.mean(rewards)),
                "mean_loss": float(np.mean(losses)),
            }
        )
    return model, history
