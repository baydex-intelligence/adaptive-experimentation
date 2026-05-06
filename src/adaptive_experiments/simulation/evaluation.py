"""Shared helpers for notebook-style bandit evaluation sweeps."""

from collections.abc import Callable, Sequence

import numpy as np

from adaptive_experiments.bandits.base import BanditPolicy
from adaptive_experiments.bandits.decaying_epsilon import DecayingEpsilon
from adaptive_experiments.bandits.random import RandomPolicy
from adaptive_experiments.bandits.thompson import ThompsonSampling
from adaptive_experiments.bandits.ucb import UCB1


def make_default_policy_factories(
    initial_epsilon: float = 0.2,
    min_epsilon: float = 0.01,
    decay_rate: float = 0.01,
) -> dict[str, Callable[[int, int], BanditPolicy]]:
    """Return standard policy constructors used in comparison sweeps."""

    def make_random_policy(n_arms: int, seed: int) -> BanditPolicy:
        return RandomPolicy(n_arms, seed=seed)

    def make_decaying_epsilon_policy(n_arms: int, seed: int) -> BanditPolicy:
        return DecayingEpsilon(
            n_arms,
            initial_epsilon=initial_epsilon,
            min_epsilon=min_epsilon,
            decay_rate=decay_rate,
            seed=seed,
        )

    def make_ucb1_policy(n_arms: int, seed: int) -> BanditPolicy:
        del seed
        return UCB1(n_arms)

    def make_thompson_policy(n_arms: int, seed: int) -> BanditPolicy:
        return ThompsonSampling(n_arms, seed=seed)

    return {
        "Random": make_random_policy,
        "DecayingEpsilon": make_decaying_epsilon_policy,
        "UCB1": make_ucb1_policy,
        "Thompson": make_thompson_policy,
    }


def default_policy_colors() -> dict[str, str]:
    """Return a stable color mapping for common policy names."""
    return {
        "Random": "tab:blue",
        "DecayingEpsilon": "tab:orange",
        "UCB1": "tab:green",
        "Thompson": "tab:red",
    }


def checkpoint_steps(
    horizon: int,
    fractions: Sequence[float] = (0.10, 0.25, 0.50, 1.00),
) -> dict[str, int]:
    """Map percentage labels to step indices within a horizon."""
    checkpoints: dict[str, int] = {}
    for fraction in fractions:
        step = max(1, int(round(horizon * fraction)))
        checkpoints[f"{int(fraction * 100)}%"] = min(step, horizon)
    return checkpoints


def best_arm_share_curve(arms: np.ndarray, best_arm: int) -> np.ndarray:
    """Cumulative share of pulls assigned to the best arm over time."""
    hits = (arms == best_arm).astype(float)
    steps = np.arange(1, len(arms) + 1, dtype=float)
    return np.cumsum(hits) / steps


def late_stage_best_arm_share(
    arms: np.ndarray,
    best_arm: int,
    window: int = 200,
) -> float:
    """Best-arm allocation share in the trailing window of pulls."""
    tail = arms[-window:] if len(arms) >= window else arms
    return float(np.mean(tail == best_arm))


def mean_confidence_interval(values: np.ndarray, z_value: float = 1.96) -> tuple[float, float]:
    """Normal-approximation confidence interval for a sample mean."""
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    margin = z_value * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean - margin, mean + margin
