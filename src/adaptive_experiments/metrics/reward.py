"""
Cumulative reward metric.
"""

import numpy as np

from adaptive_experiments.simulation.runners import StepRecord


def cumulative_reward(records: list[StepRecord]) -> np.ndarray:
    """
    Per-step cumulative reward for a single trial.

    Returns an array of shape (n_steps,).
    """
    rewards = np.array([r.reward for r in records], dtype=float)
    return rewards.cumsum()


def total_reward(records: list[StepRecord]) -> float:
    """Total reward at the end of a trial."""
    return float(sum(r.reward for r in records))
