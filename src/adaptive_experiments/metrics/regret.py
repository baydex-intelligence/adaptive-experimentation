"""
Cumulative regret metric.

Regret measures the total reward lost by not always playing the best arm.
Sublinear regret (o(T)) indicates the policy is learning over time.
"""

import numpy as np

from adaptive_experiments.simulation.runners import StepRecord


def cumulative_regret(records: list[StepRecord]) -> np.ndarray:
    """
    Per-step cumulative regret for a single trial.

    Returns an array of shape (n_steps,) where element t is the total
    regret accumulated up to and including step t.
    """
    regrets = np.array([r.best_p - r.reward for r in records], dtype=float)
    return regrets.cumsum()


def final_regret(records: list[StepRecord]) -> float:
    """Total regret at the end of a trial."""
    return float(cumulative_regret(records)[-1])
