"""Regret metrics for bandit simulations."""

import numpy as np

from adaptive_experiments.simulation.runners import StepRecord


def cumulative_pseudo_regret(records: list[StepRecord]) -> np.ndarray:
    """
    Per-step cumulative pseudo-regret for a single trial.

    Pseudo-regret at step t is best_p - chosen_p_t, which isolates decision
    quality from the extra randomness of realized Bernoulli rewards.
    """
    regrets = np.array([r.best_p - r.chosen_p for r in records], dtype=float)
    return regrets.cumsum()


def final_pseudo_regret(records: list[StepRecord]) -> float:
    """Total pseudo-regret at the end of a trial."""
    return float(cumulative_pseudo_regret(records)[-1])


def cumulative_realized_regret(records: list[StepRecord]) -> np.ndarray:
    """
    Per-step cumulative realized regret for a single trial.

    Realized regret at step t is best_p - reward_t, which keeps reward noise in
    the metric and therefore reflects a noisier sample path than pseudo-regret.
    """
    regrets = np.array([r.best_p - r.reward for r in records], dtype=float)
    return regrets.cumsum()


def final_realized_regret(records: list[StepRecord]) -> float:
    """Total realized regret at the end of a trial."""
    return float(cumulative_realized_regret(records)[-1])
