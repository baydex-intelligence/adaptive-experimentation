"""
Named scenario helpers for common bandit setups.

Each function returns a (BernoulliEnvironment, label) pair so that scripts can
reference scenarios by name without duplicating parameter lists.
"""

from adaptive_experiments.simulation.environments import BernoulliEnvironment


def easy_two_arm(seed: int = 0) -> tuple[BernoulliEnvironment, str]:
    """Two arms with a clear gap: p = [0.3, 0.6]."""
    return BernoulliEnvironment.from_probs([0.3, 0.6], seed=seed), "easy_two_arm"


def close_two_arm(seed: int = 0) -> tuple[BernoulliEnvironment, str]:
    """Two arms with a small gap: p = [0.45, 0.50]."""
    return BernoulliEnvironment.from_probs([0.45, 0.50], seed=seed), "close_two_arm"


def four_arm(seed: int = 0) -> tuple[BernoulliEnvironment, str]:
    """Four arms: p = [0.1, 0.3, 0.5, 0.7]."""
    return BernoulliEnvironment.from_probs([0.1, 0.3, 0.5, 0.7], seed=seed), "four_arm"
