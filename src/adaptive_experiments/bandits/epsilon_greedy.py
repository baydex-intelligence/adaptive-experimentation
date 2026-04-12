"""
Epsilon-greedy bandit policy.

With probability epsilon, selects a random arm (explore).
With probability 1 - epsilon, selects the arm with the highest estimated value (exploit).
"""

import numpy as np

from adaptive_experiments.bandits.base import BanditPolicy


class EpsilonGreedy(BanditPolicy):
    """
    Epsilon-greedy policy with a fixed exploration rate.

    Assumptions: epsilon in [0, 1].
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int | None = None) -> None:
        super().__init__(n_arms)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)

    def select_arm(self) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_arms))
        return int(np.argmax(self.values))
