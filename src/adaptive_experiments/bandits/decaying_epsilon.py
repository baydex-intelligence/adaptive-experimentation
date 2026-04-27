"""
Decaying epsilon-greedy bandit policy.

Explores with probability epsilon_t, where epsilon decays over time toward a
minimum floor. With probability 1 - epsilon_t, selects the arm with the highest
estimated value.
"""

import numpy as np

from adaptive_experiments.bandits.base import BanditPolicy


class DecayingEpsilon(BanditPolicy):
    """
    Epsilon-greedy policy with a decaying exploration rate.

    epsilon_t = max(min_epsilon, initial_epsilon / (1 + decay_rate * t))

    Assumptions:
        - initial_epsilon in [0, 1]
        - min_epsilon in [0, 1]
        - min_epsilon <= initial_epsilon
        - decay_rate >= 0
    """

    def __init__(
        self,
        n_arms: int,
        initial_epsilon: float = 0.2,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.01,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms)
        if not 0.0 <= initial_epsilon <= 1.0:
            raise ValueError("initial_epsilon must be in [0, 1].")
        if not 0.0 <= min_epsilon <= 1.0:
            raise ValueError("min_epsilon must be in [0, 1].")
        if min_epsilon > initial_epsilon:
            raise ValueError("min_epsilon must be <= initial_epsilon.")
        if decay_rate < 0.0:
            raise ValueError("decay_rate must be >= 0.")

        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self._rng = np.random.default_rng(seed)

    def current_epsilon(self) -> float:
        """Return the current exploration rate as a function of elapsed steps."""
        decayed = self.initial_epsilon / (1.0 + self.decay_rate * self.t)
        return max(self.min_epsilon, decayed)

    def select_arm(self) -> int:
        if self._rng.random() < self.current_epsilon():
            return int(self._rng.integers(0, self.n_arms))
        return int(np.argmax(self.values))
