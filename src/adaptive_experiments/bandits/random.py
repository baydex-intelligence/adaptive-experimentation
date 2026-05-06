"""Uniform random baseline policy for multi-armed bandits."""

import numpy as np

from adaptive_experiments.bandits.base import BanditPolicy


class RandomPolicy(BanditPolicy):
    """Select arms uniformly at random without learning."""

    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        super().__init__(n_arms)
        self._rng = np.random.default_rng(seed)

    def select_arm(self) -> int:
        return int(self._rng.integers(0, self.n_arms))
