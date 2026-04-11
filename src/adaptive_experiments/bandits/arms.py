"""
Bernoulli arm abstraction.

Each arm represents an action with a fixed but unknown probability of success.
"""

import numpy as np


class BernoulliArm:
    """
    An arm that returns 1 (success) with probability p and 0 otherwise.

    Invariants: 0 <= p <= 1.
    """

    def __init__(self, p: float, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}.")
        self.p = p
        self._rng = rng if rng is not None else np.random.default_rng()

    def pull(self) -> int:
        """Draw a Bernoulli(p) reward."""
        return int(self._rng.random() < self.p)

    def __repr__(self) -> str:
        return f"BernoulliArm(p={self.p})"
