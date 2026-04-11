"""
Bernoulli bandit environment — a collection of arms with known true probabilities.
"""

from dataclasses import dataclass

import numpy as np

from adaptive_experiments.bandits.arms import BernoulliArm


@dataclass
class BernoulliEnvironment:
    """
    A k-armed Bernoulli bandit environment.

    Fields:
        arms: list of BernoulliArm, one per action.
        best_arm: index of the arm with the highest p.
        best_p: success probability of the best arm.
    """

    arms: list[BernoulliArm]
    best_arm: int
    best_p: float

    @classmethod
    def from_probs(
        cls, probs: list[float], seed: int | None = None
    ) -> "BernoulliEnvironment":
        """
        Construct an environment from a list of success probabilities.

        Each arm gets its own independent RNG derived from seed.
        """
        rng = np.random.default_rng(seed)
        arms = [
            BernoulliArm(p, rng=np.random.default_rng(int(rng.integers(0, 2**31))))
            for p in probs
        ]
        best_arm = int(np.argmax(probs))
        return cls(arms=arms, best_arm=best_arm, best_p=probs[best_arm])

    def pull(self, arm_index: int) -> int:
        """Pull the specified arm and return the reward."""
        return self.arms[arm_index].pull()

    @property
    def n_arms(self) -> int:
        return len(self.arms)
