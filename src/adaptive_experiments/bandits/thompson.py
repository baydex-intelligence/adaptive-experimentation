"""
Thompson sampling with Beta-Bernoulli conjugate updating.

Each arm maintains a Beta(alpha, beta) posterior over its success probability.
At each step, one sample is drawn from each posterior and the arm with the
highest sample is selected.

The Beta posterior for a Bernoulli arm is:
    prior: Beta(1, 1)  (uniform)
    after s successes and f failures: Beta(1 + s, 1 + f)
"""

import numpy as np

from adaptive_experiments.bandits.base import BanditPolicy


class ThompsonSampling(BanditPolicy):
    """
    Beta-Bernoulli Thompson sampling.

    Maintains separate alpha and beta parameters per arm rather than re-using
    the generic values/counts from the base class (which tracks the mean only).

    Invariants: alpha[i] >= 1, beta[i] >= 1 for all i.
    """

    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        super().__init__(n_arms)
        self.alpha: list[float] = [1.0] * n_arms
        self.beta: list[float] = [1.0] * n_arms
        self._rng = np.random.default_rng(seed)

    def select_arm(self) -> int:
        samples = [
            self._rng.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)
        ]
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """
        Update Beta posterior for the chosen arm.

        Side effects: updates alpha[arm] or beta[arm] depending on reward;
        also calls super().update() to maintain counts and empirical mean.
        """
        super().update(arm, reward)
        if reward == 1:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0

    def reset(self) -> None:
        super().reset()
        self.alpha = [1.0] * self.n_arms
        self.beta = [1.0] * self.n_arms
