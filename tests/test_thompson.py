"""Tests for Thompson sampling (Beta-Bernoulli)."""

import pytest

from adaptive_experiments.bandits.thompson import ThompsonSampling


class TestThompsonSampling:
    def setup_method(self):
        self.ts = ThompsonSampling(n_arms=3, seed=42)

    def test_select_arm_in_range(self):
        arm = self.ts.select_arm()
        assert 0 <= arm < 3

    def test_initial_priors_uniform(self):
        assert self.ts.alpha == [1.0, 1.0, 1.0]
        assert self.ts.beta == [1.0, 1.0, 1.0]

    def test_update_increments_alpha_on_success(self):
        self.ts.update(arm=0, reward=1)
        assert self.ts.alpha[0] == 2.0
        assert self.ts.beta[0] == 1.0

    def test_update_increments_beta_on_failure(self):
        self.ts.update(arm=1, reward=0)
        assert self.ts.alpha[1] == 1.0
        assert self.ts.beta[1] == 2.0

    def test_update_other_arms_unchanged(self):
        self.ts.update(arm=0, reward=1)
        assert self.ts.alpha[1] == 1.0
        assert self.ts.alpha[2] == 1.0

    def test_reset_restores_priors(self):
        self.ts.update(arm=0, reward=1)
        self.ts.update(arm=1, reward=0)
        self.ts.reset()
        assert self.ts.alpha == [1.0, 1.0, 1.0]
        assert self.ts.beta == [1.0, 1.0, 1.0]
        assert self.ts.t == 0

    def test_converges_to_best_arm(self):
        """After many updates, the arm with highest alpha/(alpha+beta) should be selected most."""
        ts = ThompsonSampling(n_arms=2, seed=0)
        # Arm 0: 20 successes, 2 failures → strong signal
        for _ in range(20):
            ts.update(0, 1)
        for _ in range(2):
            ts.update(0, 0)
        # Arm 1: 2 successes, 20 failures → strong signal for being bad
        for _ in range(2):
            ts.update(1, 1)
        for _ in range(20):
            ts.update(1, 0)

        counts = [0, 0]
        for _ in range(1000):
            arm = ts.select_arm()
            counts[arm] += 1
        # Arm 0 should be selected the vast majority of the time.
        assert counts[0] > counts[1] * 5

    def test_counts_tracked(self):
        self.ts.update(arm=2, reward=1)
        assert self.ts.counts[2] == 1
        assert self.ts.t == 1
