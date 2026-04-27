"""Tests for decaying epsilon-greedy policy."""

import pytest

from adaptive_experiments.bandits.decaying_epsilon import DecayingEpsilon


class TestDecayingEpsilon:
    def setup_method(self):
        self.policy = DecayingEpsilon(
            n_arms=3,
            initial_epsilon=0.2,
            min_epsilon=0.01,
            decay_rate=0.05,
            seed=42,
        )

    def test_select_arm_in_range(self):
        arm = self.policy.select_arm()
        assert 0 <= arm < 3

    def test_initial_epsilon_matches_configuration(self):
        assert self.policy.current_epsilon() == pytest.approx(0.2)

    def test_epsilon_decays_after_updates(self):
        for _ in range(10):
            self.policy.update(0, 1)
        assert self.policy.current_epsilon() < 0.2

    def test_epsilon_respects_minimum_floor(self):
        for _ in range(10_000):
            self.policy.update(0, 1)
        assert self.policy.current_epsilon() == pytest.approx(0.01)

    def test_invalid_initial_epsilon_raises(self):
        with pytest.raises(ValueError):
            DecayingEpsilon(n_arms=2, initial_epsilon=1.5)

    def test_invalid_min_epsilon_raises(self):
        with pytest.raises(ValueError):
            DecayingEpsilon(n_arms=2, initial_epsilon=0.2, min_epsilon=0.3)

    def test_negative_decay_rate_raises(self):
        with pytest.raises(ValueError):
            DecayingEpsilon(n_arms=2, decay_rate=-0.1)

    def test_reset_restores_initial_schedule(self):
        for _ in range(25):
            self.policy.update(0, 1)
        self.policy.reset()
        assert self.policy.t == 0
        assert self.policy.current_epsilon() == pytest.approx(0.2)
