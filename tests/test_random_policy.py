"""Tests for the uniform random baseline policy."""

import numpy as np

from adaptive_experiments.bandits.random import RandomPolicy


def test_random_policy_returns_valid_arm_indices() -> None:
    policy = RandomPolicy(n_arms=3, seed=123)
    selections = [policy.select_arm() for _ in range(200)]
    assert all(0 <= arm < 3 for arm in selections)


def test_random_policy_seed_is_reproducible() -> None:
    left = RandomPolicy(n_arms=4, seed=7)
    right = RandomPolicy(n_arms=4, seed=7)

    left_draws = np.array([left.select_arm() for _ in range(40)], dtype=int)
    right_draws = np.array([right.select_arm() for _ in range(40)], dtype=int)

    assert np.array_equal(left_draws, right_draws)
