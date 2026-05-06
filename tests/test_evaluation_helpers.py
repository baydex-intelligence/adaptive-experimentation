"""Tests for reusable sweep-evaluation helpers."""

import numpy as np

from adaptive_experiments.simulation.evaluation import (
    best_arm_share_curve,
    checkpoint_steps,
    late_stage_best_arm_share,
    make_default_policy_factories,
    mean_confidence_interval,
)


def test_make_default_policy_factories_returns_expected_names() -> None:
    factories = make_default_policy_factories()
    assert set(factories.keys()) == {"Random", "DecayingEpsilon", "UCB1", "Thompson"}


def test_checkpoint_steps_clamps_within_horizon() -> None:
    checkpoints = checkpoint_steps(5, fractions=(0.1, 0.5, 1.0))
    assert checkpoints == {"10%": 1, "50%": 2, "100%": 5}


def test_best_arm_share_curve_is_cumulative_fraction() -> None:
    arms = np.array([1, 0, 1, 1], dtype=int)
    observed = best_arm_share_curve(arms, best_arm=1)
    expected = np.array([1.0, 0.5, 2.0 / 3.0, 0.75], dtype=float)
    assert np.allclose(observed, expected)


def test_late_stage_best_arm_share_uses_trailing_window() -> None:
    arms = np.array([0, 1, 1, 0, 1], dtype=int)
    observed = late_stage_best_arm_share(arms, best_arm=1, window=3)
    assert observed == 2.0 / 3.0


def test_mean_confidence_interval_single_observation_is_point() -> None:
    lower, upper = mean_confidence_interval(np.array([3.5], dtype=float), z_value=1.96)
    assert lower == 3.5
    assert upper == 3.5
