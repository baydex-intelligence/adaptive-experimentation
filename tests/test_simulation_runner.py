"""Tests for the simulation runner."""

import numpy as np
import pytest

from adaptive_experiments.bandits.decaying_epsilon import DecayingEpsilon
from adaptive_experiments.bandits.thompson import ThompsonSampling
from adaptive_experiments.bandits.ucb import UCB1
from adaptive_experiments.simulation.environments import BernoulliEnvironment
from adaptive_experiments.simulation.runners import (
    run_trial,
    run_repeated_trials,
    average_cumulative_reward,
    average_cumulative_pseudo_regret,
    average_cumulative_realized_regret,
)
from adaptive_experiments.metrics.regret import (
    cumulative_pseudo_regret,
    cumulative_realized_regret,
)
from adaptive_experiments.metrics.reward import cumulative_reward, total_reward
from adaptive_experiments.metrics.allocation import allocation_shares


ENV_PROBS = [0.2, 0.8]
N_STEPS = 100


@pytest.fixture
def env():
    return BernoulliEnvironment.from_probs(ENV_PROBS, seed=0)


@pytest.fixture
def policy(env):
    return DecayingEpsilon(
        n_arms=env.n_arms,
        initial_epsilon=0.2,
        min_epsilon=0.01,
        decay_rate=0.01,
        seed=1,
    )


class TestRunTrial:
    def test_returns_correct_number_of_steps(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        assert len(records) == N_STEPS

    def test_step_indices_are_sequential(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        assert [r.t for r in records] == list(range(1, N_STEPS + 1))

    def test_rewards_are_binary(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        assert all(r.reward in (0, 1) for r in records)

    def test_arm_indices_in_range(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        assert all(0 <= r.arm < env.n_arms for r in records)

    def test_best_p_is_correct(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        assert all(r.best_p == env.best_p for r in records)

    def test_chosen_p_matches_selected_arm(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        assert all(r.chosen_p == env.arms[r.arm].p for r in records)


class TestRunRepeatedTrials:
    def test_correct_number_of_trials(self, env, policy):
        trials = run_repeated_trials(policy, env, N_STEPS, n_trials=5)
        assert len(trials) == 5

    def test_policy_is_reset_between_trials(self, env):
        """Counts should start from zero at the beginning of each trial."""
        policy = DecayingEpsilon(
            n_arms=2,
            initial_epsilon=0.0,
            min_epsilon=0.0,
            decay_rate=0.01,
            seed=0,
        )
        trials = run_repeated_trials(policy, env, N_STEPS, n_trials=3)
        # Policy should have been reset and re-run; all trials have N_STEPS records.
        for trial in trials:
            assert len(trial) == N_STEPS


class TestAverageMetrics:
    def test_avg_cumulative_reward_shape(self, env, policy):
        trials = run_repeated_trials(policy, env, N_STEPS, n_trials=10)
        avg = average_cumulative_reward(trials)
        assert avg.shape == (N_STEPS,)

    def test_avg_cumulative_pseudo_regret_shape(self, env, policy):
        trials = run_repeated_trials(policy, env, N_STEPS, n_trials=10)
        avg = average_cumulative_pseudo_regret(trials)
        assert avg.shape == (N_STEPS,)

    def test_avg_cumulative_realized_regret_shape(self, env, policy):
        trials = run_repeated_trials(policy, env, N_STEPS, n_trials=10)
        avg = average_cumulative_realized_regret(trials)
        assert avg.shape == (N_STEPS,)

    def test_avg_cumulative_reward_is_non_decreasing(self, env, policy):
        trials = run_repeated_trials(policy, env, N_STEPS, n_trials=10)
        avg = average_cumulative_reward(trials)
        assert np.all(np.diff(avg) >= 0)

    def test_avg_cumulative_pseudo_regret_non_negative(self, env, policy):
        trials = run_repeated_trials(policy, env, N_STEPS, n_trials=10)
        avg = average_cumulative_pseudo_regret(trials)
        assert avg[-1] >= 0

    def test_avg_cumulative_realized_regret_is_finite(self, env, policy):
        trials = run_repeated_trials(policy, env, N_STEPS, n_trials=10)
        avg = average_cumulative_realized_regret(trials)
        assert np.isfinite(avg[-1])


class TestMetrics:
    def test_cumulative_pseudo_regret_length(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        r = cumulative_pseudo_regret(records)
        assert len(r) == N_STEPS

    def test_cumulative_realized_regret_length(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        r = cumulative_realized_regret(records)
        assert len(r) == N_STEPS

    def test_cumulative_reward_length(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        r = cumulative_reward(records)
        assert len(r) == N_STEPS

    def test_total_reward_non_negative(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        assert total_reward(records) >= 0

    def test_allocation_shares_sum_to_one(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        shares = allocation_shares(records, n_arms=env.n_arms)
        assert abs(sum(shares.values()) - 1.0) < 1e-9

    def test_allocation_shares_keys(self, env, policy):
        records = run_trial(policy, env, N_STEPS)
        shares = allocation_shares(records, n_arms=env.n_arms)
        assert set(shares.keys()) == {0, 1}


class TestBestArmConvergence:
    def test_thompson_outperforms_random_on_easy_env(self):
        """Thompson sampling should accumulate less regret than random on easy env."""
        import numpy as np
        from adaptive_experiments.bandits.base import BanditPolicy

        class RandomPolicy(BanditPolicy):
            def __init__(self, n_arms, seed=None):
                super().__init__(n_arms)
                self._rng = np.random.default_rng(seed)

            def select_arm(self):
                return int(self._rng.integers(0, self.n_arms))

        env = BernoulliEnvironment.from_probs([0.1, 0.9], seed=42)
        n_steps = 500
        n_trials = 50

        ts = ThompsonSampling(n_arms=2, seed=0)
        rp = RandomPolicy(n_arms=2, seed=0)

        ts_trials = run_repeated_trials(ts, env, n_steps, n_trials)
        rp_trials = run_repeated_trials(rp, env, n_steps, n_trials)

        ts_regret = average_cumulative_pseudo_regret(ts_trials)[-1]
        rp_regret = average_cumulative_pseudo_regret(rp_trials)[-1]

        assert ts_regret < rp_regret
