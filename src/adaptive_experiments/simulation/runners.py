"""
Simulation runner for bandit policies.

Runs a policy against an environment for T steps and returns per-step records
that can be used to compute metrics such as cumulative reward and regret.
"""

from dataclasses import dataclass

import numpy as np

from adaptive_experiments.bandits.base import BanditPolicy
from adaptive_experiments.simulation.environments import BernoulliEnvironment


@dataclass
class StepRecord:
    """Outcome of a single bandit step."""

    t: int
    arm: int
    reward: int
    best_p: float


def run_trial(
    policy: BanditPolicy,
    env: BernoulliEnvironment,
    n_steps: int,
) -> list[StepRecord]:
    """
    Run a single trial: apply policy to env for n_steps steps.

    The policy is NOT reset before running — call policy.reset() beforehand
    if starting fresh.

    Returns one StepRecord per step.
    """
    records: list[StepRecord] = []
    for t in range(1, n_steps + 1):
        arm = policy.select_arm()
        reward = env.pull(arm)
        policy.update(arm, reward)
        records.append(StepRecord(t=t, arm=arm, reward=reward, best_p=env.best_p))
    return records


def run_repeated_trials(
    policy: BanditPolicy,
    env: BernoulliEnvironment,
    n_steps: int,
    n_trials: int,
) -> list[list[StepRecord]]:
    """
    Run n_trials independent trials, resetting the policy between each.

    Returns a list of per-trial record lists.
    """
    all_trials = []
    for _ in range(n_trials):
        policy.reset()
        all_trials.append(run_trial(policy, env, n_steps))
    return all_trials


def average_cumulative_reward(trials: list[list[StepRecord]]) -> np.ndarray:
    """
    Mean cumulative reward across trials, shape (n_steps,).
    """
    rewards = np.array([[r.reward for r in trial] for trial in trials], dtype=float)
    return rewards.cumsum(axis=1).mean(axis=0)


def average_cumulative_regret(trials: list[list[StepRecord]]) -> np.ndarray:
    """
    Mean cumulative regret across trials, shape (n_steps,).

    Regret at step t = best_p - reward_t.
    """
    regrets = np.array(
        [[r.best_p - r.reward for r in trial] for trial in trials], dtype=float
    )
    return regrets.cumsum(axis=1).mean(axis=0)
