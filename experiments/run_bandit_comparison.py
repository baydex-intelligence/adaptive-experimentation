"""
Compare random, epsilon-greedy, UCB1, and Thompson sampling on a Bernoulli bandit.

Usage:
    poetry run python experiments/run_bandit_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from adaptive_experiments.bandits.base import BanditPolicy
from adaptive_experiments.bandits.decaying_epsilon import DecayingEpsilon
from adaptive_experiments.bandits.ucb import UCB1
from adaptive_experiments.bandits.thompson import ThompsonSampling
from adaptive_experiments.simulation.environments import BernoulliEnvironment
from adaptive_experiments.simulation.runners import (
    run_repeated_trials,
    average_cumulative_pseudo_regret,
    average_cumulative_reward,
)
from adaptive_experiments.metrics.allocation import allocation_shares
from adaptive_experiments.plotting.curves import plot_cumulative_regret


PROBS = [0.1, 0.3, 0.5, 0.7]
N_STEPS = 1000
N_TRIALS = 200
SEED = 0
OUTPUT_DIR = Path(__file__).parent.parent / "reports"


class RandomPolicy(BanditPolicy):
    """Uniform random baseline — selects each arm with equal probability."""

    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        super().__init__(n_arms)
        self._rng = np.random.default_rng(seed)

    def select_arm(self) -> int:
        return int(self._rng.integers(0, self.n_arms))


def main() -> None:
    env = BernoulliEnvironment.from_probs(PROBS, seed=SEED)
    n_arms = env.n_arms

    policies: dict[str, BanditPolicy] = {
        "Random": RandomPolicy(n_arms, seed=SEED),
        "DecayingEpsilon": DecayingEpsilon(
            n_arms,
            initial_epsilon=0.2,
            min_epsilon=0.01,
            decay_rate=0.01,
            seed=SEED,
        ),
        "UCB1": UCB1(n_arms),
        "Thompson": ThompsonSampling(n_arms, seed=SEED),
    }

    regret_curves: dict[str, np.ndarray] = {}

    print(f"Environment: {PROBS}  (best arm p={env.best_p})")
    print(f"Steps per trial: {N_STEPS},  Trials: {N_TRIALS}\n")
    print(f"{'Policy':<22}  {'Total regret':>14}  {'Total reward':>13}  {'Best-arm share':>15}")
    print("-" * 70)

    for name, policy in policies.items():
        trials = run_repeated_trials(policy, env, N_STEPS, N_TRIALS)
        avg_regret = average_cumulative_pseudo_regret(trials)
        avg_reward = average_cumulative_reward(trials)
        regret_curves[name] = avg_regret

        # Allocation from last trial (post-learning).
        last_trial = trials[-1]
        shares = allocation_shares(last_trial, n_arms)
        best_share = shares.get(env.best_arm, 0.0)

        print(
            f"{name:<22}  {avg_regret[-1]:>14.1f}  {avg_reward[-1]:>13.1f}  {best_share:>14.1%}"
        )

    # Plot.
    ax = plot_cumulative_regret(
        regret_curves, title="Bandit Comparison — Cumulative Pseudo-Regret"
    )
    ax.figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "bandit_comparison_regret.png"
    ax.figure.savefig(output_path, dpi=120)
    plt.close("all")
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
