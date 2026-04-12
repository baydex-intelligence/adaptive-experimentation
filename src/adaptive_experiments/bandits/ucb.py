"""
UCB1 bandit policy.

Selects the arm that maximises the upper confidence bound:
    UCB(a) = estimated_value(a) + sqrt(2 * log(t) / n(a))

Unplayed arms are always selected first to avoid division by zero.
"""

import math

from adaptive_experiments.bandits.base import BanditPolicy


class UCB1(BanditPolicy):
    """
    UCB1 policy (Auer et al., 2002).

    Achieves O(log T) regret for stationary Bernoulli bandits.
    """

    def select_arm(self) -> int:
        # Always try each arm at least once.
        for i, count in enumerate(self.counts):
            if count == 0:
                return i

        log_t = math.log(self.t)
        ucb_values = [
            self.values[i] + math.sqrt(2 * log_t / self.counts[i])
            for i in range(self.n_arms)
        ]
        return int(max(range(self.n_arms), key=lambda i: ucb_values[i]))
