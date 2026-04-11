"""
Abstract base class for bandit policies.

A policy selects an arm index at each step and updates its state with the
observed reward.
"""

from abc import ABC, abstractmethod


class BanditPolicy(ABC):
    """
    Interface for a k-armed bandit policy.

    Invariants: select_arm() returns an index in [0, n_arms).
    """

    def __init__(self, n_arms: int) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1.")
        self.n_arms = n_arms
        self.counts: list[int] = [0] * n_arms
        self.values: list[float] = [0.0] * n_arms
        self.t: int = 0

    @abstractmethod
    def select_arm(self) -> int:
        """Return the index of the arm to pull."""

    def update(self, arm: int, reward: float) -> None:
        """
        Incremental mean update for the chosen arm.

        Side effects: increments counts[arm], updates values[arm], increments t.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.t += 1

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms
        self.t = 0
