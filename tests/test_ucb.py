"""Tests for UCB1 policy."""

import pytest

from adaptive_experiments.bandits.ucb import UCB1


class TestUCB1:
    def setup_method(self):
        self.ucb = UCB1(n_arms=3)

    def test_initial_selection_cycles_through_all_arms(self):
        """Each arm must be tried once before exploitation starts."""
        arms_selected = set()
        for _ in range(3):
            arm = self.ucb.select_arm()
            arms_selected.add(arm)
            self.ucb.update(arm, 1)
        assert arms_selected == {0, 1, 2}

    def test_exploits_best_arm_after_clear_signal(self):
        """After many pulls, arm 1's UCB should exceed arms 0 and 2."""
        ucb = UCB1(n_arms=3)
        # All arms have been pulled many times so the exploration bonus is small.
        ucb.counts[0] = 100
        ucb.counts[1] = 100
        ucb.counts[2] = 100
        ucb.values[1] = 0.95
        ucb.values[0] = 0.10
        ucb.values[2] = 0.10
        ucb.t = 300

        arm = ucb.select_arm()
        assert arm == 1

    def test_unplayed_arm_selected_first(self):
        """UCB1 must explore before exploiting."""
        # Only update arm 0 and 1; arm 2 has count 0 and should be selected.
        self.ucb.update(0, 1)
        self.ucb.update(1, 0)
        arm = self.ucb.select_arm()
        assert arm == 2

    def test_reset_clears_state(self):
        self.ucb.update(0, 1)
        self.ucb.update(1, 0)
        self.ucb.reset()
        assert self.ucb.counts == [0, 0, 0]
        assert self.ucb.t == 0

    def test_returns_valid_arm_index(self):
        for _ in range(20):
            arm = self.ucb.select_arm()
            assert 0 <= arm < 3
            self.ucb.update(arm, 1)

    def test_select_arm_without_update_explores_sequentially(self):
        """Before any updates, select_arm should return each arm once in order."""
        first = self.ucb.select_arm()
        assert first == 0
