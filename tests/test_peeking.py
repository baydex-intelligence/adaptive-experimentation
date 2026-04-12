"""Tests for peeking simulation behaviour."""

import pytest

from adaptive_experiments.ab_testing.peeking import simulate_peeking, simulate_peeking_sweep


class TestSimulatePeeking:
    def test_single_look_fpr_near_alpha(self):
        """With one look, the FPR should be close to alpha."""
        result = simulate_peeking(
            p_true=0.10,
            n_per_group=1000,
            n_looks=1,
            alpha=0.05,
            n_simulations=2000,
            seed=0,
        )
        # Should be within 3 percentage points of nominal alpha.
        assert abs(result.false_positive_rate - 0.05) < 0.03

    def test_many_looks_inflate_fpr(self):
        """20 looks should give a meaningfully higher FPR than 1 look."""
        r1 = simulate_peeking(n_looks=1, n_simulations=1000, seed=1)
        r20 = simulate_peeking(n_looks=20, n_simulations=1000, seed=1)
        assert r20.false_positive_rate > r1.false_positive_rate * 1.5

    def test_result_fields(self):
        result = simulate_peeking(n_looks=5, n_simulations=200, seed=42)
        assert result.n_simulations == 200
        assert result.n_looks == 5
        assert 0.0 <= result.false_positive_rate <= 1.0

    def test_fpr_is_float_in_range(self):
        result = simulate_peeking(n_simulations=100, seed=7)
        assert isinstance(result.false_positive_rate, float)
        assert 0.0 <= result.false_positive_rate <= 1.0


class TestSimulatePeekingSweep:
    def test_returns_list_of_correct_length(self):
        results = simulate_peeking_sweep(
            looks_range=[1, 5, 10], n_simulations=100, seed=0
        )
        assert len(results) == 3

    def test_fpr_monotonically_increases(self):
        """FPR should generally increase with more looks (statistically)."""
        results = simulate_peeking_sweep(
            looks_range=[1, 5, 20],
            n_simulations=2000,
            seed=42,
        )
        fprs = [r.false_positive_rate for r in results]
        # Not guaranteed to be strictly monotone, but last should exceed first.
        assert fprs[-1] > fprs[0]
