"""Tests for two-proportion hypothesis test utilities."""

import math

import pytest

from adaptive_experiments.ab_testing.hypothesis_tests import two_proportion_ztest
from adaptive_experiments.ab_testing.effect_sizes import cohens_h, relative_lift, absolute_lift
from adaptive_experiments.ab_testing.intervals import wilson_interval, difference_interval
from adaptive_experiments.ab_testing.power import minimum_sample_size, achieved_power


class TestTwoProportionZtest:
    def test_no_difference_p_value_near_one(self):
        result = two_proportion_ztest(50, 500, 50, 500)
        assert result.p_value > 0.80

    def test_large_difference_significant(self):
        result = two_proportion_ztest(10, 1000, 200, 1000)
        assert result.p_value < 0.001

    def test_z_stat_sign(self):
        # treatment > control → positive z for 'two-sided'
        result = two_proportion_ztest(50, 500, 100, 500)
        assert result.z_stat > 0

    def test_alternative_greater(self):
        result = two_proportion_ztest(10, 100, 20, 100, alternative="greater")
        assert 0 < result.p_value < 0.5

    def test_alternative_less_returns_high_p_when_treatment_better(self):
        result = two_proportion_ztest(10, 100, 20, 100, alternative="less")
        assert result.p_value > 0.9

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError, match="Unknown alternative"):
            two_proportion_ztest(10, 100, 20, 100, alternative="bad")

    def test_equal_proportions_z_near_zero(self):
        result = two_proportion_ztest(100, 1000, 100, 1000)
        assert abs(result.z_stat) < 1e-9

    def test_result_proportions_correct(self):
        result = two_proportion_ztest(50, 200, 80, 200)
        assert math.isclose(result.p_control, 0.25)
        assert math.isclose(result.p_treatment, 0.40)


class TestEffectSizes:
    def test_cohens_h_zero_for_equal(self):
        assert math.isclose(cohens_h(0.5, 0.5), 0.0)

    def test_cohens_h_positive_when_p1_greater(self):
        assert cohens_h(0.6, 0.4) > 0

    def test_cohens_h_negative_when_p1_smaller(self):
        assert cohens_h(0.3, 0.5) < 0

    def test_relative_lift_correct(self):
        assert math.isclose(relative_lift(0.10, 0.12), 0.20, rel_tol=1e-6)

    def test_relative_lift_zero_control_raises(self):
        with pytest.raises(ValueError):
            relative_lift(0.0, 0.1)

    def test_absolute_lift(self):
        assert math.isclose(absolute_lift(0.10, 0.15), 0.05)


class TestIntervals:
    def test_wilson_contains_true_p(self):
        ci = wilson_interval(50, 100, confidence=0.95)
        assert ci.lower <= 0.5 <= ci.upper

    def test_wilson_zero_n_raises(self):
        with pytest.raises(ValueError):
            wilson_interval(0, 0)

    def test_wilson_bounds_in_unit_interval(self):
        ci = wilson_interval(0, 10)
        assert 0.0 <= ci.lower <= ci.upper <= 1.0

    def test_difference_interval_zero_diff_contains_zero(self):
        ci = difference_interval(50, 1000, 50, 1000)
        assert ci.lower < 0 < ci.upper

    def test_difference_interval_large_effect_excludes_zero(self):
        ci = difference_interval(10, 1000, 200, 1000)
        assert ci.lower > 0


class TestPower:
    def test_sample_size_increases_with_smaller_effect(self):
        n_large = minimum_sample_size(0.10, 0.15)
        n_small = minimum_sample_size(0.10, 0.12)
        assert n_small > n_large

    def test_sample_size_positive(self):
        n = minimum_sample_size(0.10, 0.15)
        assert n > 0

    def test_equal_proportions_raises(self):
        with pytest.raises(ValueError):
            minimum_sample_size(0.10, 0.10)

    def test_achieved_power_increases_with_n(self):
        p1 = achieved_power(0.10, 0.15, n_per_group=500)
        p2 = achieved_power(0.10, 0.15, n_per_group=5000)
        assert p2 > p1

    def test_achieved_power_in_unit_interval(self):
        p = achieved_power(0.10, 0.15, n_per_group=1000)
        assert 0.0 <= p <= 1.0
