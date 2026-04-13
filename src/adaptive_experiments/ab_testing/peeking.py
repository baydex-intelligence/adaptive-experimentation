"""
Peeking simulation for A/B tests.

Demonstrates false-positive inflation when a test is checked repeatedly and
stopped early once p < alpha, even when there is no true effect (H0 is true).

"""

from dataclasses import dataclass

import numpy as np

from adaptive_experiments.ab_testing.hypothesis_tests import two_proportion_ztest


@dataclass
class PeekingSimulationResult:
    """Summary of one peeking simulation run."""

    n_simulations: int
    n_looks: int
    alpha: float
    p_true: float
    n_per_group: int
    false_positive_rate: float
    """Fraction of simulations where H0 was (incorrectly) rejected."""


def simulate_peeking(
    p_true: float = 0.10,
    n_per_group: int = 1000,
    n_looks: int = 10,
    alpha: float = 0.05,
    n_simulations: int = 2000,
    seed: int = 42,
) -> PeekingSimulationResult:
    """
    Simulate the false-positive rate under repeated interim looks.

    Under H0 (both groups have p_true), the test is peeked at n_looks equally
    spaced checkpoints and stopped as soon as p < alpha.  When n_looks == 1
    the false-positive rate should be approximately alpha.  As n_looks grows,
    the false-positive rate inflates substantially above alpha. 

    Additional notes, because you're setting the actually likelihood of success
    as equal for both control and treatment groups, there actually is no difference at 
    all.

    Returns a summary with the empirical false-positive rate.
    """
    rng = np.random.default_rng(seed)
    look_points = [
        int(n_per_group * (i + 1) / n_looks) for i in range(n_looks)
    ]

    false_positives = 0

    for _ in range(n_simulations):
        control = rng.binomial(1, p_true, size=n_per_group)
        treatment = rng.binomial(1, p_true, size=n_per_group)
        rejected = False
        for n in look_points:
            s_c = int(control[:n].sum())
            s_t = int(treatment[:n].sum())
            if s_c == 0 and s_t == 0:
                continue
            result = two_proportion_ztest(s_c, n, s_t, n)
            if result.p_value < alpha:
                rejected = True
                break
        if rejected:
            false_positives += 1

    return PeekingSimulationResult(
        n_simulations=n_simulations,
        n_looks=n_looks,
        alpha=alpha,
        p_true=p_true,
        n_per_group=n_per_group,
        false_positive_rate=false_positives / n_simulations,
    )


def simulate_peeking_sweep(
    looks_range: list[int] | None = None,
    **kwargs,
) -> list[PeekingSimulationResult]:
    """
    Run simulate_peeking for a range of n_looks values.

    Useful for plotting how the false-positive rate grows with more looks.
    """
    if looks_range is None:
        looks_range = [1, 2, 5, 10, 20, 50]
    return [simulate_peeking(n_looks=n, **kwargs) for n in looks_range]
