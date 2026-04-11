"""
Two-proportion z-test for binary outcomes.

Implements the pooled-proportion z-test, which is the standard frequentist
hypothesis test for comparing two Bernoulli success rates.
"""

import math
from dataclasses import dataclass

from scipy import stats


@dataclass
class TwoProportionResult:
    """Result of a two-proportion z-test."""

    z_stat: float
    p_value: float
    p_control: float
    p_treatment: float
    n_control: int
    n_treatment: int


def two_proportion_ztest(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    alternative: str = "two-sided",
) -> TwoProportionResult:
    """
    Pooled two-proportion z-test.

    Assumptions: both n_control and n_treatment are large enough for the
    normal approximation (rule of thumb: n*p >= 5 and n*(1-p) >= 5 for both groups).

    alternative: 'two-sided', 'greater' (treatment > control), or 'less'.
    """
    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment
    p_pool = (successes_control + successes_treatment) / (n_control + n_treatment)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    if se == 0:
        z = 0.0
    else:
        z = (p_t - p_c) / se

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z)
    elif alternative == "less":
        p_value = stats.norm.cdf(z)
    else:
        raise ValueError(f"Unknown alternative '{alternative}'.")

    return TwoProportionResult(
        z_stat=z,
        p_value=p_value,
        p_control=p_c,
        p_treatment=p_t,
        n_control=n_control,
        n_treatment=n_treatment,
    )
