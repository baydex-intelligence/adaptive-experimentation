"""
Sample size and statistical power calculations for binary outcomes.

Uses the standard normal approximation for two-proportion tests.
"""

import math

from scipy import stats


def minimum_sample_size(
    p_control: float,
    p_treatment: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """
    Minimum per-group sample size for a two-proportion z-test.

    Returns the sample size needed in each group (equal allocation assumed).

    Assumptions: normal approximation is valid (both groups large).
    """
    if not (0 < p_control < 1 and 0 < p_treatment < 1):
        raise ValueError("Proportions must be strictly between 0 and 1.")
    if p_control == p_treatment:
        raise ValueError("p_control and p_treatment must differ.")

    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_beta = stats.norm.ppf(power)

    p_bar = (p_control + p_treatment) / 2
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(
            p_control * (1 - p_control) + p_treatment * (1 - p_treatment)
        )
    ) ** 2
    denominator = (p_treatment - p_control) ** 2
    return math.ceil(numerator / denominator)


def achieved_power(
    p_control: float,
    p_treatment: float,
    n_per_group: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """
    Statistical power for a two-proportion z-test at a given sample size.
    """
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    p_bar = (p_control + p_treatment) / 2
    se_null = math.sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
    se_alt = math.sqrt(
        (p_control * (1 - p_control) + p_treatment * (1 - p_treatment)) / n_per_group
    )
    effect = abs(p_treatment - p_control)
    z_beta = (effect - z_alpha * se_null) / se_alt
    return float(stats.norm.cdf(z_beta))
