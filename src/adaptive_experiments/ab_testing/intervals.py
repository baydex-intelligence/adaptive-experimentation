"""
Confidence interval helpers for binary outcomes.

Wilson's interval is preferred over the normal approximation (Wald) interval
because it has better coverage near p=0 and p=1 and for small samples.
"""

from dataclasses import dataclass

from scipy import stats


@dataclass
class ConfidenceInterval:
    lower: float
    upper: float
    center: float
    level: float


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> ConfidenceInterval:
    """
    Wilson score confidence interval for a single proportion.

    Assumptions: n >= 1.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * (p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5
    return ConfidenceInterval(
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        center=center,
        level=confidence,
    )


def difference_interval(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """
    Normal approximation confidence interval for the difference in proportions
    (treatment - control).

    Assumptions: large samples (n*p >= 5 for both groups).
    """
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment
    diff = p_t - p_c
    se = (p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment) ** 0.5
    margin = z * se
    return ConfidenceInterval(
        lower=diff - margin,
        upper=diff + margin,
        center=diff,
        level=confidence,
    )
