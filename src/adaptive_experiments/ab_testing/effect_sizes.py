"""
Effect size measures for binary outcomes.

Cohen's h is the standard effect size for comparing two proportions because it
accounts for the non-linearity of the arc-sine transformation.
"""

import math


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size between two proportions.

    Positive when p1 > p2.
    """
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def relative_lift(p_control: float, p_treatment: float) -> float:
    """
    Relative lift of treatment over control.

    Returns the fractional change: (p_treatment - p_control) / p_control.

    Assumptions: p_control > 0.
    """
    if p_control <= 0:
        raise ValueError("p_control must be positive to compute relative lift.")
    return (p_treatment - p_control) / p_control


def absolute_lift(p_control: float, p_treatment: float) -> float:
    """Absolute difference in conversion rates."""
    return p_treatment - p_control
