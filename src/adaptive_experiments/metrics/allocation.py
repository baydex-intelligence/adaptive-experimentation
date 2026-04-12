"""
Arm allocation share metric.

Shows what fraction of pulls each arm received over the course of a trial.
An optimal policy should concentrate pulls on the best arm.
"""

from adaptive_experiments.simulation.runners import StepRecord


def allocation_shares(records: list[StepRecord], n_arms: int) -> dict[int, float]:
    """
    Fraction of steps each arm was selected.

    Returns a dict mapping arm index -> share in [0, 1].
    """
    counts = {i: 0 for i in range(n_arms)}
    for r in records:
        counts[r.arm] += 1
    total = len(records)
    return {arm: count / total for arm, count in counts.items()}
