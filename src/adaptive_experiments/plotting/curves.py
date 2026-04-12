"""
Plotting helpers for bandit simulation results.

All functions accept pre-computed numpy arrays so they are decoupled from
the simulation runner.  Call plt.show() or plt.savefig() after calling these.
"""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_cumulative_regret(
    curves: dict[str, np.ndarray],
    title: str = "Cumulative Regret",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot one or more cumulative regret curves on the same axes.

    curves: mapping of policy label -> array of shape (n_steps,).
    """
    if ax is None:
        _, ax = plt.subplots()
    for label, curve in curves.items():
        ax.plot(curve, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative regret")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_cumulative_reward(
    curves: dict[str, np.ndarray],
    title: str = "Cumulative Reward",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot one or more cumulative reward curves on the same axes.
    """
    if ax is None:
        _, ax = plt.subplots()
    for label, curve in curves.items():
        ax.plot(curve, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_peeking_fpr(
    looks: Sequence[int],
    fpr: Sequence[float],
    alpha: float = 0.05,
    title: str = "False-Positive Rate vs. Number of Looks",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot empirical false-positive rate as a function of number of interim looks.

    Draws a horizontal dashed reference line at the nominal alpha level.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(looks, fpr, marker="o", label="Empirical FPR")
    ax.axhline(alpha, linestyle="--", color="red", label=f"Nominal α = {alpha}")
    ax.set_xlabel("Number of interim looks")
    ax.set_ylabel("False-positive rate")
    ax.set_title(title)
    ax.legend()
    return ax
