"""
Run peeking simulation and print a summary showing false-positive rate inflation.

Peeking simulations demonstrates the problem of relying on product teams to determine when
to stop an experiment instead of following the pre-deermined stopping rules that are much better
at guranteeing the significance of results. 

In essence, a peeking simulation is a monte-carlo audit of a particular policy. 

Usage:
    poetry run python scripts/run_peeking_simulation.py
"""

import sys
from pathlib import Path

# Allow running without installing when invoked directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt

from adaptive_experiments.ab_testing.peeking import simulate_peeking_sweep
from adaptive_experiments.plotting.curves import plot_peeking_fpr

LOOKS_RANGE = [1, 2, 5, 10, 20, 50]
N_PER_GROUP = 1000
N_SIMULATIONS = 3000
ALPHA = 0.05
SEED = 42
OUTPUT_DIR = Path(__file__).parent.parent / "reports"


def main() -> None:
    print("Running peeking simulation …")
    results = simulate_peeking_sweep(
        looks_range=LOOKS_RANGE,
        p_true=0.10,
        n_per_group=N_PER_GROUP,
        alpha=ALPHA,
        n_simulations=N_SIMULATIONS,
        seed=SEED,
    )

    print(f"\n{'Looks':>6}  {'FPR':>8}  {'Inflation':>10}")
    print("-" * 30)
    baseline = results[0].false_positive_rate
    for r in results:
        inflation = r.false_positive_rate / baseline if baseline > 0 else float("nan")
        print(f"{r.n_looks:>6}  {r.false_positive_rate:>8.3f}  {inflation:>10.2f}x")

    looks = [r.n_looks for r in results]
    fpr = [r.false_positive_rate for r in results]

    ax = plot_peeking_fpr(looks, fpr, alpha=ALPHA)
    ax.figure.tight_layout()
    output_path = OUTPUT_DIR / "peeking_fpr.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=120)
    plt.close("all")
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
