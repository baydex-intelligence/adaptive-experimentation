# Peeking Breaks A/B Tests

## Summary

This report demonstrates that repeatedly checking an A/B test and stopping
once the p-value crosses α inflates the false-positive rate (type I error)
well above the nominal level.

## Background

A standard A/B test is designed with a fixed sample size and a single
analysis at the end.  The nominal significance level α = 0.05 means that, if
H₀ is true, we expect to reject it in 5% of experiments.

When you peek at interim results and stop early, you are effectively running
multiple hypothesis tests on the same data.  Each additional look gives the
p-value more chances to dip below α by chance alone.

## Simulation Setup

- Both groups drawn from Bernoulli(p = 0.10) — no true effect.
- Per-group sample size: 1 000.
- Looks evenly spaced; stop as soon as p < 0.05.
- 3 000 independent simulations per configuration.

## Results

<!-- TODO: paste table from run_peeking_simulation.py output here -->
<!-- TODO: embed peeking_fpr.png -->

| Looks | Empirical FPR | Inflation vs. single look |
|------:|:-------------:|:-------------------------:|
| 1     | ~0.05         | 1.0×                      |
| 2     | ?             | ?                         |
| 5     | ?             | ?                         |
| 10    | ?             | ?                         |
| 20    | ?             | ?                         |
| 50    | ?             | ?                         |

## Key Takeaways

1. Even two looks can noticeably inflate the false-positive rate.
2. By 20 looks, the FPR can exceed 20–25% at a nominal 5% level.
3. Remedies: sequential testing (e.g. SPRT), alpha spending functions
   (e.g. O'Brien-Fleming), or pre-committing to a fixed sample size and
   not peeking until the end.

## References

- Johari et al. (2017). "Peeking at A/B Tests: Why It Matters, and What to Do About It."
  *ACM KDD*.
- Wald (1947). *Sequential Analysis*.
