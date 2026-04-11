# Bandit Policy Comparison

## Summary

This report compares four bandit policies on a stationary 4-armed Bernoulli
bandit: a uniform random baseline, epsilon-greedy (ε = 0.1), UCB1, and
Thompson sampling.

## Environment

- Arms: p = [0.1, 0.3, 0.5, 0.7]
- Best arm: arm 3 (p = 0.70)
- Horizon: 1 000 steps
- 200 independent trials per policy

## Policies

### Random
Selects each arm with probability 1/k.  Expected regret grows linearly
at rate `(p_best - p_mean) * T`.

### Epsilon-Greedy (ε = 0.1)
With probability ε, explores uniformly; with probability 1 − ε, exploits the
empirical best arm.  Regret is O(T) because exploration is not decayed.

### UCB1
Selects the arm that maximises `μ̂ + sqrt(2 log t / n)`.  Achieves O(log T)
regret.  No randomness; deterministic given the environment.

### Thompson Sampling
Samples from Beta(α, β) posteriors and selects the arm with the highest
sample.  Achieves O(log T) regret in expectation and typically has lower
regret constants than UCB1.

## Results

<!-- TODO: paste table from run_bandit_comparison.py output here -->
<!-- TODO: embed bandit_comparison_regret.png -->

| Policy                | Total regret | Total reward | Best-arm share |
|:----------------------|-------------:|-------------:|---------------:|
| Random                | ?            | ?            | ~25%           |
| EpsilonGreedy(ε=0.1)  | ?            | ?            | ?              |
| UCB1                  | ?            | ?            | ?              |
| Thompson              | ?            | ?            | ?              |

## Key Takeaways

1. Thompson sampling and UCB1 both achieve sublinear regret; random and
   fixed-ε greedy do not.
2. Thompson sampling tends to have lower empirical regret than UCB1 on
   Bernoulli bandits because the Beta posterior is well-matched to the
   likelihood.
3. Best-arm allocation share is a useful proxy for how quickly each policy
   concentrates on the optimal arm.

## References

- Auer et al. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem."
  *Machine Learning*.
- Thompson (1933). "On the Likelihood that One Unknown Probability Exceeds Another."
  *Biometrika*.
- Chapelle & Li (2011). "An Empirical Evaluation of Thompson Sampling." *NeurIPS*.
