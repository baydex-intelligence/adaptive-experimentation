# Production Caveats

## Summary

This report covers practical concerns that arise when moving from textbook
bandit and A/B test algorithms to real-world deployment.  These concerns are
not modelled in the current codebase — they are here to be honest about the
gap between simulation and production.

## Topics (TODO — Milestone 3)

### 1. Delayed Feedback

In many settings rewards are not observed immediately.  A conversion event
may take hours or days.  Feeding stale rewards into a bandit leads to:
- Over-optimism about recently-selected arms (they haven't had time to fail).
- Under-exploration because the posterior hasn't updated yet.

Planned: `simulation/delayed_feedback.py` — simulate arms with configurable
feedback lag and compare regret under no-delay vs. delayed settings.

### 2. Non-Stationarity

Real arm probabilities change over time (seasonality, user cohort drift,
product changes).  Standard UCB1 and Thompson sampling assume stationarity
and can converge to suboptimal arms after a distribution shift.

Planned: sliding-window UCB, discounted Thompson sampling.

### 3. Logging Bias / Counterfactual Evaluation

In a live bandit, arm pulls are not independent of past outcomes — the policy
itself determined the distribution.  Off-policy evaluation requires importance
sampling or doubly-robust estimators.

### 4. Novelty Effects

New variants often receive inflated engagement in the first days of an
experiment simply because they are new.  A bandit that adapts quickly will
exploit the novelty effect and may converge to the wrong arm.

### 5. Network Effects and SUTVA Violations

Standard A/B testing and bandits assume the stable unit treatment value
assumption (SUTVA): one user's treatment does not affect another user's
outcome.  Social features, inventory constraints, and shared infrastructure
all violate this.

### 6. Multiple Comparisons in Practice

Running many experiments simultaneously (common in large companies) increases
the family-wise error rate.  Planned: brief discussion of correction methods
(Bonferroni, Benjamini-Hochberg) and why they are hard to apply in bandit
settings.

## References

<!-- TODO: add relevant papers as each topic is implemented -->
