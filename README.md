# adaptive-experimentation

A clean, minimal Python project for learning and demonstrating **A/B testing** and
**multi-armed bandits** through simulation. Built from first principles with Bernoulli/binary
outcomes as the primary focus.

---

## Why this exists

Experimentation is a core skill in data science and ML engineering. This repo is designed to
build intuition by implementing the key algorithms from scratch, running simulations, and
producing honest write-ups of their failure modes and trade-offs. It is interview-ready in
the sense that every concept is implemented correctly and explained clearly — not in the sense
that it is production-ready software.

---

## A/B testing vs. bandits

Both techniques answer the question "which option is better?" but optimise for different goals:

| | A/B testing | Multi-armed bandits |
|---|---|---|
| **Goal** | Statistical inference — learn the true effect with confidence | Reward maximisation — accumulate as much reward as possible during the experiment |
| **Exploration** | Fixed, pre-determined allocation (e.g. 50/50) | Adaptive — exploit what you know while still exploring |
| **Output** | A p-value / confidence interval / decision | A policy that maps observations to actions |
| **Regret concept** | Not a first-class concern | Central — regret measures the cost of not always playing the best arm |

A/B testing is the right tool when causal inference is the goal and you can afford to wait
for a conclusive test. Bandits are the right tool when you must act continuously and
exploration itself has a cost.

---

## Scope and non-goals

**In scope (this repo):**
- Binary / Bernoulli outcome experiments
- Classical frequentist A/B testing (two-proportion z-test, confidence intervals, power)
- Peeking simulation showing false-positive inflation from repeated looks
- Epsilon-greedy, UCB1, and Thompson sampling bandit policies
- Simulation runner with cumulative reward, regret, and allocation metrics
- Short written case studies in `reports/`

**Out of scope (intentionally):**
- Contextual bandits or reinforcement learning
- Continuous outcome metrics (revenue, session length, etc.)
- Bayesian A/B testing beyond Thompson sampling as a bandit policy
- Production experiment infrastructure (assignment, logging, guardrails)
- Web dashboards, notebooks, or MLflow integration

---

## Repository structure

```
adaptive-experimentation/
├── pyproject.toml                      # Poetry build + dev dependencies
├── README.md
├── .gitignore
├── src/adaptive_experiments/
│   ├── ab_testing/
│   │   ├── effect_sizes.py             # Cohen's h, relative lift
│   │   ├── hypothesis_tests.py         # Two-proportion z-test
│   │   ├── intervals.py                # Wilson confidence intervals
│   │   ├── power.py                    # Sample-size / power calculations
│   │   └── peeking.py                  # Repeated-looks simulation
│   ├── bandits/
│   │   ├── arms.py                     # BernoulliArm
│   │   ├── base.py                     # Policy abstract base class
│   │   ├── epsilon_greedy.py
│   │   ├── ucb.py                      # UCB1
│   │   └── thompson.py                 # Beta-Bernoulli Thompson sampling
│   ├── simulation/
│   │   ├── environments.py             # BernoulliEnvironment
│   │   ├── runners.py                  # run_bandit_trial / run_policy
│   │   ├── scenarios.py                # Named scenario helpers
│   │   └── delayed_feedback.py         # Delayed reward stub (Milestone 3)
│   ├── metrics/
│   │   ├── regret.py                   # Cumulative regret
│   │   ├── reward.py                   # Cumulative reward
│   │   └── allocation.py               # Arm selection share
│   └── plotting/
│       └── curves.py                   # Regret / reward curves
├── scripts/
│   ├── run_peeking_simulation.py
│   └── run_bandit_comparison.py
├── reports/
│   ├── peeking-breaks-ab-tests.md
│   ├── bandit-policy-comparison.md
│   └── production-caveats.md
└── tests/
    ├── test_hypothesis_tests.py
    ├── test_peeking.py
    ├── test_thompson.py
    ├── test_ucb.py
    └── test_simulation_runner.py
```

---

## Milestone plan

### Milestone 1 — A/B testing fundamentals and peeking simulation ✅
- Two-proportion z-test with p-value and confidence interval
- Effect size (Cohen's h) and relative lift
- Minimum sample size helper
- Peeking simulation: shows that checking results early and stopping when p < 0.05
  inflates the false-positive rate well above the nominal α
- Script: `scripts/run_peeking_simulation.py`
- Report: `reports/peeking-breaks-ab-tests.md`

### Milestone 2 — Core bandit algorithms and comparison ✅
- `BernoulliArm` abstraction with explicit random seed
- Policy base class (abstract `select_arm` / `update`)
- Epsilon-greedy, UCB1, Thompson sampling
- Simulation runner with cumulative reward, regret, and allocation share
- Script: `scripts/run_bandit_comparison.py`
- Report: `reports/bandit-policy-comparison.md`

### Milestone 3 — Delayed feedback, non-stationarity, production caveats (TODO)
- Delayed reward simulation: arms whose feedback arrives with a lag
- Non-stationary arm: reward probability drifts over time
- Discussion of production concerns: logging bias, novelty effects, network effects
- Report: `reports/production-caveats.md`

---

## Getting started

```bash
# Install dependencies with Poetry
poetry install

# Run all tests
poetry run pytest

# Run peeking simulation
poetry run python scripts/run_peeking_simulation.py

# Run bandit comparison
poetry run python scripts/run_bandit_comparison.py
```

---

## Interview-ready framing

When talking about this project in an interview:

- **Peeking**: "I simulated the false-positive inflation that results from checking an A/B test
  repeatedly and stopping when the p-value crosses α. The effect is severe — at 20 looks,
  the empirical false-positive rate can exceed 20% even at a nominal 5% level."

- **Regret**: "I implemented cumulative regret as `T * p_best - sum(rewards)`. Thompson sampling
  achieves sublinear regret empirically, while epsilon-greedy with a fixed ε does not."

- **Trade-offs**: "A/B testing gives you a clean causal claim backed by a pre-registered
  hypothesis. Bandits minimize regret but make inference harder because allocation is
  adaptive and not independent of outcomes."

