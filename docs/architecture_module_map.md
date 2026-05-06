# Architecture Module Map

## Package: `adaptive_experiments`

Root: `src/adaptive_experiments/`

---

### `ab_testing/`

| Module | Public surface | Responsibility |
|--------|---------------|----------------|
| `effect_sizes.py` | `cohens_h`, `relative_lift`, `absolute_lift` | Effect size calculations for binary outcomes |
| `hypothesis_tests.py` | `two_proportion_ztest` → `TwoProportionResult` | Pooled two-proportion z-test |
| `intervals.py` | `wilson_interval`, `difference_interval` → `ConfidenceInterval` | Confidence intervals for proportions |
| `power.py` | `minimum_sample_size`, `achieved_power` | Sample size and power calculations |
| `peeking.py` | `simulate_peeking`, `simulate_peeking_sweep` → `PeekingSimulationResult` | False-positive inflation simulation |

---

### `bandits/`

| Module | Public surface | Responsibility |
|--------|---------------|----------------|
| `arms.py` | `BernoulliArm` | Single arm with fixed Bernoulli success probability |
| `base.py` | `BanditPolicy` (ABC) | Shared interface: `select_arm`, `update`, `reset` |
| `epsilon_greedy.py` | `EpsilonGreedy(BanditPolicy)` | Fixed-ε explore-exploit policy |
| `random.py` | `RandomPolicy(BanditPolicy)` | Uniform-random baseline policy |
| `ucb.py` | `UCB1(BanditPolicy)` | Upper confidence bound policy |
| `thompson.py` | `ThompsonSampling(BanditPolicy)` | Beta-Bernoulli Thompson sampling |

---

### `simulation/`

| Module | Public surface | Responsibility |
|--------|---------------|----------------|
| `environments.py` | `BernoulliEnvironment` | k-armed Bernoulli bandit environment |
| `evaluation.py` | `make_default_policy_factories`, `default_policy_colors`, `checkpoint_steps`, `best_arm_share_curve`, `late_stage_best_arm_share`, `mean_confidence_interval` | Reusable helper utilities for sweep-style experiment notebooks |
| `runners.py` | `run_trial`, `run_repeated_trials`, `average_cumulative_reward`, `average_cumulative_regret` | Execute policy against environment |
| `scenarios.py` | `easy_two_arm`, `close_two_arm`, `four_arm` | Named environment presets |
| `delayed_feedback.py` | _(stub — Milestone 3)_ | Delayed reward simulation |

---

### `metrics/`

| Module | Public surface | Responsibility |
|--------|---------------|----------------|
| `regret.py` | `cumulative_regret`, `final_regret` | Regret computation from step records |
| `reward.py` | `cumulative_reward`, `total_reward` | Reward computation from step records |
| `allocation.py` | `allocation_shares` | Arm selection frequency |

---

### `plotting/`

| Module | Public surface | Responsibility |
|--------|---------------|----------------|
| `curves.py` | `plot_cumulative_regret`, `plot_cumulative_reward`, `plot_peeking_fpr` | Matplotlib curve helpers |
| `tables.py` | `markdown_table` | Markdown table-string builder for notebook summaries |

---

## Data flow

```
BernoulliEnvironment  ←─────────────────────────────────────┐
        │                                                    │
        ▼                                                    │
  BanditPolicy.select_arm()                                  │
        │                                                    │
        ▼                                                    │
  env.pull(arm) → reward                                     │
        │                                                    │
        ▼                                                    │
  policy.update(arm, reward)                                 │
        │                                                    │
        ▼                                                    │
  StepRecord(t, arm, reward, best_p)  ──→  metrics  ──→  plotting
```

## Dependency rules

- `metrics/` and `plotting/` depend only on `simulation/runners.StepRecord` and numpy/matplotlib.
- `bandits/` depends only on numpy and `bandits/base.py`.
- `simulation/` depends on `bandits/base.py` and `bandits/arms.py`.
- `ab_testing/` depends only on numpy, scipy, and stdlib.
- No module imports from `scripts/` or `tests/`.
