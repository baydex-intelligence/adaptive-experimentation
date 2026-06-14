# Experiment Runs

Generated run artifacts live here under `experiments/runs/{run_id}/`.

Each evidence-grade run should include:

- `manifest.yaml`
- resolved config or parameter summary
- metrics table
- generated plots
- short notes on prediction, outcome, caveats, and next action

Do not treat a result as a formal claim unless its run directory contains a manifest.

Current note:

- `20260612-060907-many-arm-sweep/` is the evidence-grade many-arm run.
- `20260612-060805-many-arm-sweep/` does not contain a manifest and should not be used
  for formal claims.
