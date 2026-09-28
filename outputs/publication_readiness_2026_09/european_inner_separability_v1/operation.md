# Run and Recovery Guide

Use the native arm64 `.venv-pytorch` environment from the repository root.
CPU threads4, interop1, DataLoader workers0. The runtime rejects Rosetta before
importing Torch. This experiment uses real Torch training, not a NumPy substitute.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_inner_separability.py register
# Commit the registration before fitting.
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_inner_separability.py pilot
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_inner_separability.py train --resume
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_inner_separability.py replay_fit
# Commit training_freeze.json before making directional decisions.
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_inner_separability.py decide --resume
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_inner_separability.py replay_decide
# Commit decision_freeze.json before reading directional outcomes.
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_inner_separability.py evaluate
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_inner_separability.py replay_evaluate
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_inner_separability.py
```

Registration, complete receipts and outcomes are immutable. The commands above
document execution order, not permission to overwrite completed artifacts.
Training resume validates the registered identity and continues saved weights,
optimizer and sampler state. The paired pilot is resumed into the full run.
`replay_fit` independently retrains only the first full pair; all216 action views
and the numerical readout are replayed. Do not describe this as144 fresh replays.

Heartbeat, PID, append-only events, private checkpoints and action masks are in
`data/stage_cvpr2027_experiments/european_inner_separability_v1/`. A live heartbeat
and advancing step count distinguish slow work from an observed stall. Preserve
completed work after interruptions; do not kill a progressing job because it is
slow. The runner uses an exclusive lock and keeps10GiB free plus temporary margin.

The100-update paired pilot projected5596s for a full run; that is an extrapolation,
not measured total time. Full training completed in252.88s after loading inputs;
checkpoint-local timings and pilot-inclusive totals have separate meanings.
No CREATE job was submitted. A read-only CREATE scheduler check returned no M3W
jobs; unrelated simulation workloads and their project directory were untouched.

Only code, configuration, aggregate evidence and small hash receipts belong in
Git. Never commit raw recordings, row features, action arrays, checkpoints or the
runtime environment. Preserve unrelated staged changes. Full legacy tests,
cold raw reconstruction and independent confirmation are not performed by this
experiment. No Stage5C/SMC or deployment change.
