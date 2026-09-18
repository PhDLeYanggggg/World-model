# Microfit Reproduction And Scope

Registration `configs/m3w_source_microfit_v1.json`, frozen in commit`d87803ca`,
SHA256`facb6e5a1a6810f42f24ed69d9ffd458d964ab0e6cde28fb04518a2f8d0b4161`.
Use native arm64`.venv-pytorch/bin/python`. Recorded Torch2.12.0/NumPy2.4.6,
CPU4/inter-op1/workers0. No multiprocessing, MPS probe or x86 Conda is needed.

```sh
.venv-pytorch/bin/python scripts/run_m3w_source_microfit.py --registration configs/m3w_source_microfit_v1.json --audit-only
.venv-pytorch/bin/python scripts/run_m3w_source_microfit.py --registration configs/m3w_source_microfit_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_microfit.py --registration configs/m3w_source_microfit_v1.json --replay
.venv-pytorch/bin/python scripts/verify_analyze_m3w_source_microfit.py --registration configs/m3w_source_microfit_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_microfit.py
```

The same training command resumes optimizer/RNG state; do not run simultaneous
writers. Atomic checkpoints are written every200updates. Runtime PID/heartbeat,
logs, checkpoints, per-row predictions and input hashes live under ignored
`data/stage_cvpr2027_experiments/source_microfit_v1/`. Public outputs contain only
aggregate results, hashes, documentation and an original scientific curve.

The real100-update pilot (PID27103) took1.1522fitseconds and resumed inside the
24,000-update budget. Full run PID27150/session43413 exited0 with all12fits;
summed fit399.4116seconds, full continuation log6.6847minutes. ReplayPID27655/
session74271 completed all12forecasts exactly. Analysis/verification session30349
exited0; its completed-resume process added zero updates and preserved37artifacts.

Both microcohorts are selected from only the source complement of bookstore,
using training labels to isolate feasible trajectories. The larger cohort has
32distinct scoped agent IDs and contains all16nonzero examples from the smaller.
Selection does not read test endpoints or define an official subset. Future
targets are only training labels; c is a single full-training-complement scale.
The observed context bound and parent-coordinate scoring stay unchanged.

The analysis verifies native array alignment, per-row output bounds, six paired
training conditions, finite parameters and full budgets. Tests cover exact
original-decoder equivalence, zero initialization, both output bounds, finite
gradients with missing context, different output Jacobians, exact resume,
read-only completed resume, undefined easy ratios and fixed cohort selection.
The64-test combined regression run includes the preceding source data/model
checks; it is not a full legacy-suite run.

These are training-row memorization checks with2,000passes per row, not a
replacement for the full-source or main benchmark. Seed ranges are not a
generalization confidence interval. No held-source forecasting, sealed-role
access, new deployment, Stage5C orSMC occurs here.
