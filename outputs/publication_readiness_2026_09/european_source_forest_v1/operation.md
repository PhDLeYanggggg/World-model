# Reproduction and Runtime

Use native arm64 `.venv-pytorch/bin/python` from the repository root. All
large assets stay under `data/stage_cvpr2027_experiments/`; public records bind
their hashes, not their contents. This is an ExtraTrees estimator control,
not new neural dynamics training and not an independent test result.

## Ordered Run

1. `PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_source_forest.py register`
2. Commit the registration, protocol, config, code and tests before fitting.
3. `PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_source_forest.py pilot`
4. After the recorded resource amendment, use the checked resumable entry:
   `PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/resume_m3w_source_forest_resource_checked.py`
5. Commit `training_freeze.json` and aggregate source-fit status before transfer.
6. `PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_source_forest.py replay_fit`
7. `PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_source_forest.py decide`
8. Commit `decision_freeze.json` before the first outcome readout.
9. `PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_source_forest.py evaluate`
10. `PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_source_forest.py replay_evaluate`
11. `PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_source_forest.py report`

Do not blindly re-register a completed immutable record. A changed code/config
requires a new version. The resource wrapper verifies its amendment and the
original scientific registration; checkpoints retain the original identity.
The initial generic training entry stopped at its conservative preflight and
is not the resumable entry for this resource-amended run.

## Interruption

One process lock, CPU4, interop1, no multiprocessing. Every16 trees an atomic,
compressed checkpoint is written; events and heartbeat record PID and progress.
Resume with the checked resource wrapper. It verifies source identity,
optimization-only preprocessing, data/target/weight hashes, sklearn version,
seed and settings before adding trees. Completed fits are hash-verified and
reused. A changed-input resume must fail, not overwrite earlier work.

No scheduled background monitor was created for this bounded foreground run.
No CREATE job was submitted: the measured native local fit is feasible.

## Evidence Boundaries

Source validation uses only the source's recorded validation recordings. The
screen does not expose target labels to per-row inference. All216 transferred
action hashes must agree between causal freeze and outcome evaluation. Missing
outcomes remain missing; partial labels keep the native estimand. Undefined risk
is not a pass. Independent calibration/confirmation roles stay closed.

Exact first-fit replay checks all trees and predictions, not just a rounded
score. Full transfer replay checks frozen hashes and all readout values. This
supports reproducibility, not independent generalization or calibrated safety.
