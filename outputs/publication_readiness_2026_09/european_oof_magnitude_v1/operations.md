# Operations and Evidence Status

This source-development experiment is registered in commit35a05b87. Native
arm64 Python3.11/Torch CPU4,interop1,workers0. No DataLoader multiprocessing,
resource probing, MPS dependency or NumPy substitute for neural training.
CREATE queue was checked read-only; three unrelated pending jobs were observed.
No CREATE jobs submitted, cancelled or modified. Local access remains sufficient.

## Ordered Commands

Use `.venv-pytorch/bin/python` for all commands below. The runner writes its PID,
heartbeat and event log under the private experiment directory. Training saves
every200 updates and resumes only when identity, inputs and settings match.

1. `scripts/run_m3w_european_oof_magnitude.py --phase support`
2. Commit support before dependent training.
3. `scripts/run_m3w_european_oof_magnitude.py --phase pilot`
4. `scripts/run_m3w_european_oof_magnitude.py --phase train --resume`
5. Commit `inner_prediction_freeze.json` before fitting readouts.
6. `scripts/run_m3w_european_oof_magnitude.py --phase readouts`
7. Commit `prediction_freeze.json` before source-held evaluation.
8. `scripts/run_m3w_european_oof_magnitude.py --phase evaluate`
9. `scripts/report_m3w_european_oof_magnitude.py`
10. `scripts/plot_m3w_european_oof_magnitude.py`
11. `scripts/verify_m3w_european_oof_magnitude.py`

These commands are a recovery sequence, not proof they have completed. Current
phase and execution evidence are recorded in README_RESULTS/research_state.
The support and prediction files enforce ordering. Immutable receipts prevent
silent replacement. Interrupted, incomplete heads use the same final2000-update
budget; completed compressed heads are hash-verified and replayed, not retrained.

## Storage and Git

Private root: `data/stage_cvpr2027_experiments/european_oof_magnitude_v1/`.
Only this run's completed checkpoint is losslessly gzipped and its exact
uncompressed bytes verified before replacing that run's uncompressed file.
Full optimizer,RNG and preprocessing states are retained. Old artifacts remain
untouched. Partial fits keep ordinary atomic checkpoints. Preserve10GiB free.

Use explicit file lists and `git commit --only` to preserve the unrelated staged
work. Never add the data directory, raw inputs, images, checkpoints, scores,
feature caches, virtual environment or large detailed readout. The latter is
locally ignored and hash-bound by final verification. Git carries code, configs,
protocol, manifests, aggregate metrics, lightweight reports and scientific SVG.

## Interpretation

Real fitting, synthetic tests, exact replay and scientific lift are separate.
Even a positive cost gate is not a new trajectory policy, a physical-safety
guarantee or independent confirmation. Obs8/pred12 native annotation steps,
detector pixels only. No metric/seconds,true3D,foundation or human-gold claim.
Independent roles stay closed. Stage5C and SMC remain disabled.
