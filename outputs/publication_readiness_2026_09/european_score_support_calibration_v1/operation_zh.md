# Source-Separated Calibration: Operation Guide

This experiment does not retrain the forecast or risk networks. Frozen native
Torch checkpoints are verified by their prior seals, then rerun for causal
inference. Fresh score/decision/readout artifacts are distinct from reused weights.

Run from the repository root with native arm64 `.venv-pytorch/bin/python`.
Runtime is CPU4/interop1, workers0, no resource probing. Private artifacts and
checkpoints stay under ignored `data/stage_cvpr2027_experiments/`. Preserve10GiB.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase infer --resume
# Commit score_freeze.json before calibration.
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase calibrate --resume
# Commit decision_freeze.json before held scoring.
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase replay_infer
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase replay_calibrate
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase replay_evaluate
.venv-pytorch/bin/python scripts/report_m3w_score_support_calibration.py
.venv-pytorch/bin/python scripts/plot_m3w_score_support_calibration.py
```

`heartbeat.json` and `events.jsonl` record PID, phase and group progress. Resume
verifies already complete groups; an interrupted group is recomputed without
refitting its frozen networks. A lock prevents concurrent writers. Do not delete
locks or restart from a stale timestamp alone; first check the recorded PID.
Immutable receipts reject changed code, checkpoints, scores or outcomes.

The first registration collided with three historical filenames. Before any
inference, all three were restored byte-for-byte to commit15768a5d. The new study
uses the distinct `score_support_calibration` namespace. The old producer heartbeat
was restored from its last original terminal event; the two registration events
remain in its append-only event log for transparency. No old checkpoint, prediction
or numerical result was changed.

This guide is not a cold-download or anonymous reproduction claim. The frozen
private banks and their full parent lineage must exist. No independent role is
opened, no HPC job is submitted, and no Stage5C/SMC execution is allowed.
