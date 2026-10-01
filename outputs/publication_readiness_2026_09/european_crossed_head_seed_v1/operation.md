# Reproduction and Recovery

This is a fixed-upstream stochasticity control, not new neural dynamics training.
The 48 new ExtraTrees heads use the same upstream43 assets and source partitions
as the 24 cached original43 heads. No head seed is selected using transfer scores.
Large checkpoints remain local; public files contain hashes and light summaries.

## Ordered Run

Use native arm64 `.venv-pytorch/bin/python` from the repository root, with
`PYTHONDONTWRITEBYTECODE=1`. The runner is
`scripts/run_m3w_crossed_head_seed.py`. Its ordered phases are:

1. `register`, then commit protocol, config, code, tests and registration.
2. `pilot`: a real 16-tree resource trial, resumed into the fixed 128-tree budget.
3. `train --resume`, then commit `training_freeze.json` before transfer.
4. `replay_fit`: repeat the first full fit; compare all trees and predictions.
5. `decide`, then commit `decision_freeze.json` before outcome readout.
6. `evaluate` followed by `replay_evaluate`.
7. `report`: source/artifact seal, scoped tests and numeric summary.

The entry point checks committed phase boundaries. Immutable output mismatch
must fail. Do not edit a registered source file and overwrite this experiment;
use a new version or a separately documented amendment. Exact first-fit replay
does not mean all 48 fits were retrained twice.

## Resume and Resources

CPU threads4, interop1, DataLoader workers0. A single process lock prevents
duplicate fitting. Every16 trees an atomic checkpoint and PID heartbeat are
saved under `data/stage_cvpr2027_experiments/european_crossed_head_seed_v1/`.
Use the same `train --resume` command after interruption. Input, target, weight,
known-mask, preprocessing, library-version and upstream hashes are checked;
completed fits are verified and reused, not retrained under a new identity.

The real pilot used 6,281,412,608-byte peak RSS. All48 full new fits took
372.824561 fitting seconds cumulatively and produced106,973,492 checkpoint bytes.
The original10GiB disk reserve remains in force. No CREATE job was needed for
this bounded local run; previously completed CREATE job37602475 is not resubmitted.

## Interpretation

The readout uses12 previously exposed development localities, obs8/pred12,
stride12 raw frames, image-local detector-silver coordinates. Three head seeds
with upstream43 fixed are not three independent predictor training seeds.
Locality bootstrap intervals are nominal development summaries. Missing labels
and undefined selected-reference risks are not successes. The original2% budget
is unchanged; the moment decomposition is offline only and cannot alter actions.

Independent selection, calibration and confirmation remain closed. No metric,
seconds-level, human-gold, true3D, foundation or physical-safety claim is made.
Stage5C execution and SMC remain off. The full legacy test suite is not claimed;
the report records the exact scoped checks that ran.
