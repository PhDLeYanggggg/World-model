# Source-Checkpoint Control: Reproduction and Recovery

## Material Passport

Inputs and frozen upstream forecasters: cached_verified against the registered
closure. Training: fresh_run, 72 real Torch runs with 2,000 updates each, including
the first fit's resumed 100-update pilot. Independent source selection,
calibration and confirmation remain closed and not_run. This experiment is not
a new trajectory forecaster or a replacement for the main research objective.

## Execution Sequence

Use native arm64 `.venv-pytorch/bin/python`, CPU4/interOp1, workers0. The entry
script rejects x86_64/Rosetta before importing Torch. All commands run at the
repository root with `PYTHONDONTWRITEBYTECODE=1`.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_checkpoint.py register
.venv-pytorch/bin/python scripts/run_m3w_source_checkpoint.py pilot
.venv-pytorch/bin/python scripts/run_m3w_source_checkpoint.py train --resume
# Verify and commit training_freeze.json before decisions.
.venv-pytorch/bin/python scripts/run_m3w_source_checkpoint.py replay_fit
.venv-pytorch/bin/python scripts/run_m3w_source_checkpoint.py decide
# Verify and commit decision_freeze.json before outcomes.
.venv-pytorch/bin/python scripts/run_m3w_source_checkpoint.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_source_checkpoint.py replay_evaluate
.venv-pytorch/bin/python scripts/report_m3w_source_checkpoint.py
```

These phases enforce immutable existing files, not destructive overwrite. The
first-fit replay command requires its private replay location to be unused;
do not rerun it over an existing replay checkpoint. Preserve the completed replay
receipt instead. Evaluation replay recomputes every action and compares the
entire readout. A new experimental contrast needs a new registered directory,
not edits to sealed code or a renamed old result.

## Recovery and Provenance

- Registration commit: `691ded30`; first pilot/start commit: `1850e761`.
- Training freeze commit: `b4cbde05`; causal decision freeze: `1dba3bb6`.
- Initial full-run process finished the first fit, then the original 10GiB disk
  reserve check stopped the next fit. No checkpoint was removed. Fresh free
  space passed again; resumed PID83671 completed all 72 fits on 29 September.
- `training_freeze.json` measures 194.71 seconds for that resumed invocation,
  not the complete pilot/interrupted-run wall time. Its 144,000 parameter-update
  count is the full experiment's unique training budget, not an elapsed estimate.
- On 1 October, PID68758 replayed the first full fit, and PID68875 froze all 216
  causal views. The replay adds 2,000 verification updates beyond the unique
  training budget. Private events and checkpoints retain timestamps and traces.
- CREATE job37602475 belongs to the separate frozen-boundary diagnosis, not
  this training experiment. Observation timeout is not job failure or permission
  to restart it. No simulation-project paths, credentials or jobs are modified.

Private state is under
`data/stage_cvpr2027_experiments/european_source_checkpoint_v1/`: process lock,
heartbeat, event log, source checkpoints, RNG/optimizer state and validation
curves. Public files contain aggregate metrics and artifact hashes only. Do not
commit checkpoints, caches, trajectories, videos or third-party data.

## Verification Scope

The report verifies all checkpoint hashes, source partitions, optimization-only
preprocessing, best-step choices, validation scores and complete action/readout
replays. Separate arithmetic checks query-balanced signed MSE, per-query matched
counts and all reported error/risk denominators. A successful same-code replay
alone is not treated as independent numerical validation.

`verification.json` and `scoped_tests.txt` record actual completion only after
those checks run. The scoped suite does not establish full legacy-suite success,
cold reconstruction from every raw file, independent confirmation, CREATE
reproduction or scientific success. Those claims require their own evidence.
