# Actual-Selected-Set Calibration

## Question

Does refitting component calibration on the retained source population improve
whole-recording held-out risk without deleting useful interventions? The
[registered protocol](protocol.md) defines a source-only control, not a replacement
world model. [Prior negative results](asset_and_method_note.md) explain why another
fixed-subset loss or reference-head freeze is not being repeated.

## Current Execution

- Registration/code freeze: `34ae41b6`, pushed before source recalibration.
- Engineering checks:40 scoped tests passed on native arm64.
- Source input packets:72,38,137,774bytes; remote byte hashes verified.
- Local numerical array cache: none added;10GiB reserve unchanged.
- CREATE job:37714473,4CPU/8GiB/1hour, isolated M3W runtime.
- First scheduler observation: PENDING, not a model result or failure.
- New neural training: not_run; this experiment fits empirical calibrators only.
- Transfer/independent-role evaluation: not_run and deliberately excluded.
- Deployment: unchanged. Stage5C and SMC off.

## Run and Inspect

Use the native `.venv-pytorch` environment from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_selected_set_calibration.py inspect
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest -p no:cacheprovider tests/test_m3w_selected_set_calibration.py tests/test_m3w_selected_set_runner.py tests/test_m3w_component_calibration.py tests/test_m3w_unknown_outcome_bounds.py tests/test_m3w_selected_pool_accounting.py -q
```

Do not submit another job for this frozen run. The owned remote directory is
`/users/k24101830/m3w/european_selected_set_calibration_v1`. Per-head calibration
records live in `groups/`, progress in `heartbeat.json`, and terminal evidence in
`complete.json` and `summary.json`. The scheduler state alone is insufficient:
verify72groups, exact second-pass replay, parent OOF hashes and completion-screen
parity, then all remote file hashes. Full numerical outputs stay remote while
local disk is below reserve.

The allocated-node runner supports `--resume` only with the same frozen code,
config, input hashes and output directory. A failed/interrupted job must first be
confirmed terminal and its artifacts inspected before a separately recorded
recovery submission. Do not overwrite differing partial results, resubmit after
an uncertain submission, or run numerical work on the login node. Existing
simulation jobs are unrelated and must not be modified.

## Interpretation Rules

This is fresh source-only calibration on cached_verified forecasts from exposed
development localities. OOF is held-recording, not a fresh independent dataset.
Reference denominator, original2%budget and known/unknown-label semantics do not
change. Empty/undefined selection does not pass safety. Utility reported relative
to the full known reference mass is a separate descriptive quantity, never a
replacement risk denominator. Resubstitution cannot select the method or justify
deployment. No metric, seconds, human-gold, true3D, foundation or physical-safety
claim follows from a successful job.
