# Completed Execution and Reproduction

All runs below use the native arm64 `.venv-pytorch/bin/python`, from the project
root. The real pilot and full continuation share one fixed training budget.

| Action | PID | Execution Session | Observed Result |
| --- | ---: | ---: | --- |
| Registered pilot | 59924 | 11705 | Exit 0; 100 updates, 1.9383 fit seconds, no held scores |
| Resume fixed matrix | 59969 | 12888 | Exit 0; 95,900 additional updates; all 24 endpoints complete |
| Fresh held-source inference | 64339 | 20931 | Exit 0; all 24 models, 175,756 queries per objective/seed roster |
| Cached metric replay | 64583 | 34150 | Exit 0; exact immutable analysis and narrative match |
| Separate reducer and checkpoint replay | See local runtime history | 92334 | Exit 0; 24 checkpoints, 7,752 exact inference replays |
| Scoped integration tests | Not recorded | 98072 | Exit 0; 39 passed, 1.89 seconds |

PIDs are historical launch identifiers, not proof of current liveness. No required
training/evaluation session remains running at completion. The launch was
registered in `c6c25a4d`; `737e37bf` records the real launch before held results.

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_forecast.py --resume
.venv-pytorch/bin/python scripts/run_m3w_native_forecast.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_native_forecast.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_native_forecast.py --replay
.venv-pytorch/bin/python scripts/summarize_m3w_native_forecast_training.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_forecast.py tests/test_m3w_native_metrics.py tests/test_m3w_baseline_relative_forecaster.py tests/test_m3w_context_conditioning.py tests/test_m3w_native_forecast_verification.py -q
```

On the completed workspace, `--resume` verifies completed receipts and does not
add updates. Do not delete outputs to make a replay appear fresh. `--verify`
cannot create missing prediction archives. `--replay` reloads each final model
and rechecks fixed first/middle/last inference batches. The training-only summary
reproduces its JSON, CSV and Markdown exactly without reading held predictions.
That summary was run twice successfully, including its immutable-output check.

For genuinely absent local artifacts, the first command fits the registered
matrix. If a prior run was interrupted, it resumes the last atomic checkpoint
with optimizer and random states. All identity checks must pass; changed source,
settings, masks, row IDs or checkpoints are not silently accepted. The registration
does not authorize editing frozen files and calling the result a matched replay.

Private root: `data/stage_cvpr2027_experiments/native_forecast_v1/`.
It includes final weights, optimizer state, heartbeat, event log, row draws and
held predictions. The existing `data/stage*/` Git ignore covers this entire root.
Only source, registration/config, tests, reports and aggregate/light loss metrics
are eligible for the repository. Existing unrelated staged work is preserved.

The full historical test suite was not run: these are scoped tests for the
changed experiment and its reused prediction/metric components. No MPS/CREATE
fallback was used or implied; the real CPU training cost made local execution
reasonable. No independent calibration, closed-test readout or deployment occurs.
