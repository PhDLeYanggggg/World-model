# Nested EqMotion Execution Record

Registration `fcce08cb` was pushed before real-data fitting. The initial
100-update pilot for coupa/deathCircle seed17 completed in38.49seconds;
6,400 draws cover6,123 training rows, with zero excluded-site draws. This is a
runtime/lineage check, not a completed producer or a measured model gain.
The same fit is resumed, not restarted with an easier budget.

## Initial Live Run, 2026-09-22

The full18-fit invocation was launched as PID23846, terminal session63559.
An observed training heartbeat advanced to update150 after resuming update100.
These identifiers are historical observations; check live state before deciding
a future interruption or restart. The matrix is not complete at this entry.
Expected fitting time from the pilot is roughly7-8hours, plus inference and
verification. Slow progress is not a failure or permission to reduce the matrix.

```bash
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_nested.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_nested.py --trial coupa__deathCircle_seed17 --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_nested.py --resume
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_nested.py --cache
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_nested.py --verify --replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_eqmotion_nested.py tests/test_m3w_native_eqmotion.py tests/test_m3w_native_nested.py tests/test_m3w_native_forecast.py
```

The first three commands have been invoked. Cache and replay are not_run at
this entry.37 targeted tests pass, including real tiny EqMotion training,
bitwise resume and future-label-invariance checks. Tiny tests do not substitute
for any of the18 full registered fits.

CPU4/inter-op1/workers0/nativearm64 matches the preceding successful real
comparison. Checkpoint every200updates, heartbeat every50updates. The runner
holds a file lock. If interrupted, first verify the process is terminal, then
resume the same command; do not edit registered code/config to bypass identity
checks. A pilot checkpoint is saved at its explicit endpoint as well.

Local free space was about58GiB before launch, sufficient for these bounded
checkpoints/predictions. Current SSH configuration has no CREATE alias; no new
remote scheduler state or asset inventory was obtained. Do not infer from that
that CREATE has no jobs or data. Local data/runtime and the measured budget
justify this local run; no unrelated remote hosts were contacted.

`fresh_run`: real pilot and ongoing new fits. `cached_verified`: prior source
inputs,12outer EqMotion models,18matching Transformer samplers and366bound
dependencies. `not_run`: completed18-fit matrix at this entry, cost cache,
cost-head fitting, independent calibration and deployment. No Stage5C or SMC.
