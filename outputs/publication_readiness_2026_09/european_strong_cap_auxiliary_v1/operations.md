# Running and Recovering the Strong-Base Study

Run from the repository root with native arm64 `.venv-pytorch/bin/python`.
CPU threads4, interop1, DataLoader workers0. CREATE is checked read-only;
local execution is used subject to the real pilot and10GiB free-disk reserve.
No expensive resource probing or multiprocessing is used.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_strong_cap_auxiliary.py --phase register
# Commit registration before support.
.venv-pytorch/bin/python scripts/run_m3w_european_strong_cap_auxiliary.py --phase support
# Commit support before fitting.
.venv-pytorch/bin/python scripts/run_m3w_european_strong_cap_auxiliary.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_strong_cap_auxiliary.py --phase train --resume
# Commit all prediction hashes before held source outcomes are read.
.venv-pytorch/bin/python scripts/run_m3w_european_strong_cap_auxiliary.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_strong_cap_auxiliary.py
.venv-pytorch/bin/python scripts/plot_m3w_european_strong_cap_auxiliary.py
.venv-pytorch/bin/python scripts/verify_m3w_european_strong_cap_auxiliary.py
```

The private experiment directory is
`data/stage_cvpr2027_experiments/european_strong_cap_auxiliary_v1/`.
It contains PID-bearing heartbeat/events, model checkpoints, optimizer and
sampler states, predictions and replay logs. Checkpoint every200updates;
interruption resumes the last atomic checkpoint, not an invented final state.
An exclusive lock rejects concurrent writers. Never restart an active process
merely because it is slow. Inspect PID, heartbeat, disk and latest checkpoint.

Registration and artifacts are immutable and hash-bound. Resume rejects
changed inputs, initialization, auxiliary labels, preprocessing or settings.
Completed heads are re-inferred and compared against saved predictions.
Original-control reconstruction failure stops that view before auxiliary
training. Diagnose the implementation instead of tuning to held outcomes.

Only code, configuration, reports and aggregate metrics belong in Git.
Do not upload data, predictions, checkpoints, caches or the Python environment.
Verification is checkpoint inference/result replay, not independent retraining.
Read `verification.json` for the actual scoped tests; the full legacy suite
is not implicitly covered.
