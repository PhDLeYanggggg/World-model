# Native Nested Run and Reproduction

## Material Passport

- Mode: executed experiment and reproduction verification.
- Date: 2026-09-21.
- All required training, inference, replay and export processes exited 0.
- Local arm64 `.venv-pytorch`; no CREATE job or fresh remote-queue claim.

## Actual Execution

Registration commit `2926e917` preceded fitting. The pilot completed as
PID66582/session31479: 100 updates, 1.9043 fit seconds. PID66606/session69788
resumed it and completed 71,900 additional updates across the fixed matrix.
Total: 18 endpoints, 72,000 updates, 1,525.214 summed fitting seconds. The later
live process snapshot at 16m23s showed CPU206.0%, RSS1,830,464KiB and state Rs.
That was a liveness observation, not completion evidence; terminal exit and all
18 receipts subsequently established completion.

Cache construction PID68584/session76075 exited 0 after about 279 seconds between
the input-verified and cache-complete events. Checkpoint replay and independent
scalar verification PID68912/session50744 exited 0. Physical partition export
session5838 and exact export replay session90997 also exited 0. The combined
scoped test session61167 completed 64 tests in 2.04 seconds, without skips.

Native CPU threads4, interop1, workers0; checkpoints every200 updates and
heartbeat every50. No x86 Conda, multiprocessing workers or resource probing.
All fixed training endpoints were complete before any cost-cache inference.
No bound training source/config/registration was modified during execution.

## Commands

Run from the repository root using the native environment:

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_nested.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_native_nested.py --resume
.venv-pytorch/bin/python scripts/run_m3w_native_nested.py --cache
.venv-pytorch/bin/python scripts/run_m3w_native_nested.py --verify --replay
.venv-pytorch/bin/python scripts/export_m3w_native_cost_views.py
.venv-pytorch/bin/python scripts/export_m3w_native_cost_views.py --verify
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_nested.py tests/test_m3w_native_forecast.py tests/test_m3w_native_metrics.py tests/test_m3w_cost_validation_lineage.py tests/test_m3w_native_cost_views.py -q
```

With the completed artifacts retained, `--resume` verifies endpoint receipts and
does not train again. Verify does not construct missing caches. A process lock
prevents concurrent producers. Check the live session and heartbeat before
restarting an interrupted job; never delete checkpoints to manufacture a fresh
run or rewrite identity hashes after changing bound code.

Private outputs are under `data/stage_cvpr2027_experiments/native_nested_v1/`:
trials, predictions, supervision and head_training. Their measured file size is
658,343,931 bytes after export. These files are ignored and are not published.
Only code, configurations, reports and light aggregate metrics are synchronized.

The strict-risk choice record is a later prospective decision, not a mutation
of the frozen producer registration. No risk head, threshold or deployment was
trained or selected in this run. See [conclusions and limitations](conclusions.md).
