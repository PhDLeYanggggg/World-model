# Completed Native Gain/Harm Experiment

2026-09-21. Local native-arm64 Torch, CPU4/inter-op1/workers0. Training was real
Torch, not a NumPy fallback. Ridge is a separate exact linear control. No CREATE
job was submitted and no fresh remote queue claim is made.

## Actual Run

The code/config/registration were committed as `76849487` before fitting.
Preflight: PID70421/session67362, exit0, 336 bindings, twelve views, 175,756 rows.
The real 100-update pilot (PID70570/session41757) completed in 0.09950 fit seconds.
PID70586/session68689 resumed it from step100 and completed every fixed endpoint;
the pilot was counted once. Training, evaluation (PID70733/session18604) and
checkpoint replay (PID70775/session95432) all exited0. No required process remains
running at this completion record.

Twelve ridge plus24neural heads completed. Neural total:72,000 updates,
18,432,000 draws,22,914 parameters/head, zero unknown-label training draws.
All12paired sampler checks pass. Summed fitting time, including the pilot,
is57.00823seconds. A live snapshot at1m21s showed106.3%CPU and4,539,392KiB RSS.
These are small heads on cached features: this runtime is not the cost of
training all forecasters or the entire world model. There was no slow-run
downgrade or early stopping. Checkpoints every500updates, heartbeats every100.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_gain_harm.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_native_gain_harm.py --resume
.venv-pytorch/bin/python scripts/run_m3w_native_gain_harm.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_native_gain_harm.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_gain_harm_verifier.py tests/test_m3w_native_gain_harm.py tests/test_m3w_native_cost_views.py tests/test_m3w_native_nested.py tests/test_m3w_native_metrics.py tests/test_m3w_cost_validation_lineage.py -q
```

With complete caches, resume verifies all36receipts and performs no new training.
Do not delete completed checkpoints to manufacture a fresh-run label. Initial
evaluation used `--evaluate` only after all36heads completed. `--verify` rebuilds
the train-only preprocessors and replays all2,636,340 repeated cost-score rows
exactly; it does not reopen closed evaluation roles or tune thresholds.

The separate arithmetic verifier (PID/session recorded by the tool execution;
session16030 exit0) checks527,268 forecast rows,720scene reductions and120policy
slices without calling the experiment's ADE/FDE or conditional-report reducers.
It reads bound frozen predictions/labels only for evaluation and records a hash
of the failing query, not public raw trajectory data. Its3unit tests plus related
tests give67passed in1.52s (session28703). The full historic suite was not rerun.

Analysis SHA256:
`f265cc1c28b54bf54af45417c268a5dbf3b72fc52a7fd2399ee22574e32d590d`.
Replay report SHA256:
`f98196afefd9922d1f8fc1d506122eac851b1839a51842f2f199461cb30811d6`.
The independent report binds its own verifier code hash and the above artifacts.
Private arrays/checkpoints/logs stay in
`data/stage_cvpr2027_experiments/native_gain_harm_v1/`; they are not Git artifacts.

## Boundaries

Source inputs/forecasters are cached_verified; cost fitting and inference are
fresh_run. Independent statistical confirmation is not_run. All four source
sites have been explored. Original val/test, main/external and bookstore roles
remain closed. Labels are not human gold. Only annotation-pixel/raw-step claims
are supported. No new policy deploys; Stage5C and SMC remain off.
